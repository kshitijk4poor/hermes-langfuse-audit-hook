from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_PENDING: dict[str, dict[str, Any]] = {}
STATE_DIR = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "langfuse_audit"
STATE_PATH = STATE_DIR / "state.json"
USER_DOC = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "optimizations.md"
AUDIT_EVERY_TURNS = 10
MAX_WINDOW = 10

# ── Patterns for detecting issues in tool results ────────────────────────────

_ERROR_PATTERNS = re.compile(
    r"Error executing|Traceback|FileNotFoundError|PermissionError|"
    r"No such file|command not found|exit_code[\":\s]+[1-9]|"
    r"\"error\":\s*\"|failed to|ENOENT|EACCES|not found|"
    r"SyntaxError|ModuleNotFoundError|ImportError",
    re.IGNORECASE,
)

_EMPTY_RESULT_PATTERNS = re.compile(
    r'^(\s*\{\s*"output"\s*:\s*""\s*\}|\s*|\s*null\s*|\{\s*\})$',
)

# Truncation limit for stored args/results (keeps state.json manageable)
_MAX_STORED_CHARS = 300


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _truncate(value: Any, limit: int = _MAX_STORED_CHARS) -> str:
    s = str(value) if value is not None else ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[{len(s) - limit} more]"


def _detect_error(result: str) -> str | None:
    """Return a short error label if the tool result looks like a failure."""
    if not isinstance(result, str):
        return None
    if _ERROR_PATTERNS.search(result[:2000]):
        # Try to extract a one-liner
        for line in result[:1000].splitlines():
            line = line.strip()
            if any(kw in line.lower() for kw in ("error", "traceback", "not found", "permission", "failed")):
                return line[:150]
        return "error detected"
    return None


def _detect_empty(result: str) -> bool:
    if not isinstance(result, str):
        return result is None
    return bool(_EMPTY_RESULT_PATTERNS.match(result.strip())) or len(result.strip()) == 0


def _extract_tool_args_summary(tool_name: str, args: Any) -> str | None:
    """Extract the most useful arg for each tool type."""
    if not isinstance(args, dict):
        return None
    if tool_name == "terminal":
        return _truncate(args.get("command", ""), 200)
    if tool_name in ("read_file", "write_file"):
        return args.get("path", "")
    if tool_name == "search_files":
        return f"pattern={args.get('pattern', '')} path={args.get('path', '.')}"
    if tool_name == "patch":
        return args.get("path", "") or "(patch mode)"
    if tool_name == "browser_navigate":
        return args.get("url", "")
    if tool_name in ("web_search", "web_extract"):
        return args.get("query", "") or args.get("url", "")
    if tool_name == "delegate_task":
        return _truncate(args.get("goal", ""), 150)
    return None


# Leading shell tokens that carry no signal about what the command does.
# `cd` overwhelmingly appears as the first segment of a compound command
# (`cd /path && gh ...`), so a naive first-token prefix reports `cd` for
# essentially every terminal call and the finding becomes noise.
# `_DROP_ARG_TOKENS` consume the token AND its argument (e.g. `cd <dir>`,
# `source <file>`); `_SEGMENT_NOISE_TOKENS` mark a whole simple-command
# segment as noise (the builtin consumes the rest of the segment as args,
# e.g. `set -euo pipefail`, `export PATH=...`).
_DROP_ARG_TOKENS = {"cd", "pushd", "source", "."}
_SEGMENT_NOISE_TOKENS = {"builtin", "command", "export", "set"}


def _meaningful_command_prefix(cmd: str, max_len: int = 40) -> str:
    """Return the meaningful head token of a (possibly compound) shell command.

    Strips leading `cd <dir> &&` style prologues, shell variable assignments
    (`FOO=bar cmd`), `export`/`set`/`source` noise, and leading comment
    lines, then returns the first remaining token (truncated). Falls back
    to "" for noise-only commands (e.g. a bare `cd`) so they are dropped
    from the pattern counter.
    """
    if not isinstance(cmd, str):
        return ""
    # Skip comment lines / blank lines entirely.
    lines = [ln.strip() for ln in cmd.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return ""
    # Walk segments in order; each segment is one simple command.
    for ln in lines:
        for segment in re.split(r"&&|;", ln):
            tokens = segment.split()
            if not tokens:
                continue
            head = tokens[0]
            if head in _SEGMENT_NOISE_TOKENS:
                # `set -euo pipefail`, `export PATH=...` — the builtin
                # consumes the rest of the segment as its arguments.
                continue
            while tokens:
                head = tokens.pop(0)
                if head in _DROP_ARG_TOKENS:
                    # Drop the token AND its argument (e.g. `cd <dir>`).
                    if tokens:
                        tokens.pop(0)
                    continue
                if "=" in head and not head.startswith("-") and head.split("=")[0].replace("_", "").isalnum():
                    # Environment assignment prefix (`FOO=bar cmd`).
                    continue
                return head[:max_len]
    return ""


# ── State management ─────────────────────────────────────────────────────────

def _ensure_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"total_turns": 0, "audits": 0, "history": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"total_turns": 0, "audits": 0, "history": []}


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _append_doc(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = "# Optimization Audit Log\n\nAuto-appended findings from the Langfuse audit hook.\n"
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + "\n" + block + "\n", encoding="utf-8")


# ── Analysis engine ──────────────────────────────────────────────────────────

def _analyze_window(window: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Analyze the recent turn window and return (findings, metrics)."""
    metrics: dict[str, Any] = {}
    if not window:
        return ["No turns recorded in the current audit window."], metrics

    tool_counter: Counter[str] = Counter()
    error_details: list[dict[str, str]] = []
    empty_results: list[str] = []
    file_access: Counter[str] = Counter()  # path → count
    command_patterns: list[str] = []
    tool_sequences: list[list[str]] = []
    request_count_total = 0
    total_latency = 0.0
    slow_requests = 0
    finish_reasons: Counter[str] = Counter()
    total_result_bytes = 0
    large_results: list[dict[str, Any]] = []

    for turn in window:
        tools = turn.get("tools", [])
        requests = turn.get("requests", [])
        turn_tool_names = []

        for t in tools:
            if not isinstance(t, dict):
                continue
            name = t.get("tool_name", "unknown")
            tool_counter[name] += 1
            turn_tool_names.append(name)

            # Track errors
            err = t.get("error")
            if err:
                error_details.append({"tool": name, "error": err, "args": t.get("args_summary", "")})

            # Track empty results
            if t.get("empty_result"):
                empty_results.append(f"{name}({t.get('args_summary', '')})")

            # Track file access patterns
            args_summary = t.get("args_summary", "")
            if name in ("read_file", "write_file", "patch") and args_summary:
                file_access[args_summary] += 1

            # Track terminal commands
            if name == "terminal" and args_summary:
                command_patterns.append(args_summary)

            # Track result sizes
            size = t.get("result_size") or 0
            total_result_bytes += size
            if size > 50_000:
                large_results.append({"tool": name, "size": size, "args": args_summary})

        tool_sequences.append(turn_tool_names)
        request_count_total += len(requests)

        for req in requests:
            dur = float(req.get("api_duration") or 0.0)
            total_latency += dur
            if dur >= 10:
                slow_requests += 1
            finish_reasons[str(req.get("finish_reason") or "unknown")] += 1

    # ── Compute metrics ──────────────────────────────────────────────────
    total_tools = sum(tool_counter.values())
    avg_tools = total_tools / len(window)
    avg_requests = request_count_total / len(window)
    avg_latency = total_latency / max(request_count_total, 1)

    # Consecutive same-tool streaks across the full window
    all_tool_names = [name for seq in tool_sequences for name in seq]
    max_streak = 0
    streak_tool = ""
    if all_tool_names:
        current_streak = 1
        for i in range(1, len(all_tool_names)):
            if all_tool_names[i] == all_tool_names[i - 1]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
                    streak_tool = all_tool_names[i]
            else:
                current_streak = 1

    # Repeated file reads (same file read 3+ times)
    re_read_files = [(f, c) for f, c in file_access.items() if c >= 3]

    metrics.update({
        "turns": len(window),
        "tool_calls_total": total_tools,
        "request_count_total": request_count_total,
        "avg_tools_per_turn": round(avg_tools, 2),
        "avg_requests_per_turn": round(avg_requests, 2),
        "avg_request_latency_sec": round(avg_latency, 2),
        "slow_requests": slow_requests,
        "finish_reasons": dict(finish_reasons),
        "top_tools": tool_counter.most_common(5),
        "error_count": len(error_details),
        "empty_result_count": len(empty_results),
        "max_streak": max_streak,
        "streak_tool": streak_tool,
        "total_result_bytes": total_result_bytes,
        "sessions": sorted({t.get("session_id") for t in window if t.get("session_id")}),
    })

    # ── Generate findings ────────────────────────────────────────────────
    findings: list[str] = []

    # 1. Errors — most actionable
    if error_details:
        findings.append(f"**{len(error_details)} tool error(s) detected:**")
        for ed in error_details[:5]:
            findings.append(f"  - `{ed['tool']}({ed['args']})` → {ed['error']}")
        if len(error_details) > 5:
            findings.append(f"  - ...and {len(error_details) - 5} more")

    # 2. Empty/useless results
    if empty_results:
        findings.append(
            f"**{len(empty_results)} tool call(s) returned empty results:** "
            + ", ".join(f"`{e}`" for e in empty_results[:5])
            + (f" ...and {len(empty_results) - 5} more" if len(empty_results) > 5 else "")
        )

    # 3. Repeated file access
    if re_read_files:
        findings.append("**Same file read/written 3+ times** (consider caching or reading once with execute_code):")
        for path, count in sorted(re_read_files, key=lambda x: -x[1])[:5]:
            findings.append(f"  - `{path}` — {count}×")

    # 4. Terminal command patterns — detect repeated similar commands
    if len(command_patterns) >= 3:
        # Look for commands that share a common prefix (e.g. "grep" repeated).
        # Agents overwhelmingly run compound shell (`cd /path && gh ...`), so
        # taking the first whitespace token yields the useless prefix `cd`
        # almost every time. Strip leading `cd <dir> &&` / env prefixes first
        # so the reported prefix is the meaningful head of the command.
        cmd_prefixes: Counter[str] = Counter()
        for cmd in command_patterns:
            prefix = _meaningful_command_prefix(cmd)
            if prefix:
                cmd_prefixes[prefix] += 1
        repeated_cmds = [(p, c) for p, c in cmd_prefixes.items() if c >= 3]
        if repeated_cmds:
            findings.append("**Repeated terminal command patterns** (consider combining with `&&` or `execute_code`):")
            for prefix, count in sorted(repeated_cmds, key=lambda x: -x[1])[:3]:
                findings.append(f"  - `{prefix}` — {count}× invocations")

    # 5. Consecutive same-tool streaks
    if max_streak >= 3:
        findings.append(
            f"**Longest consecutive same-tool streak:** `{streak_tool}` × {max_streak}. "
            "Batch these into fewer calls."
        )

    # 6. Large results bloating context
    if large_results:
        findings.append(f"**{len(large_results)} tool result(s) over 50KB** (bloats context, increases latency):")
        for lr in large_results[:3]:
            findings.append(f"  - `{lr['tool']}({lr['args']})` — {lr['size']:,} bytes")

    # 7. High latency
    if avg_latency > 8.0:
        findings.append(
            f"**Avg LLM latency: {avg_latency:.1f}s** (elevated). "
            f"Total result bytes in window: {total_result_bytes:,}. "
            "Large tool outputs may be inflating prompt size."
        )

    # 8. Finish reason issues
    length_truncations = finish_reasons.get("length", 0)
    if length_truncations:
        findings.append(
            f"**{length_truncations} response(s) truncated by length.** "
            "Model ran out of output tokens — consider earlier compression or shorter intermediate outputs."
        )

    # 9. High iteration count
    if avg_requests > 3.0:
        findings.append(
            f"**High avg iterations/turn: {avg_requests:.1f}.** "
            "The model is making many round trips — check if planning or tool hints can reduce loops."
        )

    # 10. Tool distribution summary
    if tool_counter:
        tool_summary = ", ".join(f"`{name}`×{count}" for name, count in tool_counter.most_common(5))
        findings.append(f"Top tools: {tool_summary}.")

    if not findings:
        findings.append("No issues detected in the last 10 turns — behavior looks stable.")

    return findings, metrics


# ── Rendering ────────────────────────────────────────────────────────────────

def _render_block(window: list[dict[str, Any]], findings: list[str], metrics: dict[str, Any], audit_no: int) -> str:
    lines = [
        f"## Audit {audit_no} — {_now_iso()}",
        "",
        f"Window: {metrics.get('turns', 0)} turns | "
        f"{metrics.get('tool_calls_total', 0)} tool calls | "
        f"{metrics.get('request_count_total', 0)} LLM requests | "
        f"avg latency {metrics.get('avg_request_latency_sec', 0)}s",
        "",
        "### Findings",
    ]
    lines.extend(f"- {item}" for item in findings)

    # Compact turn log
    lines.extend(["", "### Turn log"])
    for turn in window:
        tools_list = turn.get("tools", [])
        tool_strs = []
        for t in tools_list[:6]:
            name = t.get("tool_name", "?")
            err_marker = " ✗" if t.get("error") else ""
            tool_strs.append(f"{name}{err_marker}")
        tools = ", ".join(tool_strs) or "none"
        if len(tools_list) > 6:
            tools += f" +{len(tools_list) - 6} more"
        lines.append(
            f"- {turn.get('timestamp', '?')} | "
            f"session={turn.get('session_id', '')[:20]} | "
            f"requests={len(turn.get('requests', []))} | "
            f"tools=[{tools}]"
        )
    return "\n".join(lines)


def _emit_highlight(findings: list[str], metrics: dict[str, Any], audit_no: int) -> None:
    # Show at most 2 findings in the console
    preview = findings[:2]
    print(f"\n[langfuse-audit] ── Audit {audit_no} ──")
    for f in preview:
        # Strip markdown bold for console
        clean = f.replace("**", "")
        print(f"[langfuse-audit]   {clean}")
    if len(findings) > 2:
        print(f"[langfuse-audit]   ...and {len(findings) - 2} more findings")
    print(f"[langfuse-audit] Full report appended to {USER_DOC}\n")


def _run_audit_if_due() -> None:
    state = _ensure_state()
    total_turns = int(state.get("total_turns", 0))
    if total_turns == 0 or total_turns % AUDIT_EVERY_TURNS != 0:
        return
    window = list(state.get("history", []))[-MAX_WINDOW:]
    findings, metrics = _analyze_window(window)
    audit_no = int(state.get("audits", 0)) + 1
    block = _render_block(window, findings, metrics, audit_no)
    _append_doc(USER_DOC, block)
    state["audits"] = audit_no
    state["last_audit_at"] = _now_iso()
    state["last_findings"] = findings[:5]
    _save_state(state)
    _emit_highlight(findings, metrics, audit_no)


# ── Hook handlers ────────────────────────────────────────────────────────────

def _get_pending(session_id: str) -> dict[str, Any]:
    key = session_id or "default"
    return _PENDING.setdefault(
        key,
        {
            "timestamp": _now_iso(),
            "session_id": session_id,
            "requests": [],
            "tools": [],
        },
    )


def on_post_llm_request(**kwargs: Any) -> None:
    with _LOCK:
        pending = _get_pending(str(kwargs.get("session_id") or ""))
        pending["requests"].append(
            {
                "api_call_count": kwargs.get("api_call_count"),
                "api_duration": kwargs.get("api_duration"),
                "finish_reason": kwargs.get("finish_reason"),
                "model": kwargs.get("model"),
                "provider": kwargs.get("provider"),
            }
        )


def on_post_tool_call(**kwargs: Any) -> None:
    with _LOCK:
        pending = _get_pending(str(kwargs.get("session_id") or ""))
        tool_name = kwargs.get("tool_name", "unknown")
        result = kwargs.get("result")
        args = kwargs.get("args")
        result_str = str(result) if result is not None else ""
        result_size = len(result_str)

        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "result_size": result_size,
        }

        # Capture summarised args
        args_summary = _extract_tool_args_summary(tool_name, args)
        if args_summary:
            entry["args_summary"] = args_summary

        # Detect errors
        error = _detect_error(result_str)
        if error:
            entry["error"] = error

        # Detect empty results
        if _detect_empty(result_str):
            entry["empty_result"] = True

        pending["tools"].append(entry)


def on_post_llm_call(**kwargs: Any) -> None:
    with _LOCK:
        session_id = str(kwargs.get("session_id") or "")
        pending = _PENDING.pop(session_id or "default", None) or _get_pending(session_id)
        pending["timestamp"] = _now_iso()
        pending["session_id"] = session_id
        pending["assistant_response_len"] = len(str(kwargs.get("assistant_response") or ""))
        state = _ensure_state()
        history = list(state.get("history", []))
        history.append(pending)
        state["history"] = history[-MAX_WINDOW:]
        state["total_turns"] = int(state.get("total_turns", 0)) + 1
        _save_state(state)
        _run_audit_if_due()


def register(ctx):
    ctx.register_hook("post_api_request", on_post_llm_request)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
