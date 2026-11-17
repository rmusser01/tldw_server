"""Utilities for Chat-side server tool auto-execution.

This module centralizes tool-call normalization and execution policy for
assistant-generated tool calls. It is intentionally independent from endpoint
plumbing so streaming and non-streaming paths can reuse the same behavior.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Tools.tool_executor import (
    ToolApprovalRequiredError,
    ToolExecutionError,
    ToolExecutor,
)


_TOOL_AUTOEXEC_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

_SAFE_IDEMPOTENCY_SEED_RE = re.compile(r"[^A-Za-z0-9._:-]+")


@dataclass
class ToolExecutionRecord:
    """One normalized/executed assistant tool call outcome."""

    tool_call_id: str
    tool_name: str | None
    ok: bool
    content: str
    result: Any | None = None
    module: str | None = None
    error: str | None = None
    skipped: bool = False
    timed_out: bool = False
    idempotency_key: str | None = None
    #: Typed flag (never inferred from ``content``): the executor refused
    #: this call with a structured MCP approval demand — an interactive
    #: approval the calling context may be unable to provide (scheduled
    #: read-only envelope terminates approval_required on it).
    approval_required: bool = False

    def to_tool_message(self) -> dict[str, Any]:
        """Return a DB-ready role=tool message payload."""
        payload: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
        if self.tool_name:
            payload["name"] = self.tool_name
        return payload

    def to_event_item(self) -> dict[str, Any]:
        """Return a JSON-serializable item for API/SSE emission."""
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "ok": self.ok,
            "result": self.result,
            "module": self.module,
            "error": self.error,
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "content": self.content,
        }


@dataclass
class ToolExecutionBatchResult:
    """Aggregate result for a batch of assistant tool calls."""

    requested_calls: int
    processed_calls: int
    execution_attempts: int
    executed_calls: int
    truncated: bool
    results: list[ToolExecutionRecord] = field(default_factory=list)

    def tool_messages(self) -> list[dict[str, Any]]:
        return [r.to_tool_message() for r in self.results]

    def event_payload(self) -> dict[str, Any]:
        return {"tool_results": [r.to_event_item() for r in self.results]}


def _safe_json_dumps(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except _TOOL_AUTOEXEC_NONCRITICAL_EXCEPTIONS:
        return json.dumps({"ok": False, "error": "Failed to serialize tool output"}, ensure_ascii=False)


def _make_result_content(
    *,
    ok: bool,
    tool_name: str | None,
    result: Any | None = None,
    module: str | None = None,
    error: str | None = None,
    skipped: bool = False,
    timed_out: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "name": tool_name,
        "result": result,
        "module": module,
        "error": error,
    }
    if skipped:
        payload["skipped"] = True
    if timed_out:
        payload["timed_out"] = True
    return _safe_json_dumps(payload)


def _matches_allow_catalog(tool_name: str, allow_catalog: list[str] | None) -> bool:
    if allow_catalog is None:
        return True
    if not allow_catalog:
        return False
    for pattern in allow_catalog:
        token = str(pattern or "").strip()
        if not token:
            continue
        if token == "*":
            return True
        if token.endswith("*"):
            if tool_name.startswith(token[:-1]):
                return True
        elif token == tool_name:
            return True
    return False


def _normalize_tool_call(entry: Any, index: int) -> tuple[str, str | None, dict[str, Any] | None, str | None]:
    """Normalize one assistant tool_call entry.

    Returns: (tool_call_id, tool_name, arguments, error)
    """
    tool_call_id = f"tool_call_{index + 1}"
    if isinstance(entry, dict):
        raw_id = entry.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            tool_call_id = raw_id.strip()
    else:
        return tool_call_id, None, None, "Malformed tool_call entry: expected object"

    function_block = entry.get("function")
    if not isinstance(function_block, dict):
        return tool_call_id, None, None, "Malformed tool_call entry: missing function object"

    raw_name = function_block.get("name")
    tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
    if not tool_name:
        return tool_call_id, None, None, "Tool call missing function.name"

    raw_args = function_block.get("arguments", {})
    if raw_args is None:
        return tool_call_id, tool_name, {}, None
    if isinstance(raw_args, dict):
        return tool_call_id, tool_name, raw_args, None
    if isinstance(raw_args, str):
        txt = raw_args.strip()
        if not txt:
            return tool_call_id, tool_name, {}, None
        try:
            parsed = json.loads(txt)
        except json.JSONDecodeError:
            return tool_call_id, tool_name, None, "Tool call arguments must be valid JSON object"
        if not isinstance(parsed, dict):
            return tool_call_id, tool_name, None, "Tool call arguments must decode to an object"
        return tool_call_id, tool_name, parsed, None

    return tool_call_id, tool_name, None, "Tool call arguments must be object or JSON string"


def build_tool_idempotency_key(
    *,
    seed: str,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Build a stable idempotency key for chat-side tool execution."""
    safe_seed = _SAFE_IDEMPOTENCY_SEED_RE.sub("_", (seed or "").strip()) or "chat"
    args_fingerprint = hashlib.sha256(_safe_json_dumps(arguments).encode("utf-8")).hexdigest()[:16]
    key = f"chat:{safe_seed}:{tool_call_id}:{tool_name}:{args_fingerprint}"
    return key[:200]


async def execute_assistant_tool_calls(
    *,
    tool_calls: Any,
    user_id: str | None,
    client_id: str | None,
    max_tool_calls: int,
    timeout_ms: int,
    allow_catalog: list[str] | None,
    attach_idempotency: bool,
    idempotency_seed: str | None = None,
    tool_executor: ToolExecutor | None = None,
) -> ToolExecutionBatchResult:
    """Execute assistant tool calls with policy checks and deterministic outcomes."""
    raw_calls = tool_calls if isinstance(tool_calls, list) else []
    requested = len(raw_calls)
    cap = max(1, int(max_tool_calls))
    selected = raw_calls[:cap]
    timeout_s = max(0.001, float(timeout_ms) / 1000.0)
    results: list[ToolExecutionRecord] = []
    attempts = 0
    successes = 0
    executor = tool_executor or ToolExecutor()
    seed = idempotency_seed or f"{user_id or 'anon'}:{client_id or 'api_client'}"

    for idx, entry in enumerate(selected):
        tool_call_id, tool_name, arguments, parse_error = _normalize_tool_call(entry, idx)

        if parse_error:
            results.append(
                ToolExecutionRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=False,
                    error=parse_error,
                    skipped=True,
                    content=_make_result_content(
                        ok=False,
                        tool_name=tool_name,
                        error=parse_error,
                        skipped=True,
                    ),
                )
            )
            continue

        assert tool_name is not None
        assert arguments is not None

        if not _matches_allow_catalog(tool_name, allow_catalog):
            deny_msg = f"Tool '{tool_name}' not permitted by chat tool allow-catalog"
            results.append(
                ToolExecutionRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=False,
                    error=deny_msg,
                    skipped=True,
                    content=_make_result_content(
                        ok=False,
                        tool_name=tool_name,
                        error=deny_msg,
                        skipped=True,
                    ),
                )
            )
            continue

        idem_key = None
        if attach_idempotency:
            idem_key = build_tool_idempotency_key(
                seed=seed,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
            )

        attempts += 1
        try:
            exec_coro = executor.execute(
                user_id=user_id,
                client_id=client_id,
                tool_name=tool_name,
                arguments=arguments,
                idempotency_key=idem_key,
                allowed_tools=allow_catalog,
            )
            raw_result = await asyncio.wait_for(exec_coro, timeout=timeout_s)
            module_name = None
            result_payload = raw_result
            if isinstance(raw_result, dict):
                module_name = raw_result.get("module")
                result_payload = raw_result.get("result", raw_result)

            successes += 1
            results.append(
                ToolExecutionRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=True,
                    result=result_payload,
                    module=module_name,
                    content=_make_result_content(
                        ok=True,
                        tool_name=tool_name,
                        result=result_payload,
                        module=module_name,
                    ),
                    idempotency_key=idem_key,
                )
            )
        except asyncio.TimeoutError:
            err = f"Tool execution timed out after {int(timeout_ms)}ms"
            results.append(
                ToolExecutionRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=False,
                    error=err,
                    timed_out=True,
                    content=_make_result_content(
                        ok=False,
                        tool_name=tool_name,
                        error=err,
                        timed_out=True,
                    ),
                    idempotency_key=idem_key,
                )
            )
        except ToolExecutionError as te:
            err = str(te)
            results.append(
                ToolExecutionRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=False,
                    error=err,
                    content=_make_result_content(
                        ok=False,
                        tool_name=tool_name,
                        error=err,
                    ),
                    idempotency_key=idem_key,
                    approval_required=isinstance(te, ToolApprovalRequiredError),
                )
            )
        except _TOOL_AUTOEXEC_NONCRITICAL_EXCEPTIONS as e:
            err = f"Tool execution failed: {e}"
            logger.warning("Unexpected tool auto-exec error for {}: {}", tool_name, e)
            results.append(
                ToolExecutionRecord(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    ok=False,
                    error=err,
                    content=_make_result_content(
                        ok=False,
                        tool_name=tool_name,
                        error=err,
                    ),
                    idempotency_key=idem_key,
                )
            )

    return ToolExecutionBatchResult(
        requested_calls=requested,
        processed_calls=len(selected),
        execution_attempts=attempts,
        executed_calls=successes,
        truncated=requested > cap,
        results=results,
    )
