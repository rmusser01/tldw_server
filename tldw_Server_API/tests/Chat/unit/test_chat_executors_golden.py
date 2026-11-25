"""Golden-output test for the chat executors (TASK-13950.25, PR A).

Every case drives ``chat_service.execute_non_stream_call`` or ``chat_service.execute_streaming_call``
against fixed fakes and records: the returned payload (or the raised error, or the SSE frames), the
persisted messages, the usage rows, the metric calls, and the order in which the stubbed collaborators
were called. Work the executors schedule in the background (topic monitoring, audit writes, moderation
review capture) is recorded as a sorted multiset, apart from the ordered trace. The record is compared
with ``golden/chat_executors_golden.json``. The golden never changes after PR A: a diff is a behaviour
change.

Every fake goes on a ``chat_service`` name, so it stays live when code moves out of that module
(moved code reads those names through ``chat_service``). The only stdlib patch is ``chat_service.time``,
with a fixed clock.

Write mode: ``TLDW_WRITE_CHAT_EXECUTORS_GOLDEN=1`` makes ``test_golden_case_matches`` write its case
into the JSON instead of asserting.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import re
import socket
import time as real_time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError
from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import PromptCostGuardrailConfig
from tldw_Server_API.app.core.Chat.tool_auto_exec import (
    ToolExecutionBatchResult,
    ToolExecutionRecord,
)
from tldw_Server_API.app.core.Usage.llm_accounting import drain_accounting

pytestmark = pytest.mark.unit

GOLDEN_PATH = Path(__file__).with_name("golden") / "chat_executors_golden.json"
WRITE_ENV = "TLDW_WRITE_CHAT_EXECUTORS_GOLDEN"
FIXED_NOW = 1_700_000_000.0

# Env vars the executors read at call time that a case sets or the machine config could set; every case
# starts without them. The three below are set explicitly instead (``_PINNED_ENV``): the first
# ``load_comprehensive_config()`` in a process applies config.txt values as env defaults, and a pinned
# value beats that default whatever the local config says.
_ENV_TO_CLEAR = (
    "CHAT_FORCE_NORMALIZE_STRING_RESPONSES",
    "CHAT_AUTO_EXECUTE_TOOLS",
    "CHAT_TOOL_AUTO_CONTINUE_ONCE",
    "CHAT_MAX_TOOL_CALLS",
    "CHAT_TOOL_TIMEOUT_MS",
    "CHAT_TOOL_ALLOW_CATALOG",
    "CHAT_TOOL_IDEMPOTENCY",
    "CHAT_INJECT_ASSISTANT_NAME",
)

_PINNED_ENV = {"STREAMS_UNIFIED": "0", "MODERATION_STREAM_BUFFER_CHARS": "1024", "CHAT_STREAM_CHANNEL_MAXSIZE": "100"}


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

_HEX32 = re.compile(r"\b[0-9a-f]{32}\b")
_CHATCMPL = re.compile(r"chatcmpl-[A-Za-z0-9.-]+")
_ISO_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)?")
_CREATED = re.compile(r'("created":\s*)\d+')
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "model_dump") and not isinstance(value, type):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=json.dumps)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return type(value).__name__  # opaque handles: the type name only


def _placeholders(text: str) -> str:
    """Stable placeholders for ids, ``stream_id``, ``created`` and timestamps in payloads and SSE frames."""
    text = _CHATCMPL.sub("chatcmpl-<id>", text)
    text = _ISO_TS.sub("<ts>", text)
    text = _UUID.sub("<uuid>", text)
    text = _HEX32.sub("<id>", text)
    return _CREATED.sub(r"\g<1>0", text)


def _sort_hash_ordered(value: Any) -> Any:
    """Sort the keys under ``generation_metadata``: ``sanitize_generation_metadata`` walks a frozenset of
    keys, so their order follows ``PYTHONHASHSEED``. Every other dict keeps its insertion order."""
    if isinstance(value, dict):
        return {
            k: (_sort_keys(v) if k == "generation_metadata" else _sort_hash_ordered(v)) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sort_hash_ordered(v) for v in value]
    return value


def _sort_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value)}
    return value


def normalize(value: Any) -> Any:
    """Plain JSON with placeholders applied; dict insertion order is kept (see ``_sort_hash_ordered``)."""
    return _sort_hash_ordered(json.loads(_placeholders(json.dumps(_jsonable(value), ensure_ascii=False))))


_VOLATILE_KW = ("latency", "duration", "elapsed", "timestamp")


def _drop_volatile(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if not any(m in k for m in _VOLATILE_KW)}


# --------------------------------------------------------------------------------------
# The recorder and the fakes
# --------------------------------------------------------------------------------------


@dataclass
class Rec:
    """Everything one case observes."""

    order: list[str] = field(default_factory=list)
    saved: list[Any] = field(default_factory=list)
    usage: list[Any] = field(default_factory=list)
    metrics: list[Any] = field(default_factory=list)
    background: list[Any] = field(default_factory=list)
    accounting: list[asyncio.Future[Any]] = field(default_factory=list)

    def tag(self, label: str) -> None:
        self.order.append(label)

    def snapshot(self) -> dict[str, Any]:
        return {
            "persisted": normalize(self.saved),
            "usage": normalize(self.usage),
            "metrics": normalize(self.metrics),
            "order": normalize(self.order),
            "background": sorted((normalize(b) for b in self.background), key=json.dumps),
        }


class _Tracker:
    def __init__(self) -> None:
        self.chunks = 0

    def add_chunk(self) -> None:
        self.chunks += 1

    def add_heartbeat(self) -> None:
        return None


class FakeMetrics:
    """Records every metric call; floats that carry a latency are dropped."""

    def __init__(self, rec: Rec) -> None:
        self._rec = rec

    @asynccontextmanager
    async def track_streaming(self, conversation_id: Any):
        tracker = _Tracker()
        try:
            yield tracker
        finally:
            self._rec.metrics.append(["track_streaming", str(conversation_id), tracker.chunks])

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("_"):
            raise AttributeError(name)

        def _call(*args: Any, **kwargs: Any) -> None:
            args = tuple(a for a in args if not isinstance(a, float))
            self._rec.metrics.append([name, list(args), _drop_volatile(kwargs)])

        return _call


class FakeProviderManager:
    def __init__(self, rec: Rec, fallback: str | None = None) -> None:
        self._rec = rec
        self._fallback = fallback

    def get_available_provider(self, exclude: Any = None) -> str | None:
        self._rec.tag(f"provider_manager.get_available_provider:{sorted(exclude or [])}")
        return self._fallback

    def record_failure(self, provider: str, error: BaseException) -> None:
        self._rec.tag(f"provider_manager.record_failure:{provider}:{type(error).__name__}")

    def record_success(self, provider: str, _latency: float) -> None:
        self._rec.tag(f"provider_manager.record_success:{provider}")


class FakeModeration:
    """Redacts ``SECRET`` and blocks ``FORBIDDEN``; both only when ``enabled``."""

    class _Policy:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled
            self.output_enabled = enabled
            self.output_action = "redact"

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled

    def get_effective_policy(self, *_a: Any, **_k: Any) -> Any:
        return self._Policy(self._enabled)

    def evaluate_action_with_match(self, text: Any, *_a: Any, **_k: Any) -> tuple[Any, ...]:
        text = str(text)
        for word, action in (("FORBIDDEN", "block"), ("SECRET", "redact")):
            if word in text:
                start = text.index(word)
                redacted = text.replace("SECRET", "[redacted]") if action == "redact" else None
                return (action, redacted, word.lower(), "default", (start, start + len(word)))
        return ("pass", None, None, None, None)

    def build_sanitized_snippet(self, _text: Any, _policy: Any, _span: Any, pattern: Any) -> Any:
        return pattern

    def check_text(self, text: Any, *_a: Any, **_k: Any) -> tuple[bool, Any]:
        text = str(text)
        for word in ("FORBIDDEN", "SECRET"):
            if word in text:
                return (True, word.lower())
        return (False, None)

    def redact_text(self, text: Any, *_a: Any, **_k: Any) -> str:
        return str(text).replace("SECRET", "[redacted]")


class FakeAudit:
    def __init__(self, rec: Rec) -> None:
        self._rec = rec

    async def log_event(self, **kwargs: Any) -> None:
        self._rec.background.append(
            ["audit", kwargs.get("action"), kwargs.get("result"), _jsonable(kwargs.get("metadata"))]
        )

    async def flush(self, **_kwargs: Any) -> None:
        return None


class FakeTopicMonitor:
    def __init__(self, rec: Rec) -> None:
        self._rec = rec

    def schedule_evaluate_and_alert(self, **kwargs: Any) -> None:
        self._rec.background.append(["topic_monitoring", _jsonable(kwargs)])


class FakeNonStreamQueue:
    """Runs the processor off the loop, as the real queue does, and returns its future."""

    def __init__(self, rec: Rec) -> None:
        self._rec = rec

    def is_running(self) -> bool:
        return True

    async def enqueue(self, *, processor: Callable[[], Any], streaming: bool = False, **_kwargs: Any) -> Any:
        self._rec.tag(f"queue.enqueue:streaming={streaming}")
        return asyncio.create_task(asyncio.to_thread(processor))


class FakeStreamQueue:
    """Bridges a queued stream into the stream channel, as the real queue worker does."""

    def __init__(self, rec: Rec) -> None:
        self._rec = rec

    def is_running(self) -> bool:
        return True

    async def enqueue(self, *, processor: Callable[[], Any], stream_channel: Any, **_kwargs: Any) -> Any:
        self._rec.tag("queue.enqueue:streaming=True")

        async def _run() -> None:
            try:
                result = await asyncio.get_running_loop().run_in_executor(None, processor)
                if hasattr(result, "__aiter__"):
                    async for chunk in result:
                        await stream_channel.put(chunk)
                else:
                    for chunk in result:
                        await stream_channel.put(chunk)
            finally:
                await stream_channel.put(None)

        asyncio.create_task(_run())
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        fut.set_result({"status": "ok"})
        return fut


def _sse(delta: dict[str, Any] | None = None, finish: str | None = None, usage: dict[str, Any] | None = None) -> str:
    chunk: dict[str, Any] = {"choices": [{"delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


def text_stream(*parts: str, usage: dict[str, Any] | None = None) -> Callable[[], Any]:
    """A provider stream factory: one content chunk per part, a stop chunk, then ``[DONE]``."""

    def _factory() -> Any:
        def _gen() -> Any:
            for part in parts:
                yield _sse({"content": part})
            yield _sse(finish="stop", usage=usage)
            yield "data: [DONE]\n\n"

        return _gen()

    return _factory


def tool_call_stream() -> Any:
    def _gen() -> Any:
        delta = {
            "tool_calls": [
                {"index": 0, "id": "c1", "type": "function", "function": {"name": "notes.search", "arguments": "{}"}}
            ]
        }
        yield _sse(delta)
        yield _sse(finish="tool_calls")
        yield "data: [DONE]\n\n"

    return _gen()


def completion(content: str | None, *, tool_calls: list[dict[str, Any]] | None = None, tag: str = "1") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-golden-{tag}",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


_TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "notes.search", "arguments": "{}"}}
_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_schema",
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
    },
}


def _certified_pre_dispatch(error: ChatProviderError) -> ChatProviderError:
    error.upstream_dispatched = False
    error.output_emitted = False
    error.allow_non_stream_fallback = True
    return error


# --------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------


@dataclass
class Case:
    kind: str  # "non_stream" or "stream"
    build: Callable[[pytest.MonkeyPatch, Rec], dict[str, Any]]


GOLDEN_CASES: dict[str, Case] = {}


def _case(case_id: str, kind: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    def _register(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        GOLDEN_CASES[case_id] = Case(kind=kind, build=fn)
        return fn

    return _register


def _request(headers: dict[str, str] | None = None) -> Any:
    return SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers=headers or {},
        state=SimpleNamespace(user_id=7, api_key_id=None, team_ids=None, org_ids=None),
    )


def _fake_refresh(rec: Rec, *, streaming: bool, model: str) -> Callable[[str], Any]:
    def _refresh(provider_name: str) -> Any:
        rec.tag(f"refresh_provider_params:{provider_name}")
        return (
            {
                "api_endpoint": provider_name,
                "api_key": f"{provider_name}-fresh-key",
                "messages_payload": [{"role": "user", "content": "hi"}],
                "model": model,
                "streaming": streaming,
            },
            model,
        )

    return _refresh


def _kwargs(
    rec: Rec,
    *,
    stream: bool,
    llm: Callable[[], Any],
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **overrides: Any,
) -> dict[str, Any]:
    cleaned_args: dict[str, Any] = {
        "api_endpoint": provider,
        "api_key": "primary-key",
        "messages_payload": [{"role": "user", "content": "hi"}],
        "model": model,
        "streaming": stream,
    }
    cleaned_args.update(overrides.pop("cleaned_args", {}))

    async def _on_success(provider_name: str) -> bool:
        rec.tag(f"on_success:{provider_name}")
        return True

    kwargs: dict[str, Any] = {
        "current_loop": asyncio.get_running_loop(),
        "cleaned_args": cleaned_args,
        "selected_provider": provider,
        "provider": provider,
        "model": model,
        "request_json": "{}",
        "request": _request(overrides.pop("headers", None)),
        "metrics": FakeMetrics(rec),
        "provider_manager": FakeProviderManager(rec, overrides.pop("fallback", None)),
        "templated_llm_payload": [{"role": "user", "content": "hi"}],
        "should_persist": True,
        "final_conversation_id": "conv-golden",
        "character_card_for_context": {"name": "Aria"},
        "chat_db": SimpleNamespace(),
        "save_message_fn": _save_message_fn(rec),
        "system_message_id": "sys-1",
        "user_message_id": "user-1",
        "audit_service": None,
        "audit_context": None,
        "client_id": "client-golden",
        "queue_execution_enabled": False,
        "enable_provider_fallback": False,
        "llm_call_func": llm,
        "refresh_provider_params": _fake_refresh(rec, streaming=stream, model=model),
        "moderation_getter": lambda: FakeModeration(False),
        "on_success": _on_success,
        "assistant_parent_message_id": "parent-1",
    }
    if stream:

        async def _commit(tokens: int) -> None:
            rec.tag(f"rg_commit:{tokens}")

        async def _refund(*_a: Any, **kw: Any) -> None:
            rec.tag(f"rg_refund:{sorted(kw)}")

        async def _on_provider_output(provider_name: str) -> bool:
            rec.tag(f"on_provider_output:{provider_name}")
            return True

        kwargs.update(rg_commit_cb=_commit, rg_refund_cb=_refund, on_provider_output=_on_provider_output)
        kwargs.pop("on_success")
        kwargs["on_success"] = _on_success
    kwargs.update(overrides)
    return kwargs


def _save_message_fn(rec: Rec) -> Callable[..., Any]:
    async def _save(_db: Any, _conv_id: Any, payload: Any, use_transaction: bool = True) -> str:
        rec.tag(f"save:{payload.get('role')}")
        rec.saved.append(payload)
        return f"m-{len(rec.saved)}"

    return _save


def _provider(rec: Rec, name: str, result: Any) -> Callable[[], Any]:
    """A provider call that records itself and returns ``result`` or raises it."""

    def _call() -> Any:
        rec.tag(f"provider:{name}")
        if isinstance(result, BaseException):
            raise result
        return result() if callable(result) else result

    return _call


def _async_provider(rec: Rec, name: str, result: Any) -> Callable[..., Any]:
    async def _call(**kwargs: Any) -> Any:
        rec.tag(f"provider_async:{name}:{kwargs.get('api_endpoint')}")
        if isinstance(result, BaseException):
            raise result
        return result

    return _call


def _sync_stream_provider(rec: Rec, name: str, factory: Callable[[], Any]) -> Callable[..., Any]:
    def _call(**kwargs: Any) -> Any:
        rec.tag(f"provider_stream:{name}:{kwargs.get('api_endpoint')}")
        return factory()

    return _call


def _autoexec(rec: Rec) -> Callable[..., Any]:
    async def _run(**kwargs: Any) -> ToolExecutionBatchResult:
        rec.tag("execute_assistant_tool_calls")
        record = ToolExecutionRecord(
            tool_call_id="c1",
            tool_name="notes.search",
            ok=True,
            result={"ok": True},
            module="notes",
            content='{"ok":true}',
        )
        return ToolExecutionBatchResult(
            requested_calls=1,
            processed_calls=1,
            execution_attempts=1,
            executed_calls=1,
            truncated=False,
            results=[record],
        )

    return _run


def _enable_autoexec(mp: pytest.MonkeyPatch, rec: Rec, *, auto_continue: bool = False) -> None:
    mp.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    mp.setattr(chat_service, "should_auto_continue_tools_once", lambda: auto_continue)
    mp.setattr(chat_service, "get_chat_max_tool_calls", lambda: 2)
    mp.setattr(chat_service, "get_chat_tool_timeout_ms", lambda: 3500)
    mp.setattr(chat_service, "get_chat_tool_allow_catalog", lambda: ["notes.*"])
    mp.setattr(chat_service, "should_attach_tool_idempotency", lambda: True)
    mp.setattr(chat_service, "execute_assistant_tool_calls", _autoexec(rec))


# ---- non-stream ----


@_case("N01_direct_persist", "non_stream")
def _n01(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    return _kwargs(rec, stream=False, llm=_provider(rec, "P", completion("Hello from the provider")))


@_case("N02_queued", "non_stream")
def _n02(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(chat_service, "get_request_queue", lambda: FakeNonStreamQueue(rec))
    return _kwargs(
        rec, stream=False, llm=_provider(rec, "P", completion("Queued reply")), queue_execution_enabled=True
    )


@_case("N03_failover", "non_stream")
def _n03(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(chat_service, "perform_chat_api_call_async", _async_provider(rec, "F", completion("From the fallback", tag="f")))
    error = _certified_pre_dispatch(ChatProviderError(provider="openai", message="primary failed", status_code=502))
    return _kwargs(
        rec,
        stream=False,
        llm=_provider(rec, "P", error),
        fallback="anthropic",
        enable_provider_fallback=True,
        refresh_provider_params=_fake_refresh(rec, streaming=False, model="claude-3-5-sonnet"),
    )


@_case("N04_no_fallback_raises", "non_stream")
def _n04(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    error = _certified_pre_dispatch(ChatProviderError(provider="openai", message="primary failed", status_code=502))
    return _kwargs(rec, stream=False, llm=_provider(rec, "P", error), fallback="anthropic")


@_case("N05_string_response", "non_stream")
def _n05(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES", "1")
    return _kwargs(rec, stream=False, llm=_provider(rec, "P", "plain text reply"))


@_case("N06_structured_deferred", "non_stream")
def _n06(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    _enable_autoexec(mp, rec, auto_continue=True)
    mp.setattr(
        chat_service,
        "perform_chat_api_call_async",
        _async_provider(rec, "P2", completion('{"answer":"ok"}', tag="cont")),
    )
    first = completion("Calling a tool", tool_calls=[_TOOL_CALL])
    return _kwargs(
        rec,
        stream=False,
        llm=_provider(rec, "P", first),
        cleaned_args={"response_format": _SCHEMA_FORMAT},
    )


@_case("N07_output_moderation_redacts", "non_stream")
def _n07(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(chat_service, "get_topic_monitoring_service", lambda: FakeTopicMonitor(rec))
    return _kwargs(
        rec,
        stream=False,
        llm=_provider(rec, "P", completion("the SECRET is out")),
        moderation_getter=lambda: FakeModeration(True),
        audit_service=FakeAudit(rec),
        audit_context=SimpleNamespace(),
    )


@_case("N08_tool_autoexec_continue", "non_stream")
def _n08(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    _enable_autoexec(mp, rec, auto_continue=True)
    mp.setattr(
        chat_service,
        "perform_chat_api_call_async",
        _async_provider(rec, "P2", completion("Final answer after the tool", tag="cont")),
    )
    first = completion("Calling a tool", tool_calls=[_TOOL_CALL])
    return _kwargs(rec, stream=False, llm=_provider(rec, "P", first))


@_case("N09_guardrails_assistant_name", "non_stream")
def _n09(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(chat_service, "INJECT_ASSISTANT_NAME", True)
    mp.setattr(
        chat_service,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=True, warn_total_estimated_tokens=1),
    )
    return _kwargs(rec, stream=False, llm=_provider(rec, "P", completion("Guarded reply")))


# ---- stream ----


@_case("S01_direct", "stream")
def _s01(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    stream = text_stream("Hello ", "world", usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
    return _kwargs(rec, stream=True, llm=_provider(rec, "P", stream))


@_case("S02_queued_worker_failover", "stream")
def _s02(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(chat_service, "get_request_queue", lambda: FakeStreamQueue(rec))
    mp.setattr(
        chat_service,
        "perform_chat_api_call",
        _sync_stream_provider(rec, "F", text_stream("fallback ", "stream")),
    )
    error = _certified_pre_dispatch(ChatProviderError(provider="openai", message="primary failed", status_code=502))
    return _kwargs(
        rec,
        stream=True,
        llm=_provider(rec, "P", error),
        fallback="anthropic",
        queue_execution_enabled=True,
        enable_provider_fallback=True,
        refresh_provider_params=_fake_refresh(rec, streaming=True, model="claude-3-5-sonnet"),
    )


@_case("S03_direct_failover", "stream")
def _s03(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(
        chat_service,
        "perform_chat_api_call",
        _sync_stream_provider(rec, "F", text_stream("fallback ", "stream")),
    )
    error = _certified_pre_dispatch(ChatProviderError(provider="openai", message="primary failed", status_code=502))
    return _kwargs(
        rec,
        stream=True,
        llm=_provider(rec, "P", error),
        fallback="anthropic",
        enable_provider_fallback=True,
        refresh_provider_params=_fake_refresh(rec, streaming=True, model="claude-3-5-sonnet"),
    )


@_case("S04_http_exception", "stream")
def _s04(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    return _kwargs(rec, stream=True, llm=_provider(rec, "P", HTTPException(status_code=503, detail="upstream down")))


@_case("S05_noncritical_error", "stream")
def _s05(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    return _kwargs(rec, stream=True, llm=_provider(rec, "P", RuntimeError("provider exploded")))


@_case("S06_chunk_moderation_holdback", "stream")
def _s06(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setenv("MODERATION_STREAM_BUFFER_CHARS", "8")
    mp.setattr(chat_service, "get_topic_monitoring_service", lambda: FakeTopicMonitor(rec))
    stream = text_stream("start of the SECRET reply ", "and then FORBIDDEN text ", "never shown")
    return _kwargs(
        rec,
        stream=True,
        llm=_provider(rec, "P", stream),
        moderation_getter=lambda: FakeModeration(True),
        audit_service=FakeAudit(rec),
        audit_context=SimpleNamespace(),
    )


@_case("S07_tool_autoexec_loop_compat", "stream")
def _s07(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    _enable_autoexec(mp, rec)
    return _kwargs(
        rec,
        stream=True,
        llm=_provider(rec, "P", tool_call_stream),
        headers={"X-TLDW-Loop-Compat": "1"},
    )


@_case("S08_streams_unified", "stream")
def _s08(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setenv("STREAMS_UNIFIED", "1")
    return _kwargs(rec, stream=True, llm=_provider(rec, "P", text_stream("unified ", "frames")))


@_case("S09_structured", "stream")
def _s09(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    return _kwargs(
        rec,
        stream=True,
        llm=_provider(rec, "P", text_stream('{"answer":"ok"}')),
        cleaned_args={"response_format": _SCHEMA_FORMAT},
    )


@_case("S10_queued_success", "stream")
def _s10(mp: pytest.MonkeyPatch, rec: Rec) -> dict[str, Any]:
    mp.setattr(chat_service, "get_request_queue", lambda: FakeStreamQueue(rec))
    return _kwargs(
        rec,
        stream=True,
        llm=_provider(rec, "P", text_stream("queued ", "stream")),
        queue_execution_enabled=True,
    )


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


class _FixedClock:
    """``chat_service.time`` with a fixed ``time()``; everything else is the real module."""

    @staticmethod
    def time() -> float:
        return FIXED_NOW

    def __getattr__(self, name: str) -> Any:
        return getattr(real_time, name)


def _install_base(mp: pytest.MonkeyPatch, rec: Rec) -> None:
    for env in _ENV_TO_CLEAR:
        mp.delenv(env, raising=False)
    for env, value in _PINNED_ENV.items():
        mp.setenv(env, value)
    mp.setattr(chat_service, "time", _FixedClock())
    mp.setattr(chat_service, "get_request_queue", lambda: None)
    mp.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    mp.setattr(chat_service, "should_auto_execute_tools", lambda: False)
    mp.setattr(chat_service, "should_auto_continue_tools_once", lambda: False)
    mp.setattr(
        chat_service,
        "load_prompt_cost_guardrail_config",
        lambda: PromptCostGuardrailConfig(enabled=False),
    )

    async def _log_llm_usage(**kwargs: Any) -> None:
        kwargs.pop("request", None)
        rec.usage.append(_drop_volatile(kwargs))

    def _submit_accounting(make_coro: Callable[[], Any]) -> bool:
        rec.tag("submit_accounting")
        rec.accounting.append(asyncio.ensure_future(make_coro()))
        return True

    async def _capture_review_item(**kwargs: Any) -> None:
        rec.background.append(["moderation_review_item", _jsonable(kwargs.get("phase")), _jsonable(kwargs.get("action"))])

    def _unexpected_provider(**kwargs: Any) -> Any:
        raise AssertionError(f"unstubbed provider call for {kwargs.get('api_endpoint')}")

    async def _unexpected_provider_async(**kwargs: Any) -> Any:
        raise AssertionError(f"unstubbed provider call for {kwargs.get('api_endpoint')}")

    mp.setattr(chat_service, "log_llm_usage", _log_llm_usage)
    mp.setattr(chat_service, "submit_accounting", _submit_accounting)
    mp.setattr(chat_service, "_capture_moderation_review_item_safely_async", _capture_review_item)
    mp.setattr(chat_service, "perform_chat_api_call", _unexpected_provider)
    mp.setattr(chat_service, "perform_chat_api_call_async", _unexpected_provider_async)


async def _drain_body(response: Any) -> list[str]:
    frames: list[str] = []
    body = response.body_iterator
    try:
        async for chunk in body:
            frames.append(chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk))
    finally:
        with contextlib.suppress(Exception):
            await body.aclose()
    return frames


def _error_record(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "status_code": getattr(error, "status_code", None),
        "detail": _jsonable(getattr(error, "detail", None)),
        "message": str(error)[:200] if not isinstance(error, HTTPException) else None,
    }


async def _settle(rec: Rec) -> None:
    """Await every task the case scheduled (accounting, audit, topic monitoring) before the snapshot."""
    for _ in range(5):
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        pending += [f for f in rec.accounting if not f.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)
    await drain_accounting()


async def run_case(
    case_id: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed: Callable[[pytest.MonkeyPatch], None] | None = None,
) -> dict[str, Any]:
    """Run one case and return its normalised record. ``seed`` exists only for the seeded-change tests."""
    case = GOLDEN_CASES[case_id]
    rec = Rec()
    _install_base(monkeypatch, rec)
    kwargs = case.build(monkeypatch, rec)
    if seed is not None:
        seed(monkeypatch)
    result: dict[str, Any]
    if case.kind == "non_stream":
        try:
            result = {"payload": normalize(await chat_service.execute_non_stream_call(**kwargs))}
        except Exception as error:  # noqa: BLE001 - the raised error is part of the record
            result = {"raised": normalize(_error_record(error))}
    else:
        try:
            response = await chat_service.execute_streaming_call(**kwargs)
        except Exception as error:  # noqa: BLE001 - the raised error is part of the record
            result = {"raised": normalize(_error_record(error))}
        else:
            frames = await _drain_body(response)
            result = {
                "media_type": response.media_type,
                "headers": normalize(dict(response.headers)),
                "frames": [_placeholders(f) for f in frames],
            }
    await _settle(rec)
    return {"result": result, **rec.snapshot()}


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def _load_golden() -> dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8")) if GOLDEN_PATH.exists() else {}


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(GOLDEN_CASES))
async def test_golden_case_matches(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actual = await run_case(case, monkeypatch)
    if os.environ.get(WRITE_ENV) == "1":
        golden = _load_golden()
        golden[case] = actual
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(_dump({k: golden[k] for k in GOLDEN_CASES if k in golden}), encoding="utf-8")
        return
    golden = _load_golden()
    assert case in golden, f"{case} has no golden; run once with {WRITE_ENV}=1 on unmodified dev code"  # nosec B101
    # Compare the JSON text: key order is part of the record.
    want, got = _dump(golden[case]), _dump(actual)
    if want != got:
        diff = "".join(difflib.unified_diff(want.splitlines(True), got.splitlines(True), "golden", "actual"))
        pytest.fail(f"{case} drifted from the golden:\n{diff}")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(GOLDEN_CASES))
async def test_golden_case_is_deterministic(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs in one process agree; a varying value belongs in the placeholders, never a blanket rule."""
    first = _dump(await run_case(case, monkeypatch))
    assert _dump(await run_case(case, monkeypatch)) == first  # nosec B101


def _seed_extra_payload_key(mp: pytest.MonkeyPatch) -> None:
    real = chat_service.jsonable_encoder

    def wrapped(value: Any, *args: Any, **kwargs: Any) -> Any:
        out = real(value, *args, **kwargs)
        if isinstance(out, dict):
            out["seeded"] = True
        return out

    mp.setattr(chat_service, "jsonable_encoder", wrapped)


def _seed_no_stream_metadata(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(chat_service, "CHAT_STREAM_INCLUDE_METADATA", False)


def _seed_extra_persisted_key(mp: pytest.MonkeyPatch) -> None:
    real = chat_service._build_assistant_message_payload

    def wrapped(**kwargs: Any) -> Any:
        payload = real(**kwargs)
        payload["seeded"] = True
        return payload

    mp.setattr(chat_service, "_build_assistant_message_payload", wrapped)


def _seed_credential_source(mp: pytest.MonkeyPatch) -> None:
    mp.setattr(chat_service, "call_credential_source", lambda _args: "seeded-source")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "section", "seed"),
    [
        pytest.param("N01_direct_persist", "result", _seed_extra_payload_key, id="payload"),
        pytest.param("S01_direct", "result", _seed_no_stream_metadata, id="frames"),
        pytest.param("N01_direct_persist", "persisted", _seed_extra_persisted_key, id="persistence"),
        pytest.param("N01_direct_persist", "usage", _seed_credential_source, id="usage"),
    ],
)
async def test_seeded_change_fails_the_comparison(
    case: str, section: str, seed: Callable[[pytest.MonkeyPatch], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = await run_case(case, monkeypatch)
    seeded = await run_case(case, monkeypatch, seed=seed)
    assert _dump(seeded[section]) != _dump(baseline[section])  # nosec B101
    assert _dump(seeded) != _dump(baseline)  # nosec B101


@pytest.mark.asyncio
async def test_every_case_stubs_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """No case opens a socket: the provider is a fake on every path."""
    attempts: list[Any] = []

    def _guard(self: socket.socket, address: Any, *_a: Any, **_k: Any) -> Any:
        attempts.append(address)
        raise OSError("network access is not allowed in the golden test")

    monkeypatch.setattr(socket.socket, "connect", _guard)
    monkeypatch.setattr(socket.socket, "connect_ex", _guard)
    for case in GOLDEN_CASES:
        await run_case(case, monkeypatch)
    assert not attempts, attempts  # nosec B101


def test_golden_covers_every_appendix_c_case() -> None:
    """Nine non-stream and ten stream cases, as the plan's Appendix C lists them."""
    assert [c[:3] for c in GOLDEN_CASES if c.startswith("N")] == [f"N{n:02d}" for n in range(1, 10)]  # nosec B101
    assert [c[:3] for c in GOLDEN_CASES if c.startswith("S")] == [f"S{n:02d}" for n in range(1, 11)]  # nosec B101
