"""Check and record every chat provider call made for a known user (LLM usage accounting spec §2, §3, §4b).

``instrument_provider_class`` wraps a provider class's own ``chat``, ``achat``,
``stream`` and ``astream``. The quota check runs before the provider method and
may raise ``LLMQuotaExceededError``; nothing else here may raise into the call.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_cost_envelope import estimate_segment_tokens
from tldw_Server_API.app.core.Usage.llm_accounting import submit_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import (
    accounted_call,
    current_llm_call_context,
    current_llm_user_id,
    in_accounted_call,
)
from tldw_Server_API.app.core.Usage.llm_quota_gate import check_user_async, check_user_sync, user_to_check
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import usage_with_service_tier

_ACCOUNTED_METHODS = ("chat", "achat", "stream", "astream")
_MARK = "__llm_accounted__"


@dataclass(frozen=True)
class _CallPlan:
    """Who to record a call for, decided when the call starts."""

    user_id: int
    operation: str
    endpoint: str


def _plan_call() -> _CallPlan | None:
    """The recording plan for a call starting now, or None when nothing is recorded."""
    try:
        if in_accounted_call():
            return None
        context = current_llm_call_context()
        if context.exempt or context.records_itself:
            return None
        user_id = current_llm_user_id()
        if user_id is None:
            return None
        return _CallPlan(user_id=int(user_id), operation=context.operation, endpoint=context.endpoint)
    except Exception as error:  # noqa: BLE001 - accounting must never break an LLM call
        logger.bind(exception_type=type(error).__name__).debug("LLM usage accounting plan failed")
        return None


def _approx_tokens(char_count: int) -> int:
    """The completion-side 4 chars/token estimate (rounded down)."""
    return max(1, char_count // 4) if char_count > 0 else 0


def _prompt_estimate(request: Any) -> int:
    """Estimated prompt tokens: the messages plus the system prompt, by the chat path's estimate."""
    try:
        if not isinstance(request, dict):
            return 1
        text = json.dumps(request.get("messages"), default=str)
        system_message = request.get("system_message")
        if isinstance(system_message, str):
            text += system_message
        return max(1, estimate_segment_tokens(text))
    except Exception:  # noqa: BLE001 - an estimate must never fail a call
        return 1


def _response_chars(response: Any) -> int:
    """Characters of assistant content in a non-streaming response."""
    total = 0
    with suppress(Exception):
        for choice in response.get("choices") or []:
            content = (choice.get("message") or {}).get("content")
            if isinstance(content, str):
                total += len(content)
    return total


def _usage_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    """Prompt and completion tokens from a provider usage block."""

    def _first(*keys: str) -> int:
        """The first numeric value among ``keys``."""
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return int(value)
        return 0

    prompt_tokens = _first("prompt_tokens", "input_tokens")
    if usage.get("prompt_tokens") is None:
        # Anthropic-shaped block: input_tokens is only the uncached remainder (Anthropic, llama.cpp /v1/messages).
        prompt_tokens += _first("cache_creation_input_tokens") + _first("cache_read_input_tokens")
    return prompt_tokens, _first("completion_tokens", "output_tokens")


class _StreamUsage:
    """Collects the usage chunk and the content length of a stream."""

    def __init__(self) -> None:
        self.usage: dict[str, Any] | None = None
        self.chars = 0

    def feed(self, item: Any) -> None:
        """Read one yielded item, which may hold several SSE lines."""
        try:
            text = item.decode("utf-8", "ignore") if isinstance(item, (bytes, bytearray)) else item
            if not isinstance(text, str):
                return
            for part in text.splitlines():
                line = part.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                # "reasoning_content" does not contain the substring '"content"', so it is listed too.
                if not any(key in payload for key in ('"usage"', '"content"', '"reasoning_content"', '"arguments"')):
                    continue
                try:
                    data = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                usage = usage_with_service_tier(data)  # an OpenAI chunk names its service tier beside the usage
                event = data.get("type")
                if event == "message_start":  # Anthropic Messages: input tokens arrive here, output in message_delta
                    message = data.get("message")
                    usage = message.get("usage") if isinstance(message, dict) else None
                if isinstance(usage, dict):
                    usage = {key: value for key, value in usage.items() if value is not None}  # null never erases a count
                    if usage:
                        self.usage = {**(self.usage or {}), **usage} if event == "message_delta" else usage
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if not isinstance(delta, dict):
                        continue
                    for text_field in (delta.get("content"), delta.get("reasoning_content")):
                        if isinstance(text_field, str):
                            self.chars += len(text_field)
                    tool_calls = delta.get("tool_calls")
                    for tool_call in tool_calls if isinstance(tool_calls, list) else []:
                        function = tool_call.get("function") if isinstance(tool_call, dict) else None
                        arguments = function.get("arguments") if isinstance(function, dict) else None
                        if isinstance(arguments, str):
                            self.chars += len(arguments)
        except Exception:  # noqa: BLE001 - a malformed chunk must never fail a stream
            return


def _submit_record(
    provider: Any,
    plan: _CallPlan,
    request: Any,
    *,
    usage: dict[str, Any] | None,
    prompt_tokens: int,
    completion_tokens: int,
    started: float,
    estimate_source: str | None,
) -> None:
    """Schedule one usage row; never raises."""
    try:
        provider_name = str(getattr(provider, "name", "") or "provider")
        model = str(request.get("model") or "") if isinstance(request, dict) else ""
        latency_ms = int((time.monotonic() - started) * 1000)

        async def _write() -> None:
            """Write the row through the shared recorder."""
            from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage

            await log_llm_usage(
                user_id=plan.user_id,
                key_id=None,
                endpoint=plan.endpoint,
                operation=plan.operation,
                provider=provider_name,
                model=model,
                status=200,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usage_metadata=usage,
                estimated=True if usage is None else None,
                estimate_source=estimate_source,
            )

        submit_accounting(_write)
    except Exception as error:  # noqa: BLE001 - accounting must never break an LLM call
        logger.bind(exception_type=type(error).__name__).debug("LLM usage accounting record failed")


def _record_response(provider: Any, plan: _CallPlan, request: Any, response: Any, started: float) -> None:
    """Record a non-streaming call from its usage block, or an estimate; never raises."""
    try:
        usage = usage_with_service_tier(response)
        if isinstance(usage, dict) and usage:
            prompt_tokens, completion_tokens = _usage_tokens(usage)
            if (prompt_tokens, completion_tokens) != (0, 0):
                _submit_record(
                    provider, plan, request, usage=usage, prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, started=started, estimate_source=None,
                )
                return
        _submit_record(
            provider, plan, request, usage=None, prompt_tokens=_prompt_estimate(request),
            completion_tokens=_approx_tokens(_response_chars(response)), started=started,
            estimate_source="response_estimate",
        )
    except Exception as error:  # noqa: BLE001 - accounting must never break an LLM call
        logger.bind(exception_type=type(error).__name__).debug("LLM usage accounting record failed")


def _record_stream(provider: Any, plan: _CallPlan, request: Any, seen: _StreamUsage, started: float) -> None:
    """Record a stream from its usage chunk, or an estimate of what was received; never raises."""
    try:
        if seen.usage is not None:
            prompt_tokens, completion_tokens = _usage_tokens(seen.usage)
            if (prompt_tokens, completion_tokens) != (0, 0):
                _submit_record(
                    provider, plan, request, usage=seen.usage, prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens, started=started, estimate_source=None,
                )
                return
        _submit_record(
            provider, plan, request, usage=None, prompt_tokens=_prompt_estimate(request),
            completion_tokens=_approx_tokens(seen.chars), started=started, estimate_source="stream_estimate",
        )
    except Exception as error:  # noqa: BLE001 - accounting must never break an LLM call
        logger.bind(exception_type=type(error).__name__).debug("LLM usage accounting record failed")


def _accounted_sync_stream(provider: Any, plan: _CallPlan, request: Any, inner: Any, started: float) -> Iterator[Any]:
    """Yield ``inner``'s items and record once when the stream ends, however it ends."""
    seen = _StreamUsage()
    iterator = iter(inner)
    try:
        while True:
            with accounted_call():
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            seen.feed(item)
            yield item
    finally:
        close = getattr(inner, "close", None)
        if callable(close):
            with suppress(Exception):
                close()
        _record_stream(provider, plan, request, seen, started)


def _close_quietly(*streams: Any) -> None:
    """Close sync iterators, ignoring their errors."""
    for stream in streams:
        close = getattr(stream, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


async def _aclose_quietly(*streams: Any) -> None:
    """Close async iterators, ignoring their errors."""
    for stream in streams:
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            with suppress(Exception):
                await aclose()


def _checked_at_first_item(user_id: int, outer: Iterator[Any], inner: Any) -> Iterator[Any]:
    """Check on the thread that reads the first item, for a stream started where it could not be checked (§4b)."""
    try:
        check_user_sync(user_id)
    except BaseException:
        _close_quietly(outer, inner)
        raise
    yield from outer


async def _async_checked_at_first_item(user_id: int, outer: Any, inner: Any) -> AsyncIterator[Any]:
    """Check before the first item, which is before an async generator's provider code runs (§4b)."""
    try:
        await check_user_async(user_id)
    except BaseException:
        await _aclose_quietly(outer, inner)
        raise
    try:
        async for item in outer:
            yield item
    finally:
        await _aclose_quietly(outer)


async def _accounted_async_stream(
    provider: Any, plan: _CallPlan, request: Any, inner: Any, started: float
) -> AsyncIterator[Any]:
    """Async form of ``_accounted_sync_stream``."""
    seen = _StreamUsage()
    iterator = inner.__aiter__()
    try:
        while True:
            with accounted_call():
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    return
            seen.feed(item)
            yield item
    finally:
        aclose = getattr(inner, "aclose", None)
        if callable(aclose):
            with suppress(Exception):
                await aclose()
        _record_stream(provider, plan, request, seen, started)


def _wrap_chat(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Record after a sync chat call."""

    @functools.wraps(fn)
    def chat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        if user_id is not None:
            check_user_sync(user_id)
        plan = _plan_call()
        if plan is None:
            if user_id is None:
                return fn(self, request, *args, **kwargs)
            with accounted_call():  # checked here, so a nested provider method is not checked again
                return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            response = fn(self, request, *args, **kwargs)
        _record_response(self, plan, request, response, started)
        return response

    return chat


def _wrap_achat(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Record after an async chat call."""

    @functools.wraps(fn)
    async def achat(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        if user_id is not None:
            await check_user_async(user_id)
        plan = _plan_call()
        if plan is None:
            if user_id is None:
                return await fn(self, request, *args, **kwargs)
            with accounted_call():  # checked here, so a nested provider method is not checked again
                return await fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            response = await fn(self, request, *args, **kwargs)
        _record_response(self, plan, request, response, started)
        return response

    return achat


def _wrap_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Check before the provider method, then return a generator that records when the stream ends.

    The check, and the plan, are fixed when stream() is called. Where the check
    cannot run then (on the event loop's thread), it runs at the first item.
    """

    @functools.wraps(fn)
    def stream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        unchecked = user_id is not None and not check_user_sync(user_id)
        plan = _plan_call()
        if plan is None and not unchecked:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if not hasattr(inner, "__iter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        outer = inner if plan is None else _accounted_sync_stream(self, plan, request, inner, started)
        return _checked_at_first_item(user_id, outer, inner) if unchecked else outer

    return stream


def _wrap_astream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Check before the provider code runs, and return an async generator that records when the stream ends.

    An awaitable result stays awaitable, and is checked before it is awaited.
    """

    @functools.wraps(fn)
    def astream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        user_id = user_to_check()
        plan = _plan_call()
        if plan is None and user_id is None:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if inspect.isawaitable(inner):

            async def _resolve() -> Any:
                """Check, await the provider's result, then wrap the iterator it resolves to."""
                if user_id is not None:
                    try:
                        await check_user_async(user_id)
                    except BaseException:
                        _close_quietly(inner)
                        raise
                with accounted_call():
                    resolved = await inner
                if plan is None or not hasattr(resolved, "__aiter__"):
                    return resolved
                return _accounted_async_stream(self, plan, request, resolved, started)

            return _resolve()
        if not hasattr(inner, "__aiter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        outer = inner if plan is None else _accounted_async_stream(self, plan, request, inner, started)
        return outer if user_id is None else _async_checked_at_first_item(user_id, outer, inner)

    if inspect.iscoroutinefunction(fn):
        inspect.markcoroutinefunction(astream)
    return astream


def record_direct_response(provider: str, request: dict[str, Any], response: Any, started: float) -> None:
    """Record a successful provider call made without an adapter (the Messages API native path); never raises."""
    plan = _plan_call()
    if plan is not None:
        _record_response(SimpleNamespace(name=provider), plan, request, response, started)


def accounted_direct_stream(
    provider: str, request: dict[str, Any], stream: AsyncIterator[Any], started: float
) -> AsyncIterator[Any]:
    """Return ``stream`` wrapped to record once when it ends, or unchanged when nothing is recorded.

    For a provider stream made without an adapter. Each item must hold whole SSE events.
    """
    plan = _plan_call()
    if plan is None:
        return stream
    return _accounted_async_stream(SimpleNamespace(name=provider), plan, request, stream, started)


_WRAPPERS: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {
    "chat": _wrap_chat,
    "achat": _wrap_achat,
    "stream": _wrap_stream,
    "astream": _wrap_astream,
}


def instrument_provider_class(cls: type) -> None:
    """Wrap the call methods ``cls`` defines itself. Inherited and abstract methods are left alone."""
    for name in _ACCOUNTED_METHODS:
        fn = cls.__dict__.get(name)
        if fn is None or not inspect.isfunction(fn):
            continue
        if getattr(fn, _MARK, False) or getattr(fn, "__isabstractmethod__", False):
            continue
        wrapped = _WRAPPERS[name](fn)
        setattr(wrapped, _MARK, True)
        setattr(cls, name, wrapped)
