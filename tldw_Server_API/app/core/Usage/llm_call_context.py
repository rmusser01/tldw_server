"""Who an LLM call is for, and how it is accounted (LLM usage accounting spec §1)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from tldw_Server_API.app.core.DB_Management.scope_context import get_scope


@dataclass(frozen=True)
class LLMCallContext:
    """How LLM calls made in the current context are attributed and accounted."""

    user_id: int | None = None
    operation: str = "llm_call"
    endpoint: str = "internal"
    exempt: bool = False
    records_itself: bool = False
    prechecked: bool = False


@dataclass(frozen=True)
class LLMCallState:
    """A snapshot of the call context for a thread or queue hand-off."""

    context: LLMCallContext
    in_accounted_call: bool


_DEFAULT_CONTEXT = LLMCallContext()
_CTX: ContextVar[LLMCallContext] = ContextVar("llm_call_context", default=_DEFAULT_CONTEXT)
_IN_ACCOUNTED_CALL: ContextVar[bool] = ContextVar("llm_in_accounted_call", default=False)


def current_llm_call_context() -> LLMCallContext:
    """The call context in effect."""
    return _CTX.get()


def current_llm_user_id() -> int | None:
    """The user the next LLM call is for: the explicit one, else the content scope's, else None."""
    explicit = _CTX.get().user_id
    if explicit is not None:
        return explicit
    scope = get_scope()
    return scope.user_id if scope is not None else None


@contextmanager
def llm_call_context(**overrides: Any) -> Iterator[LLMCallContext]:
    """Overlay fields on the current call context for the duration of the block."""
    token = _CTX.set(replace(_CTX.get(), **overrides))
    try:
        yield _CTX.get()
    finally:
        _CTX.reset(token)


async def with_llm_call_context(awaitable: Awaitable[Any], **overrides: Any) -> Any:
    """Await ``awaitable`` with the overrides applied; for marking a long call without re-indenting it.

    The overrides apply to a coroutine this function awaits itself. A task that was
    started earlier has already copied its context and does not see them.
    """
    with llm_call_context(**overrides):
        return await awaitable


@contextmanager
def accounted_call() -> Iterator[None]:
    """Mark that an accounted provider call is running, so nested provider methods don't account again."""
    token = _IN_ACCOUNTED_CALL.set(True)
    try:
        yield
    finally:
        _IN_ACCOUNTED_CALL.reset(token)


def in_accounted_call() -> bool:
    """True inside an accounted provider call."""
    return _IN_ACCOUNTED_CALL.get()


def capture_llm_call_state() -> LLMCallState:
    """Snapshot the call state, resolving the user so the receiver doesn't need the content scope."""
    return LLMCallState(
        context=replace(_CTX.get(), user_id=current_llm_user_id()),
        in_accounted_call=_IN_ACCOUNTED_CALL.get(),
    )


def run_with_llm_call_state(state: LLMCallState | None, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with a captured call state applied; with no state, just call it."""
    if state is None:
        return fn(*args, **kwargs)
    ctx_token = _CTX.set(state.context)
    flag_token = _IN_ACCOUNTED_CALL.set(state.in_accounted_call)
    try:
        return fn(*args, **kwargs)
    finally:
        _IN_ACCOUNTED_CALL.reset(flag_token)
        _CTX.reset(ctx_token)
