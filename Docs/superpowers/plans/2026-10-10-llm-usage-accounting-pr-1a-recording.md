# LLM Usage Accounting PR 1a (mechanism and recording) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every LLM chat call made for a known user writes exactly one `llm_usage_log` row, whichever module makes it, with no new refusals.

**Architecture:** `ChatProvider.__init_subclass__` wraps each provider's `chat`, `achat`, `stream` and `astream` with a recorder. The user comes from an explicit LLM call context or the existing content scope. Writes are scheduled on one event loop. The recorders that exist today keep running, and their calls are marked so the new recorder skips them.

**Tech Stack:** Python 3.12, FastAPI, `contextvars`, asyncio, pytest (asyncio mode auto), SQLite/Postgres AuthNZ pool.

**Spec:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1, §2, §3, §5, the embeddings predicate in §6, Delivery PR 1a). Read it before any task.

## Global Constraints

- **No new refusals.** This PR adds no quota check. `prechecked` exists on the context but nothing reads it yet.
- **Accounting never breaks an LLM call.** Every accounting step catches its own errors and logs. A provider call returns or raises exactly as it does today.
- **Never `asyncio.run`** in accounting code. Writes are scheduled with `submit_accounting`.
- **Carry only LLM call state into threads.** Do not use `contextvars.copy_context()` for the bounded pools or the queue. Other context variables are not passed to those threads today.
- **No caller changes** beyond the sites this plan names.
- **Record only when a user is known** and the context is neither `exempt` nor `records_itself`.
- **What is recorded.** A non-streaming call is recorded only when it succeeds. A stream is recorded once however it ends (completion, a provider error or an early close), from what was received by then.
- **Row defaults:** `operation="llm_call"`, `endpoint="internal"`, `key_id=None`, `status=200`.
- **Embeddings predicate, verbatim:** `COALESCE(operation, '') <> 'embeddings'`.
- **Test-first.** Each task's tests are written and seen failing before its code.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Commits** end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git stash`. Never `--no-verify`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below). Tests run with `TLDW_TEST_NO_DOCKER=1`.

## Review Focus

Inputs the spec implies and a person would hit. Each has a test in the task named.

1. **A stream the consumer abandons without closing.** The generator is finalized later, possibly on another thread. Expected: one row is still scheduled and nothing raises. (Task 2)
2. **Accounting itself fails** (the recorder raises, or no loop is available). Expected: the LLM call's result is unchanged. (Tasks 1 and 2)
3. **A provider whose `astream` returns an awaitable** that resolves to an iterator, not an async generator. Expected: it still works and records once. (Task 2)
4. **Nested provider methods:** `achat` calling `chat`, `astream` iterating `stream`, a subclass calling `super().chat()`. Expected: one row. (Task 2)
5. **A subclass that inherits its methods** from an instrumented parent (`CustomOpenAIAdapter2`). Expected: one row, not zero and not two. (Task 2)

## File Structure

| File | Responsibility |
|---|---|
| `tldw_Server_API/app/core/Usage/llm_call_context.py` (new) | Who a call is for and how it is accounted; the state snapshot for thread and queue hand-offs. |
| `tldw_Server_API/app/core/Usage/llm_accounting.py` (new) | The accounting loop, and scheduling a write on it from any thread. |
| `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (new) | The four method wrappers and the recorder. |
| `tldw_Server_API/app/core/LLM_Calls/providers/base.py` | `__init_subclass__` hook. |
| `tldw_Server_API/app/core/Chat/bounded_daemon.py` | Carries the call state into pool threads. |
| `tldw_Server_API/app/core/Chat/request_queue.py` | Carries the call state from enqueue to the processor. |
| `tldw_Server_API/app/main.py` | Captures the accounting loop at startup. |
| Marked call sites | `endpoints/chat.py`, `Chat/chat_route_resolver.py`, `endpoints/character_chat_sessions.py`, `Claims_Extraction/claims_engine.py`, `AuthNZ/byok_testing.py`, `Setup/first_chat_verifier.py`. |
| `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` | The embeddings predicate. |

---

### Task 1: The call context and the accounting loop

**Files:**
- Create: `tldw_Server_API/app/core/Usage/llm_call_context.py`
- Create: `tldw_Server_API/app/core/Usage/llm_accounting.py`
- Modify: `tldw_Server_API/app/main.py` (the `lifespan` function, right after `_startup_trace("lifespan: entered")`)
- Test: `tldw_Server_API/tests/Usage/test_llm_call_context.py`, `tldw_Server_API/tests/Usage/test_llm_accounting_loop.py`

**Interfaces:**
- Consumes: `get_scope()` and `scoped_context(...)` from `tldw_Server_API.app.core.DB_Management.scope_context`.
- Produces, from `llm_call_context.py`:
  - `LLMCallContext` (frozen dataclass: `user_id: int | None = None`, `operation: str = "llm_call"`, `endpoint: str = "internal"`, `exempt: bool = False`, `records_itself: bool = False`, `prechecked: bool = False`)
  - `current_llm_call_context() -> LLMCallContext`
  - `current_llm_user_id() -> int | None`
  - `llm_call_context(**overrides)` (context manager)
  - `async with_llm_call_context(awaitable, **overrides) -> Any`
  - `accounted_call()` (context manager) and `in_accounted_call() -> bool`
  - `LLMCallState`, `capture_llm_call_state() -> LLMCallState`, `run_with_llm_call_state(state, fn, *args, **kwargs) -> Any`
- Produces, from `llm_accounting.py`:
  - `set_accounting_loop(loop | None) -> None`, `get_accounting_loop() -> AbstractEventLoop | None`
  - `submit_accounting(make_coro: Callable[[], Awaitable[Any]]) -> bool`
  - `async drain_accounting(timeout: float = 5.0) -> None`

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_call_context.py`:

```python
"""Who an LLM call is for, and the state carried across threads (LLM usage accounting spec §1)."""

import threading

import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Usage.llm_call_context import (
    LLMCallContext,
    accounted_call,
    capture_llm_call_state,
    current_llm_call_context,
    current_llm_user_id,
    in_accounted_call,
    llm_call_context,
    run_with_llm_call_state,
    with_llm_call_context,
)

pytestmark = pytest.mark.unit


def test_defaults_have_no_user_and_the_generic_labels() -> None:
    """With nothing set, there is no user and the row labels are the generic ones."""
    assert current_llm_user_id() is None
    assert current_llm_call_context() == LLMCallContext()
    assert current_llm_call_context().operation == "llm_call"
    assert current_llm_call_context().endpoint == "internal"


def test_the_content_scope_supplies_the_user() -> None:
    """A request's content scope is the user when no explicit one is set."""
    with scoped_context(user_id=9):
        assert current_llm_user_id() == 9
    assert current_llm_user_id() is None


def test_an_explicit_user_wins_over_the_scope_and_is_restored() -> None:
    """An explicit user overrides the scope, and leaving the block restores the previous value."""
    with scoped_context(user_id=9):
        with llm_call_context(user_id=4, operation="workflow"):
            assert current_llm_user_id() == 4
            assert current_llm_call_context().operation == "workflow"
            with llm_call_context(records_itself=True):
                assert current_llm_call_context().records_itself is True
                assert current_llm_call_context().operation == "workflow"
            assert current_llm_call_context().records_itself is False
        assert current_llm_user_id() == 9


async def test_with_llm_call_context_applies_while_the_awaitable_runs() -> None:
    """The awaitable sees the overrides; the caller does not afterwards."""

    async def _probe() -> bool:
        """Report the flag seen while running."""
        return current_llm_call_context().exempt

    assert await with_llm_call_context(_probe(), exempt=True) is True
    assert current_llm_call_context().exempt is False


def test_state_snapshot_carries_the_resolved_user_into_a_raw_thread() -> None:
    """A raw thread sees the snapshot's user, flags and nested-call marker, and nothing without it."""
    seen: dict[str, object] = {}

    def _probe(tag: str) -> None:
        """Record what this thread sees."""
        seen[tag] = (current_llm_user_id(), current_llm_call_context().records_itself, in_accounted_call())

    with scoped_context(user_id=6), llm_call_context(records_itself=True), accounted_call():
        state = capture_llm_call_state()
    bare = threading.Thread(target=_probe, args=("bare",))
    carried = threading.Thread(target=run_with_llm_call_state, args=(state, _probe, "carried"))
    bare.start(), carried.start()
    bare.join(), carried.join()
    assert seen["bare"] == (None, False, False)
    assert seen["carried"] == (6, True, True)
    assert run_with_llm_call_state(None, lambda: "ran") == "ran"
```

`tldw_Server_API/tests/Usage/test_llm_accounting_loop.py`:

```python
"""Accounting writes run on one event loop, from any thread (LLM usage accounting spec §5)."""

import asyncio
import threading

import pytest

from tldw_Server_API.app.core.Usage import llm_accounting

pytestmark = pytest.mark.unit


@pytest.fixture
async def accounting_loop(monkeypatch: pytest.MonkeyPatch) -> asyncio.AbstractEventLoop:
    """Make the test's loop the accounting loop for this test only."""
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    return loop


async def test_a_write_from_the_loop_thread_runs_on_the_loop(accounting_loop) -> None:
    """Scheduling from the accounting loop's own thread runs the coroutine there."""
    ran: list[asyncio.AbstractEventLoop] = []

    async def _write() -> None:
        """Record which loop ran this."""
        ran.append(asyncio.get_running_loop())

    assert llm_accounting.submit_accounting(_write) is True
    await llm_accounting.drain_accounting()
    assert ran == [accounting_loop]


async def test_a_write_from_a_plain_thread_runs_on_the_accounting_loop(accounting_loop) -> None:
    """A thread with no loop hands the write to the accounting loop."""
    ran: list[asyncio.AbstractEventLoop] = []

    async def _write() -> None:
        """Record which loop ran this."""
        ran.append(asyncio.get_running_loop())

    results: list[bool] = []
    worker = threading.Thread(target=lambda: results.append(llm_accounting.submit_accounting(_write)))
    worker.start()
    await asyncio.to_thread(worker.join)
    await llm_accounting.drain_accounting()
    assert results == [True] and ran == [accounting_loop]


async def test_a_write_from_a_thread_with_its_own_loop_still_runs_on_the_accounting_loop(accounting_loop) -> None:
    """A workflow-engine-style thread with its own loop never runs accounting on that loop."""
    ran: list[asyncio.AbstractEventLoop] = []

    async def _write() -> None:
        """Record which loop ran this."""
        ran.append(asyncio.get_running_loop())

    async def _other_loop_main() -> None:
        """Schedule a write from a second event loop."""
        assert llm_accounting.submit_accounting(_write) is True

    worker = threading.Thread(target=lambda: asyncio.run(_other_loop_main()))
    worker.start()
    await asyncio.to_thread(worker.join)
    await llm_accounting.drain_accounting()
    assert ran == [accounting_loop]


def test_with_no_loop_anywhere_nothing_is_scheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no accounting loop and no running loop, the write is skipped and nothing raises."""
    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    made: list[int] = []

    async def _write() -> None:
        """Never runs."""
        made.append(1)

    assert llm_accounting.submit_accounting(_write) is False
    assert made == []


async def test_a_failing_write_is_logged_and_never_raises(accounting_loop) -> None:
    """A write that raises does not escape to the caller or to drain."""

    async def _boom() -> None:
        """Fail like a database outage."""
        raise RuntimeError("db down")

    assert llm_accounting.submit_accounting(_boom) is True
    await llm_accounting.drain_accounting()


def test_a_closed_accounting_loop_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured loop that has closed is not used."""
    loop = asyncio.new_event_loop()
    loop.close()
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    assert llm_accounting.get_accounting_loop() is None


def test_a_stopped_accounting_loop_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured loop that is no longer running is not used, so no write can sit forever."""
    loop = asyncio.new_event_loop()
    loop.run_until_complete(asyncio.sleep(0))
    monkeypatch.setattr(llm_accounting, "_accounting_loop", loop)
    try:
        assert llm_accounting.get_accounting_loop() is None
        assert llm_accounting.submit_accounting(lambda: asyncio.sleep(0)) is False
    finally:
        loop.close()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_call_context.py tldw_Server_API/tests/Usage/test_llm_accounting_loop.py`
Expected: two collection errors: `ModuleNotFoundError` for `llm_call_context`, and `ImportError: cannot import name 'llm_accounting'`.

- [ ] **Step 3: Write `llm_call_context.py`**

```python
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
    """Await ``awaitable`` with the overrides applied; for marking a long call without re-indenting it."""
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
```

- [ ] **Step 4: Write `llm_accounting.py`**

```python
"""Run LLM usage accounting on one event loop (LLM usage accounting spec §5).

The AuthNZ pool is bound to one loop, and on Postgres using it from another
loop recreates it. Accounting writes are therefore scheduled on the server's
loop from wherever the LLM call ran, and never through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

_accounting_loop: asyncio.AbstractEventLoop | None = None
_pending: set[Any] = set()


def set_accounting_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Record the loop accounting runs on (the server's loop, set at startup)."""
    global _accounting_loop
    _accounting_loop = loop


def get_accounting_loop() -> asyncio.AbstractEventLoop | None:
    """The accounting loop, or None when none was set, it has closed, or it is not running."""
    loop = _accounting_loop
    # A stopped loop would accept a threadsafe hand-off and never run it.
    if loop is None or loop.is_closed() or not loop.is_running():
        return None
    return loop


def _on_done(future: Any) -> None:
    """Forget a finished write and log its failure."""
    _pending.discard(future)
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.bind(exception_type=type(error).__name__).warning("LLM usage accounting write failed")


def submit_accounting(make_coro: Callable[[], Awaitable[Any]]) -> bool:
    """Schedule ``make_coro()`` on the accounting loop from any thread; never raises.

    Fire-and-forget. Returns False when there is no loop to run it on.
    """
    coro: Any = None
    try:
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        target = get_accounting_loop() or running
        if target is None:
            return False
        coro = make_coro()
        future = target.create_task(coro) if running is target else asyncio.run_coroutine_threadsafe(coro, target)
        _pending.add(future)
        future.add_done_callback(_on_done)
        return True
    except Exception as error:  # noqa: BLE001 - accounting must never break an LLM call
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        logger.bind(exception_type=type(error).__name__).warning("LLM usage accounting could not be scheduled")
        return False


async def drain_accounting(timeout: float = 5.0) -> None:
    """Wait until scheduled writes have finished (tests and shutdown)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _pending and loop.time() < deadline:
        await asyncio.sleep(0.01)
```

- [ ] **Step 5: Capture the loop at startup**

In `tldw_Server_API/app/main.py`, in `lifespan`, directly after the line `_startup_trace("lifespan: entered")`, add:

```python
    from tldw_Server_API.app.core.Usage.llm_accounting import set_accounting_loop

    set_accounting_loop(asyncio.get_running_loop())
```

`asyncio` is already imported at the top of `main.py`.

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_call_context.py tldw_Server_API/tests/Usage/test_llm_accounting_loop.py`
Expected: 12 passed.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/app/core/Usage/llm_accounting.py \
  tldw_Server_API/app/main.py tldw_Server_API/tests/Usage/test_llm_call_context.py \
  tldw_Server_API/tests/Usage/test_llm_accounting_loop.py
git commit -m "feat(usage): LLM call context and a single accounting loop (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Record in the provider base class

**Files:**
- Create: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/base.py` (add `__init_subclass__` to `ChatProvider`, after the `async_chat_is_native` class attribute)
- Test: `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`

**Interfaces:**
- Consumes: everything Task 1 produces; `log_llm_usage` from `tldw_Server_API.app.core.Usage.usage_tracker` (keyword-only: `user_id, key_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, usage_metadata, estimated, estimate_source`).
- Produces: `instrument_provider_class(cls) -> None`. Every `ChatProvider` subclass's own `chat`/`achat`/`stream`/`astream` are wrapped at class creation and carry `__llm_accounted__ = True`.
- Stream items are SSE lines, `str` like `data: {...}`. A usage chunk carries a top-level `usage` object.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`:

```python
"""Every chat provider call for a known user records one usage row (LLM usage accounting spec §2, §3)."""

import asyncio
import gc
import inspect
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context

pytestmark = pytest.mark.unit

REQUEST = {"model": "m1", "messages": [{"role": "user", "content": "hello there, how are you"}]}
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
LINES = [
    'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n',
    "data: [DONE]\n\n",
]


class _FakeProvider(ChatProvider):
    """A provider with native-looking async methods that call its sync ones."""

    name = "fake-accounted"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion with a usage block."""
        return {"choices": [{"message": {"content": "hello"}}], "usage": dict(USAGE)}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield two content chunks, the second with usage, then DONE."""
        yield from LINES

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Run the sync chat on a thread, as production adapters do."""
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        """Iterate the sync stream, as production adapters do."""
        for line in self.stream(request, timeout=timeout):
            yield line


class _NoUsageProvider(_FakeProvider):
    """A provider that reports no usage and sends no usage chunk."""

    name = "fake-no-usage"

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion without a usage block."""
        return {"choices": [{"message": {"content": "x" * 40}}]}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Yield content chunks with no usage."""
        yield 'data: {"choices":[{"delta":{"content":"abcdefgh"}}]}\n\n'
        yield "data: [DONE]\n\n"


class _InheritingProvider(_FakeProvider):
    """Defines no call methods of its own."""

    name = "fake-inheriting"


class _SuperCallingProvider(_FakeProvider):
    """Overrides chat and delegates to the parent."""

    name = "fake-super"

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Delegate to the parent's chat."""
        return super().chat(request, timeout=timeout)


class _AwaitableAstreamProvider(_FakeProvider):
    """Its astream is a coroutine that resolves to an async iterator."""

    name = "fake-awaitable-astream"

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:  # type: ignore[override]
        """Return an async iterator instead of being one."""

        async def _gen() -> AsyncIterator[str]:
            """Yield the canned lines."""
            for line in LINES:
                yield line

        return _gen()


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, and make the test's loop the accounting loop."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    return captured


async def _settled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wait for scheduled writes, then return the rows."""
    await llm_accounting.drain_accounting()
    return rows


async def test_chat_on_a_thread_records_one_row_for_the_context_user(rows) -> None:
    """A sync chat on a worker thread writes one row with the provider's own usage."""
    with llm_call_context(user_id=7):
        response = await asyncio.to_thread(_FakeProvider().chat, REQUEST)
    assert response["usage"] == USAGE
    (row,) = await _settled(rows)
    assert (row["user_id"], row["provider"], row["model"]) == (7, "fake-accounted", "m1")
    assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
    assert (row["operation"], row["endpoint"], row["key_id"], row["status"]) == ("llm_call", "internal", None, 200)
    assert row["usage_metadata"] == USAGE and row["estimated"] is None


async def test_the_content_scope_user_is_recorded_and_labels_come_from_the_context(rows) -> None:
    """With only a content scope, that user is recorded; operation and endpoint follow the context."""
    with scoped_context(user_id=9), llm_call_context(operation="rag", endpoint="POST:/api/v1/rag/search"):
        _FakeProvider().chat(REQUEST)
    (row,) = await _settled(rows)
    assert (row["user_id"], row["operation"], row["endpoint"]) == (9, "rag", "POST:/api/v1/rag/search")


async def test_no_user_exempt_and_self_recorded_calls_write_nothing(rows) -> None:
    """No row without a user, for an exempt call, or for a call that records itself."""
    _FakeProvider().chat(REQUEST)
    with llm_call_context(user_id=7, exempt=True):
        _FakeProvider().chat(REQUEST)
    with llm_call_context(user_id=7, records_itself=True):
        _FakeProvider().chat(REQUEST)
        list(_FakeProvider().stream(REQUEST))
    assert await _settled(rows) == []


async def test_nested_provider_methods_record_once(rows) -> None:
    """achat calling chat, astream iterating stream, and super().chat() each write one row."""
    with llm_call_context(user_id=7):
        await _FakeProvider().achat(REQUEST)
        assert [line async for line in _FakeProvider().astream(REQUEST)] == LINES
        _SuperCallingProvider().chat(REQUEST)
    assert [row["provider"] for row in await _settled(rows)] == ["fake-accounted", "fake-accounted", "fake-super"]


async def test_an_inheriting_subclass_records_once(rows) -> None:
    """A subclass with no methods of its own is recorded once, under its own name."""
    with llm_call_context(user_id=7):
        _InheritingProvider().chat(REQUEST)
    (row,) = await _settled(rows)
    assert row["provider"] == "fake-inheriting"


async def test_a_fully_consumed_stream_records_the_usage_chunk(rows) -> None:
    """A stream read to the end writes one row from the provider's usage chunk."""
    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    assert inspect.isgenerator(stream)
    assert list(stream) == LINES
    (row,) = await _settled(rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 2)
    assert row["usage_metadata"] == {"prompt_tokens": 5, "completion_tokens": 2}


async def test_a_stream_closed_early_records_an_estimate(rows) -> None:
    """Closing after the first chunk still writes one row, estimated from what was received."""
    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    assert next(stream) == LINES[0]
    stream.close()
    (row,) = await _settled(rows)
    assert row["estimated"] is True and row["estimate_source"] == "stream_estimate"
    assert row["completion_tokens"] == 1 and row["prompt_tokens"] >= 1


async def test_an_abandoned_stream_still_records_and_never_raises(rows) -> None:
    """A stream dropped without close is finalized by the collector and still writes its row."""
    with llm_call_context(user_id=7):
        stream = _FakeProvider().stream(REQUEST)
    next(stream)
    del stream
    gc.collect()
    assert len(await _settled(rows)) == 1


async def test_a_response_without_usage_is_estimated(rows) -> None:
    """With no usage block, tokens are estimated at four characters each and the row is marked."""
    with llm_call_context(user_id=7):
        _NoUsageProvider().chat(REQUEST)
        list(_NoUsageProvider().stream(REQUEST))
    chat_row, stream_row = await _settled(rows)
    assert chat_row["estimated"] is True and chat_row["estimate_source"] == "response_estimate"
    assert chat_row["completion_tokens"] == 10 and chat_row["usage_metadata"] is None
    assert stream_row["estimate_source"] == "stream_estimate" and stream_row["completion_tokens"] == 2


async def test_an_awaitable_astream_still_works_and_records_once(rows) -> None:
    """An astream that returns an awaitable resolves to a working iterator and writes one row."""
    with llm_call_context(user_id=7):
        pending = _AwaitableAstreamProvider().astream(REQUEST)
        assert inspect.isawaitable(pending)
        assert [line async for line in await pending] == LINES
    assert len(await _settled(rows)) == 1


async def test_an_accounting_failure_does_not_change_the_call(monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """When the recorder raises, the provider's result is returned unchanged."""

    async def _boom(**kwargs: Any) -> None:
        """Fail like a database outage."""
        raise RuntimeError("db down")

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _boom)
    with llm_call_context(user_id=7):
        assert _FakeProvider().chat(REQUEST)["usage"] == USAGE
        assert list(_FakeProvider().stream(REQUEST)) == LINES
    await llm_accounting.drain_accounting()


async def test_a_midstream_provider_error_reraises_the_same_object_and_still_records(rows) -> None:
    """A provider that fails after one chunk raises its own exception, and the one row is still written."""
    failure = ValueError("upstream broke")

    class _FailingStreamProvider(_FakeProvider):
        """Yields one chunk, then fails."""

        name = "fake-failing-stream"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield one content chunk and raise."""
            yield LINES[0]
            raise failure

    with llm_call_context(user_id=7):
        stream = _FailingStreamProvider().stream(REQUEST)
    assert next(stream) == LINES[0]
    with pytest.raises(ValueError) as raised:
        next(stream)
    assert raised.value is failure
    (row,) = await _settled(rows)
    assert row["estimate_source"] == "stream_estimate" and row["completion_tokens"] == 1


async def test_a_non_iterable_stream_result_is_passed_through_untouched(rows) -> None:
    """A broken provider that returns None from stream() still returns None, and nothing is recorded."""

    class _BrokenStreamProvider(_FakeProvider):
        """Returns nothing iterable."""

        name = "fake-broken-stream"

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:  # type: ignore[override]
            """Return None, as a misbehaving adapter might."""
            return None  # type: ignore[return-value]

    with llm_call_context(user_id=7):
        assert _BrokenStreamProvider().stream(REQUEST) is None
    assert await _settled(rows) == []


def test_instrumenting_keeps_type_identity_and_abstractness() -> None:
    """Instances keep their class, methods are marked, and an abstract subclass stays abstract."""
    assert type(_FakeProvider()) is _FakeProvider
    for method in ("chat", "stream", "achat", "astream"):
        assert getattr(_FakeProvider.__dict__[method], "__llm_accounted__", False) is True
    assert "chat" not in _InheritingProvider.__dict__

    class _StillAbstract(ChatProvider):
        """Implements only capabilities."""

        def capabilities(self) -> dict[str, Any]:
            """No capabilities."""
            return {}

    with pytest.raises(TypeError):
        _StillAbstract()  # type: ignore[abstract]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`
Expected: every test that expects a row fails with `ValueError: not enough values to unpack` or an empty-list assertion, and `test_instrumenting_keeps_type_identity_and_abstractness` fails on `__llm_accounted__`.

- [ ] **Step 3: Write `usage_accounting.py`**

```python
"""Record usage for every chat provider call made for a known user (LLM usage accounting spec §2, §3).

``instrument_provider_class`` wraps a provider class's own ``chat``, ``achat``,
``stream`` and ``astream``. Nothing here may raise into the provider call.
"""

from __future__ import annotations

import functools
import inspect
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Usage.llm_accounting import submit_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import (
    accounted_call,
    current_llm_call_context,
    current_llm_user_id,
    in_accounted_call,
)

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
    except Exception:  # noqa: BLE001 - accounting must never break an LLM call
        logger.opt(exception=True).debug("LLM usage accounting plan failed")
        return None


def _approx_tokens(char_count: int) -> int:
    """Four characters per token, the estimate the chat endpoint already uses."""
    return max(1, char_count // 4) if char_count > 0 else 0


def _prompt_estimate(request: Any) -> int:
    """Estimated prompt tokens from the request's messages."""
    try:
        messages = request.get("messages") if isinstance(request, dict) else None
        return max(1, len(json.dumps(messages, default=str)) // 4)
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
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return 0

    return _first("prompt_tokens", "input_tokens"), _first("completion_tokens", "output_tokens")


class _StreamUsage:
    """Collects the usage chunk and the content length of a stream."""

    def __init__(self) -> None:
        self.usage: dict[str, Any] | None = None
        self.chars = 0

    def feed(self, item: Any) -> None:
        """Read one SSE line."""
        try:
            line = item.decode("utf-8", "ignore") if isinstance(item, (bytes, bytearray)) else item
            if not isinstance(line, str):
                return
            line = line.strip()
            if not line.startswith("data:"):
                return
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                return
            data = json.loads(payload)
            if not isinstance(data, dict):
                return
            usage = data.get("usage")
            if isinstance(usage, dict) and usage:
                self.usage = usage
            for choice in data.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                content = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(content, str):
                    self.chars += len(content)
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
    except Exception:  # noqa: BLE001 - accounting must never break an LLM call
        logger.opt(exception=True).debug("LLM usage accounting record failed")


def _record_response(provider: Any, plan: _CallPlan, request: Any, response: Any, started: float) -> None:
    """Record a non-streaming call from its usage block, or an estimate."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if isinstance(usage, dict) and usage:
        prompt_tokens, completion_tokens = _usage_tokens(usage)
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


def _record_stream(provider: Any, plan: _CallPlan, request: Any, seen: _StreamUsage, started: float) -> None:
    """Record a stream from its usage chunk, or an estimate of what was received."""
    if seen.usage is not None:
        prompt_tokens, completion_tokens = _usage_tokens(seen.usage)
        _submit_record(
            provider, plan, request, usage=seen.usage, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, started=started, estimate_source=None,
        )
        return
    _submit_record(
        provider, plan, request, usage=None, prompt_tokens=_prompt_estimate(request),
        completion_tokens=_approx_tokens(seen.chars), started=started, estimate_source="stream_estimate",
    )


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
        plan = _plan_call()
        if plan is None:
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
        plan = _plan_call()
        if plan is None:
            return await fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            response = await fn(self, request, *args, **kwargs)
        _record_response(self, plan, request, response, started)
        return response

    return achat


def _wrap_stream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a generator that records when the stream ends. The plan is fixed when stream() is called."""

    @functools.wraps(fn)
    def stream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _plan_call()
        if plan is None:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if not hasattr(inner, "__iter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        return _accounted_sync_stream(self, plan, request, inner, started)

    return stream


def _wrap_astream(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return an async generator that records when the stream ends; an awaitable result stays awaitable."""

    @functools.wraps(fn)
    def astream(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        plan = _plan_call()
        if plan is None:
            return fn(self, request, *args, **kwargs)
        started = time.monotonic()
        with accounted_call():
            inner = fn(self, request, *args, **kwargs)
        if inspect.isawaitable(inner):

            async def _resolve() -> Any:
                """Await the provider's result, then wrap the iterator it resolves to."""
                with accounted_call():
                    resolved = await inner
                if not hasattr(resolved, "__aiter__"):
                    return resolved
                return _accounted_async_stream(self, plan, request, resolved, started)

            return _resolve()
        if not hasattr(inner, "__aiter__"):
            return inner  # a broken provider's result reaches the caller's own check unchanged
        return _accounted_async_stream(self, plan, request, inner, started)

    return astream


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
```

Three notes on behaviour:
- `astream` in a subclass is an `async def` with `yield`, so calling it returns an async generator, not an awaitable. The wrapper is a plain `def`, and it returns an async generator too, so callers see the same kind of value.
  - One function-level difference: `inspect.isasyncgenfunction(Adapter.astream)` is now `False`. No code checks that on a chat adapter.
- The plan and the start time are fixed when `stream()` or `astream()` is called. The iterator may run later on another thread, where the context is different.
- A stream whose provider iterator raises midway re-raises the same exception object and still writes its one row, from what was received.

- [ ] **Step 4: Add the hook to `ChatProvider`**

In `tldw_Server_API/app/core/LLM_Calls/providers/base.py`, inside `class ChatProvider(ABC):`, directly after the line `    async_chat_is_native: bool = False`, add:

```python

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Record usage for every call a provider subclass makes (LLM usage accounting spec §2)."""
        super().__init_subclass__(**kwargs)
        from tldw_Server_API.app.core.LLM_Calls.usage_accounting import instrument_provider_class

        instrument_provider_class(cls)
```

The import is inside the method so `base.py` keeps its import-time dependencies.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`
Expected: 14 passed.

- [ ] **Step 6: Run the provider and registry suites for regressions**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/LLM_Adapters`
Expected: the same pass and fail counts as on the task's base commit. Compare against a base run in a detached worktree if anything fails. The identity and type assertions named in the spec (§2) must still pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/LLM_Calls/providers/base.py \
  tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py
git commit -m "feat(usage): record every chat provider call for a known user in the provider base class (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Carry the call state into pool threads and through the chat queue

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/bounded_daemon.py` (`BoundedDaemonPool.start`, lines 121-150)
- Modify: `tldw_Server_API/app/core/Chat/request_queue.py` (`QueuedRequest`, `RequestQueue.enqueue`, `_process_request`)
- Test: `tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py`

**Interfaces:**
- Consumes: `capture_llm_call_state`, `run_with_llm_call_state`, `LLMCallState` from Task 1.
- Produces: any target started through `BoundedDaemonPool.start` runs with the starter's call state. `QueuedRequest.llm_state` holds the enqueuer's state, and the processor runs with it.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py`:

```python
"""The LLM call state reaches pool threads and queued processors (LLM usage accounting spec §1)."""

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool, await_bounded_sync_call
from tldw_Server_API.app.core.Chat.request_queue import RequestPriority, RequestQueue
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)

pytestmark = pytest.mark.unit


def _probe() -> tuple[int | None, bool, bool]:
    """What the current thread sees."""
    context = current_llm_call_context()
    return current_llm_user_id(), context.records_itself, context.exempt


async def test_a_pool_thread_sees_the_starter_s_call_state() -> None:
    """A bounded sync call sees the scope user and the flags of the code that started it."""
    pool = BoundedDaemonPool(capacity=2)
    with scoped_context(user_id=6), llm_call_context(records_itself=True):
        marked = await await_bounded_sync_call(_probe, pool=pool, exhaustion_message="full")
    bare = await await_bounded_sync_call(_probe, pool=pool, exhaustion_message="full")
    assert marked == (6, True, False)
    assert bare == (None, False, False)


async def test_a_queued_processor_runs_with_the_enqueuer_s_call_state() -> None:
    """The queue's workers start with no call state; the processor still sees the enqueuer's."""
    queue = RequestQueue(max_queue_size=4, max_concurrent=1)
    await queue.start(num_workers=1)
    try:
        with llm_call_context(user_id=4, records_itself=True):
            marked: asyncio.Future[Any] = await queue.enqueue(
                "req-marked", {}, "client", RequestPriority.NORMAL, processor=_probe
            )
        bare: asyncio.Future[Any] = await queue.enqueue(
            "req-bare", {}, "client", RequestPriority.NORMAL, processor=_probe
        )
        assert await asyncio.wait_for(marked, timeout=5) == (4, True, False)
        assert await asyncio.wait_for(bare, timeout=5) == (None, False, False)
    finally:
        await queue.stop()


async def test_a_queued_stream_factory_runs_with_the_enqueuer_s_call_state() -> None:
    """A streaming request's factory also runs with the enqueuer's state."""
    queue = RequestQueue(max_queue_size=4, max_concurrent=1)
    await queue.start(num_workers=1)
    seen: list[tuple[int | None, bool, bool]] = []

    def _factory() -> Any:
        """Record the state, then return an empty stream."""
        seen.append(_probe())
        return iter(())

    channel: asyncio.Queue[Any] = asyncio.Queue()
    try:
        with llm_call_context(user_id=4, records_itself=True):
            future = await queue.enqueue(
                "req-stream", {}, "client", RequestPriority.NORMAL,
                processor=_factory, streaming=True, stream_channel=channel,
            )
        await asyncio.wait_for(asyncio.shield(future), timeout=5)
    finally:
        await queue.stop()
    assert seen == [(4, True, False)]
```

Two facts these tests rely on, confirmed by a dry run:
- A non-streaming request's future resolves to the processor's own return value (`request_queue.py:611`, `:741`).
- A streaming request's future resolves when pumping finishes, without the test draining the channel.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py`
Expected: the three tests fail, each seeing `(None, False, False)` where the marked state is expected.

- [ ] **Step 3: Carry the state in `BoundedDaemonPool.start`**

In `tldw_Server_API/app/core/Chat/bounded_daemon.py`, add to the imports:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import (
    capture_llm_call_state,
    run_with_llm_call_state,
)
```

Replace this block in `BoundedDaemonPool.start`:

```python
        self._acquire_capacity(exhaustion_message)

        def guarded_target() -> None:
            try:
                target()
            finally:
```

with:

```python
        self._acquire_capacity(exhaustion_message)
        # Only the LLM call state crosses into the thread; no other context does.
        llm_state = capture_llm_call_state()

        def guarded_target() -> None:
            try:
                run_with_llm_call_state(llm_state, target)
            finally:
```

- [ ] **Step 4: Carry the state through the queue**

In `tldw_Server_API/app/core/Chat/request_queue.py`:

Add the import next to the other `tldw_Server_API` imports:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import (
    LLMCallState,
    capture_llm_call_state,
    run_with_llm_call_state,
)
```

Add a field at the end of `QueuedRequest`, after `stream_terminal_error_emitted`:

```python
    # The enqueuer's LLM call state; the queue's worker tasks have none of their own.
    llm_state: Optional[LLMCallState] = field(compare=False, default=None)
```

In `RequestQueue.enqueue`, add one argument to the `QueuedRequest(...)` construction, after the `stream_factory_timeout=(...)` argument:

```python
                llm_state=capture_llm_call_state(),
```

In `_process_request`, change both `fn = partial(` constructions (the non-streaming one near line 715 and the streaming one near line 908) from:

```python
                fn = partial(
                    request.processor,
                    *request.processor_args,
                    **request.processor_kwargs,
                )
```

to:

```python
                fn = partial(
                    run_with_llm_call_state,
                    request.llm_state,
                    request.processor,
                    *request.processor_args,
                    **request.processor_kwargs,
                )
```

The two blocks are indented differently; keep each block's own indentation.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py`
Expected: 3 passed.

- [ ] **Step 6: Run the Chat unit suite for regressions**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Chat/unit`
Expected: the same pass and fail counts as on the task's base commit.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Chat/bounded_daemon.py tldw_Server_API/app/core/Chat/request_queue.py \
  tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py
git commit -m "feat(usage): carry the LLM call state into pool threads and queued chat processors (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Mark the calls that record themselves, and the exempt calls

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (two pipeline calls; one router call)
- Modify: `tldw_Server_API/app/core/Chat/chat_route_resolver.py` (router call)
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` (router call)
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py` (extract and verify calls)
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_testing.py` (two health calls)
- Modify: `tldw_Server_API/app/core/Setup/first_chat_verifier.py` (one call)
- Test: new `tldw_Server_API/tests/Chat/integration/test_chat_completions_marks_self_recorded.py`, new `tldw_Server_API/tests/Setup/test_first_chat_verifier_exempt.py`, and additions to `tests/Chat/unit/test_chat_route_resolver.py`, `tests/Chat/integration/test_chat_endpoint_auto_routing.py`, `tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`, `tests/Claims/test_claims_engine_modes.py`, `tests/AuthNZ_Unit/test_byok_validation_boundary.py`.

**Interfaces:**
- Consumes: `llm_call_context`, `with_llm_call_context`, `current_llm_call_context` from Task 1; the pool propagation from Task 3.
- Produces: no row from the base-class recorder for the chat pipeline, the router calls, the claims extract and verify calls, provider key validation and first-chat verification. The existing recorders are unchanged.

Every edit below wraps one call. No argument, return value or exception handling changes.

- [ ] **Step 1: Write the failing tests**

Each test replaces the provider call with a probe that reports the flag it sees.

New file `tldw_Server_API/tests/Setup/test_first_chat_verifier_exempt.py`:

```python
"""First-chat verification neither records nor is gated (LLM usage accounting spec §1)."""

import pytest

from tldw_Server_API.app.core.Setup import first_chat_verifier
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context

pytestmark = pytest.mark.unit


async def test_first_chat_verification_runs_exempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The verification call sees exempt=True."""
    seen: list[bool] = []

    async def _probe(**kwargs: object) -> dict:
        """Report the flag, then return a minimal completion."""
        seen.append(current_llm_call_context().exempt)
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr(first_chat_verifier, "perform_chat_api_call_async", _probe)
    await first_chat_verifier._call_chat_completion(provider="openai", model="m", prompt="hi")
    assert seen == [True]
```

New file `tldw_Server_API/tests/Chat/integration/test_chat_completions_marks_self_recorded.py`, modelled on `tests/Chat/integration/test_chat_llm_tokens_quota_402.py` (same fixtures, same header construction):

```python
"""/chat/completions records its own usage, so its provider call is marked (LLM usage accounting spec §3)."""

import pytest

pytestmark = pytest.mark.integration

from tldw_Server_API.app.api.v1.endpoints import chat as chat_ep
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_call_context


def test_chat_completions_marks_its_provider_call_as_self_recorded(
    client, auth_token, mock_chacha_db, setup_dependencies, configure_for_mock_server, monkeypatch
):
    """The non-streaming pipeline's provider call runs with records_itself=True."""
    seen: list[bool] = []

    def _dispatch(*args: object, **kwargs: object) -> dict:
        """Report the flag seen on the pool thread, then return a completion."""
        seen.append(current_llm_call_context().records_itself)
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "local-llm",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(chat_ep, "perform_chat_api_call", _dispatch)
    monkeypatch.setitem(chat_ep.API_KEYS, "local-llm", "dummy-key")
    request_data = ChatCompletionRequest(
        model="local-llm",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        api_provider="local-llm",
    )
    settings = get_settings()
    headers = {"X-CSRF-Token": client.csrf_token}
    if settings.AUTH_MODE == "multi_user":
        headers["Authorization"] = auth_token
    else:
        headers["X-API-KEY"] = auth_token

    response = client.post("/api/v1/chat/completions", json=request_data.model_dump(), headers=headers)

    assert response.status_code == 200, response.text
    assert seen == [True]
```

The canned completion satisfies `_nonstream_provider_result_is_usable` (`chat_service.py:3155`): a dict with a non-empty `choices` list carrying text. `chat_ep.perform_chat_api_call` is the seam the pipeline calls in both paths (`chat.py:5317`).

Add a streaming case to the same file: the same request with `stream=True`, and a `_dispatch` that appends the flag and returns `iter(['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n"])`. Read the response body to the end, then assert `seen == [True]`.

In `tldw_Server_API/tests/Chat/unit/test_chat_route_resolver.py`, add a test that calls `_select_auto_llm_router_choice` directly. The file's `router_dependencies` fixture can't be reused: it lists one candidate, and the router call is skipped unless there are at least two. Shape the call like `tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py:128-160`, with:
- `RouterRequest(model="auto", surface="chat", latest_user_turn=..., scope=...)`;
- `RoutingPolicy(request_model="auto", server_default_provider="openai", boundary_mode="server_default_provider")`;
- two candidates: `[{"provider": "openai", "model": "gpt-test"}, {"provider": "openai", "model": "gpt-alt"}]`;
- a `provider_listing` whose `openai` entry has a `default_model`;
- `request=SimpleNamespace(state=SimpleNamespace(user_id="user-1", api_key_id="key-1"))` and `current_user=SimpleNamespace(id="user-1", id_int=1)`;
- `resolve_provider_api_key_fn=lambda *a, **k: ("k", {})`;
- `resolve_byok_credentials_fn` returning an object with `api_key`, `app_config` and an `async def touch_last_used()`;
- `log_model_router_usage_fn`, an async no-op;
- `perform_chat_api_call_async_fn`, an async probe that appends `current_llm_call_context().records_itself` and returns `{"choices": [{"message": {"content": '{"provider":"openai","model":"gpt-test"}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}`.

Assert the list is `[True]`. The probe runs inline in the same task, so it sees the mark without any pool.

In `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py`, add the same probe to one existing test that reaches the router call in `chat.py` (the call near line 1617): wrap the fake that already replaces `perform_chat_api_call_async` so it appends the flag before delegating, and assert every recorded flag is `True`.

In `tldw_Server_API/tests/Claims/test_claims_engine_modes.py`, add two tests using the file's existing construction, `ClaimsEngine(_analyze_stub)`. The stub appends `current_llm_call_context().records_itself` and returns the canned text the file already uses. One test drives `extract`, one drives `verify`. Assert `[True]` each. The stub runs on a `SYNC_ADAPTER_CALL_POOL` thread, through `_run_bounded_claims_analyze`, so the probe reads the state Task 3 carries there, not the test's own context.

In `tldw_Server_API/tests/AuthNZ_Unit/test_byok_validation_boundary.py`, extend `test_provider_validation_dispatch_binds_authentic_candidate_credentials`, in both `branch` cases. Nothing in that file replaces `_run_bounded_health_call`; it installs a fake adapter and an observed pool (`_install_boundary`), or a fake `chat_api_call` (`_install_legacy_boundary`).
- In the `adapter` branch, make the fake adapter's `chat` append `current_llm_call_context().exempt` before returning.
- In the `legacy` branch, make the `call` fake do the same.

Assert the list is `[True]` after the request. Both run on the observed pool's thread, which inherits `BoundedDaemonPool.start`, so they see the Task 3 state.

For the character-chat router call, extend `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py`. It already monkeypatches `character_chat_sessions.perform_chat_api_call_async` with `async def provider_call(**kwargs)` and calls `_select_auto_character_llm_router_choice` directly (near lines 102-128). In the `a-to-b` case, make `provider_call` append `current_llm_call_context().records_itself`, and assert every recorded flag is `True`.

- [ ] **Step 2: Run the new tests and confirm they fail**

Run each file touched in Step 1. Expected: every new assertion fails with `[False] == [True]`.

- [ ] **Step 3: Mark the chat pipeline**

In `tldw_Server_API/app/api/v1/endpoints/chat.py`, add to the imports from `tldw_Server_API.app.core.Usage`:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context, with_llm_call_context
```

In `_execute_stream_response`, change `return await execute_streaming_call(` to `return await with_llm_call_context(execute_streaming_call(`, and change that call's closing line from `)` to `), records_itself=True)`. Do the same in `_execute_non_stream_response` for `execute_non_stream_call(`. The result has this shape:

```python
                        return await with_llm_call_context(execute_streaming_call(
                            current_loop=current_loop,
                            # ... every existing argument, unchanged ...
                        ), records_itself=True)
```

The provider factory runs on a pool thread started inside that call, so it carries the flag. The stream object fixes its plan when `stream()` is called, so iterating it after the block ends changes nothing.

- [ ] **Step 4: Mark the three router calls**

Each router call has its own recorder, `log_model_router_usage`. Wrap the awaited call in a `with` block.

`chat.py`, the call near line 1617:

```python
        try:
            with llm_call_context(records_itself=True):
                response = await await_owned_worker(
                    perform_chat_api_call_async(
                        # ... existing arguments, unchanged ...
                    ),
                    on_cancel_result=_mark_late_router_choice,
                )
```

`core/Chat/chat_route_resolver.py`, in `execute_router_call` (add the import `from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context`):

```python
        try:
            with llm_call_context(records_itself=True):
                return await perform_chat_api_call_async_fn(
                    # ... existing arguments, unchanged ...
                )
        finally:
            await byok_resolution.touch_last_used()
```

`endpoints/character_chat_sessions.py`, in `_execute_router_call` (add the same import):

```python
        with llm_call_context(records_itself=True):
            result = await await_owned_worker(
                perform_chat_api_call_async(
                    # ... existing arguments, unchanged ...
                ),
                on_cancel_result=_mark_late_router_choice,
            )
```

- [ ] **Step 5: Mark the claims extract and verify calls**

In `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py`, add the import `from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context`. In `extract` (the call near line 837) and in `verify` (the call near line 1202), wrap only the `_run_bounded_claims_analyze` await:

```python
            start_time = time.time()
            with llm_call_context(records_itself=True):
                raw = await _run_bounded_claims_analyze(
                    self._analyze,
                    # ... existing arguments, unchanged ...
                )
            latency_s = time.time() - start_time
```

Do not mark the APS proposition extractor call near line 1486 (`strategy.chunk(..., llm_call_func=self._analyze)`). No recorder covers it, so the base-class recorder must.

- [ ] **Step 6: Mark the exempt calls**

`tldw_Server_API/app/core/AuthNZ/byok_testing.py` (add the same import): wrap each of the two `result = await _run_bounded_health_call(` calls in `test_provider_credentials`:

```python
                with llm_call_context(exempt=True):
                    result = await _run_bounded_health_call(
                        call_adapter,
                        provider=provider_norm,
                        timeout_seconds=effective_timeout,
                    )
```

and the same around the `call_legacy_provider` one.

`tldw_Server_API/app/core/Setup/first_chat_verifier.py` (add the same import): in `_call_chat_completion`:

```python
    with llm_call_context(exempt=True):
        return await perform_chat_api_call_async(
            api_endpoint=provider,
            model=model,
            messages_payload=[{"role": "user", "content": prompt}],
            stream=False,
        )
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run every file touched in Step 1. Expected: all new tests pass, and no existing test in those files changes result.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Chat/chat_route_resolver.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_engine.py tldw_Server_API/app/core/AuthNZ/byok_testing.py \
  tldw_Server_API/app/core/Setup/first_chat_verifier.py tldw_Server_API/tests
git commit -m "feat(usage): mark the LLM calls that record themselves and the exempt ones (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Exclude embeddings from the LLM token counter

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` (`sum_user_llm_tokens_since`, lines 250-263)
- Test: `tldw_Server_API/tests/Usage/test_llm_token_counter_excludes_embeddings.py`

**Interfaces:**
- Produces: `sum_user_llm_tokens_since` ignores rows whose `operation` is `embeddings` and counts rows whose `operation` is NULL. PR 2's cost counter reuses the module constant `_LLM_GENERATION_PREDICATE`.

- [ ] **Step 1: Write the failing test**

`tldw_Server_API/tests/Usage/test_llm_token_counter_excludes_embeddings.py`:

```python
"""The monthly LLM token counter leaves embeddings out and keeps rows with no operation (spec §6)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_tables

pytestmark = pytest.mark.unit


async def test_embeddings_are_excluded_and_null_operations_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat 100 + NULL-operation 10 are counted; embeddings 1000 and another user's 5 are not."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///./Databases/users_test_emb_{uuid.uuid4().hex}.sqlite")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    for user_id, operation, tokens in ((1, "chat", 100), (1, None, 10), (1, "embeddings", 1000), (2, "chat", 5)):
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, operation, total_tokens) VALUES (?, ?, ?)",
            user_id, operation, tokens,
        )

    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await AuthnzUsageRepo(pool).sum_user_llm_tokens_since(user_id=1, since=since) == 110.0
```

`DatabasePool.execute(query, *args)` takes its parameters as separate arguments (`AuthNZ/database.py:1496`), as written above.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_token_counter_excludes_embeddings.py`
Expected: FAIL with `assert 1110.0 == 110.0`.

- [ ] **Step 3: Add the predicate**

In `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`, add a module constant after the imports:

```python
# LLM quota counters leave embeddings out. ``operation`` is nullable, so a bare
# ``<>`` would also drop rows with no operation. This is a constant, never user
# input, so the queries that interpolate it carry ``nosec B608``.
_LLM_GENERATION_PREDICATE = "COALESCE(operation, '') <> 'embeddings'"
```

In `sum_user_llm_tokens_since`, replace the query string:

```python
            "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log WHERE user_id = ? AND ts >= ?",
```

with:

```python
            "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log "  # nosec B608
            f"WHERE user_id = ? AND ts >= ? AND {_LLM_GENERATION_PREDICATE}",
```

Bandit reports B608 on the first line of the concatenated string, so the comment goes there. Keep it to `# nosec B608` alone: trailing words make Bandit warn that they are not test names.

and extend the docstring's first paragraph with: `Embeddings rows are excluded (LLM usage accounting spec §6).`

- [ ] **Step 4: Run the test and confirm it passes**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_token_counter_excludes_embeddings.py tldw_Server_API/tests/Usage/test_quota_resolver.py`
Expected: 13 passed. `test_llm_tokens_month_query_passes_naive_utc_on_postgres` still receives two arguments, because the predicate adds no parameter.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py \
  tldw_Server_API/tests/Usage/test_llm_token_counter_excludes_embeddings.py
git commit -m "fix(usage): the monthly LLM token counter leaves embeddings out and keeps NULL-operation rows (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: An end-to-end row, ADR-067 and the docs

**Files:**
- Test: `tldw_Server_API/tests/Usage/test_llm_usage_recorded_end_to_end.py`
- Create: `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`, following `Docs/ADR/000-template.md`; add its row to the index in `Docs/ADR/README.md`.
- Modify: `Docs/Operations/Usage_Quotas.md`
- Modify: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (Status line only)
- Backlog: TASK-13437 notes.

**Interfaces:**
- Consumes: everything from Tasks 1 to 5.

- [ ] **Step 1: Write the end-to-end test**

It uses a real SQLite `llm_usage_log`, a real `ChatProviderRegistry`, and `perform_chat_api_call_async` through the bounded pool, with the user supplied only by the content scope.

`tldw_Server_API/tests/Usage/test_llm_usage_recorded_end_to_end.py`:

```python
"""A call through the shared layer writes one real llm_usage_log row for the scope user (spec §1 to §3)."""

import asyncio
import uuid
from collections.abc import Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.tests.provider_credential_test_helpers import resolved_request_fields_async
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_cache_columns, _ensure_llm_tables

pytestmark = pytest.mark.integration


class _E2EProvider(ChatProvider):
    """A sync-only provider, like most production adapters."""

    name = "e2e-fake"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion with usage."""
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Unused."""
        return iter(())


async def test_one_row_is_written_for_the_scope_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """RAG-style usage: scope set, no explicit context, call made through perform_chat_api_call_async."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///./Databases/users_test_e2e_{uuid.uuid4().hex}.sqlite")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)
    await _ensure_llm_cache_columns(pool)

    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("e2e-fake", _E2EProvider)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())

    fields = await resolved_request_fields_async("e2e-fake", api_key="k", app_config={}, model="e2e-model")
    with scoped_context(user_id=41):
        result = await chat_service.perform_chat_api_call_async(
            api_endpoint="e2e-fake", **fields, messages_payload=[{"role": "user", "content": "hi"}],
            model="e2e-model", streaming=False,
        )
    assert result["choices"][0]["message"]["content"] == "ok"
    await llm_accounting.drain_accounting()

    rows = await pool.fetchall(
        "SELECT user_id, operation, endpoint, provider, model, prompt_tokens, completion_tokens "
        "FROM llm_usage_log WHERE provider = ?",
        "e2e-fake",
    )
    assert len(rows) == 1
    row = dict(rows[0]) if not isinstance(rows[0], dict) else rows[0]
    assert (row["user_id"], row["operation"], row["endpoint"]) == (41, "llm_call", "internal")
    assert (row["model"], row["prompt_tokens"], row["completion_tokens"]) == ("e2e-model", 30, 12)
```

Two facts this test relies on, confirmed by a dry run: `DatabasePool.fetchall(query, *args)` exists (`AuthNZ/database.py:1555`), and `resolved_request_fields_async` accepts a provider name that isn't registered anywhere else.

- [ ] **Step 2: Run the test and confirm it passes**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_usage_recorded_end_to_end.py`
Expected: 1 passed. To show the test can fail, comment out the `instrument_provider_class(cls)` line in `base.py`, re-run, record the failure (`assert 0 == 1`), and restore the line.

- [ ] **Step 3: Write ADR-067**

Confirm the number is free: `git ls-tree --name-only origin/dev Docs/ADR/ | grep -E '/06[7-9]'` prints nothing, and no open PR adds one (`gh pr list --repo rmusser01/tldw_server --state open --limit 100 --json number,files -q '.[] | select(any(.files[]; .path | test("^Docs/ADR/06[7-9]"))) | .number'`). If 067 is taken, use the next free number in the file name, the title and every mention.

Content, each statement checked against the code:
- **Header:** Status "Accepted", today's date, Decision owner "Repository owner (@rmusser01)", Related task TASK-13437, Related spec `Docs/Design/2026-10-09-llm-usage-accounting-design.md`.
- **Context:** only `/chat/completions`, the router, the claims engine and embeddings recorded LLM usage. About 39 files call the shared layer and about 23 call adapters directly, and none of them recorded.
- **Decision:** "LLM chat calls are accounted in `ChatProvider.__init_subclass__`: each provider's own `chat`, `achat`, `stream` and `astream` record one `llm_usage_log` row for the user the call is for. The user comes from an explicit `llm_call_context` or the content scope. Writes run on the server's event loop."
- **Alternatives considered:**
  - A wrapper object returned by the registry. It breaks adapter identity and type checks, and the strict-transport check.
  - Recording in `perform_chat_api_call*`. It misses the direct adapter callers.
  - Removing the existing recorders. It breaks the claims budget guard and router analytics.
- **Consequences:**
  - More rows are written, so anything that sums `llm_usage_log` sees more usage.
  - Calls with no known user are not recorded.
  - Callers that record themselves must mark their call with `records_itself`.
  - A test that replaces a provider method is not accounted.
  - Embeddings are excluded from the LLM token counter.

- [ ] **Step 4: Update the operator page**

In `Docs/Operations/Usage_Quotas.md`, in the row or paragraph for `limits.llm_tokens_per_month` (near line 68), keep "enforced on `/chat/completions` only" and add, in the page's own style:

- The monthly count now includes every LLM chat call recorded for the user, not just chat completions: RAG generation, character chat, summarization, evaluations and other calls made during a request. Users reach a token limit sooner than before.
- That includes calls a request triggers without the user asking for them directly, such as query rewriting and classification.
- Provider key validation and first-run chat verification are never counted.
- Embedding tokens no longer count toward it.
- Calls made by background jobs, workflow runs and WebSocket sessions are recorded in a later change.
- The count depends on `LLM_USAGE_ENABLED` (default on). With it off, nothing is recorded and the limit never triggers.

- [ ] **Step 5: Update the spec status, refresh the mirror, run the docs tests**

In the spec, change the Status line to `- **Status:** Approved; PR 1a in progress`.

```bash
bash Helper_Scripts/refresh_docs_published.sh
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Docs
```

Expected: `tests/Docs` passes. Commit only the mirror changes your edits caused (the ADR and its index).

- [ ] **Step 6: Update the backlog**

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13437 --status "In Progress" --append-notes "Verified 2026-10-09: most LLM paths were not recorded at all (only chat completions, the router, claims and embeddings wrote llm_usage_log). Spec: Docs/Design/2026-10-09-llm-usage-accounting-design.md. PR 1a records every chat provider call for a known user in ChatProvider.__init_subclass__, keeps the existing recorders (marked records_itself), carries the call state into pool threads and the chat queue, and excludes embeddings from the token counter. No new refusals. Next: PR 1b (WorkerSDK, workflow engine, WebSockets, executor sites), PR 2 (the check and the cost keys)."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13436 --append-notes "Re-scoped by the owner on 2026-10-09: a general LLM cost quota (limits.llm_cost_per_day_usd and limits.llm_cost_per_month_usd) replaces evaluation-only cost caps, and the dead evaluation cost caps are deleted. Spec: Docs/Design/2026-10-09-llm-usage-accounting-design.md, sections 7 and 8; delivered in PR 2 and PR 3 of that spec."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task normalize --check backlog/tasks/task-13436*.md backlog/tasks/task-13437*.md
```

Expected: `normalize --check` exits 0.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/tests/Usage/test_llm_usage_recorded_end_to_end.py Docs/ADR Docs/Published \
  Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md
git add -A backlog/tasks
git commit -m "docs(usage): ADR-067 and the operator note for recorded LLM usage; an end-to-end recording test (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Ship

- [ ] **Step 1: Rebase and re-check**

```bash
git fetch origin dev
git rebase origin/dev
$PY Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Expected: the fingerprint matches. This PR changes no route or schema; if it drifts, the drift came from `dev`, so refresh it with `--fingerprint`.

- [ ] **Step 2: Run the sweep**

Run every test file that references a changed symbol, plus the suites around the changed code:

```bash
git grep -l -e llm_usage_log -e log_llm_usage -e perform_chat_api_call -e ChatProvider -e BoundedDaemonPool \
  -e RequestQueue -e sum_user_llm_tokens_since -e llm_tokens_this_month -e _run_bounded_health_call \
  -e first_chat_verifier -e log_model_router_usage -- 'tldw_Server_API/tests/**/test_*.py' > /tmp/pr1a_tests.txt
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider --timeout=600 $(cat /tmp/pr1a_tests.txt) \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/LLM_Adapters \
  tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Claims tldw_Server_API/tests/Docs tldw_Server_API/tests/lint
```

Use the session scratchpad for the list file if `/tmp` is not allowed.

Every failure needs evidence before it is called pre-existing: it fails the same way at `origin/dev` in a detached worktree, or its import graph excludes every changed file. A failure that passes when run alone is a load or ordering flake only if it also passes alone at `origin/dev`.

- [ ] **Step 3: Static checks**

```bash
uvx bandit -q -ll $(git diff --name-only --diff-filter=AM origin/dev HEAD -- 'tldw_Server_API/app/*.py')
```

Expected: 0 medium-or-higher issues. Ruff per-rule counts on the changed app files must not rise against `origin/dev`.

- [ ] **Step 4: Open the PR against `dev`**

The body covers:
- what is recorded now, and what is not yet (jobs, workflows, WebSockets);
- that there are no new refusals;
- that the monthly token count now includes more calls and excludes embeddings;
- the review focus items and their tests;
- the sweep results, with each failure's evidence;
- the `## Change summary` waiver section (the owner confirmed on 2026-10-09 that the waiver covers the spec 2 follow-up PRs).

Queue it alone. After the merge, delete the branch.
