"""Background roots get a user, and an executor hand-off keeps it (LLM usage accounting spec §1)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Usage import llm_roots
from tldw_Server_API.app.core.Usage.llm_call_context import (
    bind_llm_call_user,
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)
from tldw_Server_API.app.core.Usage.llm_roots import (
    LLMStateThreadPoolExecutor,
    ensure_llm_state_executor,
    llm_call_root,
)

pytestmark = pytest.mark.unit


def _probe() -> tuple[int | None, str]:
    """What the current thread sees."""
    return current_llm_user_id(), current_llm_call_context().operation


async def test_a_root_sets_its_user_and_operation_and_restores_them() -> None:
    """Inside the root the owner is the user; afterwards the previous state is back."""
    with llm_call_context(user_id=3):
        with llm_call_root("7", operation="workflow"):
            assert _probe() == (7, "workflow")
        assert _probe() == (3, "llm_call")


async def test_a_root_with_no_usable_owner_keeps_the_user_already_set() -> None:
    """A missing or non-numeric owner never clears a user set by an outer root."""
    with llm_call_context(user_id=3):
        for owner in ("", None, "u"):
            with llm_call_root(owner):
                assert current_llm_user_id() == 3
    with llm_call_root(""):
        assert current_llm_user_id() is None


async def test_binding_a_user_lasts_for_the_rest_of_the_task_only() -> None:
    """A bound user stays for the task that bound it and does not reach a sibling task."""

    async def _session(user_id: int | None) -> int | None:
        """Bind, yield to the loop, then report."""
        bind_llm_call_user(user_id)
        await asyncio.sleep(0)
        return current_llm_user_id()

    assert await asyncio.gather(_session(5), _session(None), _session(9)) == [5, None, 9]
    assert current_llm_user_id() is None


async def test_run_in_executor_drops_the_user_until_the_executor_is_installed() -> None:
    """The default executor loses the user; the installed one carries it, for this hand-off only."""
    loop = asyncio.get_running_loop()
    # The first assertion needs a loop nothing has installed on: pytest-asyncio gives each test its own.
    with llm_call_context(user_id=7):
        assert await loop.run_in_executor(None, _probe) == (None, "llm_call")
    assert ensure_llm_state_executor() is True
    with llm_call_context(user_id=7, operation="rag"):
        assert await loop.run_in_executor(None, _probe) == (7, "rag")
    assert await loop.run_in_executor(None, _probe) == (None, "llm_call")


async def test_installing_the_executor_is_idempotent_and_keeps_to_thread_working() -> None:
    """A second install keeps the same executor, and asyncio.to_thread still carries the user."""
    loop = asyncio.get_running_loop()
    ensure_llm_state_executor()
    first = loop._default_executor  # noqa: SLF001 - the stdlib loop's only way to read the default executor
    assert isinstance(first, LLMStateThreadPoolExecutor)
    ensure_llm_state_executor(loop)
    assert loop._default_executor is first  # noqa: SLF001
    with llm_call_context(user_id=4):
        assert await asyncio.to_thread(_probe) == (4, "llm_call")


def test_on_uvloop_the_executor_is_installed_once_and_carries_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server runs on uvloop, whose loop does not expose its default executor to Python."""
    uvloop = pytest.importorskip("uvloop")
    created: list[LLMStateThreadPoolExecutor] = []

    class _Counting(LLMStateThreadPoolExecutor):
        """Counts how many executors get built."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(llm_roots, "LLMStateThreadPoolExecutor", _Counting)

    async def _main() -> tuple[int | None, str]:
        """Enter three roots on one loop, then hand a call to the executor."""
        for owner in ("1", "2", "3"):
            with llm_call_root(owner):
                pass
        with llm_call_context(user_id=7):
            return await asyncio.get_running_loop().run_in_executor(None, _probe)

    loop = uvloop.new_event_loop()
    try:
        assert not hasattr(loop, "_default_executor")
        assert loop.run_until_complete(_main()) == (7, "llm_call")
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()
    assert len(created) == 1


def test_with_no_running_loop_nothing_is_installed() -> None:
    """Outside an event loop there is nothing to install on, and nothing raises."""
    assert ensure_llm_state_executor() is False


async def test_a_root_installs_the_executor_on_its_own_loop() -> None:
    """Entering a root is enough for an executor hand-off inside it to keep the user."""
    loop = asyncio.get_running_loop()
    with llm_call_root(8):
        assert await loop.run_in_executor(None, _probe) == (8, "llm_call")
