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
