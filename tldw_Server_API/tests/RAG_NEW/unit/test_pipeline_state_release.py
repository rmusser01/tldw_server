"""A run's ``PipelineState`` must hold nothing once the call returns (TASK-13950.24 PR N, I1).

The published closures capture ``state`` and are stored on it, a reference cycle. Without
``PipelineState.release()`` the retriever, documents and credential handles stay alive until a
cyclic GC. ``release()`` clears every slot, which breaks that cycle.

These tests pin that the state holds nothing once the run returns. They do not pin that the
state object itself dies by refcount. A refcount check under ``gc.disable()`` passed locally but
failed on CI (Linux, Python 3.12.16) with ``release()`` in place. The cause is not established.
One candidate is a reference cycle elsewhere in the call chain, such as an exception traceback,
which would keep caller frames, and the ``state`` local in them, alive until a cyclic GC.
"""

from __future__ import annotations

import contextlib

import pytest

from tldw_Server_API.app.core.RAG.rag_service.pipeline import run_state
from tldw_Server_API.tests.RAG_NEW.unit.test_unified_pipeline_golden import run_case

pytestmark = pytest.mark.unit


def _set_slots(state: run_state.PipelineState) -> list[str]:
    still_set = []
    for name in run_state.PipelineState.__slots__:
        with contextlib.suppress(UnboundLocalError):
            getattr(state, name)
            still_set.append(name)
    return still_set


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["G01_minimal", "G02_answer", "G03_rerank_citations"])
async def test_state_holds_nothing_when_the_run_returns(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[run_state.PipelineState] = []

    class SpyState(run_state.PipelineState):  # has a __dict__: catches writes outside the slots
        def __init__(self, rcfg) -> None:
            super().__init__(rcfg)
            seen.append(self)

    monkeypatch.setattr(run_state, "PipelineState", SpyState)
    await run_case(case, monkeypatch)
    assert len(seen) == 1  # nosec B101
    assert _set_slots(seen[0]) == []  # nosec B101
    assert vars(seen[0]) == {}  # nosec B101


def test_release_is_tolerant_of_unset_slots_and_repeatable() -> None:
    from tldw_Server_API.app.core.RAG.rag_service.pipeline.run_config import RagRunConfig

    state = run_state.PipelineState(RagRunConfig.from_kwargs(query="q"))  # only __init__ ran: most slots unset
    state.release()
    state.release()
    with pytest.raises(UnboundLocalError):
        state.rcfg  # noqa: B018


@pytest.mark.asyncio
async def test_a_body_exception_propagates_unchanged_and_state_is_still_released(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up

    seen: list[run_state.PipelineState] = []

    class SpyState(run_state.PipelineState):
        def __init__(self, rcfg) -> None:
            super().__init__(rcfg)
            seen.append(self)

    boom = KeyError("boom")  # not in the pipeline's error ladder, so it escapes run_rag_pipeline

    def fail(*_a, **_k):
        raise boom

    monkeypatch.setattr(run_state, "PipelineState", SpyState)
    monkeypatch.setattr(up._query_prep, "apply_guardrail_defaults", fail)
    with pytest.raises(KeyError) as info:
        await run_case("G01_minimal", monkeypatch)
    assert info.value is boom  # nosec B101
    assert _set_slots(seen[0]) == []  # nosec B101
