"""A run's ``PipelineState`` must not outlive the call (TASK-13950.24 PR N, I1).

The published closures capture ``state`` and are stored on it, a reference cycle. Without
``PipelineState.release()`` the retriever, documents and credential handles stay alive until a
cyclic GC. Garbage collection is disabled here so only refcounting can free them.
"""

from __future__ import annotations

import gc
import weakref

import pytest

from tldw_Server_API.app.core.RAG.rag_service.pipeline import run_state
from tldw_Server_API.tests.RAG_NEW.unit.test_unified_pipeline_golden import run_case

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["G01_minimal", "G02_answer", "G03_rerank_citations"])
async def test_state_is_freed_by_refcount_when_the_run_returns(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    refs: list[weakref.ref] = []

    class SpyState(run_state.PipelineState):  # no __slots__: instances can be weak-referenced
        def __init__(self, rcfg) -> None:
            super().__init__(rcfg)
            refs.append(weakref.ref(self))

    monkeypatch.setattr(run_state, "PipelineState", SpyState)
    gc.collect()
    gc.disable()
    try:
        await run_case(case, monkeypatch)
        alive = [r for r in refs if r() is not None]
    finally:
        gc.enable()
        gc.collect()
    assert refs and alive == []  # nosec B101


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
    with pytest.raises(UnboundLocalError):
        seen[0].rcfg  # noqa: B018
