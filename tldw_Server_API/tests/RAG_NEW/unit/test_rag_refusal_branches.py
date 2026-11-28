"""A moderation refusal in an optional RAG stage is labelled and never a verdict (TASK-13950.27 PR J).

Covers the refusal branches of ``pipeline/post_processing`` (faithfulness), ``pipeline/rerank`` and
``post_generation_verifier``: the stage reports ``llm_policy_refused`` (not ``provider_unavailable``), the answer
stays unverified, and the prior document order stands.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.RAG.rag_service import faithfulness as faithfulness_module
from tldw_Server_API.app.core.RAG.rag_service import post_generation_verifier as verifier_module
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up
from tldw_Server_API.app.core.RAG.rag_service.pipeline import post_processing, rerank
from tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier import PostGenerationVerifier
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit

REFUSAL = "Input violates moderation policy"


def _docs() -> list[Document]:
    return [Document(id=str(i), content=f"doc {i}", metadata={"source": DataSource.MEDIA_DB}) for i in (1, 2, 3)]


# ---- post_processing: faithfulness -------------------------------------------------------------------------------


class _RefusingLLM:
    async def generate(self, prompt: str) -> str:
        raise LLMPolicyRefusal(REFUSAL)


def _finalize_inputs(llm: Any, credential_runtime: Any) -> tuple[Any, Any]:
    rcfg = SimpleNamespace(
        observability=SimpleNamespace(
            ground_truth_doc_ids=None, enable_monitoring=False, debug_mode=False, metrics_k=5
        ),
        verification=SimpleNamespace(enable_faithfulness_eval=True),
        runtime=SimpleNamespace(faithfulness_llm=llm, credential_runtime=credential_runtime),
    )
    result = SimpleNamespace(
        total_time=0.0, timings={}, metadata={}, errors=[], documents=_docs(), generated_answer="An answer.",
        cache_hit=False,
    )
    return rcfg, SimpleNamespace(start_time=time.time(), result=result, metrics=None, query="q")


@pytest.mark.parametrize("credential_runtime", [None, object()], ids=["no-runtime", "runtime"])
async def test_a_refused_faithfulness_verification_is_unscored_and_labelled(credential_runtime):
    rcfg, state = _finalize_inputs(_RefusingLLM(), credential_runtime)

    await post_processing.finalize_run(rcfg, state)

    recorded = state.result.metadata["faithfulness"]
    assert recorded["failure_code"] == "llm_policy_refused"
    assert recorded["verification_available"] is False
    assert recorded["faithfulness_score"] is None
    assert state.result.errors == []


@pytest.mark.parametrize("credential_runtime", [None, object()], ids=["no-runtime", "runtime"])
async def test_a_refusal_escaping_the_evaluator_is_not_relabelled_provider_unavailable(
    monkeypatch, credential_runtime
):
    async def refuse(self, response: str, context: str) -> Any:
        raise LLMPolicyRefusal(REFUSAL)

    monkeypatch.setattr(faithfulness_module.FaithfulnessEvaluator, "evaluate_detailed", refuse)
    rcfg, state = _finalize_inputs(_RefusingLLM(), credential_runtime)

    await post_processing.finalize_run(rcfg, state)

    assert state.result.metadata["faithfulness"] == {
        "failure_code": "llm_policy_refused",
        "verification_available": False,
    }
    assert state.result.errors == []


# ---- rerank ----------------------------------------------------------------------------------------------------


class _Degradations:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, str, str, str]] = []

    def record(self, *, component: str, from_value: str, to_value: str, reason: str) -> None:
        self.recorded.append((component, from_value, to_value, reason))


async def test_a_refused_rerank_skips_the_stage_with_its_reason_and_keeps_the_order(monkeypatch):
    class _Reranker:
        async def rerank(self, query: str, documents: list[Any]) -> Any:
            raise LLMPolicyRefusal(REFUSAL)

    monkeypatch.setattr(up, "create_reranker", lambda *_a, **_k: _Reranker())
    degradations = _Degradations()

    async def resilient_call(_name: str, fn: Any, *args: Any) -> Any:
        return await fn(*args)

    documents = _docs()
    rcfg = SimpleNamespace(
        rerank=SimpleNamespace(rerank_top_k=5, rerank_min_relevance_prob=None, rerank_sentinel_margin=None),
        runtime=SimpleNamespace(credential_runtime=None),
    )
    state = SimpleNamespace(
        include_rerank_snapshots=False, selected_strategy="two_tier", rerank_config=None, llm_client=None,
        reranker_credential_handle=None, reranking_strategy="two_tier", query="q", top_k=5, rerank_start=time.time(),
        metrics=None, _otel_span_rk=None, resilient_call=resilient_call,
        record_profile_degradation=degradations.record,
        result=SimpleNamespace(documents=documents, metadata={}, timings={}),
    )

    await rerank.apply_rerank(rcfg, state)

    assert [d.id for d in state.result.documents] == ["1", "2", "3"]
    assert degradations.recorded == [("reranking_strategy", "two_tier", "skipped", "llm_policy_refused")]
    assert state.reranking_strategy == "two_tier"  # not degraded to hybrid: no second attempt


# ---- post_generation_verifier ------------------------------------------------------------------------------------


async def test_a_refused_claims_check_leaves_the_answer_unverified_with_its_reason():
    async def refusing_runner(**_kwargs: Any) -> Any:
        raise LLMPolicyRefusal(REFUSAL)

    verifier = PostGenerationVerifier(claims_runner=refusing_runner, max_retries=1, unsupported_threshold=0.10)

    out = await verifier.verify_and_maybe_fix(query="What is RAG?", answer="RAG is X.", base_documents=_docs())

    assert (out.verification_available, out.failure_code, out.reason) == (
        False, "llm_policy_refused", "verification_unavailable"
    )
    assert out.fixed is False and out.new_answer is None
    assert out.total_claims == 0  # no verdict: neither "all supported" nor "all unsupported"


async def test_a_refused_recheck_marks_the_answer_unverified_with_its_reason(monkeypatch):
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", None)

    class _Generator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "A repaired answer."}

    monkeypatch.setattr(verifier_module, "AnswerGenerator", _Generator)
    attempts = {"n": 0}

    async def runner(**_kwargs: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return {"claims": [{"id": "c1"}, {"id": "c2"}], "summary": {"supported": 0, "refuted": 1, "nei": 1}}
        raise LLMPolicyRefusal(REFUSAL)

    verifier = PostGenerationVerifier(claims_runner=runner, max_retries=1, unsupported_threshold=0.10)

    out = await verifier.verify_and_maybe_fix(query="What is RAG?", answer="RAG is X.", base_documents=_docs())

    assert attempts["n"] == 2
    assert (out.verification_available, out.failure_code, out.reason) == (
        False, "llm_policy_refused", "verification_unavailable"
    )


async def test_a_refused_regeneration_propagates_instead_of_regen_failed(monkeypatch):
    monkeypatch.setattr(verifier_module, "MultiDatabaseRetriever", None)

    class _Generator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            raise LLMPolicyRefusal(REFUSAL)

    monkeypatch.setattr(verifier_module, "AnswerGenerator", _Generator)

    async def runner(**_kwargs: Any) -> Any:
        return {"claims": [{"id": "c1"}], "summary": {"supported": 0, "refuted": 1, "nei": 0}}

    verifier = PostGenerationVerifier(claims_runner=runner, max_retries=1, unsupported_threshold=0.10)

    with pytest.raises(LLMPolicyRefusal):
        await verifier.verify_and_maybe_fix(query="What is RAG?", answer="RAG is X.", base_documents=_docs())
