"""Provider/model isolation at the unified-pipeline evidence-chain boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as unified_pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("enable_citations", [False, True], ids=["citations-off", "citations-on"])
async def test_direct_evidence_chain_builders_keep_concurrent_provider_models_isolated(
    monkeypatch: pytest.MonkeyPatch,
    enable_citations: bool,
) -> None:
    document = Document(
        id="provider-isolation",
        content="Credential snapshots remain request-owned across evidence-chain stages.",
        source=DataSource.MEDIA_DB,
        metadata={"title": "Provider isolation"},
        score=0.9,
    )
    runtimes = {"alpha": object(), "beta": object()}
    selected = {
        "alpha": ("anthropic", "claude-alpha"),
        "beta": ("openai", "gpt-beta"),
    }
    calls: list[dict[str, Any]] = []
    pre_generation_queries: set[str] = set()
    both_pre_generation_calls_entered = asyncio.Event()
    release_pre_generation_calls = asyncio.Event()

    class FakeRetriever:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def retrieve(self, *_args: Any, **_kwargs: Any) -> list[Document]:
            return [document]

    class FakeAnswerGenerator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": f"Answer for {kwargs['query']}."}

    class FakeCitationGenerator:
        async def generate_citations(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                academic_citations=[],
                chunk_citations=[],
                inline_markers={},
                citation_map={},
            )

    class CapturingEvidenceChainBuilder:
        def __init__(self, **kwargs: Any) -> None:
            self.provider = kwargs.get("llm_provider")
            self.model = kwargs.get("llm_model")
            self.runtime = kwargs.get("credential_runtime")

        async def build_chains(self, **kwargs: Any) -> Any:
            query = str(kwargs["query"])
            phase = "post" if kwargs.get("generated_answer") else "pre"
            calls.append(
                {
                    "query": query,
                    "phase": phase,
                    "provider": self.provider,
                    "model": self.model,
                    "runtime": self.runtime,
                }
            )
            if phase == "pre":
                pre_generation_queries.add(query)
                if len(pre_generation_queries) == 2:
                    both_pre_generation_calls_entered.set()
                await release_pre_generation_calls.wait()
            return SimpleNamespace(
                chains=[],
                overall_confidence=0.0,
                multi_hop_detected=False,
                metadata={
                    "total_nodes": 0,
                    "total_claims": 0,
                    "supported_claims": 0,
                },
            )

    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(unified_pipeline, "CitationGenerator", FakeCitationGenerator)
    monkeypatch.setattr(
        unified_pipeline,
        "EvidenceChainBuilder",
        CapturingEvidenceChainBuilder,
    )

    async def run(label: str) -> Any:
        provider, model = selected[label]
        return await unified_pipeline.unified_rag_pipeline(
            query=label,
            sources=["media_db"],
            generation_provider=provider,
            generation_model=model,
            enable_cache=False,
            enable_citations=enable_citations,
            enable_evidence_chains=True,
            enable_generation=True,
            enable_reranking=False,
            enable_pre_retrieval_clarification=False,
            credential_runtime=runtimes[label],
        )

    tasks = [asyncio.create_task(run(label)) for label in selected]
    try:
        await asyncio.wait_for(both_pre_generation_calls_entered.wait(), timeout=2.0)
        release_pre_generation_calls.set()
        results = await asyncio.gather(*tasks)
    finally:
        release_pre_generation_calls.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    assert all(result.generated_answer for result in results)
    assert len(calls) == 4
    for label, (provider, model) in selected.items():
        request_calls = [call for call in calls if call["query"] == label]
        assert [call["phase"] for call in request_calls] == ["pre", "post"]
        assert all(call["provider"] == provider for call in request_calls)
        assert all(call["model"] == model for call in request_calls)
        assert all(call["runtime"] is runtimes[label] for call in request_calls)
