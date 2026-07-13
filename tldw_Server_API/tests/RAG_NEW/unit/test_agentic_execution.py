from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as agentic_execution
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import (
    AgenticConfig,
    AgenticToolbox,
    build_agentic_derived_evidence,
    build_agentic_execution_context,
)
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit


class _CredentialRuntime:
    def __init__(self, provider: str, base_url: str = "https://agentic-embeddings.example/v1"):
        section = "openai_api" if provider == "openai" else "huggingface_api"
        self.handle = SimpleNamespace(
            provider=provider,
            api_key="runtime-agentic-key",
            app_config={section: {"base_url": base_url}},
            credentials_resolved=True,
        )
        self.resolved: list[str] = []
        self.marked: list[Any] = []

    async def resolve(self, provider: str):
        self.resolved.append(provider)
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


def test_build_agentic_derived_evidence_tracks_actual_lineage_only():
    retrieved = RetrievedEvidence(
        documents=[
            {"id": "doc-1", "content": "one"},
            {"id": "doc-2", "content": "two"},
        ],
        metadata={},
    )

    derived = build_agentic_derived_evidence(
        retrieved_evidence=retrieved,
        synthetic_chunk={"id": "synthetic", "content": "merged"},
        derived_from_document_ids=("doc-2",),
        coarse_docs_window=[{"id": "doc-1"}, {"id": "doc-2"}],
    )

    assert isinstance(derived, DerivedEvidence)
    assert derived.derived_from_document_ids == ("doc-2",)
    assert derived.metadata["coarse_docs"] == [{"id": "doc-1"}, {"id": "doc-2"}]


def test_agentic_toolbox_open_section_prefers_db_structure_lookup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeDb:
        def lookup_section_by_heading(self, media_id: int, heading: str):
            captured["media_id"] = media_id
            captured["heading"] = heading
            return (7, 21, "Results")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.rag_enable_structure_index",
        lambda default=True: True,
    )
    monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", lambda: FakeDb())

    doc = Document(
        id="doc-1",
        content="# Intro\nalpha\n\n# Results\nbeta\n",
        metadata={"title": "Paper", "media_id": 42},
        source=DataSource.MEDIA_DB,
    )
    toolbox = AgenticToolbox([doc], AgenticConfig(enable_section_index=True))

    assert toolbox.open_section(doc, "Results") == (7, 21)
    assert captured == {"media_id": 42, "heading": "Results"}


def test_build_agentic_execution_context_derives_effective_payload_and_config() -> None:
    resolved_request = ResolvedRAGRequest(
        query="agentic config from canonical contracts",
        strategy="agentic",
        payload={
            "query": "agentic config from canonical contracts",
            "strategy": "agentic",
            "sources": ["notes"],
            "search_mode": "vector",
            "top_k": 5,
            "min_score": 0.15,
            "index_namespace": "tenant-a",
            "agentic_max_tool_calls": 6,
            "agentic_enable_tools": True,
            "agentic_coverage_target": 0.92,
            "agentic_enable_metrics": False,
            "agentic_debug_trace": False,
            "debug_mode": True,
        },
        index_namespace="tenant-a",
        rag_profile="balanced",
        user_id="1",
        feedback_user_id="1",
    )
    retrieval_plan = RetrievalPlan(
        query=resolved_request.query,
        sources=("notes",),
        search_mode="vector",
        top_k=5,
        min_score=0.15,
        index_namespace="tenant-a",
    )

    effective_payload, agentic_config = build_agentic_execution_context(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        payload_override={
            **resolved_request.payload,
            "agentic_top_k_docs": 4,
            "agentic_window_chars": 1400,
        },
    )

    assert effective_payload["sources"] == ["notes"]
    assert effective_payload["search_mode"] == "vector"
    assert effective_payload["top_k"] == 5
    assert effective_payload["min_score"] == 0.15
    assert effective_payload["index_namespace"] == "tenant-a"
    assert agentic_config.top_k_docs == 4
    assert agentic_config.window_chars == 1400
    assert agentic_config.max_tool_calls == 6
    assert agentic_config.enable_tools is True
    assert agentic_config.coverage_target == 0.92
    assert agentic_config.enable_metrics is False
    assert agentic_config.debug_trace is True


@pytest.mark.asyncio
async def test_tool_loop_uses_runtime_credentials_for_hosted_provider_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    embedding_settings = {
        "default_model_id": "openai:text-embedding-3-small",
        "models": {
            "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
        },
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        captured.update(
            texts=texts,
            app_config=app_config,
            model_id_override=model_id_override,
            kwargs=kwargs,
        )
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    doc = Document(
        id="agentic-provider-doc",
        content="alpha paragraph\n\nbeta paragraph",
        source=DataSource.MEDIA_DB,
        metadata={},
    )
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
        agentic_provider_embedding_model_id="openai:text-embedding-3-small",
    )

    await agentic_execution.tool_loop(
        [doc],
        "alpha",
        cfg,
        credential_runtime=runtime,
    )

    assert runtime.resolved == ["openai"]
    assert runtime.marked == [runtime.handle]
    assert captured["texts"][0] == "alpha"
    assert captured["kwargs"] == {
        "api_key_override": "runtime-agentic-key",
        "base_url_override": "https://agentic-embeddings.example/v1",
        "credentials_resolved": True,
    }


@pytest.mark.asyncio
async def test_tool_loop_provider_embedding_latency_consumes_time_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    now = 0.0
    search_calls = 0
    embedding_settings = {
        "default_model_id": "openai:text-embedding-3-small",
        "models": {
            "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
        },
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        nonlocal now
        now = 2.0
        return [[1.0, 0.0] for _ in texts]

    def search_within(*args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        return [(0, 5)]

    monkeypatch.setattr(agentic_execution, "_now", lambda: now)
    monkeypatch.setattr(
        agentic_execution,
        "make_default_registry",
        lambda toolbox: {"search_within": search_within},
    )
    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        time_budget_sec=1.0,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
        agentic_provider_embedding_model_id="openai:text-embedding-3-small",
    )

    content, citations, trace = await agentic_execution.tool_loop(
        [Document(id="slow", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
        "alpha",
        cfg,
        credential_runtime=_CredentialRuntime("openai"),
    )

    assert content == "alpha"
    assert len(citations) == 1
    assert trace == []
    assert search_calls == 0


@pytest.mark.asyncio
async def test_tool_loop_local_huggingface_embeddings_do_not_resolve_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    embedding_settings = {
        "default_model_id": "huggingface:local-model",
        "models": {
            "huggingface:local-model": SimpleNamespace(provider="huggingface"),
        },
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
        agentic_provider_embedding_model_id="huggingface:local-model",
    )

    await agentic_execution.tool_loop(
        [Document(id="local", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
        "alpha",
        cfg,
        credential_runtime=runtime,
    )

    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured["kwargs"] == {}


@pytest.mark.asyncio
async def test_tool_loop_runtime_local_api_uses_exact_endpoint_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    embedding_settings = {
        "default_model_id": "local_api:agentic-model",
        "models": {
            "local_api:agentic-model": SimpleNamespace(
                provider="local_api",
                api_url="https://agentic-local.example/embeddings",
                api_key="configured-key-must-not-be-used",
            ),
        },
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
        agentic_provider_embedding_model_id="local_api:agentic-model",
    )

    await agentic_execution.tool_loop(
        [Document(id="local-api", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
        "alpha",
        cfg,
        credential_runtime=runtime,
    )

    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured["kwargs"] == {
        "api_key_override": None,
        "base_url_override": "https://agentic-local.example/embeddings",
        "credentials_resolved": True,
    }


@pytest.mark.asyncio
async def test_agentic_provider_vector_cache_identity_includes_authorized_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    calls: list[str] = []
    embedding_settings = {
        "default_model_id": "openai:text-embedding-3-small",
        "models": {
            "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
        },
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        calls.append(kwargs["base_url_override"])
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
        agentic_provider_embedding_model_id="openai:text-embedding-3-small",
    )
    doc = Document(id="cache-doc", content="alpha", source=DataSource.MEDIA_DB, metadata={})

    await agentic_execution.tool_loop(
        [doc],
        "alpha",
        cfg,
        credential_runtime=_CredentialRuntime("openai", "https://endpoint-a.example/v1"),
    )
    await agentic_execution.tool_loop(
        [doc],
        "alpha",
        cfg,
        credential_runtime=_CredentialRuntime("openai", "https://endpoint-b.example/v1"),
    )

    assert calls == ["https://endpoint-a.example/v1", "https://endpoint-b.example/v1"]
    assert all("runtime-agentic-key" not in key for key in agentic_execution._INTRA_DOC_VEC_CACHE)
