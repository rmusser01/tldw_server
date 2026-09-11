import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as agentic_execution
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.LLM_Calls.openai_credentials import (
    OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG,
)
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
    def __init__(
        self,
        provider: str,
        base_url: str = "https://agentic-embeddings.example/v1",
        api_key: str = "runtime-agentic-key",
    ):
        section = "openai_api" if provider == "openai" else "huggingface_api"
        self.handle = SimpleNamespace(
            provider=provider,
            api_key=api_key,
            app_config={section: {"base_url": base_url}},
            credentials_resolved=True,
        )
        self.resolved: list[str] = []
        self.marked: list[Any] = []

    async def resolve(self, provider: str, *, model: str | None = None):
        del model
        self.resolved.append(provider)
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


def _runtime_api_key(kwargs: dict[str, Any]) -> str | None:
    handle = kwargs.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY)
    value = getattr(handle, "api_key", None)
    return value if isinstance(value, str) else None


def _runtime_base_url(kwargs: dict[str, Any]) -> str | None:
    handle = kwargs.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY)
    app_config = getattr(handle, "app_config", None)
    section = app_config.get("openai_api") if isinstance(app_config, dict) else None
    value = section.get("base_url") if isinstance(section, dict) else None
    return value if isinstance(value, str) else None


def _runtime_embedding_kwargs(runtime: _CredentialRuntime) -> dict[str, Any]:
    return {
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: runtime.handle,
        OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG: True,
    }


def _configure_agentic_provider_embedding_test(
    monkeypatch: pytest.MonkeyPatch,
    create_embeddings_batch: Any,
) -> tuple[Document, AgenticConfig]:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    embedding_settings = {
        "default_model_id": "openai:text-embedding-3-small",
        "models": {
            "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
        },
    }
    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", create_embeddings_batch)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    return (
        Document(
            id="credential-isolation-doc",
            content="alpha paragraph\n\nbeta paragraph",
            source=DataSource.MEDIA_DB,
            metadata={},
        ),
        AgenticConfig(
            top_k_docs=1,
            max_tool_calls=1,
            enable_metrics=False,
            agentic_use_provider_embeddings_within=True,
            agentic_provider_embedding_model_id="openai:text-embedding-3-small",
        ),
    )


def _capture_agentic_query_vectors(
    monkeypatch: pytest.MonkeyPatch,
    document_id: str,
) -> dict[str, list[list[float]]]:
    original_registry_factory = agentic_execution.make_default_registry
    captured: dict[str, list[list[float]]] = {}

    def capture(toolbox: AgenticToolbox):
        api_key = str(_runtime_api_key(toolbox.embedding_call_kwargs) or "legacy")
        captured.setdefault(api_key, []).append(toolbox._query_vecs[document_id].tolist())
        return original_registry_factory(toolbox)

    monkeypatch.setattr(agentic_execution, "make_default_registry", capture)
    return captured


def _credential_specific_vectors(api_key: str | None) -> list[list[float]]:
    query_vector = [0.0, 1.0] if api_key == "runtime-b-key" else [1.0, 0.0]
    return [query_vector, [1.0, 0.0], [0.0, 1.0]]


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
    assert captured["kwargs"] == _runtime_embedding_kwargs(runtime)


@pytest.mark.asyncio
async def test_explicit_agentic_embeddings_isolate_identical_requests_by_credential_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatches: list[str] = []

    def fake_create(texts, _app_config, _model_id_override=None, **kwargs):
        assert texts == ["same query", "alpha paragraph", "beta paragraph"]
        api_key = _runtime_api_key(kwargs)
        dispatches.append(api_key)
        return _credential_specific_vectors(api_key)

    doc, cfg = _configure_agentic_provider_embedding_test(monkeypatch, fake_create)
    captured_vectors = _capture_agentic_query_vectors(monkeypatch, doc.id)
    runtime_a = _CredentialRuntime("openai", api_key="runtime-a-key")
    runtime_b = _CredentialRuntime("openai", api_key="runtime-b-key")

    await agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime_a)
    await agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime_b)

    assert dispatches == ["runtime-a-key", "runtime-b-key"]
    assert runtime_a.marked == [runtime_a.handle]
    assert runtime_b.marked == [runtime_b.handle]
    assert captured_vectors == {
        "runtime-a-key": [[1.0, 0.0]],
        "runtime-b-key": [[0.0, 1.0]],
    }
    assert agentic_execution._INTRA_DOC_VEC_CACHE == {}


@pytest.mark.asyncio
async def test_repeated_explicit_agentic_embedding_calls_always_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatches: list[str] = []

    def fake_create(_texts, _app_config, _model_id_override=None, **kwargs):
        api_key = _runtime_api_key(kwargs)
        dispatches.append(api_key)
        return _credential_specific_vectors(api_key)

    doc, cfg = _configure_agentic_provider_embedding_test(monkeypatch, fake_create)
    runtime = _CredentialRuntime("openai", api_key="runtime-a-key")

    await agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime)
    await agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime)

    assert dispatches == ["runtime-a-key", "runtime-a-key"]
    assert runtime.marked == [runtime.handle, runtime.handle]
    assert agentic_execution._INTRA_DOC_VEC_CACHE == {}


@pytest.mark.asyncio
async def test_legacy_agentic_embedding_calls_still_use_vector_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatches = 0

    def fake_create(_texts, _app_config, _model_id_override=None, **kwargs):
        nonlocal dispatches
        assert kwargs == {}
        dispatches += 1
        return _credential_specific_vectors(None)

    doc, cfg = _configure_agentic_provider_embedding_test(monkeypatch, fake_create)

    await agentic_execution.tool_loop([doc], "same query", cfg)
    await agentic_execution.tool_loop([doc], "same query", cfg)

    assert dispatches == 1
    assert len(agentic_execution._INTRA_DOC_VEC_CACHE) == 1


@pytest.mark.concurrent
@pytest.mark.asyncio
async def test_concurrent_explicit_agentic_embedding_calls_do_not_share_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_mark_started = threading.Event()
    release_first_mark = threading.Event()
    dispatches: list[str] = []

    class BlockingMarkRuntime(_CredentialRuntime):
        async def mark_used(self, handle: Any) -> None:
            await super().mark_used(handle)
            first_mark_started.set()
            if not release_first_mark.wait(timeout=5):
                raise AssertionError("timed out waiting to release the first runtime")

    def fake_create(_texts, _app_config, _model_id_override=None, **kwargs):
        api_key = _runtime_api_key(kwargs)
        dispatches.append(api_key)
        return _credential_specific_vectors(api_key)

    doc, cfg = _configure_agentic_provider_embedding_test(monkeypatch, fake_create)
    captured_vectors = _capture_agentic_query_vectors(monkeypatch, doc.id)
    runtime_a = BlockingMarkRuntime("openai", api_key="runtime-a-key")
    runtime_b = _CredentialRuntime("openai", api_key="runtime-b-key")

    def run(runtime: _CredentialRuntime):
        return asyncio.run(agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime))

    first_task = asyncio.create_task(asyncio.to_thread(run, runtime_a))
    try:
        assert await asyncio.to_thread(first_mark_started.wait, 5)
        await asyncio.to_thread(run, runtime_b)
    finally:
        release_first_mark.set()
        await first_task

    assert dispatches == ["runtime-a-key", "runtime-b-key"]
    assert runtime_a.marked == [runtime_a.handle]
    assert runtime_b.marked == [runtime_b.handle]
    assert captured_vectors == {
        "runtime-a-key": [[1.0, 0.0]],
        "runtime-b-key": [[0.0, 1.0]],
    }
    assert agentic_execution._INTRA_DOC_VEC_CACHE == {}


@pytest.mark.asyncio
async def test_failed_explicit_agentic_runtime_cannot_poison_another_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatches: list[str] = []

    class FailingMarkRuntime(_CredentialRuntime):
        async def mark_used(self, handle: Any) -> None:
            await super().mark_used(handle)
            raise RuntimeError("usage mark failed")

    def fake_create(_texts, _app_config, _model_id_override=None, **kwargs):
        api_key = _runtime_api_key(kwargs)
        dispatches.append(api_key)
        return _credential_specific_vectors(api_key)

    doc, cfg = _configure_agentic_provider_embedding_test(monkeypatch, fake_create)
    captured_vectors = _capture_agentic_query_vectors(monkeypatch, doc.id)
    runtime_a = FailingMarkRuntime("openai", api_key="runtime-a-key")
    runtime_b = _CredentialRuntime("openai", api_key="runtime-b-key")

    with pytest.raises(RuntimeError, match="usage mark failed"):
        await agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime_a)
    await agentic_execution.tool_loop([doc], "same query", cfg, credential_runtime=runtime_b)

    assert dispatches == ["runtime-a-key", "runtime-b-key"]
    assert runtime_b.marked == [runtime_b.handle]
    assert captured_vectors["runtime-b-key"] == [[0.0, 1.0]]
    assert agentic_execution._INTRA_DOC_VEC_CACHE == {}


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
        captured.update(app_config=app_config, kwargs=kwargs)
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
@pytest.mark.parametrize("provider_name", ["local_api", "local"])
async def test_tool_loop_runtime_remote_local_uses_exact_selected_deployment(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    embedding_settings = {
        "default_model_id": f"{provider_name}:agentic-model",
        "models": {
            f"{provider_name}:agentic-model": SimpleNamespace(
                provider=provider_name,
                api_url="https://agentic-local.example/embeddings",
                api_key="configured-agentic-key",
            ),
        },
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        captured.update(app_config=app_config, kwargs=kwargs)
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
        agentic_provider_embedding_model_id=f"{provider_name}:agentic-model",
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
        "api_key_override": "configured-agentic-key",
        "base_url_override": "https://agentic-local.example/embeddings",
        "credentials_resolved": True,
    }
    assert "configured-agentic-key" not in repr(captured["app_config"])
    assert "configured-agentic-key" not in repr(agentic_execution._INTRA_DOC_VEC_CACHE)


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
        calls.append(_runtime_base_url(kwargs))
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


@pytest.mark.asyncio
async def test_tool_loop_uses_one_scrubbed_embedding_config_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    secret_values = {
        "server-openai-key",
        "model-openai-key",
        "nested-credential",
        "changed-local-key",
    }
    snapshots = iter(
        [
            {
                "openai_api": {"api_key": "server-openai-key"},
                "EMBEDDING_CONFIG": {
                    "default_model_id": "openai:model-a",
                    "models": {
                        "openai:model-a": {
                            "provider": "openai",
                            "model_name_or_path": "model-a",
                            "api_key": "model-openai-key",
                            "credentials": {"token": "nested-credential"},
                        }
                    },
                },
            },
            {
                "EMBEDDING_CONFIG": {
                    "default_model_id": "local_api:changed-model",
                    "models": {
                        "local_api:changed-model": {
                            "provider": "local_api",
                            "model_name_or_path": "changed-model",
                            "api_url": "https://changed-local.example/embeddings",
                            "api_key": "changed-local-key",
                        }
                    },
                }
            },
        ]
    )
    load_calls = 0
    captured: dict[str, Any] = {}

    def load_config():
        nonlocal load_calls
        load_calls += 1
        return next(snapshots)

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        captured.update(app_config=app_config, model_id=model_id_override, kwargs=kwargs)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(core_config, "load_comprehensive_config", load_config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()

    await agentic_execution.tool_loop(
        [Document(id="snapshot", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
        "alpha",
        AgenticConfig(
            top_k_docs=1,
            max_tool_calls=1,
            enable_metrics=False,
            agentic_use_provider_embeddings_within=True,
        ),
        credential_runtime=_CredentialRuntime("openai"),
    )

    assert load_calls == 1
    assert captured["app_config"]["embedding_config"]["default_model_id"] == "openai:model-a"
    assert "changed-local.example" not in repr(captured["app_config"])
    assert "openai_api" not in captured["app_config"]
    assert not any(secret in repr(captured["app_config"]) for secret in secret_values)
    runtime_kwargs = captured["kwargs"]
    assert _runtime_api_key(runtime_kwargs) == "runtime-agentic-key"
    assert runtime_kwargs[OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG] is True


@pytest.mark.asyncio
async def test_agentic_provider_vector_cache_identity_includes_config_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    current = {"model_id": "openai:model-a"}
    dispatched_models: list[str] = []

    def load_config():
        model_id = current["model_id"]
        return {
            "EMBEDDING_CONFIG": {
                "default_model_id": model_id,
                "models": {model_id: {"provider": "openai", "model_name_or_path": model_id}},
            }
        }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        dispatched_models.append(app_config["embedding_config"]["default_model_id"])
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(core_config, "load_comprehensive_config", load_config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
    )
    doc = Document(id="model-cache", content="alpha", source=DataSource.MEDIA_DB, metadata={})
    runtime = _CredentialRuntime("openai")

    await agentic_execution.tool_loop([doc], "alpha", cfg, credential_runtime=runtime)
    current["model_id"] = "openai:model-b"
    await agentic_execution.tool_loop([doc], "alpha", cfg, credential_runtime=runtime)

    assert dispatched_models == ["openai:model-a", "openai:model-b"]
    assert all("runtime-agentic-key" not in key for key in agentic_execution._INTRA_DOC_VEC_CACHE)


@pytest.mark.asyncio
async def test_legacy_local_api_vector_cache_identity_includes_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    current = {"endpoint": "https://local-a.example/embeddings"}
    dispatched_endpoints: list[str] = []

    def load_config():
        return {
            "EMBEDDING_CONFIG": {
                "default_model_id": "local_api:model-a",
                "models": {
                    "local_api:model-a": {
                        "provider": "local_api",
                        "model_name_or_path": "model-a",
                        "api_url": current["endpoint"],
                    }
                },
            }
        }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        model = app_config["embedding_config"]["models"]["local_api:model-a"]
        dispatched_endpoints.append(model["api_url"])
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(core_config, "load_comprehensive_config", load_config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    cfg = AgenticConfig(
        top_k_docs=1,
        max_tool_calls=1,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
    )
    doc = Document(id="local-endpoint-cache", content="alpha", source=DataSource.MEDIA_DB, metadata={})

    await agentic_execution.tool_loop([doc], "alpha", cfg)
    current["endpoint"] = "https://local-b.example/embeddings"
    await agentic_execution.tool_loop([doc], "alpha", cfg)

    assert dispatched_endpoints == [
        "https://local-a.example/embeddings",
        "https://local-b.example/embeddings",
    ]


@pytest.mark.asyncio
async def test_toolbox_disables_provider_embeddings_after_first_document_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings import async_embeddings
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    calls = 0
    embedding_settings = {
        "default_model_id": "openai:model-a",
        "models": {"openai:model-a": {"provider": "openai", "model_name_or_path": "model-a"}},
    }

    def fail_create(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise async_embeddings.EmbeddingProviderError("openai", code="provider_failure")

    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fail_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()
    metadata: dict[str, Any] = {}
    docs = [
        Document(id=f"failure-{idx}", content="alpha", source=DataSource.MEDIA_DB, metadata={})
        for idx in range(3)
    ]

    content, citations, _ = await agentic_execution.tool_loop(
        docs,
        "alpha",
        AgenticConfig(
            top_k_docs=3,
            max_tool_calls=3,
            enable_metrics=False,
            agentic_use_provider_embeddings_within=True,
        ),
        credential_runtime=_CredentialRuntime("openai"),
        stage_metadata=metadata,
    )

    assert calls == 1
    assert content
    assert citations
    assert metadata == {
        "embedding_coverage": "degraded",
        "failure_code": "provider_unavailable",
    }


@pytest.mark.asyncio
async def test_tool_loop_does_not_start_planner_after_embedding_setup_exhausts_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    now = 0.0
    planner_calls = 0
    embedding_settings = {
        "default_model_id": "openai:model-a",
        "models": {"openai:model-a": {"provider": "openai", "model_name_or_path": "model-a"}},
    }

    def fake_create(texts, app_config, model_id_override=None, **kwargs):
        nonlocal now
        now = 2.0
        return [[1.0, 0.0] for _ in texts]

    class Planner:
        def __init__(self, *args, **kwargs):
            nonlocal planner_calls
            planner_calls += 1

        async def generate(self, **kwargs):
            raise AssertionError("planner must not run after the deadline")

    monkeypatch.setattr(agentic_execution, "_now", lambda: now)
    monkeypatch.setattr(agentic_execution, "AnswerGenerator", Planner)
    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    agentic_execution._INTRA_DOC_VEC_CACHE.clear()

    content, citations, _ = await agentic_execution.tool_loop(
        [Document(id="deadline", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
        "alpha",
        AgenticConfig(
            top_k_docs=1,
            max_tool_calls=1,
            time_budget_sec=1.0,
            enable_metrics=False,
            use_llm_planner=True,
            agentic_use_provider_embeddings_within=True,
        ),
        credential_runtime=_CredentialRuntime("openai"),
    )

    assert planner_calls == 0
    assert content == "alpha"
    assert len(citations) == 1


@pytest.mark.asyncio
async def test_optional_planner_uses_runtime_and_degrades_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

    runtime = object()
    planner_runtimes: list[Any] = []
    planner_generate_calls = 0

    class Planner:
        def __init__(self, *args, **kwargs):
            planner_runtimes.append(kwargs.get("credential_runtime"))

        async def generate(self, **_kwargs):
            nonlocal planner_generate_calls
            planner_generate_calls += 1
            raise ByokResolutionError("credential_store_unavailable", "openai")

    monkeypatch.setattr(agentic_execution, "AnswerGenerator", Planner)
    metadata: dict[str, Any] = {}

    content, citations, _ = await agentic_execution.tool_loop(
        [Document(id="planner", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
        "alpha",
        AgenticConfig(
            top_k_docs=1,
            max_tool_calls=1,
            enable_metrics=False,
            use_llm_planner=True,
        ),
        credential_runtime=runtime,
        stage_metadata=metadata,
    )

    assert planner_runtimes == [runtime]
    assert planner_generate_calls == 1
    assert content == "alpha"
    assert len(citations) == 1
    assert metadata == {
        "planner": {
            "failure_code": "credential_store_unavailable",
            "verification_available": False,
        },
    }


@pytest.mark.asyncio
async def test_concurrent_planner_failures_keep_bounded_metadata_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

    entered = asyncio.Event()
    release = asyncio.Event()
    entered_count = 0

    class Runtime:
        def __init__(self, failure_code: str) -> None:
            self.failure_code = failure_code

    class Planner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.runtime = kwargs["credential_runtime"]

        async def generate(self, **_kwargs: object) -> dict[str, str]:
            nonlocal entered_count
            entered_count += 1
            if entered_count == 2:
                entered.set()
            await release.wait()
            raise ByokResolutionError(self.runtime.failure_code, "openai")

    monkeypatch.setattr(agentic_execution, "AnswerGenerator", Planner)
    metadata_a: dict[str, Any] = {}
    metadata_b: dict[str, Any] = {}

    async def run(runtime: Runtime, metadata: dict[str, Any]) -> None:
        await agentic_execution.tool_loop(
            [Document(id="planner", content="alpha", source=DataSource.MEDIA_DB, metadata={})],
            "alpha",
            AgenticConfig(
                top_k_docs=1,
                max_tool_calls=1,
                enable_metrics=False,
                use_llm_planner=True,
            ),
            credential_runtime=runtime,
            stage_metadata=metadata,
        )

    task_a = asyncio.create_task(run(Runtime("credential_store_unavailable"), metadata_a))
    task_b = asyncio.create_task(run(Runtime("credential_scope_revoked"), metadata_b))
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    release.set()
    await asyncio.gather(task_a, task_b)

    assert metadata_a == {
        "planner": {
            "failure_code": "credential_store_unavailable",
            "verification_available": False,
        }
    }
    assert metadata_b == {
        "planner": {
            "failure_code": "credential_scope_revoked",
            "verification_available": False,
        }
    }
