"""Sanitizer coverage for agentic chunker fallback logs."""

import sys
import types
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.RAG.rag_service import agentic_chunker as ac
from tldw_Server_API.app.core.RAG.rag_service import claims as claims_mod
from tldw_Server_API.app.core.RAG.rag_service import database_retrievers
from tldw_Server_API.app.core.RAG.rag_service import generation as generation_mod
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit


_SENSITIVE_MESSAGE = (
    "backend exploded at /private/rag-secret.db "
    "api_key=sk-test-private-token"
)
_SENSITIVE_FRAGMENTS = (
    "/private/",
    "rag-secret.db",
    "sk-test-private-token",
    "api_key=",
    "backend exploded",
)


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(str(message))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(str(message))

    def opt(self, *args: object, **kwargs: object) -> "_LoggerStub":
        return self


class _EmptyRetriever:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
        return []


class _ExplodingRetriever:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
        raise RuntimeError(_SENSITIVE_MESSAGE)


def _doc() -> Document:
    return Document(
        id="agentic-sanitize-doc",
        content="Transformers rely on attention for long-range dependencies.",
        metadata={"title": "Transformer", "source": "media_db"},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )


def _assert_no_sensitive_log_fragments(messages: list[str]) -> None:
    rendered = "\n".join(messages)
    for fragment in _SENSITIVE_FRAGMENTS:
        assert fragment not in rendered


@pytest.fixture(autouse=True)
def _clear_agentic_caches() -> None:
    ac.clear_agentic_caches()


@pytest.mark.asyncio
async def test_coarse_retrieval_fallback_warning_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _ExplodingRetriever)

    result = await ac.agentic_rag_pipeline(
        query="coarse retrieval sanitizer",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(enable_metrics=False),
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert result.metadata["coarse_docs"] == []
    assert any("Agentic coarse retrieval failed" in msg for msg in logger_stub.warnings)
    _assert_no_sensitive_log_fragments(logger_stub.warnings)


@pytest.mark.asyncio
async def test_coarse_retrieval_propagates_required_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_error = ChatAuthenticationError(
        "Embedding provider authentication failed.",
        provider="openai",
    )

    class _ProviderFailingRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            raise provider_error

    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _ProviderFailingRetriever)

    with pytest.raises(ChatAuthenticationError) as exc_info:
        await ac.agentic_rag_pipeline(
            query="required provider failure",
            sources=["media_db"],
            search_mode="fts",
            agentic=ac.AgenticConfig(enable_metrics=False),
            enable_generation=False,
            enable_citations=False,
            credential_runtime=object(),
        )

    assert exc_info.value is provider_error
    assert logger_stub.warnings == []


@pytest.mark.asyncio
async def test_coarse_retrieval_keeps_legacy_typed_failure_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProviderFailingRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            raise ChatAuthenticationError("legacy provider failure", provider="openai")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _ProviderFailingRetriever)

    result = await ac.agentic_rag_pipeline(
        query="legacy provider fallback",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(enable_metrics=False),
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert result.metadata["coarse_docs"] == []
    assert any("Agentic coarse retrieval failed" in msg for msg in logger_stub.warnings)


@pytest.mark.asyncio
async def test_media_db_fallback_retrieval_warning_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _EmptyRetriever)
    monkeypatch.setattr(database_retrievers, "MediaDBRetriever", _ExplodingRetriever)

    result = await ac.agentic_rag_pipeline(
        query="media fallback sanitizer",
        sources=["media_db"],
        media_db_path=str(tmp_path / "media.db"),
        search_mode="fts",
        agentic=ac.AgenticConfig(enable_metrics=False),
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert result.metadata["coarse_docs"] == []
    assert any("Agentic Media DB fallback retrieval failed" in msg for msg in logger_stub.warnings)
    _assert_no_sensitive_log_fragments(logger_stub.warnings)


@pytest.mark.asyncio
async def test_vlm_late_chunking_skip_debug_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)

    class _DocRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            return [_doc()]

    def _raise_backend_error(name: str | None = None) -> None:
        raise RuntimeError(_SENSITIVE_MESSAGE)

    registry_module = types.ModuleType("vlm_registry_stub")
    registry_module.get_backend = _raise_backend_error
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry",
        registry_module,
    )
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _DocRetriever)

    result = await ac.agentic_rag_pipeline(
        query="vlm sanitizer",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(
            top_k_docs=1,
            agentic_enable_vlm_late_chunking=True,
            enable_metrics=False,
        ),
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert result.errors == []
    assert logger_stub.debugs == ["Agentic VLM late chunking skipped"]
    _assert_no_sensitive_log_fragments(logger_stub.debugs)


@pytest.mark.asyncio
async def test_generation_fallback_uses_bounded_error_and_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)

    class _DocRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            return [_doc()]

    class _ExplodingGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def generate(self, *args: object, **kwargs: object) -> dict[str, str]:
            raise RuntimeError(_SENSITIVE_MESSAGE)

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _DocRetriever)
    monkeypatch.setattr(generation_mod, "AnswerGenerator", _ExplodingGenerator)

    result = await ac.agentic_rag_pipeline(
        query="generation sanitizer",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(top_k_docs=1, enable_metrics=False),
        enable_generation=True,
        enable_citations=False,
    )

    assert result.generated_answer is None
    assert result.errors == ["Answer generation failed"]
    assert _SENSITIVE_MESSAGE not in result.errors
    assert logger_stub.warnings == ["Agentic generation failed"]
    _assert_no_sensitive_log_fragments(logger_stub.warnings)


@pytest.mark.asyncio
async def test_claims_verification_skip_debug_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)

    class _DocRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            return [_doc()]

    class _AnswerGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def generate(self, *args: object, **kwargs: object) -> dict[str, str]:
            return {"answer": "Attention lets transformers connect distant tokens."}

    class _ExplodingClaimsEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, *args: object, **kwargs: object) -> dict[str, object]:
            raise RuntimeError(_SENSITIVE_MESSAGE)

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _DocRetriever)
    monkeypatch.setattr(generation_mod, "AnswerGenerator", _AnswerGenerator)
    monkeypatch.setattr(claims_mod, "ClaimsEngine", _ExplodingClaimsEngine, raising=False)

    result = await ac.agentic_rag_pipeline(
        query="claims sanitizer",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(top_k_docs=1, enable_metrics=False),
        enable_generation=True,
        enable_claims=True,
        enable_citations=False,
    )

    assert result.generated_answer == "Attention lets transformers connect distant tokens."
    assert "claims" not in result.metadata
    assert logger_stub.debugs == ["Agentic claims verification skipped"]
    _assert_no_sensitive_log_fragments(logger_stub.debugs)


@pytest.mark.asyncio
async def test_agentic_pipeline_threads_runtime_to_retrieval_and_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object()
    captured: dict[str, object] = {}

    class _DocRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["retrieval_runtime"] = kwargs.get("credential_runtime")

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            return [_doc()]

    async def fake_tool_loop(*args: object, **kwargs: object):
        captured["tool_runtime"] = kwargs.get("credential_runtime")
        return "grounded chunk", [{"document_id": _doc().id, "start": 0, "end": 8}], []

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _DocRetriever)
    monkeypatch.setattr(ac, "_tool_loop", fake_tool_loop)

    result = await ac.agentic_rag_pipeline(
        query="runtime propagation",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(enable_tools=True, enable_metrics=False),
        credential_runtime=runtime,
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert captured == {
        "retrieval_runtime": runtime,
        "tool_runtime": runtime,
    }


@pytest.mark.asyncio
async def test_agentic_hosted_embedding_failure_uses_hash_fallback_and_bounded_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

    class _DocRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            return [_doc()]

    class _FailingRuntime:
        async def resolve(self, provider: str):
            raise ByokResolutionError("credential_store_unavailable", provider)

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _DocRetriever)
    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {
            "EMBEDDING_CONFIG": {
                "default_model_id": "openai:text-embedding-3-small",
                "models": {
                    "openai:text-embedding-3-small": {"provider": "openai"},
                },
            }
        },
    )

    result = await ac.agentic_rag_pipeline(
        query="attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(
            enable_tools=True,
            enable_metrics=False,
            agentic_use_provider_embeddings_within=True,
            agentic_provider_embedding_model_id="openai:text-embedding-3-small",
        ),
        credential_runtime=_FailingRuntime(),
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert result.metadata["agentic_embeddings"] == {
        "embedding_coverage": "degraded",
        "failure_code": "credential_store_unavailable",
    }


@pytest.mark.asyncio
async def test_agentic_outer_cache_cannot_hide_later_runtime_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    class _DocRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def retrieve(self, *args: object, **kwargs: object) -> list[Document]:
            return [_doc()]

    class _WorkingRuntime:
        def __init__(self) -> None:
            self.handle = SimpleNamespace(
                provider="openai",
                api_key="runtime-key",
                app_config={"openai_api": {"base_url": "https://embedding.example/v1"}},
                credentials_resolved=True,
            )

        async def resolve(self, provider: str):
            return self.handle

        async def mark_used(self, handle: object) -> None:
            pass

    class _FailingRuntime:
        async def resolve(self, provider: str):
            raise ByokResolutionError("credential_store_unavailable", provider)

    embedding_settings = {
        "default_model_id": "openai:text-embedding-3-small",
        "models": {
            "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
        },
    }
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _DocRetriever)
    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config",
        lambda: {"EMBEDDING_CONFIG": embedding_settings},
    )
    monkeypatch.setattr(
        Embeddings_Create,
        "create_embeddings_batch",
        lambda texts, *args, **kwargs: [[1.0, 0.0] for _ in texts],
    )
    cfg = ac.AgenticConfig(
        enable_tools=True,
        enable_metrics=False,
        agentic_use_provider_embeddings_within=True,
        agentic_provider_embedding_model_id="openai:text-embedding-3-small",
    )

    first = await ac.agentic_rag_pipeline(
        query="attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        credential_runtime=_WorkingRuntime(),
        enable_generation=False,
        enable_citations=False,
    )
    second = await ac.agentic_rag_pipeline(
        query="attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        credential_runtime=_FailingRuntime(),
        enable_generation=False,
        enable_citations=False,
    )

    assert first.documents
    assert second.metadata["agentic_embeddings"] == {
        "embedding_coverage": "degraded",
        "failure_code": "credential_store_unavailable",
    }
