"""Sanitizer coverage for agentic chunker fallback logs."""

import sys
import types

import pytest

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
    assert logger_stub.warnings == ["Agentic coarse retrieval failed"]
    _assert_no_sensitive_log_fragments(logger_stub.warnings)


@pytest.mark.asyncio
async def test_media_db_fallback_retrieval_warning_omits_raw_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(ac, "logger", logger_stub)
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", _EmptyRetriever)
    monkeypatch.setattr(database_retrievers, "MediaDBRetriever", _ExplodingRetriever)

    result = await ac.agentic_rag_pipeline(
        query="media fallback sanitizer",
        sources=["media_db"],
        media_db_path="/tmp/media.db",
        search_mode="fts",
        agentic=ac.AgenticConfig(enable_metrics=False),
        enable_generation=False,
        enable_citations=False,
    )

    assert result.documents
    assert result.metadata["coarse_docs"] == []
    assert logger_stub.warnings == ["Agentic Media DB fallback retrieval failed"]
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
async def test_generation_fallback_warning_omits_raw_exception_but_preserves_result_error(
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
    assert result.errors == [_SENSITIVE_MESSAGE]
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
