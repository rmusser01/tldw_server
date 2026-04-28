import pytest

from tldw_Server_API.app.core.RAG.rag_service import evidence_chains
from tldw_Server_API.app.core.RAG.rag_service.evidence_chains import EvidenceChainBuilder
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(str(message))

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))


@pytest.mark.asyncio
async def test_llm_fact_extraction_fallback_log_omits_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evidence_chains, "logger", logger_stub)

    builder = EvidenceChainBuilder(enable_llm_extraction=True)

    async def fail_llm_extract(_doc: Document, _query: str):
        raise RuntimeError("fact extraction failed for /private/rag/evidence.db?token=secret-token")

    monkeypatch.setattr(builder, "_llm_extract_facts", fail_llm_extract)
    document = Document(
        id="doc-1",
        content="RAG systems use evidence chains to support answers.",
        source=DataSource.MEDIA_DB,
        score=0.9,
        metadata={"title": "Evidence"},
    )

    facts = await builder._extract_facts_from_document(document, "evidence chains")

    assert facts
    assert logger_stub.debugs == ["LLM fact extraction failed, using heuristic"]
    joined = "\n".join(logger_stub.debugs)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "evidence.db" not in joined


@pytest.mark.asyncio
async def test_llm_extract_facts_warning_omits_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(evidence_chains, "logger", logger_stub)

    from tldw_Server_API.app.core.RAG.rag_service import generation

    class FailingAnswerGenerator:
        def __init__(self, **_kwargs) -> None:
            pass

        async def generate(self, **_kwargs):
            raise RuntimeError("provider failed for /private/rag/evidence.db?token=secret-token")

    monkeypatch.setattr(generation, "AnswerGenerator", FailingAnswerGenerator)

    builder = EvidenceChainBuilder(enable_llm_extraction=True)
    document = Document(
        id="doc-1",
        content="RAG systems use evidence chains to support answers.",
        source=DataSource.MEDIA_DB,
        score=0.9,
        metadata={"title": "Evidence"},
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        await builder._llm_extract_facts(document, "evidence chains")

    assert logger_stub.warnings == ["LLM fact extraction failed"]
    joined = "\n".join(logger_stub.warnings)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "evidence.db" not in joined
