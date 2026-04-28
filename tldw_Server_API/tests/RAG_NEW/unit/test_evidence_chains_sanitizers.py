import pytest

from tldw_Server_API.app.core.RAG.rag_service import evidence_chains
from tldw_Server_API.app.core.RAG.rag_service.evidence_chains import EvidenceChainBuilder
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str) -> None:
        self.debugs.append(str(message))


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
