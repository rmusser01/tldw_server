import pytest

from tldw_Server_API.app.core.RAG.rag_service import citations
from tldw_Server_API.app.core.RAG.rag_service import evidence_chains
from tldw_Server_API.app.core.RAG.rag_service.citations import (
    AcademicCitationFormatter,
    CitationGenerator,
)
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


class _MetricsStub:
    def increment(self, *_args, **_kwargs) -> None:
        return None


def test_citation_date_parse_log_omits_raw_value_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(citations, "logger", logger_stub)
    monkeypatch.setattr(citations, "get_metrics_registry", lambda: _MetricsStub())

    raw_date = "not-a-date /private/rag/citation-source.json?token=secret-token"
    formatter = AcademicCitationFormatter()

    formatted = formatter._format_date(raw_date, style="mla")

    assert formatted == raw_date
    assert logger_stub.debugs == ["Citation date parse failed; returning raw"]
    joined = "\n".join(logger_stub.debugs)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "Invalid isoformat" not in joined


@pytest.mark.asyncio
async def test_evidence_chain_building_fallback_log_omits_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(citations, "logger", logger_stub)

    async def fail_build_chains(self, *_args, **_kwargs):
        raise RuntimeError(
            "chain build failed for /private/rag/evidence.db?token=secret-token"
        )

    monkeypatch.setattr(
        evidence_chains.EvidenceChainBuilder,
        "build_chains",
        fail_build_chains,
    )

    document = Document(
        id="chunk-1",
        content="RAG systems use evidence chains to support answers.",
        source=DataSource.MEDIA_DB,
        score=0.9,
        metadata={"title": "Evidence Chains", "author": "Ada Lovelace"},
        source_document_id="doc-1",
    )
    generator = CitationGenerator()

    result, chain_result = await generator.generate_citations_with_chains(
        documents=[document],
        query="evidence chains",
        generated_answer="Evidence chains support RAG answers.",
    )

    assert chain_result is None
    assert result.chunk_citations
    assert result.chunk_citations[0].chunk_id == "chunk-1"
    assert logger_stub.warnings == ["Evidence chain building failed"]
    joined = "\n".join(logger_stub.warnings)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "evidence.db" not in joined
