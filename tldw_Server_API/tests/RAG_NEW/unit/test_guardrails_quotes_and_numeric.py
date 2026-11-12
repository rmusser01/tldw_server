import pytest

from tldw_Server_API.app.core.RAG.rag_service.guardrails import (
    build_quote_citations,
    check_numeric_fidelity,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


def test_build_quote_citations_basic():


    docs = [
        Document(
            id="d1",
            content="This is an example passage with a distinctive quoted span for testing.",
            source=DataSource.MEDIA_DB,
            metadata={},
            score=0.9,
        )
    ]
    answer = 'The system found "a distinctive quoted span" relevant to the query.'
    qc = build_quote_citations(answer, docs)
    assert isinstance(qc, dict)
    assert qc.get("total", 0) >= 1
    assert qc.get("supported", 0) >= 1
    # Find the specific quote entry
    quotes = qc.get("quotes", [])
    assert any(e.get("citations") for e in quotes), "Expected at least one citation for the quoted text"
    # Verify at least one citation has verified True
    verified_any = False
    for e in quotes:
        for c in e.get("citations", []):
            if c.get("verified") is True:
                verified_any = True
                break
        if verified_any:
            break
    assert verified_any, "Expected at least one verified citation"


def test_numeric_fidelity_normalization_handles_units_and_currency():


     # Answer uses currency and word/percent units
    answer = "Revenue reached $1.5 million and margins improved by 23%."
    # Sources use normalized numeric forms
    docs = [
        Document(
            id="d2",
            content="The company reported revenue of 1,500,000 with an increase of 23 percent over last year.",
            source=DataSource.MEDIA_DB,
            metadata={},
            score=0.8,
        )
    ]
    nf = check_numeric_fidelity(answer, docs)
    assert nf.missing == set(), f"Unexpected missing tokens: {nf.missing}"
    # Ensure present contains both tokens in normalized form
    assert any(tok.endswith("m") for tok in nf.present), "Expected million mapping to 'm' suffix"
    assert any(tok.endswith("%") for tok in nf.present), "Expected percent mapping to '%' suffix"


def test_unmatched_quote_does_not_count_as_supported():
    doc = Document(id="vega", content="Baseline calibration took three days.", metadata={})

    result = build_quote_citations('The report states "calibration took ten days".', [doc])

    assert result["quotes"] == [{"text": "calibration took ten days", "citations": []}]
    assert result["coverage"] == 0
