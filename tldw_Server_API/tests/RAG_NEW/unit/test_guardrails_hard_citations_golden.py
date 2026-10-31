import pytest

from tldw_Server_API.app.core.RAG.rag_service.guardrails import build_hard_citations
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def test_build_hard_citations_offsets_golden():

    text = "We ran 42 experiments. The findings were consistent across trials."
    d = Document(id="gold1", content=text, metadata={"title": "Paper"}, source=DataSource.MEDIA_DB, score=0.9)
    answer = "We ran 42 experiments. The findings were consistent across trials."
    hc = build_hard_citations(answer, [d])
    assert isinstance(hc, dict) and hc.get("sentences")
    for entry in hc["sentences"]:
        s = entry.get("text", "")
        cites = entry.get("citations") or []
        assert cites, "Expected at least one citation per sentence"
        st, en = cites[0].get("start", 0), cites[0].get("end", 0)
        assert text[int(st) : int(en)].strip() in {s.strip(), s.strip()[: len(text[int(st) : int(en)].strip())]}


def test_unmatched_answer_does_not_receive_an_arbitrary_source_span():
    doc = Document(id="vega", content="Baseline calibration took three days.", metadata={})

    result = build_hard_citations("Baseline calibration took ten days.", [doc])

    assert result["sentences"] == [{"text": "Baseline calibration took ten days.", "citations": []}]
    assert result["coverage"] == 0


def test_partial_middle_match_does_not_support_the_entire_sentence():
    shared = "The greenhouse pilot measured water use across ten plots over twenty-one days"
    doc = Document(id="vega", content=f"Original {shared} with no cost data.", metadata={})

    result = build_hard_citations(f"Invented {shared} with a cost of one million dollars.", [doc])

    assert result["supported"] == 0


@pytest.mark.parametrize("status", ["refuted", "misquoted", "numerical_error", "unverified"])
def test_claim_verdicts_that_are_not_verified_do_not_count_as_supported(status):
    text = "Baseline calibration took three days."
    doc = Document(id="vega", content=text, metadata={})
    payload = [
        {
            "text": "Baseline calibration took ten days.",
            "status": status,
            "citations": [{"doc_id": "vega", "start": 0, "end": len(text)}],
            "evidence": [{"doc_id": "vega", "snippet": text}],
        }
    ]

    assert build_hard_citations(payload[0]["text"], [doc], payload)["supported"] == 0


@pytest.mark.parametrize("doc_id,start,end", [("missing", 0, 10), ("vega", -1, 10), ("vega", 0, 500), ("vega", 10, 10)])
def test_claim_citations_require_a_returned_document_and_valid_offsets(doc_id, start, end):
    text = "Baseline calibration took three days."
    doc = Document(id="vega", content=text, metadata={})
    payload = [
        {
            "text": text,
            "status": "verified",
            "citations": [{"doc_id": doc_id, "start": start, "end": end}],
            "evidence": [{"doc_id": "vega", "snippet": text}],
        }
    ]

    assert build_hard_citations(text, [doc], payload)["supported"] == 0


def test_verified_paraphrase_keeps_the_actual_source_evidence_span():
    source = "Baseline calibration took three days."
    doc = Document(id="vega", content="Introduction. " + source, metadata={})
    claim = "Calibration lasted three days."
    payload = [
        {
            "text": claim,
            "status": "verified",
            "label": "supported",
            "citations": [{"doc_id": "vega", "start": 0, "end": len(claim)}],
            "evidence": [{"doc_id": "vega", "snippet": source}],
        }
    ]

    result = build_hard_citations(claim, [doc], payload)

    assert result["sentences"][0]["citations"] == [{"doc_id": "vega", "start": 14, "end": 14 + len(source)}]
    assert result["coverage"] == 1


def test_unmatched_claim_evidence_cannot_turn_an_arbitrary_span_into_support():
    doc = Document(id="vega", content="Baseline calibration took three days.", metadata={})
    claim = "Calibration lasted ten days."
    payload = [
        {
            "text": claim,
            "status": "verified",
            "citations": [{"doc_id": "vega", "start": 0, "end": 20}],
            "evidence": [{"doc_id": "vega", "snippet": "Unrelated invented evidence."}],
        }
    ]

    assert build_hard_citations(claim, [doc], payload)["supported"] == 0
