"""Real claim-verifier authority ranking across multiple source documents."""

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import Claim, HybridClaimVerifier
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SourceAuthority, VerificationStatus

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["llm", "nli"])
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ([{}, {}], SourceAuthority.SECONDARY),
        ([{"source_type": "primary"}, {}], SourceAuthority.PRIMARY),
        ([{}, {"source_type": "primary"}], SourceAuthority.PRIMARY),
        ([{"doi": "10.0000/controlled"}], SourceAuthority.PEER_REVIEWED),
        ([], SourceAuthority.SECONDARY),
    ],
    ids=["equal-secondary", "primary-first", "primary-last", "single-peer-reviewed", "no-evidence"],
)
def test_verifier_selects_ranked_enum_and_preserves_multi_source_evidence(
    mode: str, metadata: list[dict[str, Any]], expected: SourceAuthority
) -> None:
    """External inference is controlled; classification, verification and citations are real."""

    def analyze(*_args: Any, **_kwargs: Any) -> str:
        return '{"label":"supported","confidence":0.95,"rationale":"The supplied sources state this fact."}'

    verifier = HybridClaimVerifier(analyze)
    if mode == "nli":
        verifier._nli = lambda _request: [[{"label": "entailment", "score": 0.95}]]
    text = "The telescope is named Selene."
    documents = [
        Document(id=f"source-{index}", content=text, metadata=dict(value), score=0.9)
        for index, value in enumerate(metadata)
    ]
    result = asyncio.run(
        verifier.verify(Claim(id="claim-1", text=text), "What is the telescope named?", documents, mode=mode)
    )

    assert result.source_authority is expected  # nosec B101 -- Pytest verifies the returned authority remains the highest-ranked enum.
    assert [item.doc_id for item in result.evidence] == [item.id for item in documents]  # nosec B101 -- Preserve every supplied evidence identity and its stable order.
    if documents:
        assert result.status is VerificationStatus.VERIFIED  # nosec B101 -- Multiple enums must not silently downgrade successful NLI to unverified.
        assert {item["doc_id"] for item in result.citations} == {item.id for item in documents}  # nosec B101 -- Each real source remains represented by a citation.
