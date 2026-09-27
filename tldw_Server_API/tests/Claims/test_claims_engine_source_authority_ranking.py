"""Real verifier paths rank evidence enums by authority, not object ordering."""

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
    Claim,
    HybridClaimVerifier,
)
from tldw_Server_API.app.core.RAG.rag_service.types import (
    Document,
    SourceAuthority,
    VerificationStatus,
)

pytestmark = pytest.mark.unit

METADATA_BY_RANK = {
    1: {},
    2: {"source_type": "whitepaper"},
    3: {"doi": "10.0000/example"},
    4: {"url": "https://example.gov/report"},
    5: {"source_type": "primary"},
}


async def verify_ranks(ranks, mode):
    def analyze(*args, **kwargs):
        return '{"label":"supported","confidence":0.99,"rationale":"Evidence supports claim."}'

    verifier = HybridClaimVerifier(analyze)
    if mode == "nli":
        # Replace the optional model only; evidence selection and decisions stay real.
        verifier._nli = lambda inputs: [[{"label": "entailment", "score": 0.99}]]
    documents = [
        Document(
            id=str(index),
            content="The sensor is in the garden.",
            metadata=METADATA_BY_RANK[rank],
        )
        for index, rank in enumerate(ranks)
    ]
    return await verifier.verify(
        Claim(id="claim", text="The sensor is in the garden."),
        query="Where is the sensor?",
        base_documents=documents,
        mode=mode,
        doc_only_mode=True,
    )


@pytest.mark.parametrize("mode", ["llm", "nli"])
@pytest.mark.parametrize(
    "ranks,expected",
    [
        ([], SourceAuthority.SECONDARY),
        ([2], SourceAuthority.INDUSTRY),
        ([1, 1], SourceAuthority.SECONDARY),
        ([1, 5, 3], SourceAuthority.PRIMARY),
        ([4, 2, 3], SourceAuthority.GOVERNMENT),
    ],
)
def test_verifier_preserves_highest_authority_enum(ranks, expected, mode):
    result = asyncio.run(verify_ranks(ranks, mode))
    assert result.source_authority is expected
    if ranks:
        assert result.status is VerificationStatus.VERIFIED
        assert len(result.citations) == len(ranks)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
    st.sampled_from(["llm", "nli"]),
)
def test_authority_rank_is_order_independent(ranks, mode):
    forward = asyncio.run(verify_ranks(ranks, mode))
    reverse = asyncio.run(verify_ranks(list(reversed(ranks)), mode))
    assert forward.source_authority.value == max(ranks)
    assert reverse.source_authority is forward.source_authority
    assert forward.status is reverse.status is VerificationStatus.VERIFIED
