from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
    Claim,
    ClaimVerification,
    VerificationResult,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, VerificationStatus


pytestmark = pytest.mark.unit
_UNIT_ID_SUFFIX = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=16,
)
_UNIT_IDS = st.builds(
    lambda prefix, suffix: f"{prefix}:{suffix}",
    st.sampled_from(["quiz", "flashcard", "slide", "table", "mindmap"]),
    _UNIT_ID_SUFFIX,
)


def _noop_analyze(*args: Any, **kwargs: Any) -> str:
    return '{"claims": []}'


class _UnitIdEngine:
    async def extract_claims_only(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("explicit artifact unit claims should not run extraction")

    async def verify_claims_only(self, claims: list[Claim], *args: Any, **kwargs: Any) -> VerificationResult:
        return VerificationResult(
            verifications=[
                ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.VERIFIED,
                    confidence=0.9,
                )
                for claim in claims
            ],
            summary={"total": len(claims)},
        )


@settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    unit_ids=st.lists(
        _UNIT_IDS,
        min_size=1,
        max_size=8,
        unique=True,
    )
)
def test_unit_ids_round_trip_through_verification_mapping(
    monkeypatch: pytest.MonkeyPatch,
    unit_ids: list[str],
) -> None:
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    monkeypatch.setattr(artifact_verification, "ClaimsEngine", lambda analyze_fn: _UnitIdEngine())

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="quiz",
            units=[
                ArtifactVerificationUnit(
                    unit_id=unit_id,
                    text=f"{unit_id} fact.",
                    claims=[f"{unit_id} fact."],
                )
                for unit_id in unit_ids
            ],
            source_documents=[
                Document(id="source", content=" ".join(f"{unit_id} fact." for unit_id in unit_ids), metadata={})
            ],
            generation_provider="llamacpp",
            generation_model="local-test",
            analyze_fn=_noop_analyze,
        )

        assert {unit.unit_id for unit in result.unit_results} == set(unit_ids)
        for claim in result.report["claims"]:
            assert any(claim["claim_id"].startswith(f"{unit_id}:") for unit_id in unit_ids)

    asyncio.run(_run())
