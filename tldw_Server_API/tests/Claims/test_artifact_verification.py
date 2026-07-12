from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import (
    Claim,
    ClaimVerification,
    ExtractionResult,
    VerificationResult,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, VerificationStatus


pytestmark = pytest.mark.unit


def _noop_analyze(*args: Any, **kwargs: Any) -> str:
    return '{"claims": []}'


class _FakeClaimsEngine:
    def __init__(self, _analyze_fn: Any, statuses: list[VerificationStatus]):
        self.analyze_fn = _analyze_fn
        self.statuses = statuses
        self.verify_calls: list[dict[str, Any]] = []

    async def extract_claims_only(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("explicit artifact unit claims should not run extraction")

    async def verify_claims_only(self, claims: list[Claim], *args: Any, **kwargs: Any) -> VerificationResult:
        self.verify_calls.append({"claims": claims, "kwargs": kwargs})
        self.analyze_fn(
            "global-claims-default",
            "verify",
            "prompt",
            None,
            "system",
            model_override="global-claims-model",
        )
        return VerificationResult(
            verifications=[
                ClaimVerification(
                    claim=claim,
                    status=self.statuses[index],
                    confidence=0.92,
                    rationale=f"{self.statuses[index].value} rationale",
                )
                for index, claim in enumerate(claims)
            ],
            summary={"total": len(claims)},
        )


class _EngineRef:
    engine: _FakeClaimsEngine | None = None

    def __getattr__(self, name: str) -> Any:
        if self.engine is None:
            raise AttributeError(name)
        return getattr(self.engine, name)


def _patch_engine(monkeypatch: pytest.MonkeyPatch, statuses: list[VerificationStatus]) -> _FakeClaimsEngine:
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification

    ref = _EngineRef()

    def _factory(analyze_fn: Any) -> _FakeClaimsEngine:
        ref.engine = _FakeClaimsEngine(analyze_fn, statuses)
        return ref.engine

    monkeypatch.setattr(artifact_verification, "ClaimsEngine", _factory)
    return ref  # type: ignore[return-value]


def test_verification_fails_without_selected_source_documents() -> None:
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="quiz",
            units=[
                ArtifactVerificationUnit(
                    unit_id="quiz:q1:answer",
                    text="Acme revenue was $10 million.",
                    claims=["Acme revenue was $10 million."],
                )
            ],
            source_documents=[],
            generation_provider="llamacpp",
            generation_model="local-test",
            analyze_fn=_noop_analyze,
        )

        assert result.verdict == "failed"
        assert result.metadata["reason"] == "no_source_documents"
        assert result.report["total_claims"] == 0

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("status", "expected_verdict"),
    [
        (VerificationStatus.VERIFIED, "grounded"),
        (VerificationStatus.UNVERIFIED, "needs_revision"),
        (VerificationStatus.CONTESTED, "needs_revision"),
        (VerificationStatus.MISLEADING, "needs_revision"),
        (VerificationStatus.REFUTED, "failed"),
        (VerificationStatus.HALLUCINATION, "failed"),
        (VerificationStatus.NUMERICAL_ERROR, "failed"),
        (VerificationStatus.MISQUOTED, "failed"),
        (VerificationStatus.CITATION_NOT_FOUND, "failed"),
    ],
)
def test_claim_statuses_map_to_artifact_verdicts(
    monkeypatch: pytest.MonkeyPatch,
    status: VerificationStatus,
    expected_verdict: str,
) -> None:
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    engine = _patch_engine(monkeypatch, [status])
    source = Document(
        id="source-1",
        content="Acme revenue was $9 million.",
        metadata={"title": "Annual report"},
    )

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="quiz",
            units=[
                ArtifactVerificationUnit(
                    unit_id="quiz:q3:answer",
                    text="Acme revenue was $10 million.",
                    claims=["Acme revenue was $10 million."],
                )
            ],
            source_documents=[source],
            generation_provider="llamacpp",
            generation_model="local-test",
            analyze_fn=_noop_analyze,
        )

        assert result.verdict == expected_verdict
        assert result.unit_results[0].unit_id == "quiz:q3:answer"
        assert result.unit_results[0].statuses == [status.value]
        assert result.report["claims"][0]["claim_id"].startswith("quiz:q3:answer:")
        assert engine.verify_calls[0]["kwargs"]["doc_only_mode"] is True
        assert engine.verify_calls[0]["kwargs"]["retrieve_fn"] is None

    asyncio.run(_run())


def test_provider_and_model_are_recorded_in_result_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    _patch_engine(monkeypatch, [VerificationStatus.VERIFIED])

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="flashcards",
            units=[
                ArtifactVerificationUnit(
                    unit_id="flashcard:7:back",
                    text="The trial enrolled 42 participants.",
                    claims=["The trial enrolled 42 participants."],
                )
            ],
            source_documents=[
                Document(
                    id="study",
                    content="The trial enrolled 42 participants.",
                    metadata={"title": "Study"},
                )
            ],
            generation_provider="llamacpp",
            generation_model="http://127.0.0.1:9099",
            analyze_fn=_noop_analyze,
        )

        assert result.metadata["generation_provider"] == "llamacpp"
        assert result.metadata["generation_model"] == "http://127.0.0.1:9099"
        assert result.metadata["verification_provider"] == "llamacpp"
        assert result.metadata["verification_model"] == "http://127.0.0.1:9099"
        assert result.metadata["verification_llm_is_default"] is True

    asyncio.run(_run())


def test_claims_verification_provider_override_is_used_and_marked_non_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    calls: list[dict[str, Any]] = []

    def _capture_analyze(
        api_endpoint: str | None,
        input_data: Any,
        prompt: str | None,
        api_key: str | None,
        system_message: str | None,
        **kwargs: Any,
    ) -> str:
        calls.append({"api_endpoint": api_endpoint, "model_override": kwargs.get("model_override")})
        return '{"claims": []}'

    _patch_engine(monkeypatch, [VerificationStatus.VERIFIED])

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="flashcards",
            units=[
                ArtifactVerificationUnit(
                    unit_id="flashcard:7:back",
                    text="The trial enrolled 42 participants.",
                    claims=["The trial enrolled 42 participants."],
                )
            ],
            source_documents=[
                Document(
                    id="study",
                    content="The trial enrolled 42 participants.",
                    metadata={"title": "Study"},
                )
            ],
            generation_provider="llamacpp",
            generation_model="generation-model",
            verification_provider="openrouter",
            verification_model="claims-verifier-model",
            analyze_fn=_capture_analyze,
        )

        assert calls
        assert calls[-1] == {
            "api_endpoint": "openrouter",
            "model_override": "claims-verifier-model",
        }
        assert result.metadata["verification_provider"] == "openrouter"
        assert result.metadata["verification_model"] == "claims-verifier-model"
        assert result.metadata["verification_llm_is_default"] is False
        assert result.metadata["verification_llm_differs_from_generation"] is True

    asyncio.run(_run())


def test_claims_verification_config_default_is_used_when_request_omits_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )
    from tldw_Server_API.app.core.config import settings

    calls: list[dict[str, Any]] = []
    keys = ["CLAIMS_VERIFICATION_PROVIDER", "CLAIMS_VERIFICATION_MODEL"]
    snapshot = {key: settings.get(key) for key in keys}

    def _capture_analyze(
        api_endpoint: str | None,
        input_data: Any,
        prompt: str | None,
        api_key: str | None,
        system_message: str | None,
        **kwargs: Any,
    ) -> str:
        calls.append({"api_endpoint": api_endpoint, "model_override": kwargs.get("model_override")})
        return '{"claims": []}'

    _patch_engine(monkeypatch, [VerificationStatus.VERIFIED])

    async def _run() -> None:
        settings["CLAIMS_VERIFICATION_PROVIDER"] = "openrouter"
        settings["CLAIMS_VERIFICATION_MODEL"] = "claims-default-model"
        try:
            result = await verify_generated_artifact_against_sources(
                artifact_type="flashcards",
                units=[
                    ArtifactVerificationUnit(
                        unit_id="flashcard:7:back",
                        text="The trial enrolled 42 participants.",
                        claims=["The trial enrolled 42 participants."],
                    )
                ],
                source_documents=[
                    Document(
                        id="study",
                        content="The trial enrolled 42 participants.",
                        metadata={"title": "Study"},
                    )
                ],
                generation_provider="llamacpp",
                generation_model="generation-model",
                analyze_fn=_capture_analyze,
            )

            assert calls[-1] == {
                "api_endpoint": "openrouter",
                "model_override": "claims-default-model",
            }
            assert result.metadata["verification_provider"] == "openrouter"
            assert result.metadata["verification_model"] == "claims-default-model"
            assert result.metadata["verification_llm_source"] == "config"

            override = await verify_generated_artifact_against_sources(
                artifact_type="flashcards",
                units=[
                    ArtifactVerificationUnit(
                        unit_id="flashcard:8:back",
                        text="The trial enrolled 42 participants.",
                        claims=["The trial enrolled 42 participants."],
                    )
                ],
                source_documents=[
                    Document(
                        id="study",
                        content="The trial enrolled 42 participants.",
                        metadata={"title": "Study"},
                    )
                ],
                generation_provider="llamacpp",
                generation_model="generation-model",
                verification_provider="request-provider",
                verification_model="request-model",
                analyze_fn=_capture_analyze,
            )
            assert calls[-1] == {
                "api_endpoint": "request-provider",
                "model_override": "request-model",
            }
            assert override.metadata["verification_llm_source"] == "request"
        finally:
            for key, value in snapshot.items():
                if value is None:
                    settings.pop(key, None)
                else:
                    settings[key] = value

    asyncio.run(_run())


def test_text_cap_prevents_grounded_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    _patch_engine(monkeypatch, [VerificationStatus.VERIFIED])
    monkeypatch.setattr(artifact_verification, "_MAX_TEXT_CHARS_BY_ARTIFACT", {"quiz": 12})

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="quiz",
            units=[
                ArtifactVerificationUnit(
                    unit_id="quiz:q1",
                    text="This generated question text is intentionally longer than the cap.",
                    claims=["The generated question text is longer than the cap."],
                )
            ],
            source_documents=[
                Document(
                    id="source",
                    content="This generated question text is intentionally longer than the cap.",
                    metadata={},
                )
            ],
            generation_provider="llamacpp",
            generation_model="local-test",
            analyze_fn=_noop_analyze,
        )

        assert result.verdict == "needs_revision"
        assert "text" in result.metadata["cap_hit"]
        assert result.unit_results[0].verdict == "needs_revision"
        assert result.unit_results[0].metadata["text_truncated"] is True

    asyncio.run(_run())


def test_no_claims_preserves_truncated_unit_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    monkeypatch.setattr(artifact_verification, "_MAX_TEXT_CHARS_BY_ARTIFACT", {"quiz": 12})

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="quiz",
            units=[
                ArtifactVerificationUnit(
                    unit_id="quiz:q1",
                    text="This generated question text exceeds the configured cap.",
                    claims=[],
                )
            ],
            source_documents=[Document(id="source", content="Source text", metadata={})],
            generation_provider="llamacpp",
            generation_model="local-test",
            analyze_fn=_noop_analyze,
        )

        assert result.unit_results[0].metadata["reason"] == "no_claims"
        assert result.unit_results[0].metadata["text_truncated"] is True
        assert result.unit_results[0].metadata["original_text_length"] > 12

    asyncio.run(_run())


def test_explicit_claim_cap_prevents_grounded_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    engine = _patch_engine(monkeypatch, [VerificationStatus.VERIFIED])
    monkeypatch.setattr(artifact_verification, "_MAX_CLAIMS_PER_UNIT_BY_ARTIFACT", {"quiz": 1})

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="quiz",
            units=[
                ArtifactVerificationUnit(
                    unit_id="quiz:q1",
                    text="Alpha is supported. Beta is supported.",
                    claims=["Alpha is supported.", "Beta is supported."],
                )
            ],
            source_documents=[
                Document(
                    id="source",
                    content="Alpha is supported. Beta is supported.",
                    metadata={},
                )
            ],
            generation_provider="llamacpp",
            generation_model="local-test",
            analyze_fn=_noop_analyze,
        )

        assert result.verdict == "needs_revision"
        assert "claims" in result.metadata["cap_hit"]
        assert len(engine.verify_calls[0]["claims"]) == 1
        assert result.unit_results[0].metadata["claims_truncated"] is True

    asyncio.run(_run())


def test_prose_units_use_claims_engine_extraction_and_keep_unit_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Claims_Extraction import artifact_verification
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
        ArtifactVerificationUnit,
        verify_generated_artifact_against_sources,
    )

    class _ExtractingEngine:
        def __init__(self, _analyze_fn: Any):
            self.extraction_calls = 0

        async def extract_claims_only(self, answer: str, *args: Any, **kwargs: Any) -> ExtractionResult:
            self.extraction_calls += 1
            return ExtractionResult(
                claims=[Claim(id="c1", text="The summary says the trial enrolled 42 participants.")],
                extractor_mode="llm",
            )

        async def verify_claims_only(self, claims: list[Claim], *args: Any, **kwargs: Any) -> VerificationResult:
            return VerificationResult(
                verifications=[
                    ClaimVerification(
                        claim=claim,
                        status=VerificationStatus.VERIFIED,
                        confidence=0.91,
                    )
                    for claim in claims
                ],
                summary={"total": len(claims)},
            )

    engine = _ExtractingEngine(_noop_analyze)
    monkeypatch.setattr(artifact_verification, "ClaimsEngine", lambda analyze_fn: engine)

    async def _run() -> None:
        result = await verify_generated_artifact_against_sources(
            artifact_type="audio_summary",
            units=[
                ArtifactVerificationUnit(
                    unit_id="audio_script:paragraph:4",
                    text="The summary says the trial enrolled 42 participants.",
                )
            ],
            source_documents=[
                Document(
                    id="study",
                    content="The trial enrolled 42 participants.",
                    metadata={"title": "Study"},
                )
            ],
            generation_provider="llamacpp",
            generation_model="generation-model",
            analyze_fn=_noop_analyze,
        )

        assert engine.extraction_calls == 1
        assert result.verdict == "grounded"
        assert result.report["claims"][0]["claim_id"] == "audio_script:paragraph:4:c1"
        assert result.unit_results[0].claim_ids == ["audio_script:paragraph:4:c1"]

    asyncio.run(_run())
