"""Exercise CI biology fixtures through the real generation and verification code."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    verify_generated_artifact_against_sources,
)
from tldw_Server_API.app.core.Flashcards.verification import build_flashcard_verification_units
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.Workflows.adapters.content import generation

from mock_openai_server.mock_openai.config import MockConfig
from mock_openai_server.mock_openai.server import app, get_config_instance

SOURCE = (
    "The mitochondria is the powerhouse of the cell.\n\n"
    "DNA stands for deoxyribonucleic acid.\n\n"
    "Photosynthesis converts light energy into chemical energy.\n\n"
    "The human body has 206 bones.\n\n"
    "Water boils at 100 degrees Celsius at sea level."
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "verdict", "verified_count"),
    [
        (None, "grounded", 5),
        ("swapped_answer", "needs_revision", 4),
        ("extra_claim", "needs_revision", 4),
        ("missing_evidence", "needs_revision", 0),
    ],
)
def test_ci_biology_fixture_verifies_only_declared_answers_and_source(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None,
    verdict: str,
    verified_count: int,
) -> None:
    """The real ClaimsEngine must accept all five known cards and reject altered inputs."""
    root = Path(__file__).resolve().parents[2]
    config = MockConfig.from_file(
        root / "apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/ci-journeys.json"
    )
    monkeypatch.setitem(app.dependency_overrides, get_config_instance, lambda: config)
    with TestClient(app) as client:

        def complete(
            system_message: str, messages: list[dict[str, Any]], model: str
        ) -> dict[str, Any]:
            """Send the application prompt to the configured provider HTTP endpoint."""
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer sk-ci-controlled-downstream"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system_message}, *messages],
                },
            )
            response.raise_for_status()
            return response.json()

        async def generate(**kwargs: Any) -> dict[str, Any]:
            """Route generation through provider HTTP, retaining application parsing."""
            return complete(kwargs["system_message"], kwargs["messages"], kwargs["model"])

        def judge(
            _provider: str | None,
            _claim: str,
            prompt: str,
            _api_key: str | None,
            system_message: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            """Route actual semantic judge prompts through the same provider fixture."""
            return complete(
                system_message,
                [{"role": "user", "content": prompt}],
                kwargs["model_override"],
            )

        # Replace only the downstream call; generation parsing and the grounding gate stay real.
        monkeypatch.setattr(generation, "perform_chat_api_call_async", generate)
        generated = asyncio.run(
            generation.run_flashcard_generate_adapter(
                {"text": SOURCE, "num_cards": 5, "provider": "openai", "model": "gpt-4.1-mini"}, {}
            )
        )
        assert generated["count"] == 5
        cards = generated["flashcards"]
        evidence = SOURCE
        if mutation == "swapped_answer":
            cards[0]["back"] = "Photosynthesis."
        elif mutation == "extra_claim":
            cards[0]["back"] += " It produces ATP."
        elif mutation == "missing_evidence":
            evidence = "DNA stands for deoxyribonucleic acid."
        result = asyncio.run(
            verify_generated_artifact_against_sources(
                artifact_type="flashcards",
                units=build_flashcard_verification_units(cards),
                source_documents=[Document(id="flashcards-source", content=evidence, metadata={})],
                generation_provider="openai",
                generation_model="gpt-4.1-mini",
                analyze_fn=judge,
            )
        )
    assert result.verdict == verdict
    assert result.report["verified_count"] == verified_count
