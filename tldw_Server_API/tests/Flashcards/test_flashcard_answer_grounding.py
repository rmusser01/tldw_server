"""Verify card meaning without treating an isolated source word as a claim."""

import asyncio
import json

import pytest

from tldw_Server_API.app.api.v1.endpoints.flashcards import _build_flashcard_verification_units
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import verify_generated_artifact_against_sources
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import HybridClaimVerifier
from tldw_Server_API.app.core.RAG.rag_service.types import Document


pytestmark = pytest.mark.unit

SOURCE = (
    "Mitochondria produce energy in cells. DNA stands for deoxyribonucleic acid. "
    "Photosynthesis converts light energy into chemical energy. "
    "The adult human skeleton has 206 bones. Water boils at 100 degrees Celsius at sea level."
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("Mitochondria", "grounded"), ("Photosynthesis", "failed"), ("Mitochondria produce ATP", "failed")],
)
def test_verifier_receives_question_relationship_and_rejects_wrong_answers(monkeypatch, answer, expected):
    """Only the external judge is replaced; unit construction and verification stay real."""

    async def no_local_model(_self):
        return None

    monkeypatch.setattr(HybridClaimVerifier, "_get_nli", no_local_model)

    def judge(_provider, claim, prompt, *_args, **_kwargs):
        assert SOURCE in prompt
        supported = claim == "Question: What produces energy in cells?\nAnswer: Mitochondria"
        return json.dumps({"label": "supported" if supported else "refuted", "confidence": 0.95})

    result = asyncio.run(
        verify_generated_artifact_against_sources(
            artifact_type="flashcards",
            units=_build_flashcard_verification_units(
                [{"front": "What produces energy in cells?", "back": answer, "model_type": "basic"}]
            ),
            source_documents=[Document(id="flashcards-source", content=SOURCE, metadata={})],
            generation_provider="llamacpp",
            generation_model="test-model",
            analyze_fn=judge,
        )
    )

    assert result.verdict == expected
    assert result.report["claims"][0]["evidence"][0]["snippet"] == SOURCE


def test_question_context_does_not_drop_other_generated_claims():
    units = _build_flashcard_verification_units(
        [
            {
                "front": "What produces energy in cells?",
                "back": "Mitochondria",
                "notes": "An unsupported note",
                "extra": "An unsupported extra",
            }
        ]
    )
    assert [unit.text for unit in units] == [
        "Question: What produces energy in cells?\nAnswer: Mitochondria",
        "An unsupported note",
        "An unsupported extra",
    ]


@pytest.mark.parametrize("card_plan", [None, [{"card_type": "basic", "count": 1}]])
def test_generation_instructs_model_to_use_only_supplied_facts(monkeypatch, card_plan):
    from tldw_Server_API.app.core.Workflows.adapters.content import generation

    sent = {}

    async def generate(**kwargs):
        sent.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [{"front": "What produces energy?", "back": "Mitochondria", "generation_type": "basic"}]
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(generation, "perform_chat_api_call_async", generate)
    result = asyncio.run(
        generation.run_flashcard_generate_adapter(
            {
                "text": SOURCE,
                "num_cards": 1,
                "card_plan": card_plan,
                "provider": "llamacpp",
            },
            {},
        )
    )
    assert result["count"] == 1
    assert SOURCE in sent["messages"][0]["content"]
    assert "Use only facts explicitly stated in the source" in sent["system_message"]


def test_quoted_source_term_does_not_prove_an_incorrect_answer(monkeypatch):
    async def no_local_model(_self):
        return None

    monkeypatch.setattr(HybridClaimVerifier, "_get_nli", no_local_model)

    def judge(*_args, **_kwargs):
        return '{"label": "refuted", "confidence": 0.95, "rationale": "Source says energy, not light."}'

    result = asyncio.run(
        verify_generated_artifact_against_sources(
            artifact_type="flashcards",
            units=_build_flashcard_verification_units([{"front": 'What do "Mitochondria" produce?', "back": "Light"}]),
            source_documents=[Document(id="source", content=SOURCE, metadata={})],
            generation_provider="llamacpp",
            generation_model="test",
            analyze_fn=judge,
        )
    )
    assert result.verdict == "failed"
