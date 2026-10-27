"""Qualify outgoing flashcard instructions through the real generation and claims path."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.flashcards import generate_flashcards
from tldw_Server_API.app.api.v1.schemas.flashcards import FlashcardGenerateRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Claims_Extraction import artifact_verification
from tldw_Server_API.app.core.Claims_Extraction.claims_engine import HybridClaimVerifier
from tldw_Server_API.app.core.Workflows.adapters.content import generation

pytestmark = pytest.mark.unit

FACTS = [
    "Mitochondria produce energy in cells.",
    "DNA stands for deoxyribonucleic acid.",
    "Photosynthesis converts light energy into chemical energy.",
    "The adult human skeleton has 206 bones.",
    "Water boils at 100 degrees Celsius at sea level.",
]
SOURCE = " ".join(FACTS)
QUESTIONS = [
    "What produces energy in cells?",
    "What does DNA stand for?",
    "What does photosynthesis convert?",
    "How many bones does the adult human skeleton have?",
    "At what temperature does water boil at sea level?",
]
PROVIDER = "ollama"
MODEL = "qwen2.5:0.5b"
MIXED_PLAN = [
    {"card_type": "basic", "count": 1},
    {"card_type": "basic_reverse", "count": 1},
    {"card_type": "cloze", "count": 1},
    {"card_type": "true_false", "count": 2},
]


def _cards(types: list[str]) -> list[dict[str, Any]]:
    """Independent source-fact fixtures, never generated from production prompt text."""
    cards = []
    for fact, question, card_type in zip(FACTS, QUESTIONS, types, strict=True):
        front = question
        if card_type == "cloze":
            front = "{{c1::" + fact + "}}"
        elif card_type == "true_false":
            front = "True or false: " + fact
        cards.append({
            "front": front,
            "back": ("True. " if card_type == "true_false" else "") + fact,
            "tags": ["biology"],
            "generation_type": card_type,
        })
    return cards


@dataclass
class LlmSeams:
    """Capture external inference only; endpoint/adapter/verification remain real."""

    cards: list[dict[str, Any]] = field(default_factory=lambda: _cards(["basic"] * 5))
    judge_label: str = "supported"
    generation_calls: list[dict[str, Any]] = field(default_factory=list)
    judge_calls: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def llm_seams(monkeypatch: pytest.MonkeyPatch) -> LlmSeams:
    seams = LlmSeams()

    async def external_generate(**kwargs: Any) -> dict[str, Any]:
        seams.generation_calls.append(kwargs)
        return {"choices": [{"message": {"content": json.dumps(seams.cards)}}]}

    def external_judge(
        provider: str,
        claim: Any,
        prompt: str | None,
        _api_key: str | None,
        system: str | None,
        **kwargs: Any,
    ) -> str:
        seams.judge_calls.append({
            "provider": provider,
            "model": kwargs.get("model_override"),
            "claim": claim,
            "prompt": prompt,
            "system": system,
        })
        return json.dumps({
            "label": seams.judge_label,
            "confidence": 0.95,
            "rationale": "Independent fixed external judge verdict.",
        })

    async def external_nli_unavailable(_self: HybridClaimVerifier) -> None:
        return None

    monkeypatch.setattr(generation, "perform_chat_api_call_async", external_generate)
    monkeypatch.setattr(artifact_verification, "_default_analyze_fn", external_judge)
    monkeypatch.setattr(HybridClaimVerifier, "_get_nli", external_nli_unavailable)
    return seams


def _request(**options: Any) -> FlashcardGenerateRequest:
    return FlashcardGenerateRequest(
        text=SOURCE,
        num_cards=5,
        provider=PROVIDER,
        model=MODEL,
        claims_verification_provider=PROVIDER,
        claims_verification_model=MODEL,
        **options,
    )


@pytest.mark.parametrize(
    ("request_options", "generation_types"),
    [
        ({"card_type": "basic"}, ["basic"] * 5),
        ({"card_type": "basic_reverse"}, ["basic_reverse"] * 5),
        ({"card_type": "cloze"}, ["cloze"] * 5),
        ({"card_plan": [{"card_type": "basic", "count": 5}]}, ["basic"] * 5),
        ({"card_plan": MIXED_PLAN}, ["basic", "basic_reverse", "cloze", "true_false", "true_false"]),
    ],
    ids=["legacy-basic", "legacy-reverse", "legacy-cloze", "planned-basic", "planned-mix"],
)
@pytest.mark.asyncio
async def test_endpoint_requests_field_types_and_source_cards_without_copyable_sample_values(
    llm_seams: LlmSeams,
    request_options: dict[str, Any],
    generation_types: list[str],
) -> None:
    """Red checks outgoing instructions, not fabricated model adherence to a new phrase."""
    llm_seams.cards = _cards(generation_types)
    response = await generate_flashcards(
        _request(**request_options),
        current_user=User(id=1, username="prompt-contract-user"),
    )

    assert len(llm_seams.generation_calls) == 1  # nosec B101 - pytest contract assertion.
    sent = llm_seams.generation_calls[0]
    assert sent["messages"] == [{"role": "user", "content": f"Generate flashcards from:\n\n{SOURCE}"}]  # nosec B101 - pytest contract assertion.
    assert (sent["api_provider"], sent["model"]) == (PROVIDER, MODEL)  # nosec B101 - pytest contract assertion.
    assert (sent["max_tokens"], sent["temperature"]) == (4000, 0.7)  # nosec B101 - pytest contract assertion.
    system = sent["system_message"]
    assert "Generate 5 flashcards" in system  # nosec B101 - pytest contract assertion.
    assert "Use only facts explicitly stated in the source" in system  # nosec B101 - pytest contract assertion.
    # A value-bearing array is copyable content, rather than a field/type description.
    assert re.search(r'\[\s*\{\s*"front"\s*:\s*"', system) is None  # nosec B101 - pytest contract assertion.
    assert all(token in system.lower() for token in ("front", "back", "string", "tags", "distinct"))  # nosec B101 - pytest contract assertion.
    if "card_plan" in request_options:
        for row in request_options["card_plan"]:
            assert f"exactly {row['count']} {row['card_type']} flashcard" in system  # nosec B101 - pytest contract assertion.
        assert "generation_type" in system  # nosec B101 - pytest contract assertion.

    assert response["count"] == 5  # nosec B101 - pytest contract assertion.
    assert [(card["front"], card["back"]) for card in response["flashcards"]] == [(card["front"], card["back"]) for card in llm_seams.cards]  # nosec B101 - pytest contract assertion.
    assert Counter(card["generation_type"] for card in response["flashcards"]) == Counter(generation_types)  # nosec B101 - pytest contract assertion.
    assert [card["model_type"] for card in response["flashcards"]] == ["basic" if value == "true_false" else value for value in generation_types]  # nosec B101 - pytest contract assertion.
    verification = response["claim_verification"]
    assert verification["verdict"] == "grounded"  # nosec B101 - pytest contract assertion.
    assert len(verification["unitResults"]) == len(llm_seams.judge_calls) > 0  # nosec B101 - pytest contract assertion.
    assert all(SOURCE in call["prompt"] for call in llm_seams.judge_calls)  # nosec B101 - pytest contract assertion.
    assert {(call["provider"], call["model"]) for call in llm_seams.judge_calls} == {(PROVIDER, MODEL)}  # nosec B101 - pytest contract assertion.
    assert all(f"Question: {card['front']}\nAnswer: {card['back']}" in {call["claim"] for call in llm_seams.judge_calls} for card in llm_seams.cards)  # nosec B101 - pytest contract assertion.


@pytest.mark.parametrize(("label", "verdict"), [("nei", "failed"), ("refuted", "failed")])
@pytest.mark.asyncio
async def test_endpoint_retains_real_claims_refusal_for_unsupported_drafts(
    llm_seams: LlmSeams, label: str, verdict: str
) -> None:
    llm_seams.cards = [{"front": "Q", "back": "A", "tags": []} for _ in range(5)]
    llm_seams.judge_label = label
    with pytest.raises(HTTPException) as raised:
        await generate_flashcards(
            _request(card_type="basic"),
            current_user=User(id=1, username="prompt-contract-user"),
        )

    assert raised.value.status_code == 422  # nosec B101 - pytest refusal assertion.
    assert raised.value.detail["code"] == "claim_verification_failed"  # nosec B101 - pytest refusal assertion.
    verification = raised.value.detail["claimVerification"]
    assert verification["verdict"] == verdict  # nosec B101 - pytest refusal assertion.
    assert len(verification["unitResults"]) == len(llm_seams.judge_calls) == 10  # nosec B101 - pytest refusal assertion.
    assert all(unit["verdict"] != "grounded" for unit in verification["unitResults"])  # nosec B101 - pytest refusal assertion.
    assert all(SOURCE in call["prompt"] for call in llm_seams.judge_calls)  # nosec B101 - pytest refusal assertion.


@pytest.mark.asyncio
async def test_endpoint_retains_empty_generation_refusal_before_claims(llm_seams: LlmSeams) -> None:
    llm_seams.cards = []
    with pytest.raises(HTTPException) as raised:
        await generate_flashcards(
            _request(card_type="basic"),
            current_user=User(id=1, username="prompt-contract-user"),
        )

    assert (raised.value.status_code, raised.value.detail) == (422, "Flashcard generation returned no usable cards")  # nosec B101 - pytest refusal assertion.
    assert len(llm_seams.generation_calls) == 1  # nosec B101 - pytest refusal assertion.
    assert llm_seams.judge_calls == []  # nosec B101 - pytest refusal assertion.
