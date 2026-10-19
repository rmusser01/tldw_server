"""Real HTTP/provider transport contracts for source-paired distinct cards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import uuid4

import httpx
import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import flashcards, study_suggestions
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    verify_generated_artifact_against_sources,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Flashcards.verification import build_flashcard_verification_units
from tldw_Server_API.app.core.LLM_Calls import adapter_registry, provider_config_resolution
from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration
MODEL = "qwen2.5:0.5b"
SOURCE = (
    "The mitochondria is the powerhouse of the cell. "
    "DNA stands for deoxyribonucleic acid. "
    "Photosynthesis converts light energy into chemical energy. "
    "The human body has 206 bones. "
    "Water boils at 100 degrees Celsius at sea level. Primary revision one."
)
VALID = [
    {"front": "What is the powerhouse of the cell?", "back": "The mitochondria.", "tags": []},
    {"front": "What does DNA stand for?", "back": "Deoxyribonucleic acid.", "tags": []},
    {"front": "What does photosynthesis convert?", "back": "Light energy into chemical energy.", "tags": []},
    {"front": "How many bones does the human body have?", "back": "206 bones.", "tags": []},
    {"front": "At what temperature does water boil at sea level?", "back": "100 degrees Celsius.", "tags": []},
]
OBSERVED = [
    {"front": "The mitochondria is the powerhouse of the cell.", "back": "DNA stands for deoxyribonucleic acid.", "tags": []},
    {"front": "Photosynthesis converts light energy into chemical energy.", "back": "The human body has 206 bones.", "tags": []},
    {"front": "Water boils at 100 degrees Celsius at sea level.", "back": "Primary revision one.", "tags": []},
    {"front": "Primary revision one.", "back": "The human body has 206 bones.", "tags": []},
]


@dataclass
class InferenceTransport:
    """Only external inference is substituted; payload construction stays real."""

    cards: list[dict] = field(default_factory=lambda: [dict(card) for card in VALID])
    judge: str = '{"label":"supported","confidence":0.95,"rationale":"Fixed external inference."}'
    requests: list[dict] = field(default_factory=list)
    responses_closed: int = 0

    def fetch(self, **kwargs):
        payload = kwargs["json"]
        assert kwargs["configured_endpoint"].matches(kwargs["url"])  # nosec B101 - real boundary contract
        self.requests.append(payload)
        is_judge = "fact-checking judge" in payload["messages"][0]["content"]
        content = self.judge if is_judge else json.dumps(self.cards)
        response = httpx.Response(200, request=httpx.Request("POST", kwargs["url"]),
                                  json={"choices": [{"message": {"role": "assistant", "content": content}}]})
        transport = self

        class Response:
            def raise_for_status(self):
                response.raise_for_status()

            def json(self):
                return response.json()

            def close(self):
                response.close()
                transport.responses_closed += 1

        return Response()


@pytest.fixture
def inference_transport(private_pair_paths, monkeypatch):
    """Configure a private synthetic server origin and inject its transport only."""
    config = {"ollama_api": {"api_url": "http://127.0.0.1:19443/v1", "model": MODEL,
                             "api_retries": 0, "streaming": False}}
    monkeypatch.setattr(provider_config_resolution, "load_settings", lambda: config)
    monkeypatch.setattr(local_adapters, "load_settings", lambda: config)
    monkeypatch.setitem(settings, "CLAIMS_LLM_PROVIDER", "ollama")
    monkeypatch.setitem(settings, "CLAIMS_LLM_MODEL", MODEL)
    monkeypatch.setitem(settings, "CLAIMS_ENABLE_LOCAL_NLI", False)
    transport = InferenceTransport()
    adapter = adapter_registry.get_registry().get_adapter("ollama")
    monkeypatch.setattr(adapter, "http_fetcher", transport.fetch)
    return transport


@pytest.fixture
def private_pair_paths(tmp_path, monkeypatch):
    """Supported auxiliary fixture paths remain bound through app teardown."""
    for key, path in {
        "USER_DB_BASE_DIR": tmp_path / "users",
        "JOBS_DB_PATH": tmp_path / "jobs.db",
        "MONITORING_ALERTS_DB": tmp_path / "alerts.db",
        "SYSTEM_LOG_FILE_PATH": tmp_path / "system.log",
        "WORKFLOWS_ARTIFACTS_DIR": tmp_path / "artifacts",
    }.items():
        monkeypatch.setenv(key, str(path))


@pytest.fixture
def pair_client(private_pair_paths, single_user_cookie_client, tmp_path):
    """Official real AuthNZ/lifespan; only the private content DB is substituted."""
    client, key = single_user_cookie_client
    for endpoint in (flashcards, study_suggestions):
        if not any(getattr(route, "path", "") == "/api/v1" + endpoint.router.routes[0].path for route in client.app.routes):
            client.app.include_router(endpoint.router, prefix="/api/v1")
    db = CharactersRAGDB(str(tmp_path / "cards.db"), client_id="pair-contract")
    client.app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        yield client, {"X-API-KEY": key}, db
    finally:
        client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
        db.close_connection()


def _request(**options):
    return {"text": SOURCE, "num_cards": 5, "provider": "ollama", "model": MODEL,
            "claims_verification_provider": "ollama", "claims_verification_model": MODEL, **options}


@pytest.mark.parametrize("planned", [False, True])
def test_basic_generation_sends_question_and_direct_answer_contract(pair_client, inference_transport, planned):
    options = {"card_plan": [{"card_type": "basic", "count": 5}]} if planned else {"card_type": "basic"}
    if planned:
        for card in inference_transport.cards:
            card["generation_type"] = "basic"
    client, headers, db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request(**options))
    assert response.status_code == 200, response.text  # nosec B101 - real endpoint contract
    payload = inference_transport.requests[0]
    assert payload["messages"][1] == {"role": "user", "content": f"Generate flashcards from:\n\n{SOURCE}"}  # nosec B101
    assert payload["model"] == MODEL  # nosec B101
    system = payload["messages"][0]["content"].lower()
    assert "a question or statement" not in system  # nosec B101 - catches ambiguous basic contract
    assert "directly answers the front question" in system  # nosec B101 - outgoing semantic contract
    assert [(card["front"], card["back"]) for card in response.json()["flashcards"]] == [(card["front"], card["back"]) for card in VALID]  # nosec B101
    assert db.list_decks() == []  # nosec B101 - generation does not persist
    assert inference_transport.responses_closed == len(inference_transport.requests) == 6  # nosec B101


@pytest.mark.parametrize(("card_type", "instruction"), [
    ("cloze", "cloze deletion"),
    ("basic_reverse", "bidirectional"),
    ("true_false", "true/false"),
])
def test_other_card_types_retain_their_semantics(pair_client, inference_transport, card_type, instruction):
    for card in inference_transport.cards:
        card["generation_type"] = card_type
    client, headers, _db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers,
                           json=_request(card_plan=[{"card_type": card_type, "count": 5}]))
    assert response.status_code == 200, response.text  # nosec B101
    assert instruction in inference_transport.requests[0]["messages"][0]["content"].lower()  # nosec B101
    assert {card["generation_type"] for card in response.json()["flashcards"]} == {card_type}  # nosec B101


@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("normalized_variant", [False, True])
def test_duplicate_pairs_cannot_satisfy_requested_count(pair_client, inference_transport, planned, normalized_variant):
    cards = [dict(card) for card in VALID[:4]]
    duplicate = dict(cards[0])
    if normalized_variant:
        duplicate["front"] = "  WHAT  is the POWERHOUSE\nof the cell? "
        duplicate["back"] = " the\tMITOCHONDRIA. "
    cards.append(duplicate)
    if planned:
        for card in cards:
            card["generation_type"] = "basic"
    inference_transport.cards = cards
    client, headers, db = pair_client
    options = {"card_plan": [{"card_type": "basic", "count": 5}]} if planned else {"card_type": "basic"}
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request(**options))
    assert response.status_code == 422, response.text  # nosec B101 - clear causal refusal diagnostic
    assert response.json()["detail"] == "duplicate_flashcards"  # nosec B101
    assert len(inference_transport.requests) == 1  # nosec B101 - refuse before verification
    assert db.list_decks() == []  # nosec B101 - no duplicate drafts persisted


def test_same_pair_in_different_requested_types_is_not_a_duplicate(pair_client, inference_transport):
    inference_transport.cards = [
        {**VALID[0], "generation_type": "basic"},
        {**VALID[0], "generation_type": "basic_reverse"},
    ]
    client, headers, _db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request(
        num_cards=2, card_plan=[{"card_type": "basic", "count": 1}, {"card_type": "basic_reverse", "count": 1}]))
    assert response.status_code == 200, response.text  # nosec B101
    assert response.json()["count"] == 2  # nosec B101 - resolved generation type participates in identity


def test_observed_unique_wrong_pairs_retain_semantic_refusal_and_report(pair_client, inference_transport):
    inference_transport.cards = [dict(card) for card in OBSERVED]
    inference_transport.judge = '{"label":"nei","confidence":0.5,"rationale":"Unrelated answer."}'
    client, headers, db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request(num_cards=4))
    assert response.status_code == 422, response.text  # nosec B101
    report = response.json()["detail"]
    assert report["code"] == "claim_verification_failed"  # nosec B101
    back_units = [unit for unit in report["claimVerification"]["unitResults"] if unit["unit_id"].endswith(":back")]
    assert len(back_units) == 4 and all(unit["statuses"] == ["unverified"] for unit in back_units)  # nosec B101
    judge_inputs = [request["messages"][1]["content"] for request in inference_transport.requests[1:]]
    assert all(SOURCE in prompt for prompt in judge_inputs)  # nosec B101
    assert any("Question: Water boils at 100 degrees Celsius at sea level.\nAnswer: Primary revision one." in prompt for prompt in judge_inputs)  # nosec B101
    assert db.list_decks() == []  # nosec B101


def test_empty_generation_keeps_existing_422(pair_client, inference_transport):
    inference_transport.cards = []
    client, headers, _db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request())
    assert (response.status_code, response.json()["detail"]) == (422, "Flashcard generation returned no usable cards")  # nosec B101


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.asyncio
async def test_workflow_duplicate_generation_records_failed_step_and_run(tmp_path, inference_transport, monkeypatch, continuation):
    monkeypatch.setenv("WORKFLOWS_DISABLE_QUOTAS", "1")
    inference_transport.cards = [dict(VALID[0]), dict(VALID[0])]
    db = WorkflowsDatabase(str(tmp_path / "workflows.db"))
    run_id = str(uuid4())
    steps = [{"id": "cards", "type": "flashcard_generate", "retry": 0,
              "config": {"text": SOURCE, "num_cards": 2, "provider": "ollama", "model": MODEL}}]
    if continuation:
        steps.insert(0, {"id": "before", "type": "log", "config": {"message": "already complete"}})
    db.create_run(run_id=run_id, tenant_id="default", user_id="1", inputs={}, workflow_id=None,
                  definition_version=1, definition_snapshot={"name": "Distinct source pairs", "version": 1, "steps": steps})
    engine = WorkflowEngine(db)
    if continuation:
        db.update_run_status(run_id, status="waiting_human")
        await engine.continue_run(run_id, after_step_id="before", last_outputs={"logged": True})
    else:
        await engine.start_run(run_id)
    assert db.get_run(run_id).status == "failed"  # nosec B101 - actual persisted terminal state
    step = db.get_latest_step_run(run_id=run_id, step_id="cards")
    assert step["status"] == "failed" and "duplicate_flashcards" in step["error"]  # nosec B101
    outputs = json.loads(step.get("outputs_json") or "{}")
    assert not outputs.get("flashcards")  # nosec B101 - no duplicate artifacts
    assert len(inference_transport.requests) == 1  # nosec B101


def test_study_suggestion_real_configuration_refusal_does_not_create_deck(pair_client, inference_transport):
    """The caller currently lacks a provider; characterize its truthful refusal."""
    client, headers, db = pair_client
    quiz_id = db.create_quiz(name="Source facts", source_bundle_json=[])
    db.create_question(quiz_id=quiz_id, question_type="multiple_choice", question_text="What does DNA stand for?",
                       options=["Deoxyribonucleic acid", "Light"], correct_answer=0,
                       explanation="DNA stands for deoxyribonucleic acid.")
    attempt = db.start_attempt(quiz_id)
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz", activity_type="quiz_attempt", anchor_type="quiz_attempt", anchor_id=int(attempt["id"]),
        suggestion_type="study_suggestions", payload_json={"topics": [{"id": "topic-1", "display_label": "Biology", "selected": True}]})
    response = client.post(f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions", headers=headers, json={
        "target_service": "flashcards", "target_type": "deck", "action_kind": "follow_up_flashcards",
        "selected_topic_ids": ["topic-1"], "generator_version": "v1", "force_regenerate": True})
    assert (response.status_code, response.json()["detail"]) == (400, "flashcard_generate_error")  # nosec B101
    assert inference_transport.requests == []  # nosec B101 - precisely states the earlier refusal boundary
    assert db.list_decks(include_deleted=True) == []  # nosec B101 - no partial persistence


@pytest.mark.parametrize("judge", [
    '{"label":"nei","confidence":0.5,"rationale":"Uncertain."}',
    "malformed judge output",
])
@pytest.mark.asyncio
async def test_zero_score_supplied_literal_evidence_is_uncertain_not_unsupported(inference_transport, judge):
    """Factual fronts require semantics; unranked supplied evidence is not absence."""
    inference_transport.judge = judge
    claim = "Photosynthesis converts light energy into chemical energy."
    result = await verify_generated_artifact_against_sources(
        artifact_type="flashcards", units=build_flashcard_verification_units([{"front": claim}]),
        source_documents=[Document(id="literal-source", content=SOURCE, metadata={})],
        generation_provider="ollama", generation_model=MODEL,
    )
    assert all(SOURCE in request["messages"][1]["content"] for request in inference_transport.requests)  # nosec B101
    assert result.unit_results[0].statuses == ["unverified"]  # nosec B101 - no score-only verification or false unsupported label
    assert result.verdict == "needs_revision"  # nosec B101 - never grounded on uncertainty


@pytest.mark.parametrize(("label", "status", "verdict"), [
    ("supported", "verified", "grounded"),
    ("refuted", "refuted", "failed"),
])
@pytest.mark.asyncio
async def test_front_semantic_guard_preserves_positive_and_refuted_judgments(inference_transport, label, status, verdict):
    inference_transport.judge = json.dumps({"label": label, "confidence": 0.95})
    claim = "Water boils at 100 degrees Celsius at sea level."
    result = await verify_generated_artifact_against_sources(
        artifact_type="flashcards", units=build_flashcard_verification_units([{"front": claim}]),
        source_documents=[Document(id="literal-source", content=SOURCE, metadata={})],
        generation_provider="ollama", generation_model=MODEL,
    )
    assert result.unit_results[0].statuses == [status]  # nosec B101
    assert result.verdict == verdict  # nosec B101 - requires semantics does not bypass judgment


def test_observed_duplicate_drafts_are_refused_before_judging(pair_client, inference_transport):
    inference_transport.cards = [*(dict(card) for card in OBSERVED), dict(OBSERVED[0])]
    client, headers, db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request())
    assert response.status_code == 422, response.text  # nosec B101
    assert response.json()["detail"] == "duplicate_flashcards"  # nosec B101
    assert len(inference_transport.requests) == 1 and db.list_decks() == []  # nosec B101


def test_unrelated_adapter_configuration_error_retains_400(pair_client, inference_transport, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "false")
    client, headers, _db = pair_client
    response = client.post("/api/v1/flashcards/generate", headers=headers, json=_request(provider="unknown-provider"))
    assert response.status_code == 400, response.text  # nosec B101
    assert response.json()["detail"] == "flashcard_generate_error"  # nosec B101
    assert inference_transport.requests == []  # nosec B101
