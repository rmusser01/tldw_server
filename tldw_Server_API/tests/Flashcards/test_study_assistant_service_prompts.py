"""Owner-customized study guidance through real HTTP, storage and persistence."""

import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import flashcards, quizzes, service_prompts
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.Flashcards import study_assistant

pytestmark = pytest.mark.integration
GROUNDING = (
    "You are a focused study assistant. Stay strictly within the provided flashcard or quiz-question context. "
    "Do not broaden to unrelated material or invent external facts. "
)
ACTIONS = {
    "explain": ("explain", "Explain the material clearly and stay anchored to the provided study context only."),
    "mnemonic": ("mnemonic", "Offer one memorable mnemonic tied directly to the provided study context only."),
    "follow_up": (
        "followup",
        "Answer the follow-up question using only the provided study context and thread history.",
    ),
    "freeform": (
        "freeform",
        "Answer the learner message using only the provided study context and keep the response concise.",
    ),
}


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Keep authorization identities, storage, prompt assembly and response writes real."""
    state = SimpleNamespace(owner=1, calls=[], reads=[], databases={}, prompts={}, paths={}, during_model=None)
    for owner in (1, 2):
        db = CharactersRAGDB(str(tmp_path / f"study-{owner}.db"), client_id="study-prompts-test")
        state.databases[owner] = db
        state.prompts[owner] = PromptsDatabase(tmp_path / f"prompts-{owner}.db", "study-prompts-test")
        deck = db.add_deck("Biology")
        card = db.add_flashcard({"deck_id": deck, "front": "What is a nephron?", "back": "Kidney functional unit."})
        quiz = db.create_quiz(name="Biology")
        question = db.create_question(
            quiz_id=quiz,
            question_type="multiple_choice",
            question_text="Which organ contains nephrons?",
            correct_answer=0,
            options=["Kidney", "Lung"],
            explanation="Nephrons filter blood in the kidney.",
        )
        attempt = db.start_attempt(quiz)
        db.submit_attempt(int(attempt["id"]), answers=[{"question_id": question, "user_answer": 1}])
        state.paths[owner] = {
            "flashcard": f"/api/v1/flashcards/{card}/assistant",
            "quiz": f"/api/v1/quizzes/attempts/{attempt['id']}/questions/{question}/assistant",
        }

    async def user() -> User:
        """Supply the current authenticated owner."""
        return User(id=state.owner, username=f"owner-{state.owner}")

    async def database() -> AsyncIterator[CharactersRAGDB]:
        """Close the response endpoint's connection on its own event-loop thread."""
        db = state.databases[state.owner]
        try:
            yield db
        finally:
            db.close_connection()

    async def prompts(request: Request, current_user: User) -> PromptsDatabase:
        """Select real prompt storage by captured authenticated identity."""
        state.reads.append(current_user.id)
        return state.prompts[current_user.id]

    def settings_prompts() -> PromptsDatabase:
        """Use the same owner's database for the real Settings API."""
        return state.prompts[state.owner]

    def principal() -> AuthPrincipal:
        """Supply identity for Settings' expected-user authorization guard."""
        return AuthPrincipal(
            kind="user",
            user_id=state.owner,
            username=f"owner-{state.owner}",
            subject=f"user:{state.owner}",
            token_type="access",
        )

    async def model(**kwargs: Any) -> dict[str, Any]:
        """Capture the provider boundary without replacing generation or persistence."""
        state.calls.append(kwargs)
        if state.during_model:
            state.during_model()
        return {"choices": [{"message": {"role": "assistant", "content": "A nephron filters blood."}}]}

    app = FastAPI()
    app.include_router(flashcards.router, prefix="/api/v1")
    app.include_router(quizzes.router, prefix="/api/v1")
    app.include_router(service_prompts.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = user
    app.dependency_overrides[get_chacha_db_for_user] = database
    app.dependency_overrides[get_auth_principal] = principal
    app.dependency_overrides[Prompts_DB_Deps.get_prompts_db_for_user] = settings_prompts
    monkeypatch.setattr(Prompts_DB_Deps, "get_prompts_db_for_user", prompts)
    monkeypatch.setattr(study_assistant, "perform_chat_api_call_async", model)
    with TestClient(app, raise_server_exceptions=False) as client:
        state.client = client
        yield state
    for db in (*state.databases.values(), *state.prompts.values()):
        db.close_connection()


def save(context: SimpleNamespace, action: str, guidance: str) -> None:
    """Save a literal override, using actual optimistic revision handling."""
    db = context.prompts[context.owner]
    prompt_id = f"study.assistant.{ACTIONS[action][0]}"
    row = db.get_service_prompt_override(prompt_id)
    db.save_service_prompt_override(prompt_id, {"guidance": guidance}, row.revision if row else None)


def respond(context: SimpleNamespace, surface: str, action: str = "explain") -> dict[str, Any]:
    """Request a real response and expose HTTP failures clearly."""
    response = context.client.post(
        context.paths[context.owner][surface] + "/respond",
        json={
            "action": action,
            "message": "Explain {literally}.",
            "provider": "openai",
            "model": "test-model",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("surface", ["flashcard", "quiz"])
@pytest.mark.parametrize("action", list(ACTIONS))
def test_saved_guidance_reaches_model_and_preserves_context(
    context: SimpleNamespace, surface: str, action: str
) -> None:
    """Ignoring overrides or losing fixed context changes the observable model request."""
    baseline = respond(context, surface, action)
    default_call = context.calls[-1]
    assert default_call["system_message"] == GROUNDING + ACTIONS[action][1]
    save(context, action, "Teach in French with {literal} examples.")
    result = respond(context, surface, action)
    call = context.calls[-1]
    assert call["system_message"] == GROUNDING + "Teach in French with {literal} examples."
    assert "Study context JSON:" in call["messages"][0]["content"]
    assert "Kidney" in call["messages"][0]["content"]
    assert call["messages"][0]["content"].endswith("Learner message: Explain {literally}.")
    assert (call["api_provider"], call["model"], call["temperature"], call["max_tokens"]) == (
        "openai",
        "test-model",
        0.3,
        1000,
    )
    assert (
        result["assistant_message"]["content"] == baseline["assistant_message"]["content"] == "A nephron filters blood."
    )


@pytest.mark.parametrize("surface", ["flashcard", "quiz"])
def test_owner_isolation_and_action_selection(context: SimpleNamespace, surface: str) -> None:
    """Owner-one guidance cannot leak to another action or owner's model request."""
    save(context, "explain", "Owner one guidance.")
    respond(context, surface)
    assert context.calls[-1]["system_message"] == GROUNDING + "Owner one guidance."
    respond(context, surface, "mnemonic")
    assert context.calls[-1]["system_message"] == GROUNDING + ACTIONS["mnemonic"][1]
    context.owner = 2
    respond(context, surface)
    assert context.calls[-1]["system_message"] == GROUNDING + ACTIONS["explain"][1]
    assert context.reads == [1, 1, 2]


@pytest.mark.parametrize("surface", ["flashcard", "quiz"])
def test_invalid_guidance_fails_before_model_or_response_writes(context: SimpleNamespace, surface: str) -> None:
    """Corrupt saved data must not silently use defaults or append response messages."""
    path = context.paths[context.owner][surface]
    before = context.client.get(path).json()
    save(context, "explain", "")
    response = context.client.post(path + "/respond", json={"action": "explain"})
    assert response.status_code == 500
    assert context.calls == []
    after = context.client.get(path).json()
    assert after["messages"] == before["messages"] == []
    assert after["thread"]["version"] == before["thread"]["version"]


@pytest.mark.parametrize("surface", ["flashcard", "quiz"])
def test_fact_check_ignores_prompt_storage(context: SimpleNamespace, surface: str) -> None:
    """Unrelated corrupt customization must not affect structured fact-checking."""
    save(context, "explain", "")
    respond(context, surface, "fact_check")
    assert context.reads == []
    assert context.calls[-1]["system_message"] == (
        GROUNDING
        + "Fact-check the learner explanation against the provided study context and return structured corrections."
        " Return a JSON object with keys: verdict, corrections, missing_points, next_prompt, response_text."
    )


@pytest.mark.parametrize("action", list(ACTIONS))
def test_settings_save_and_reset_reach_both_surfaces(context: SimpleNamespace, action: str) -> None:
    """The real Settings API must change generation and restore packaged defaults."""
    path = f"/api/v1/service-prompts/study.assistant.{ACTIONS[action][0]}"
    headers = {"X-TLDW-Expected-User-Id": "1"}
    saved = context.client.put(
        path,
        headers=headers,
        json={
            "parts": {"guidance": "Use concrete {literal} examples."},
            "expected_revision": None,
        },
    )
    assert saved.status_code == 200, saved.text
    for surface in ("flashcard", "quiz"):
        respond(context, surface, action)
        assert context.calls[-1]["system_message"] == GROUNDING + "Use concrete {literal} examples."
    reset = context.client.delete(path, headers=headers, params={"expected_revision": saved.json()["revision"]})
    assert reset.status_code == 200, reset.text
    for surface in ("flashcard", "quiz"):
        respond(context, surface, action)
        assert context.calls[-1]["system_message"] == GROUNDING + ACTIONS[action][1]


@pytest.mark.parametrize("surface", ["flashcard", "quiz"])
def test_prompt_snapshot_survives_edit_and_owner_change(context: SimpleNamespace, surface: str) -> None:
    """Edits and identity changes while awaiting the model affect only later requests."""
    save(context, "explain", "Original guidance.")

    def change_configuration() -> None:
        """Change the saved original scope then switch the next request's identity."""
        save(context, "explain", "Edited guidance.")
        context.owner = 2

    context.during_model = change_configuration
    result = respond(context, surface)
    assert context.calls[-1]["system_message"] == GROUNDING + "Original guidance."
    context.during_model = None
    respond(context, surface)
    assert context.calls[-1]["system_message"] == GROUNDING + ACTIONS["explain"][1]
    context.owner = 1
    history = context.client.get(context.paths[1][surface]).json()["messages"]
    assert history[-1]["id"] == result["assistant_message"]["id"]
    respond(context, surface)
    assert context.calls[-1]["system_message"] == GROUNDING + "Edited guidance."
    assert context.reads == [1, 2, 1]


@pytest.mark.parametrize("invalid", [False, True])
def test_prompt_read_and_close_share_worker(
    context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, invalid: bool
) -> None:
    """Read-time connections must close even when saved guidance is corrupt."""
    if invalid:
        save(context, "explain", "")
    db = context.prompts[1]
    read = db.get_service_prompt_override
    close = db.close_connection
    events: list[tuple[str, int]] = []

    def record_read(definition_id: str) -> Any:
        """Record the actual storage read's worker without bypassing it."""
        events.append(("read", threading.get_ident()))
        return read(definition_id)

    def record_close() -> None:
        """Record the actual thread-local cleanup."""
        events.append(("close", threading.get_ident()))
        close()

    monkeypatch.setattr(db, "get_service_prompt_override", record_read)
    monkeypatch.setattr(db, "close_connection", record_close)
    response = context.client.post(context.paths[1]["flashcard"] + "/respond", json={"action": "explain"})
    assert response.status_code == (500 if invalid else 200)
    assert [name for name, _ in events] == ["read", "close"]
    assert events[0][1] == events[1][1] != threading.get_ident()


def test_storage_failure_is_safe_and_does_not_write(context: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Storage error details cannot leak through HTTP or the contextual error log."""

    def fail_read(definition_id: str) -> Any:
        """Simulate a failing external storage boundary with sensitive details."""
        raise RuntimeError("private /secret/account.db prompt=private-guidance")

    messages: list[str] = []

    def capture_log(message: Any) -> None:
        """Collect real Loguru output for privacy and correlation assertions."""
        messages.append(str(message))

    monkeypatch.setattr(context.prompts[1], "get_service_prompt_override", fail_read)
    sink = logger.add(capture_log, level="ERROR")
    try:
        response = context.client.post(context.paths[1]["quiz"] + "/respond", json={"action": "explain"})
    finally:
        logger.remove(sink)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to load study assistant guidance"
    assert context.calls == []
    assert "user_id=1, action=explain" in "".join(messages)
    assert "private" not in response.text + "".join(messages)
    assert context.client.get(context.paths[1]["quiz"]).json()["messages"] == []
