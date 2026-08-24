from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import get_prompts_db_for_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.Writing import note_title as note_title_module


class _CountingPromptsDatabase(PromptsDatabase):
    def __init__(self, db_path: Path, client_id: str) -> None:
        super().__init__(db_path, client_id)
        self.read_definition_ids: list[str] = []

    def get_service_prompt_override(self, definition_id: str):
        self.read_definition_ids.append(definition_id)
        return super().get_service_prompt_override(definition_id)


class _RecordingAdapter:
    def __init__(self, titles: list[str]) -> None:
        self.titles = iter(titles)
        self.payloads: list[dict[str, object]] = []

    def chat(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": next(self.titles)}}]}


@pytest.fixture()
def client_user_only(monkeypatch, tmp_path: Path):
    """Use full app profile so Notes endpoints are registered."""
    # Force full app profile for these tests
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    import importlib

    from tldw_Server_API.app import main as app_main
    from tldw_Server_API.app.api.v1.endpoints import notes as notes_module
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

    # Reload after env tweaks so router gating sees MINIMAL_TEST_APP=0
    importlib.reload(app_main)
    fastapi_app = app_main.app
    prompts_db = _CountingPromptsDatabase(
        tmp_path / "notes-title-service-prompts.sqlite",
        "notes-title-service-prompt-test",
    )

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True)

    async def resolve_test_prompts_db(_request, _current_user):
        return prompts_db

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_prompts_db_for_user] = lambda: prompts_db
    monkeypatch.setattr(
        notes_module,
        "get_prompts_db_for_user",
        resolve_test_prompts_db,
    )
    fastapi_app.state.notes_title_prompts_db = prompts_db
    with TestClient(fastapi_app) as client:
        yield client
    fastapi_app.dependency_overrides.clear()
    prompts_db.close_connection()


def _enable_llm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    titles: list[str],
) -> _RecordingAdapter:
    settings = {
        "NOTES_TITLE_LLM_ENABLED": True,
        "NOTES_TITLE_DEFAULT_STRATEGY": "heuristic",
        "NOTES_TITLE_MAX_LEN": 1_000,
    }
    from tldw_Server_API.app.api.v1.endpoints import notes as notes_module

    monkeypatch.setattr(notes_module, "core_settings", settings)
    monkeypatch.setattr(note_title_module, "core_settings", settings)
    adapter = _RecordingAdapter(titles)
    monkeypatch.setattr(
        note_title_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda provider: adapter if provider == "openai" else None),
    )
    return adapter


def _save_title_override(client: TestClient) -> _CountingPromptsDatabase:
    prompts_db: _CountingPromptsDatabase = client.app.state.notes_title_prompts_db
    prompts_db.save_service_prompt_override(
        "notes.title.generate",
        {
            "system": "Write titles in the saved account style.",
            "title_instruction": "Create an account-specific title",
        },
        expected_revision=None,
    )
    return prompts_db


def _corrupt_title_override(client: TestClient) -> _CountingPromptsDatabase:
    prompts_db = _save_title_override(client)
    prompts_db.get_connection().execute(
        "UPDATE ServicePromptOverrides SET parts_json = ? WHERE definition_id = ?",
        ("{", "notes.title.generate"),
    )
    prompts_db.get_connection().commit()
    return prompts_db


def _make_prompts_db_unavailable(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import notes as notes_module

    async def unavailable_direct(_request, _current_user):
        raise HTTPException(status_code=500, detail="Prompts DB unavailable")

    async def unavailable_dependency():
        raise HTTPException(status_code=500, detail="Prompts DB unavailable")

    monkeypatch.setattr(
        notes_module,
        "get_prompts_db_for_user",
        unavailable_direct,
    )
    client.app.dependency_overrides[get_prompts_db_for_user] = unavailable_dependency


def test_create_note_uses_one_owner_scoped_title_prompt(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_db = _save_title_override(client_user_only)
    adapter = _enable_llm(monkeypatch, titles=["Saved prompt title"])

    response = client_user_only.post(
        "/api/v1/notes/",
        json={
            "content": "Content whose heuristic title differs",
            "auto_title": True,
            "title_strategy": "llm",
            "title_max_len": 80,
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["title"] == "Saved prompt title"
    assert prompts_db.read_definition_ids == ["notes.title.generate"]
    assert adapter.payloads[0]["system_message"] == "Write titles in the saved account style."


def test_suggest_title_uses_one_owner_scoped_title_prompt(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_db = _save_title_override(client_user_only)
    adapter = _enable_llm(monkeypatch, titles=["Suggested saved title"])

    response = client_user_only.post(
        "/api/v1/notes/title/suggest",
        json={
            "content": "Content whose heuristic title differs",
            "title_strategy": "llm",
            "title_max_len": 80,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["title"] == "Suggested saved title"
    assert prompts_db.read_definition_ids == ["notes.title.generate"]
    assert adapter.payloads[0]["system_message"] == "Write titles in the saved account style."


def test_bulk_create_reuses_one_owner_scoped_title_prompt(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_db = _save_title_override(client_user_only)
    adapter = _enable_llm(
        monkeypatch,
        titles=["First saved title", "Second saved title"],
    )

    response = client_user_only.post(
        "/api/v1/notes/bulk",
        json={
            "notes": [
                {
                    "content": "First note content",
                    "auto_title": True,
                    "title_strategy": "llm",
                    "title_max_len": 80,
                },
                {
                    "content": "Second note content",
                    "auto_title": True,
                    "title_strategy": "llm",
                    "title_max_len": 80,
                },
            ]
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 2
    assert [result["note"]["title"] for result in response.json()["results"]] == [
        "First saved title",
        "Second saved title",
    ]
    assert prompts_db.read_definition_ids == ["notes.title.generate"]
    assert [payload["system_message"] for payload in adapter.payloads] == [
        "Write titles in the saved account style.",
        "Write titles in the saved account style.",
    ]


def test_bulk_llm_then_heuristic_preserves_legacy_heuristic_call_shape(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_db = _save_title_override(client_user_only)
    _enable_llm(monkeypatch, titles=[])
    calls: list[tuple[str, bool]] = []

    from tldw_Server_API.app.api.v1.endpoints import notes as notes_module

    def generate_title(content: str, options=None, **kwargs) -> str:
        has_service_prompt = "service_prompt" in kwargs
        calls.append((options.strategy, has_service_prompt))
        if options.strategy == "heuristic" and has_service_prompt:
            raise TypeError("legacy heuristic call received service_prompt")
        return f"{options.strategy}: {content}"

    monkeypatch.setattr(notes_module, "generate_note_title", generate_title)

    response = client_user_only.post(
        "/api/v1/notes/bulk",
        json={
            "notes": [
                {
                    "content": "LLM item",
                    "auto_title": True,
                    "title_strategy": "llm",
                },
                {
                    "content": "Heuristic item",
                    "auto_title": True,
                    "title_strategy": "heuristic",
                },
            ]
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 2
    assert prompts_db.read_definition_ids == ["notes.title.generate"]
    assert calls == [("llm", True), ("heuristic", False)]


def test_corrupt_owner_title_prompt_fails_before_model_or_persistence(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_db = _corrupt_title_override(client_user_only)
    adapter = _enable_llm(monkeypatch, titles=["Must not be used"])
    before = client_user_only.get("/api/v1/notes/")
    assert before.status_code == 200, before.text

    response = client_user_only.post(
        "/api/v1/notes/",
        json={
            "content": "This note must not be persisted",
            "auto_title": True,
            "title_strategy": "llm",
            "title_max_len": 80,
        },
    )

    assert response.status_code == 500, response.text
    assert prompts_db.read_definition_ids == ["notes.title.generate"]
    assert adapter.payloads == []
    list_response = client_user_only.get("/api/v1/notes/")
    assert list_response.status_code == 200, list_response.text
    assert list_response.json()["notes"] == before.json()["notes"]


def test_bulk_corrupt_owner_title_prompt_fails_each_item_before_model_or_persistence(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_db = _corrupt_title_override(client_user_only)
    adapter = _enable_llm(monkeypatch, titles=["Must not be used"])
    before = client_user_only.get("/api/v1/notes/")
    assert before.status_code == 200, before.text

    response = client_user_only.post(
        "/api/v1/notes/bulk",
        json={
            "notes": [
                {
                    "content": "First note must not be persisted",
                    "auto_title": True,
                    "title_strategy": "llm",
                },
                {
                    "content": "Second note must not be persisted",
                    "auto_title": True,
                    "title_strategy": "llm",
                },
            ]
        },
    )

    assert response.status_code == 207, response.text
    assert response.json()["created_count"] == 0
    assert response.json()["failed_count"] == 2
    assert prompts_db.read_definition_ids == [
        "notes.title.generate",
        "notes.title.generate",
    ]
    assert adapter.payloads == []
    after = client_user_only.get("/api/v1/notes/")
    assert after.status_code == 200, after.text
    assert after.json()["notes"] == before.json()["notes"]


def test_active_llm_prompts_db_failure_stops_before_model_or_persistence(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _enable_llm(monkeypatch, titles=["Must not be used"])
    _make_prompts_db_unavailable(client_user_only, monkeypatch)
    before = client_user_only.get("/api/v1/notes/")
    assert before.status_code == 200, before.text

    response = client_user_only.post(
        "/api/v1/notes/",
        json={
            "content": "This note must not be persisted",
            "auto_title": True,
            "title_strategy": "llm",
            "title_max_len": 80,
        },
    )

    assert response.status_code == 500, response.text
    assert adapter.payloads == []
    list_response = client_user_only.get("/api/v1/notes/")
    assert list_response.status_code == 200, list_response.text
    assert list_response.json()["notes"] == before.json()["notes"]


def test_explicit_title_does_not_acquire_prompts_db(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_prompts_db_unavailable(client_user_only, monkeypatch)

    response = client_user_only.post(
        "/api/v1/notes/",
        json={"title": "Explicit title", "content": "Explicit-title content"},
    )

    assert response.status_code == 201, response.text
    assert response.json()["title"] == "Explicit title"


def test_disabled_llm_strategy_uses_heuristic_without_loading_prompt(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import notes as notes_module

    disabled_settings = {
        "NOTES_TITLE_LLM_ENABLED": False,
        "NOTES_TITLE_DEFAULT_STRATEGY": "heuristic",
        "NOTES_TITLE_MAX_LEN": 1_000,
    }
    monkeypatch.setattr(notes_module, "core_settings", disabled_settings)
    monkeypatch.setattr(note_title_module, "core_settings", disabled_settings)
    _make_prompts_db_unavailable(client_user_only, monkeypatch)

    response = client_user_only.post(
        "/api/v1/notes/",
        json={
            "content": "Feature-disabled title content",
            "auto_title": True,
            "title_strategy": "llm",
            "title_max_len": 80,
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["title"] == "Feature-disabled title content"
    assert client_user_only.app.state.notes_title_prompts_db.read_definition_ids == []


def test_create_note_with_auto_title(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _make_prompts_db_unavailable(client_user_only, monkeypatch)
    resp = client_user_only.post(
        "/api/v1/notes/",
        json={
            "content": "# Heading\nSome content body explaining things.",
            "auto_title": True,
            "title_strategy": "heuristic",
            "title_max_len": 250,
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["title"]
    assert len(data["title"]) <= 250
    assert data["content"].startswith("# Heading") or data["content"].startswith("Heading")
    assert client_user_only.app.state.notes_title_prompts_db.read_definition_ids == []


def test_bulk_create_with_auto_title(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _make_prompts_db_unavailable(client_user_only, monkeypatch)
    payload = {
        "notes": [
            {
                "content": "Intro line\nDetails...",
                "auto_title": True,
                "title_strategy": "heuristic",
                "title_max_len": 250,
            }
        ]
    }
    resp = client_user_only.post("/api/v1/notes/bulk", json=payload)
    assert resp.status_code in (200, 207), resp.text
    data = resp.json()
    assert data["created_count"] >= 1
    assert data["results"][0]["success"] is True
    note = data["results"][0]["note"]
    assert note["title"]
    assert len(note["title"]) <= 250
    assert client_user_only.app.state.notes_title_prompts_db.read_definition_ids == []


def test_suggest_title_endpoint(
    client_user_only: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    _make_prompts_db_unavailable(client_user_only, monkeypatch)
    resp = client_user_only.post(
        "/api/v1/notes/title/suggest",
        json={
            "content": "[Deep Dive](https://example.com) — A long read about AI.\nMore text.",
            "title_strategy": "heuristic",
            "title_max_len": 50,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["title"]
    assert len(data["title"]) <= 50
    assert client_user_only.app.state.notes_title_prompts_db.read_definition_ids == []
