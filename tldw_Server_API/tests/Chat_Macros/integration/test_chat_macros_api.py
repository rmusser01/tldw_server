from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

os.environ.setdefault("MINIMAL_TEST_APP", "1")
os.environ.setdefault("TEST_MODE", "1")
_routes_disable = {
    part.strip() for part in str(os.environ.get("ROUTES_DISABLE", "")).split(",") if part and part.strip()
}
_routes_disable.update({"media", "audio", "audio-websocket"})
os.environ["ROUTES_DISABLE"] = ",".join(sorted(_routes_disable))

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

pytestmark = pytest.mark.integration

PREFIX = "/api/v1/chat/macros"
TEST_USER_ID = 4242


@dataclass(slots=True)
class MacroApiClient:
    client: TestClient
    db: CharactersRAGDB


class FakeJobManager:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return {"id": len(self.created), **kwargs}


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MacroApiClient]:
    from tldw_Server_API.app.main import app as fastapi_app

    user_base = tmp_path / "user_databases" / str(TEST_USER_ID)
    user_base.mkdir(parents=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))

    db = CharactersRAGDB(db_path=user_base / "ChaChaNotes.db", client_id="chat_macros_api_test")
    principal = AuthPrincipal(
        kind="user",
        user_id=TEST_USER_ID,
        username="macro-api-user",
        roles=["user"],
        permissions=["chat.macros.read", "chat.macros.write"],
        subject=f"user:{TEST_USER_ID}",
        token_type="access",
    )

    async def override_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(principal=principal)
        request.state.user_id = TEST_USER_ID
        return principal

    def override_chacha_db() -> CharactersRAGDB:
        return db

    monkeypatch.setattr(DatabasePaths, "get_user_base_directory", staticmethod(lambda uid: user_base))
    fastapi_app.dependency_overrides[auth_deps.get_auth_principal] = override_principal
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db

    try:
        with TestClient(fastapi_app) as client:
            yield MacroApiClient(client=client, db=db)
    finally:
        fastapi_app.dependency_overrides.clear()
        db.close_connection()


def _macro_yaml(name: str = "daily_digest", command: str | None = None) -> str:
    return (
        "schema_version: 1\n"
        f"name: {name}\n"
        f"command: {command or name}\n"
        "description: Test macro.\n"
        "steps:\n"
        "  - id: prompt\n"
        "    type: prompt\n"
        "    output: answer\n"
        "    prompt: Say hi.\n"
    )


def test_list_get_and_settings_round_trip(api_client: MacroApiClient):
    response = api_client.client.get(PREFIX)
    assert response.status_code == 200, response.text
    body = response.json()
    wrapup = next(item for item in body["macros"] if item["name"] == "wrapup")
    assert wrapup["command"] == "wrapup"
    assert wrapup["source"] == "builtin"
    assert wrapup["immutable"] is True

    detail = api_client.client.get(f"{PREFIX}/wrapup")
    assert detail.status_code == 200, detail.text
    assert detail.json()["definition"]["name"] == "wrapup"
    assert "schema_version: 1" in detail.json()["raw"]

    settings = api_client.client.get(f"{PREFIX}/settings")
    assert settings.status_code == 200, settings.text
    assert "default" in settings.json()["settings"]["output_profiles"]

    updated = api_client.client.put(
        f"{PREFIX}/settings",
        json={
            "settings": {
                "output_profiles": {
                    "compact": {
                        "format": "single_response",
                        "sections": ["summary"],
                    }
                }
            }
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["settings"]["output_profiles"]["compact"]["format"] == "single_response"


def test_macro_crud_validate_and_clone(api_client: MacroApiClient):
    invalid = api_client.client.post(
        f"{PREFIX}/validate",
        json={
            "raw": (
                "schema_version: 1\n"
                "name: bad\n"
                "command: bad\n"
                "permissions:\n"
                "  tool_calls: [web]\n"
                "steps: []\n"
            )
        },
    )
    assert invalid.status_code == 200, invalid.text
    assert invalid.json()["valid"] is False
    assert "tool_calls" in invalid.json()["error"]

    created = api_client.client.post(
        PREFIX,
        json={"name": "daily_digest", "raw": _macro_yaml(), "supporting_files": {"notes.txt": "alpha"}},
    )
    assert created.status_code == 201, created.text
    assert created.json()["summary"]["source"] == "user"
    assert created.json()["supporting_files"] == {"notes.txt": "alpha"}

    updated = api_client.client.put(
        f"{PREFIX}/daily_digest",
        json={"raw": _macro_yaml("daily_digest", "team_digest")},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["summary"]["command"] == "team_digest"

    deleted = api_client.client.delete(f"{PREFIX}/daily_digest")
    assert deleted.status_code == 204, deleted.text
    missing = api_client.client.get(f"{PREFIX}/daily_digest")
    assert missing.status_code == 404

    cloned = api_client.client.post(f"{PREFIX}/wrapup/clone", json={"name": "my_wrapup", "command": "my_wrapup"})
    assert cloned.status_code == 201, cloned.text
    assert cloned.json()["summary"]["name"] == "my_wrapup"
    assert cloned.json()["definition"]["command"] == "my_wrapup"


def test_run_detail_and_cancel(api_client: MacroApiClient):
    defaulted = api_client.client.post(
        f"{PREFIX}/run",
        json={
            "macro_name": "wrapup",
            "args": {"question": ["Check fallback profile"]},
            "mode": "background",
            "output_profile": "missing_profile",
        },
    )
    assert defaulted.status_code == 202, defaulted.text
    defaulted_detail = api_client.client.get(f"{PREFIX}/runs/{defaulted.json()['run_id']}")
    assert defaulted_detail.status_code == 200, defaulted_detail.text
    assert defaulted_detail.json()["run"]["output_profile"] == "default"
    assert defaulted_detail.json()["run"]["normalized_args"]["output_profile"] == "default"

    created = api_client.client.post(
        f"{PREFIX}/run",
        json={
            "macro_name": "wrapup",
            "args": {"question": ["Check blockers"]},
            "mode": "background",
            "surface": "chat",
            "conversation_id": "conv-1",
            "output_profile": "default",
            "context_snapshot": {"message_count": 3},
        },
    )
    assert created.status_code == 202, created.text
    run = created.json()
    assert run["status"] == "pending"
    assert run["run_id"]
    assert run["detail_url"].endswith(f"/runs/{run['run_id']}")

    repo = ChatMacroRepository(api_client.db)
    repo.upsert_branch(
        run["run_id"],
        step_id="summary",
        label="Summary",
        status="failed",
        error_code="provider_error",
        error_message="provider failed with api_key=sk-secret123456",
    )
    repo.upsert_branch(
        run["run_id"],
        step_id="decisions",
        label="Decisions",
        status="failed",
        error_code="provider_error",
        error_message='Authorization: Bearer bearer-secret x-api-key: AIzaSecret {"api_key":"json-secret"} token: raw-token',
    )

    detail = api_client.client.get(f"{PREFIX}/runs/{run['run_id']}")
    assert detail.status_code == 200, detail.text
    detail_body = detail.json()
    assert detail_body["run"]["normalized_args"]["question"] == ["Check blockers"]
    errors = [branch["error"] for branch in detail_body["branches"]]
    assert {branch["error_code"] for branch in detail_body["branches"]} == {"provider_error"}
    for secret in ("sk-secret", "bearer-secret", "AIzaSecret", "json-secret", "raw-token"):
        assert all(secret not in error for error in errors)

    cancelled = api_client.client.post(f"{PREFIX}/runs/{run['run_id']}/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancel_requested"
    assert cancelled.json()["cancel_requested_at"]


def test_run_endpoint_enqueues_background_job(api_client: MacroApiClient):
    from tldw_Server_API.app.main import app as fastapi_app

    job_manager = FakeJobManager()
    fastapi_app.dependency_overrides[try_get_job_manager] = lambda: job_manager
    try:
        created = api_client.client.post(
            f"{PREFIX}/run",
            json={
                "macro_name": "wrapup",
                "args": {"question": ["Check blockers"]},
                "mode": "background",
                "surface": "chat",
                "conversation_id": "conv-1",
                "output_profile": "default",
                "context_snapshot": {"message_count": 3},
            },
        )
    finally:
        fastapi_app.dependency_overrides.pop(try_get_job_manager, None)

    assert created.status_code == 202, created.text
    run = created.json()
    assert len(job_manager.created) == 1
    queued = job_manager.created[0]
    assert queued["domain"] == "chat_macros"
    assert queued["queue"] == "default"
    assert queued["job_type"] == "chat_macro_run"
    assert queued["owner_user_id"] == str(TEST_USER_ID)
    assert queued["payload"]["macro_run_id"] == run["run_id"]
    assert queued["payload"]["user_id"] == str(TEST_USER_ID)
    assert queued["payload"]["macro_digest"]
    assert queued["payload"]["normalized_args"] == {
        "preset": "general",
        "keep_forks": False,
        "mode": "background",
        "output_profile": "default",
        "sync": False,
        "include_branches": False,
        "question": ["Check blockers"],
    }


def test_run_endpoint_returns_503_when_jobs_manager_is_unavailable(api_client: MacroApiClient):
    from tldw_Server_API.app.main import app as fastapi_app

    fastapi_app.dependency_overrides[try_get_job_manager] = lambda: None
    try:
        created = api_client.client.post(
            f"{PREFIX}/run",
            json={
                "macro_name": "wrapup",
                "args": {"question": ["Check blockers"]},
                "mode": "background",
                "surface": "chat",
                "conversation_id": "conv-1",
                "output_profile": "default",
            },
        )
    finally:
        fastapi_app.dependency_overrides.pop(try_get_job_manager, None)

    assert created.status_code == 503, created.text
    assert created.json()["detail"] == "Jobs manager unavailable."
