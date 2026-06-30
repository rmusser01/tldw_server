"""
Integration tests for Writing Playground endpoints using a real ChaChaNotes DB.
"""

import configparser
import importlib
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration


def _has_tiktoken() -> bool:
    try:
        from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
            TokenizerUnavailable,
            resolve_tiktoken_encoding,
        )
    except Exception:
        return False
    try:
        resolve_tiktoken_encoding("gpt-3.5-turbo")
    except TokenizerUnavailable:
        return False
    except Exception:
        return False
    return True


@pytest.fixture()
def client_with_writing_db(tmp_path, monkeypatch):
    db_path = tmp_path / "writing_integration.db"
    db = CharactersRAGDB(str(db_path), client_id="integration_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return db

    # Keep this suite focused on writing routes and avoid heavyweight media/audio imports.
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_DISABLE", "media,audio")
    monkeypatch.setenv("SKIP_AUDIO_ROUTERS_IN_TESTS", "1")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app
    fastapi_app.state.writing_test_db = db

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


def test_writing_sessions_crud_and_clone(client_with_writing_db: TestClient):
    client = client_with_writing_db

    create_resp = client.post(
        "/api/v1/writing/sessions",
        json={"name": "Session One", "payload": {"text": "Draft 1"}},
    )
    assert create_resp.status_code == 201, create_resp.text
    session = create_resp.json()
    session_id = session["id"]

    list_resp = client.get("/api/v1/writing/sessions")
    assert list_resp.status_code == 200, list_resp.text
    listed = list_resp.json()
    assert listed["total"] >= 1
    assert listed["pagination"] == {
        "mode": "offset",
        "limit": 100,
        "offset": 0,
        "total": listed["total"],
        "has_more": False,
        "next_offset": None,
    }
    assert listed["has_more"] is False
    assert listed["next_offset"] is None
    assert any(item["id"] == session_id for item in listed["sessions"])

    get_resp = client.get(f"/api/v1/writing/sessions/{session_id}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()
    assert fetched["payload"]["text"] == "Draft 1"
    version = fetched["version"]

    update_resp = client.patch(
        f"/api/v1/writing/sessions/{session_id}",
        json={"name": "Session One Updated", "payload": {"text": "Draft 2"}},
        headers={"expected-version": str(version)},
    )
    assert update_resp.status_code == 200, update_resp.text
    updated = update_resp.json()
    assert updated["version"] == version + 1
    assert updated["payload"]["text"] == "Draft 2"

    stale_update = client.patch(
        f"/api/v1/writing/sessions/{session_id}",
        json={"name": "Session One Stale"},
        headers={"expected-version": str(version)},
    )
    assert stale_update.status_code in (400, 409)

    clone_resp = client.post(
        f"/api/v1/writing/sessions/{session_id}/clone",
        json={"name": "Session One Clone"},
    )
    assert clone_resp.status_code == 200, clone_resp.text
    clone = clone_resp.json()
    assert clone["version_parent_id"] == session_id
    assert clone["name"] == "Session One Clone"

    refreshed = client.get(f"/api/v1/writing/sessions/{session_id}").json()
    del_resp = client.delete(
        f"/api/v1/writing/sessions/{session_id}",
        headers={"expected-version": str(refreshed["version"])},
    )
    assert del_resp.status_code in (200, 204)
    assert client.get(f"/api/v1/writing/sessions/{session_id}").status_code == 404


def test_writing_templates_and_themes_crud(client_with_writing_db: TestClient):
    client = client_with_writing_db

    tmpl_resp = client.post(
        "/api/v1/writing/templates",
        json={"name": "Template A", "payload": {"preset": "alpha"}},
    )
    assert tmpl_resp.status_code == 201, tmpl_resp.text
    template = tmpl_resp.json()
    tmpl_version = template["version"]

    list_templates = client.get("/api/v1/writing/templates")
    assert list_templates.status_code == 200
    templates_payload = list_templates.json()
    assert templates_payload["total"] == 1
    assert templates_payload["pagination"] == {
        "mode": "offset",
        "limit": 100,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }
    assert templates_payload["has_more"] is False
    assert templates_payload["next_offset"] is None

    upd_tmpl = client.patch(
        "/api/v1/writing/templates/Template A",
        json={"is_default": True},
        headers={"expected-version": str(tmpl_version)},
    )
    assert upd_tmpl.status_code == 200, upd_tmpl.text
    updated_template = upd_tmpl.json()
    assert updated_template["is_default"] is True

    del_tmpl = client.delete(
        "/api/v1/writing/templates/Template A",
        headers={"expected-version": str(updated_template["version"])},
    )
    assert del_tmpl.status_code in (200, 204)
    assert client.get("/api/v1/writing/templates/Template A").status_code == 404

    theme_resp = client.post(
        "/api/v1/writing/themes",
        json={"name": "Theme A", "class_name": "theme-a", "css": ".theme-a { color: #111; }", "order": 2},
    )
    assert theme_resp.status_code == 201, theme_resp.text
    theme = theme_resp.json()
    theme_version = theme["version"]
    assert theme["order"] == 2

    list_themes = client.get("/api/v1/writing/themes")
    assert list_themes.status_code == 200
    themes_payload = list_themes.json()
    assert themes_payload["total"] == 1
    assert themes_payload["pagination"] == {
        "mode": "offset",
        "limit": 100,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }
    assert themes_payload["has_more"] is False
    assert themes_payload["next_offset"] is None

    upd_theme = client.patch(
        "/api/v1/writing/themes/Theme A",
        json={"order": 1},
        headers={"expected-version": str(theme_version)},
    )
    assert upd_theme.status_code == 200, upd_theme.text
    updated_theme = upd_theme.json()
    assert updated_theme["order"] == 1

    del_theme = client.delete(
        "/api/v1/writing/themes/Theme A",
        headers={"expected-version": str(updated_theme["version"])},
    )
    assert del_theme.status_code in (200, 204)
    assert client.get("/api/v1/writing/themes/Theme A").status_code == 404


def test_writing_snapshot_export_import_replace(client_with_writing_db: TestClient):
    client = client_with_writing_db

    session_resp = client.post(
        "/api/v1/writing/sessions",
        json={"name": "Snapshot Session", "payload": {"text": "draft"}},
    )
    assert session_resp.status_code == 201, session_resp.text

    template_resp = client.post(
        "/api/v1/writing/templates",
        json={"name": "Snapshot Template", "payload": {"inst_pre": "[INST]"}},
    )
    assert template_resp.status_code == 201, template_resp.text

    theme_resp = client.post(
        "/api/v1/writing/themes",
        json={
            "name": "Snapshot Theme",
            "class_name": "snapshot-theme",
            "css": ".snapshot-theme { color: #111; }",
            "order": 1,
        },
    )
    assert theme_resp.status_code == 201, theme_resp.text

    export_resp = client.get("/api/v1/writing/snapshot/export")
    assert export_resp.status_code == 200, export_resp.text
    exported = export_resp.json()
    assert exported["version"] == 1
    assert exported["counts"]["sessions"] >= 1
    assert exported["counts"]["templates"] >= 1
    assert exported["counts"]["themes"] >= 1
    assert any(item["name"] == "Snapshot Session" for item in exported["sessions"])
    assert any(item["name"] == "Snapshot Template" for item in exported["templates"])
    assert any(item["name"] == "Snapshot Theme" for item in exported["themes"])

    import_resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": "replace",
            "snapshot": {
                "sessions": [
                    {
                        "id": "snapshot-restore-session-1",
                        "name": "Restored Session",
                        "payload": {"text": "restored"},
                        "schema_version": 1,
                    }
                ],
                "templates": [
                    {
                        "name": "Restored Template",
                        "payload": {"inst_pre": "<U>"},
                        "schema_version": 1,
                        "is_default": True,
                    }
                ],
                "themes": [
                    {
                        "name": "Restored Theme",
                        "class_name": "restored-theme",
                        "css": ".restored-theme { color: #123; }",
                        "schema_version": 1,
                        "is_default": True,
                        "order": 0,
                    }
                ],
            },
        },
    )
    assert import_resp.status_code == 200, import_resp.text
    imported = import_resp.json()
    assert imported["imported"]["sessions"] == 1
    assert imported["imported"]["templates"] == 1
    assert imported["imported"]["themes"] == 1

    sessions_payload = client.get("/api/v1/writing/sessions").json()
    template_payload = client.get("/api/v1/writing/templates").json()
    theme_payload = client.get("/api/v1/writing/themes").json()

    session_names = {item["name"] for item in sessions_payload["sessions"]}
    template_names = {item["name"] for item in template_payload["templates"]}
    theme_names = {item["name"] for item in theme_payload["themes"]}

    assert "Restored Session" in session_names
    assert "Snapshot Session" not in session_names
    assert "Restored Template" in template_names
    assert "Snapshot Template" not in template_names
    assert "Restored Theme" in theme_names
    assert "Snapshot Theme" not in theme_names


def test_writing_snapshot_import_merge_preserves_existing(client_with_writing_db: TestClient):
    client = client_with_writing_db

    assert (
        client.post(
            "/api/v1/writing/sessions",
            json={"name": "Existing Session", "payload": {"text": "existing"}},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/v1/writing/templates",
            json={"name": "Existing Template", "payload": {"inst_pre": "[E]"}},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/v1/writing/themes",
            json={"name": "Existing Theme", "class_name": "existing-theme", "css": ".existing-theme{}", "order": 1},
        ).status_code
        == 201
    )

    import_resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": "merge",
            "snapshot": {
                "sessions": [
                    {
                        "id": "snapshot-merge-session-1",
                        "name": "Merged Session",
                        "payload": {"text": "merged"},
                        "schema_version": 1,
                    }
                ],
                "templates": [
                    {
                        "name": "Merged Template",
                        "payload": {"inst_pre": "<M>"},
                        "schema_version": 1,
                        "is_default": False,
                    }
                ],
                "themes": [
                    {
                        "name": "Merged Theme",
                        "class_name": "merged-theme",
                        "css": ".merged-theme { color: #0a0; }",
                        "schema_version": 1,
                        "is_default": False,
                        "order": 2,
                    }
                ],
            },
        },
    )
    assert import_resp.status_code == 200, import_resp.text
    imported = import_resp.json()
    assert imported["mode"] == "merge"
    assert imported["imported"]["sessions"] == 1
    assert imported["imported"]["templates"] == 1
    assert imported["imported"]["themes"] == 1

    sessions_payload = client.get("/api/v1/writing/sessions").json()
    template_payload = client.get("/api/v1/writing/templates").json()
    theme_payload = client.get("/api/v1/writing/themes").json()

    session_names = {item["name"] for item in sessions_payload["sessions"]}
    template_names = {item["name"] for item in template_payload["templates"]}
    theme_names = {item["name"] for item in theme_payload["themes"]}

    assert "Existing Session" in session_names
    assert "Merged Session" in session_names
    assert "Existing Template" in template_names
    assert "Merged Template" in template_names
    assert "Existing Theme" in theme_names
    assert "Merged Theme" in theme_names


@pytest.mark.parametrize("mode", ["merge", "replace"])
def test_snapshot_import_restores_soft_deleted_template_and_theme_names(
    client_with_writing_db: TestClient,
    mode: str,
):
    client = client_with_writing_db
    db = client.app.state.writing_test_db

    template_create = client.post(
        "/api/v1/writing/templates",
        json={"name": "Soft Deleted Template", "payload": {"inst_pre": "[S]"}},
    )
    assert template_create.status_code == 201, template_create.text
    template_created = template_create.json()

    theme_create = client.post(
        "/api/v1/writing/themes",
        json={
            "name": "Soft Deleted Theme",
            "class_name": "soft-deleted-theme",
            "css": ".soft-deleted-theme { color: #246; }",
            "order": 3,
        },
    )
    assert theme_create.status_code == 201, theme_create.text
    theme_created = theme_create.json()

    template_delete = client.delete(
        "/api/v1/writing/templates/Soft Deleted Template",
        headers={"expected-version": str(template_created["version"])},
    )
    assert template_delete.status_code in (200, 204), template_delete.text
    theme_delete = client.delete(
        "/api/v1/writing/themes/Soft Deleted Theme",
        headers={"expected-version": str(theme_created["version"])},
    )
    assert theme_delete.status_code in (200, 204), theme_delete.text

    template_deleted_row = db.execute_query(
        "SELECT id, version, deleted FROM writing_templates WHERE name = ?",
        ("Soft Deleted Template",),
    ).fetchone()
    theme_deleted_row = db.execute_query(
        "SELECT id, version, deleted FROM writing_themes WHERE name = ?",
        ("Soft Deleted Theme",),
    ).fetchone()
    assert template_deleted_row["deleted"] == 1
    assert theme_deleted_row["deleted"] == 1

    import_resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": mode,
            "snapshot": {
                "sessions": [],
                "templates": [
                    {
                        "name": "Soft Deleted Template",
                        "payload": {"inst_pre": "[R]"},
                        "schema_version": 1,
                        "is_default": True,
                    }
                ],
                "themes": [
                    {
                        "name": "Soft Deleted Theme",
                        "class_name": "restored-theme",
                        "css": ".restored-theme { color: #579; }",
                        "schema_version": 1,
                        "is_default": False,
                        "order": 5,
                    }
                ],
            },
        },
    )
    assert import_resp.status_code == 200, import_resp.text

    template_row = db.execute_query(
        "SELECT id, version, deleted, payload_json, is_default FROM writing_templates WHERE name = ?",
        ("Soft Deleted Template",),
    ).fetchone()
    theme_row = db.execute_query(
        "SELECT id, version, deleted, class_name, css, is_default, order_index FROM writing_themes WHERE name = ?",
        ("Soft Deleted Theme",),
    ).fetchone()

    assert template_row["id"] == template_deleted_row["id"]
    assert theme_row["id"] == theme_deleted_row["id"]
    assert template_row["deleted"] == 0
    assert theme_row["deleted"] == 0
    assert template_row["version"] == template_deleted_row["version"] + 1
    assert theme_row["version"] == theme_deleted_row["version"] + 1
    assert template_row["is_default"] == 1
    assert theme_row["order_index"] == 5
    assert "restored-theme" in theme_row["class_name"]


def test_snapshot_import_replace_rolls_back_when_db_import_fails_after_soft_delete(
    client_with_writing_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = client_with_writing_db
    db = client.app.state.writing_test_db

    keep_session_resp = client.post(
        "/api/v1/writing/sessions",
        json={"name": "Keep Session", "payload": {"text": "keep"}},
    )
    assert keep_session_resp.status_code == 201, keep_session_resp.text
    assert client.post(
        "/api/v1/writing/templates",
        json={"name": "Keep Template", "payload": {"inst_pre": "[K]"}},
    ).status_code == 201
    assert client.post(
        "/api/v1/writing/themes",
        json={"name": "Keep Theme", "class_name": "keep-theme", "css": ".keep-theme{}", "order": 1},
    ).status_code == 201

    original_transaction = db.transaction

    class FailingConnectionProxy:
        def __init__(self, conn):
            self._conn = conn
            self._triggered = False

        def execute(self, sql, params=()):
            cursor = self._conn.execute(sql, params)
            normalized_sql = " ".join(str(sql).split())
            if (not self._triggered) and "UPDATE writing_sessions SET deleted = 1" in normalized_sql:
                self._triggered = True
                raise RuntimeError("inject failure after replace soft-delete begins")
            return cursor

        def __getattr__(self, item):
            return getattr(self._conn, item)

    @contextmanager
    def failing_transaction():
        with original_transaction() as conn:
            yield FailingConnectionProxy(conn)

    monkeypatch.setattr(db, "transaction", failing_transaction)

    resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": "replace",
            "snapshot": {
                "sessions": [
                    {
                        "id": "replace-rollback-session",
                        "name": "Restored Session",
                        "payload": {"text": "new"},
                        "schema_version": 1,
                    }
                ],
                "templates": [
                    {
                        "name": "Restored Template",
                        "payload": {"inst_pre": "[R]"},
                        "schema_version": 1,
                        "is_default": False,
                    }
                ],
                "themes": [
                    {
                        "name": "Restored Theme",
                        "class_name": "restored-theme",
                        "css": ".restored-theme { color: #579; }",
                        "schema_version": 1,
                        "is_default": False,
                        "order": 2,
                    }
                ],
            },
        },
    )

    assert resp.status_code == 500, resp.text
    assert resp.json()["detail"] == "Unexpected error while processing writing snapshot import"
    sessions = client.get("/api/v1/writing/sessions").json()["sessions"]
    templates = client.get("/api/v1/writing/templates").json()["templates"]
    themes = client.get("/api/v1/writing/themes").json()["themes"]
    assert {item["name"] for item in sessions} == {"Keep Session"}
    assert {item["name"] for item in templates} == {"Keep Template"}
    assert {item["name"] for item in themes} == {"Keep Theme"}


def test_writing_snapshot_import_rejects_blank_session_name(
    client_with_writing_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = client_with_writing_db
    assert (
        client.post(
            "/api/v1/writing/sessions",
            json={"name": "Keep Session", "payload": {"text": "keep"}},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/v1/writing/templates",
            json={"name": "Keep Template", "payload": {"inst_pre": "[K]"}},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/v1/writing/themes",
            json={"name": "Keep Theme", "class_name": "keep-theme", "css": ".keep-theme{}", "order": 1},
        ).status_code
        == 201
    )
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    soft_delete_calls: list[tuple[str, str]] = []

    def track_soft_delete_session(self, session_id, expected_version):
        soft_delete_calls.append(("session", str(session_id)))
        raise AssertionError("replace-mode session soft-delete should not run for blank import")

    def track_soft_delete_template(self, name, expected_version):
        soft_delete_calls.append(("template", str(name)))
        raise AssertionError("replace-mode template soft-delete should not run for blank import")

    def track_soft_delete_theme(self, name, expected_version):
        soft_delete_calls.append(("theme", str(name)))
        raise AssertionError("replace-mode theme soft-delete should not run for blank import")

    monkeypatch.setattr(writing_endpoints.CharactersRAGDB, "soft_delete_writing_session", track_soft_delete_session)
    monkeypatch.setattr(writing_endpoints.CharactersRAGDB, "soft_delete_writing_template", track_soft_delete_template)
    monkeypatch.setattr(writing_endpoints.CharactersRAGDB, "soft_delete_writing_theme", track_soft_delete_theme)

    resp = client.post(
        "/api/v1/writing/snapshot/import",
        json={
            "mode": "replace",
            "snapshot": {
                "sessions": [{"name": "   ", "payload": {"text": "ignored"}, "schema_version": 1}],
                "templates": [],
                "themes": [],
            },
        },
    )

    assert resp.status_code == 400, resp.text
    assert "session name" in resp.json()["detail"].lower()
    assert soft_delete_calls == []
    sessions = client.get("/api/v1/writing/sessions").json()["sessions"]
    templates = client.get("/api/v1/writing/templates").json()["templates"]
    themes = client.get("/api/v1/writing/themes").json()["themes"]
    assert {item["name"] for item in sessions} == {"Keep Session"}
    assert {item["name"] for item in templates} == {"Keep Template"}
    assert {item["name"] for item in themes} == {"Keep Theme"}


def test_writing_capabilities_basic(client_with_writing_db: TestClient):
    client = client_with_writing_db

    version_resp = client.get("/api/v1/writing/version")
    assert version_resp.status_code == 200, version_resp.text
    assert version_resp.json()["version"] == 1

    resp = client.get("/api/v1/writing/capabilities", params={"include_providers": "false"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["version"] == 1
    assert data["server"]["sessions"] is True
    assert data["server"]["defaults_catalog"] is True
    assert data["server"]["snapshots"] is True
    assert data["server"]["wordclouds"] is True
    assert data["server"]["detokenize"] is True
    assert data["server"]["token_probabilities"]["inline_reroll"] is True
    assert data["server"]["context"]["author_note_depth_mode"] == "insertion"
    assert data["server"]["context"]["context_order"] is True
    assert data["server"]["context"]["context_budget"] is True
    assert data["providers"] is None


def test_writing_defaults_catalog(client_with_writing_db: TestClient):
    client = client_with_writing_db

    resp = client.get("/api/v1/writing/defaults")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["version"] == 1
    assert isinstance(data["templates"], list)
    assert isinstance(data["themes"], list)
    assert data["templates"]
    assert data["themes"]
    assert data["templates"][0]["name"] == "default"
    assert data["templates"][0]["is_default"] is True
    assert data["themes"][0]["name"] == "default"
    assert data["themes"][0]["is_default"] is True


def test_writing_wordclouds_flow(client_with_writing_db: TestClient):
    client = client_with_writing_db

    text = "Alpha beta beta gamma gamma gamma"
    create_resp = client.post(
        "/api/v1/writing/wordclouds",
        json={"text": text, "options": {"stopwords": []}},
    )
    assert create_resp.status_code in (200, 202), create_resp.text
    payload = create_resp.json()
    assert payload["status"] in ("ready", "queued", "running", "failed")
    assert payload["id"]

    if payload["status"] == "ready":
        words = {entry["text"]: entry["weight"] for entry in payload["result"]["words"]}
        assert words["gamma"] == 3
        assert words["beta"] == 2

    cached_resp = client.post(
        "/api/v1/writing/wordclouds",
        json={"text": text, "options": {"stopwords": []}},
    )
    assert cached_resp.status_code in (200, 202), cached_resp.text
    cached = cached_resp.json()
    assert cached["id"] == payload["id"]

    get_resp = client.get(f"/api/v1/writing/wordclouds/{payload['id']}")
    assert get_resp.status_code == 200, get_resp.text
    fetched = get_resp.json()
    assert fetched["id"] == payload["id"]


def test_writing_wordclouds_empty_result(client_with_writing_db: TestClient):
    client = client_with_writing_db

    text = "the and of"
    resp = client.post(
        "/api/v1/writing/wordclouds",
        json={"text": text},
    )
    assert resp.status_code in (200, 202), resp.text
    data = resp.json()
    assert data["status"] in ("ready", "queued", "running", "failed")
    if data["status"] == "ready":
        assert data["result"]["words"] == []


def test_get_wordcloud_returns_404_for_unknown_id(client_with_writing_db: TestClient):
    client = client_with_writing_db

    resp = client.get("/api/v1/writing/wordclouds/does-not-exist")

    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": "Wordcloud not found"}


def test_get_wordcloud_returns_failed_result(client_with_writing_db: TestClient, monkeypatch: pytest.MonkeyPatch):
    client = client_with_writing_db
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    def boom(*_args, **_kwargs):
        raise RuntimeError("wordcloud failed")

    monkeypatch.setattr(writing_endpoints, "_compute_wordcloud", boom)

    create_resp = client.post("/api/v1/writing/wordclouds", json={"text": "alpha beta"})
    assert create_resp.status_code == 200, create_resp.text
    payload = create_resp.json()
    assert payload["id"]
    assert payload["status"] == "failed"
    assert payload["cached"] is False
    assert payload["error"] == "Wordcloud generation failed"
    assert payload["result"] is None

    get_resp = client.get(f"/api/v1/writing/wordclouds/{payload['id']}")
    assert get_resp.status_code == 200, get_resp.text
    fetched = get_resp.json()
    assert fetched["id"] == payload["id"]
    assert fetched["status"] == "failed"
    assert fetched["cached"] is False
    assert fetched["error"] == "Wordcloud generation failed"
    assert fetched["result"] is None


def test_writing_capabilities_provider_tokenizers(client_with_writing_db: TestClient, monkeypatch):
    client = client_with_writing_db

    async def fake_get_configured_providers_async(include_deprecated: bool = False):
        return {
            "default_provider": "openai",
            "providers": [
                {
                    "name": "openai",
                    "models": ["gpt-3.5-turbo", "definitely-not-a-real-model"],
                    "capabilities": {},
                }
            ],
        }

    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    monkeypatch.setattr(writing_endpoints, "get_configured_providers_async", fake_get_configured_providers_async)

    resp = client.get("/api/v1/writing/capabilities")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    providers = data["providers"]
    assert providers
    tokenizers = providers[0]["tokenizers"]
    assert "gpt-3.5-turbo" in tokenizers
    assert "definitely-not-a-real-model" in tokenizers
    assert tokenizers["definitely-not-a-real-model"]["available"] is False
    assert tokenizers["definitely-not-a-real-model"]["count_accuracy"] == "unavailable"
    if _has_tiktoken():
        assert tokenizers["gpt-3.5-turbo"]["available"] is True
        assert tokenizers["gpt-3.5-turbo"]["tokenizer"].startswith("tiktoken:")
        assert tokenizers["gpt-3.5-turbo"]["kind"] == "tiktoken"
        assert tokenizers["gpt-3.5-turbo"]["source"].startswith("tiktoken.encoding_for_model")
        assert tokenizers["gpt-3.5-turbo"]["detokenize"] is True
        assert tokenizers["gpt-3.5-turbo"]["count_accuracy"] == "exact"
        assert tokenizers["gpt-3.5-turbo"]["strict_mode_effective"] is False
    else:
        assert tokenizers["gpt-3.5-turbo"]["available"] is False
        assert "not available" in tokenizers["gpt-3.5-turbo"]["error"].lower()


def test_writing_capabilities_includes_extra_body_compat(client_with_writing_db: TestClient, monkeypatch):
    client = client_with_writing_db

    async def fake_get_configured_providers_async(include_deprecated: bool = False):
        return {
            "default_provider": "openai",
            "providers": [
                {
                    "name": "openai",
                    "models": ["gpt-4o-mini"],
                    "capabilities": {},
                }
            ],
        }

    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    monkeypatch.setattr(writing_endpoints, "get_configured_providers_async", fake_get_configured_providers_async)
    monkeypatch.setattr(
        writing_endpoints,
        "_tokenizer_support",
        lambda _provider, _model: writing_endpoints.WritingTokenizerSupport(available=False, error="test"),
    )

    resp = client.get("/api/v1/writing/capabilities")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    providers = data["providers"]
    assert providers
    assert "extra_body_compat" in providers[0]
    assert isinstance(providers[0]["extra_body_compat"]["known_params"], list)
    assert "model_extra_body_compat" in providers[0]
    assert "gpt-4o-mini" in providers[0]["model_extra_body_compat"]
    assert isinstance(providers[0]["model_extra_body_compat"]["gpt-4o-mini"]["known_params"], list)


def test_writing_capabilities_requested_extra_body_compat_unknown_fallback(client_with_writing_db: TestClient):
    client = client_with_writing_db
    resp = client.get(
        "/api/v1/writing/capabilities",
        params={
            "include_providers": "false",
            "provider": "definitely-unknown-provider",
            "model": "definitely-unknown-model",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    requested = data["requested"]
    assert requested is not None
    assert requested["extra_body_compat"]["supported"] is False
    assert requested["extra_body_compat"]["known_params"] == []


def test_writing_capabilities_requested_extra_body_compat_reflects_strict_runtime(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    monkeypatch.setenv("LOCAL_LLM_STRICT_OPENAI_COMPAT", "true")
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_tokenizer",
        lambda _provider, _model: (object(), "test-tokenizer"),
    )
    resp = client.get(
        "/api/v1/writing/capabilities",
        params={
            "include_providers": "false",
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    requested = data["requested"]
    assert requested is not None
    assert requested["extra_body_compat"]["supported"] is False
    assert "strict_openai_compat" in str(requested["extra_body_compat"]["effective_reason"])


def test_writing_capabilities_requested_reports_non_exact_count_accuracy(
    client_with_writing_db: TestClient,
):
    client = client_with_writing_db
    resp = client.get(
        "/api/v1/writing/capabilities",
        params={
            "include_providers": "false",
            "provider": "deepseek",
            "model": "gpt-3.5-turbo",
        },
    )
    assert resp.status_code == 200, resp.text
    requested = resp.json()["requested"]
    assert requested is not None
    assert requested["count_accuracy"] == "unavailable"
    assert requested["strict_mode_effective"] is False


def test_writing_capabilities_requested_reflects_strict_token_counting_runtime(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    monkeypatch.setenv("STRICT_TOKEN_COUNTING", "true")
    resp = client.get(
        "/api/v1/writing/capabilities",
        params={
            "include_providers": "false",
            "provider": "deepseek",
            "model": "gpt-3.5-turbo",
        },
    )
    assert resp.status_code == 200, resp.text
    requested = resp.json()["requested"]
    assert requested is not None
    assert requested["count_accuracy"] == "unavailable"
    assert requested["strict_mode_effective"] is True


def test_writing_capabilities_provider_tokenizers_reflect_strict_runtime(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    monkeypatch.setenv("STRICT_TOKEN_COUNTING", "true")

    async def fake_get_configured_providers_async(include_deprecated: bool = False):
        return {
            "default_provider": "openai",
            "providers": [
                {
                    "name": "openai",
                    "models": ["gpt-3.5-turbo", "definitely-not-a-real-model"],
                    "capabilities": {},
                }
            ],
        }

    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    monkeypatch.setattr(writing_endpoints, "get_configured_providers_async", fake_get_configured_providers_async)

    resp = client.get("/api/v1/writing/capabilities")
    assert resp.status_code == 200, resp.text
    tokenizers = resp.json()["providers"][0]["tokenizers"]
    assert tokenizers["gpt-3.5-turbo"]["strict_mode_effective"] is True
    assert tokenizers["definitely-not-a-real-model"]["strict_mode_effective"] is True


def test_writing_tokenize_and_count(client_with_writing_db: TestClient):
    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    client = client_with_writing_db
    count_resp = client.post(
        "/api/v1/writing/token-count",
        json={"provider": "openai", "model": "gpt-3.5-turbo", "text": "Hello world"},
    )
    assert count_resp.status_code == 200, count_resp.text
    count_payload = count_resp.json()
    assert count_payload["count"] >= 1
    assert count_payload["meta"]["tokenizer_kind"] == "tiktoken"
    assert count_payload["meta"]["tokenizer_source"].startswith("tiktoken.encoding_for_model")
    assert count_payload["meta"]["detokenize_available"] is True
    assert count_payload["meta"]["count_accuracy"] == "exact"
    assert count_payload["meta"]["strict_mode_effective"] is False

    tokenize_resp = client.post(
        "/api/v1/writing/tokenize",
        json={
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "text": "Hello world",
            "options": {"include_strings": False},
        },
    )
    assert tokenize_resp.status_code == 200, tokenize_resp.text
    tokens = tokenize_resp.json()
    assert tokens["strings"] is None
    assert tokens["meta"]["tokenizer_kind"] == "tiktoken"
    assert tokens["meta"]["tokenizer_source"].startswith("tiktoken.encoding_for_model")
    assert tokens["meta"]["detokenize_available"] is True
    assert tokens["meta"]["count_accuracy"] == "exact"
    assert tokens["meta"]["strict_mode_effective"] is False


def test_writing_detokenize_roundtrip(client_with_writing_db: TestClient):
    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    client = client_with_writing_db
    source_text = "Hello world"
    tokenize_resp = client.post(
        "/api/v1/writing/tokenize",
        json={
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "text": source_text,
            "options": {"include_strings": False},
        },
    )
    assert tokenize_resp.status_code == 200, tokenize_resp.text
    token_ids = tokenize_resp.json()["ids"]
    assert token_ids

    detok_resp = client.post(
        "/api/v1/writing/detokenize",
        json={
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "ids": token_ids,
        },
    )
    assert detok_resp.status_code == 200, detok_resp.text
    payload = detok_resp.json()
    assert payload["text"] == source_text
    assert isinstance(payload["strings"], list)
    assert payload["meta"]["tokenizer_kind"] == "tiktoken"
    assert payload["meta"]["tokenizer_source"].startswith("tiktoken.encoding_for_model")
    assert payload["meta"]["detokenize_available"] is True
    assert payload["meta"]["count_accuracy"] == "exact"
    assert payload["meta"]["strict_mode_effective"] is False


def test_writing_detokenize_unavailable(client_with_writing_db: TestClient):
    client = client_with_writing_db
    resp = client.post(
        "/api/v1/writing/detokenize",
        json={"provider": "openai", "model": "definitely-not-a-real-model", "ids": [1, 2]},
    )
    assert resp.status_code == 422, resp.text


def test_writing_tokenize_provider_model_mismatch(client_with_writing_db: TestClient):
    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    client = client_with_writing_db
    resp = client.post(
        "/api/v1/writing/token-count",
        json={"provider": "deepseek", "model": "gpt-3.5-turbo", "text": "Mismatch OK"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] >= 1


def test_writing_tokenize_provider_model_mismatch_strict_rejected(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    if not _has_tiktoken():
        pytest.skip("tiktoken not available")

    client = client_with_writing_db
    monkeypatch.setenv("STRICT_TOKEN_COUNTING", "true")
    resp = client.post(
        "/api/v1/writing/token-count",
        json={"provider": "deepseek", "model": "gpt-3.5-turbo", "text": "Mismatch strict"},
    )
    assert resp.status_code == 422, resp.text
    assert "exact tokenizer unavailable" in str(resp.json().get("detail", "")).lower()


def test_writing_token_count_allows_exact_count_only_tokenizer(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    monkeypatch.setenv("STRICT_TOKEN_COUNTING", "true")
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    class _FakeCountOnlyTokenizer:
        def count_tokens(self, text: str) -> int:
            if text == "Hello strict":
                return 3
            return 1

    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_tokenizer",
        lambda _provider, _model: (
            _FakeCountOnlyTokenizer(),
            "anthropic:remote-count",
            "provider-native-count",
            "anthropic.http.count_tokens",
            False,
            "exact",
            True,
        ),
    )

    count_resp = client.post(
        "/api/v1/writing/token-count",
        json={"provider": "anthropic", "model": "claude-opus-4-20250514", "text": "Hello strict"},
    )
    assert count_resp.status_code == 200, count_resp.text
    payload = count_resp.json()
    assert payload["count"] == 3
    assert payload["meta"]["count_accuracy"] == "exact"
    assert payload["meta"]["strict_mode_effective"] is True
    assert payload["meta"]["tokenizer_kind"] == "provider-native-count"


def test_writing_tokenize_rejects_count_only_tokenizer(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    monkeypatch.setenv("STRICT_TOKEN_COUNTING", "true")
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    class _FakeCountOnlyTokenizer:
        def count_tokens(self, text: str) -> int:  # noqa: ARG002
            return 2

    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_tokenizer",
        lambda _provider, _model: (
            _FakeCountOnlyTokenizer(),
            "google:remote-count",
            "provider-native-count",
            "google.http.count_tokens",
            False,
            "exact",
            True,
        ),
    )

    tokenize_resp = client.post(
        "/api/v1/writing/tokenize",
        json={"provider": "google", "model": "gemini-2.5-flash", "text": "Hello"},
    )
    assert tokenize_resp.status_code == 422, tokenize_resp.text
    assert "tokenize not available" in str(tokenize_resp.json().get("detail", "")).lower()


def test_writing_tokenize_unavailable(client_with_writing_db: TestClient):
    client = client_with_writing_db
    resp = client.post(
        "/api/v1/writing/token-count",
        json={"provider": "openai", "model": "definitely-not-a-real-model", "text": "Hello"},
    )
    assert resp.status_code == 422, resp.text
    detail = str(resp.json().get("detail", "")).lower()
    assert "not available" in detail or "unavailable" in detail


def test_writing_tokenize_prefers_provider_native_tokenizer(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    class _FakeNativeTokenizer:
        def encode(self, text: str):
            if text == "Hello native":
                return [10, 20]
            if text == "x":
                return [1]
            return [99]

        def decode(self, token_ids):
            if list(token_ids) == [10]:
                return "He"
            if list(token_ids) == [20]:
                return "llo native"
            if list(token_ids) == [10, 20]:
                return "Hello native"
            return ""

    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_provider_native_tokenizer",
        lambda _provider, _model: (
            _FakeNativeTokenizer(),
            "native:fake",
            "provider-native",
            "llama.http.tokenize",
            True,
        ),
    )

    def _fail_tiktoken(_model: str):
        raise AssertionError("tiktoken fallback should not be used when native tokenizer resolves")

    monkeypatch.setattr(writing_endpoints, "_resolve_tiktoken_encoding", _fail_tiktoken)

    count_resp = client.post(
        "/api/v1/writing/token-count",
        json={"provider": "llama", "model": "local-model", "text": "Hello native"},
    )
    assert count_resp.status_code == 200, count_resp.text
    count_payload = count_resp.json()
    assert count_payload["count"] == 2
    assert count_payload["meta"]["tokenizer"] == "native:fake"
    assert count_payload["meta"]["tokenizer_kind"] == "provider-native"
    assert count_payload["meta"]["tokenizer_source"] == "llama.http.tokenize"
    assert count_payload["meta"]["detokenize_available"] is True
    assert count_payload["meta"]["count_accuracy"] == "exact"
    assert count_payload["meta"]["strict_mode_effective"] is False

    tokenize_resp = client.post(
        "/api/v1/writing/tokenize",
        json={"provider": "llama", "model": "local-model", "text": "Hello native"},
    )
    assert tokenize_resp.status_code == 200, tokenize_resp.text
    tokenize_payload = tokenize_resp.json()
    assert tokenize_payload["ids"] == [10, 20]
    assert tokenize_payload["strings"] == ["He", "llo native"]
    assert tokenize_payload["meta"]["tokenizer_kind"] == "provider-native"
    assert tokenize_payload["meta"]["count_accuracy"] == "exact"

    detok_resp = client.post(
        "/api/v1/writing/detokenize",
        json={"provider": "llama", "model": "local-model", "ids": [10, 20]},
    )
    assert detok_resp.status_code == 200, detok_resp.text
    detok_payload = detok_resp.json()
    assert detok_payload["text"] == "Hello native"
    assert detok_payload["meta"]["tokenizer_kind"] == "provider-native"
    assert detok_payload["meta"]["count_accuracy"] == "exact"


def test_writing_capabilities_requested_native_tokenizer_metadata(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_tokenizer_details",
        lambda _provider, _model: (
            object(),
            "native:fake",
            "provider-native",
            "llama.http.tokenize",
            True,
            "exact",
            False,
        ),
    )

    resp = client.get(
        "/api/v1/writing/capabilities",
        params={
            "include_providers": "false",
            "provider": "llama",
            "model": "local-model",
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    requested = payload.get("requested")
    assert requested is not None
    assert requested["tokenizer_available"] is True
    assert requested["tokenizer"] == "native:fake"
    assert requested["tokenizer_kind"] == "provider-native"
    assert requested["tokenizer_source"] == "llama.http.tokenize"
    assert requested["detokenize_available"] is True
    assert requested["count_accuracy"] == "exact"
    assert requested["strict_mode_effective"] is False


def test_writing_capabilities_requested_downgrades_failed_native_exact(
    client_with_writing_db: TestClient,
    monkeypatch,
):
    client = client_with_writing_db
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    class _FailingNativeTokenizer:
        def encode(self, text: str):  # noqa: ARG002
            raise RuntimeError("endpoint down")

        def decode(self, token_ids):  # noqa: ARG002
            return ""

    class _FakeFallbackTokenizer:
        name = "cl100k_base"

        def encode(self, text: str, disallowed_special=()):  # noqa: ARG002
            return [1, 2, 3]

        def decode(self, token_ids):  # noqa: ARG002
            return "fallback"

    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_provider_native_tokenizer",
        lambda _provider, _model: (
            _FailingNativeTokenizer(),
            "native:fake",
            "provider-native",
            "ollama.http.tokenize",
            True,
        ),
    )
    monkeypatch.setattr(
        writing_endpoints,
        "_resolve_tiktoken_encoding",
        lambda _model: _FakeFallbackTokenizer(),
    )

    resp = client.get(
        "/api/v1/writing/capabilities",
        params={
            "include_providers": "false",
            "provider": "ollama",
            "model": "local-model",
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    requested = payload.get("requested")
    assert requested is not None
    assert requested["tokenizer_available"] is False
    assert requested["count_accuracy"] == "unavailable"


@pytest.mark.parametrize(
    "provider,section,endpoint_field,api_key_field,endpoint,expected_base_url,expected_label",
    [
        (
            "llama",
            "Local-API",
            "llama_api_IP",
            "llama_api_key",
            "http://127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:8080",
            "llama.cpp",
        ),
        (
            "kobold",
            "Local-API",
            "kobold_api_IP",
            "kobold_api_key",
            "http://127.0.0.1:5001/api/v1/generate",
            "http://127.0.0.1:5001",
            "kobold.cpp",
        ),
        (
            "ooba",
            "Local-API",
            "ooba_api_IP",
            "ooba_api_key",
            "http://127.0.0.1:5000/v1/chat/completions",
            "http://127.0.0.1:5000",
            "oobabooga",
        ),
        (
            "tabby",
            "Local-API",
            "tabby_api_IP",
            "tabby_api_key",
            "http://127.0.0.1:5000/api/v1/generate",
            "http://127.0.0.1:5000",
            "tabbyapi",
        ),
        (
            "vllm",
            "Local-API",
            "vllm_api_IP",
            "vllm_api_key",
            "http://127.0.0.1:8000/v1/chat/completions",
            "http://127.0.0.1:8000",
            "vllm",
        ),
        (
            "aphrodite",
            "Local-API",
            "aphrodite_api_IP",
            "aphrodite_api_key",
            "http://127.0.0.1:2242/v1/chat/completions",
            "http://127.0.0.1:2242",
            "aphrodite",
        ),
        (
            "custom_openai_api",
            "API",
            "custom_openai_api_ip",
            "custom_openai_api_key",
            "http://127.0.0.1:8088/v1/chat/completions",
            "http://127.0.0.1:8088",
            "custom-openai-api",
        ),
    ],
)
def test_resolve_provider_native_tokenizer_configured_local_provider_matrix(
    monkeypatch,
    provider: str,
    section: str,
    endpoint_field: str,
    api_key_field: str,
    endpoint: str,
    expected_base_url: str,
    expected_label: str,
):
    from tldw_Server_API.app.api.v1.endpoints import writing as writing_endpoints

    parser = configparser.ConfigParser()
    parser.add_section(section)
    parser.set(section, endpoint_field, endpoint)
    parser.set(section, api_key_field, "test-key")
    monkeypatch.setattr(writing_endpoints, "load_comprehensive_config", lambda: parser)

    captured = {}

    class _FakeAdapter:
        def __init__(self, *, base_url: str, model: str | None, api_key: str | None, timeout_seconds: float = 10.0) -> None:
            captured["base_url"] = base_url
            captured["model"] = model
            captured["api_key"] = api_key
            captured["timeout_seconds"] = timeout_seconds

        def encode(self, text: str, disallowed_special=()):  # noqa: ARG002
            return [1] if text is not None else []

        def count_tokens(self, text: str) -> int:
            return len(self.encode(text))

    monkeypatch.setattr(writing_endpoints, "_ProviderNativeTokenizerHTTPAdapter", _FakeAdapter)

    adapter, tokenizer_name, tokenizer_kind, tokenizer_source, detokenize_available = (
        writing_endpoints._resolve_provider_native_tokenizer(provider, "local-model")
    )

    assert isinstance(adapter, _FakeAdapter)
    assert captured["base_url"] == expected_base_url
    assert captured["model"] == "local-model"
    assert captured["api_key"] == "test-key"

    assert tokenizer_kind == "provider-native"
    assert tokenizer_name == f"{expected_label}:remote"
    assert tokenizer_source == f"{expected_label}.http.tokenize"
    assert detokenize_available is True
