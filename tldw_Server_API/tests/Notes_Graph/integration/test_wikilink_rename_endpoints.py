"""API coverage for updating ``[[Old title]]`` links after a rename (#3110).

- ``POST /api/v1/notes/wikilinks/referrers`` counts the live notes that link
  to a title.
- ``POST /api/v1/notes/wikilinks/rewrite`` rewrites those links to the renamed
  note, one note at a time under optimistic locking.
- ``POST /api/v1/notes/wikilinks/rewrite/undo`` restores the previous text.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

REFERRERS = "/api/v1/notes/wikilinks/referrers"
REWRITE = "/api/v1/notes/wikilinks/rewrite"
UNDO = "/api/v1/notes/wikilinks/rewrite/undo"


def _make_token(scope: str) -> str:
    svc = JWTService(get_settings())
    return svc.create_virtual_access_token(
        user_id=1, username="tester", role="user", scope=scope, ttl_minutes=5,
    )


@pytest.fixture()
def client_and_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_SECRET_KEY", "wikilink_rename_tests_secret_1234567890")
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()

    db = CharactersRAGDB(str(tmp_path / "wikilink_rename_endpoints.db"), client_id="1")

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            roles=["user"],
            permissions=["notes.graph.read", "notes.graph.write"],
        )

    from tldw_Server_API.app import main as app_main
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return db

    importlib.reload(app_main)
    fastapi_app = app_main.app
    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client, db

    fastapi_app.dependency_overrides.clear()
    db.close_connection()

    # Restore the default lightweight test profile for later tests.
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    reset_settings()
    importlib.reload(app_main)
    reset_settings()


def _headers(**extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_token('notes')}", **extra}


def _create_note(client: TestClient, title: str, content: str = "body") -> str:
    resp = client.post("/api/v1/notes/", json={"title": title, "content": content}, headers=_headers())
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _get_note(client: TestClient, note_id: str) -> dict:
    resp = client.get(f"/api/v1/notes/{note_id}", headers=_headers())
    assert resp.status_code == 200, resp.text
    return resp.json()


def _put_note(client: TestClient, note_id: str, payload: dict, expected_version: int) -> dict:
    resp = client.put(
        f"/api/v1/notes/{note_id}",
        json=payload,
        headers=_headers(**{"expected-version": str(expected_version)}),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _renamed_library(client: TestClient) -> dict[str, str]:
    """Rename "Old title" to "New title" through the API, with two notes linking to it."""

    renamed = _create_note(client, "Old title", "the renamed note")
    linker_a = _create_note(client, "Linker A", "Intro [[Old title]] and plain Old title.")
    linker_b = _create_note(client, "Linker B", "Twice: [[old  TITLE]] then [[ Old title ]].")
    _create_note(client, "Unrelated", "Links [[Old title 2]] only.")
    _put_note(client, renamed, {"title": "New title"}, expected_version=1)
    return {"renamed": renamed, "a": linker_a, "b": linker_b}


def _referrers(client: TestClient, **payload) -> dict:
    resp = client.post(REFERRERS, json=payload, headers=_headers())
    assert resp.status_code == 200, resp.text
    return resp.json()


def _rewrite_targets(referrers: dict) -> list[dict]:
    return [{"id": note["id"], "expected_version": note["version"]} for note in referrers["notes"]]


def _undo_targets(rewrite: dict) -> list[dict]:
    return [
        {"id": result["id"], "expected_version": result["version"], "replacements": result["replacements"]}
        for result in rewrite["results"]
        if result["status"] == "updated"
    ]


def test_referrers_counts_the_notes_a_rename_left_unresolved(client_and_db) -> None:
    client, db = client_and_db
    ids = _renamed_library(client)
    trashed = _create_note(client, "Trashed linker", "Gone [[Old title]].")
    db.soft_delete_note(trashed, expected_version=1)

    payload = _referrers(client, title="old title", exclude_note_id=ids["renamed"], unresolved_only=True)

    assert payload["title"] == "old title"
    assert payload["count"] == 2
    assert payload["next_after_note_id"] is None
    assert {note["id"]: (note["title"], note["version"]) for note in payload["notes"]} == {
        ids["a"]: ("Linker A", 1),
        ids["b"]: ("Linker B", 1),
    }


def test_referrers_is_empty_when_nothing_links_to_the_title(client_and_db) -> None:
    client, _db = client_and_db
    _renamed_library(client)

    payload = _referrers(client, title="Never linked")

    assert payload == {"title": "Never linked", "count": 0, "notes": [], "next_after_note_id": None}


def test_referrers_pages_with_a_cursor(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    expected = sorted([ids["a"], ids["b"]])

    first = _referrers(client, title="Old title", exclude_note_id=ids["renamed"], limit=1)
    second = _referrers(
        client,
        title="Old title",
        exclude_note_id=ids["renamed"],
        limit=1,
        after_note_id=first["next_after_note_id"],
    )

    assert (first["count"], [note["id"] for note in first["notes"]]) == (2, expected[:1])
    assert first["next_after_note_id"] == expected[0]
    assert (second["count"], [note["id"] for note in second["notes"]]) == (2, expected[1:])
    assert second["next_after_note_id"] is None


@pytest.mark.parametrize(
    "payload",
    [{"title": ""}, {"title": "x" * 1025}, {"title": "Old title", "limit": 201}, {"title": "Old title", "extra": 1}],
)
def test_referrers_validates_its_input(client_and_db, payload) -> None:
    client, _db = client_and_db

    assert client.post(REFERRERS, json=payload, headers=_headers()).status_code == 422


def test_rewrite_updates_the_linking_notes_and_reports_each_one(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    referrers = _referrers(client, title="Old title", exclude_note_id=ids["renamed"], unresolved_only=True)

    resp = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": _rewrite_targets(referrers)},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert {key: payload[key] for key in ("old_title", "new_title", "link_form", "replacement")} == {
        "old_title": "Old title",
        "new_title": "New title",
        "link_form": "title",
        "replacement": "[[New title]]",
    }
    assert payload["new_title_shared"] is False
    assert (payload["updated_count"], payload["skipped_count"]) == (2, 0)
    by_id = {result["id"]: result for result in payload["results"]}
    assert by_id[ids["a"]] == {
        "id": ids["a"],
        "title": "Linker A",
        "status": "updated",
        "version": 2,
        "replaced_count": 1,
        "replacements": [{"token_index": 0, "original": "[[Old title]]"}],
    }
    assert [item["original"] for item in by_id[ids["b"]]["replacements"]] == [
        "[[old  TITLE]]",
        "[[ Old title ]]",
    ]
    assert _get_note(client, ids["a"])["content"] == "Intro [[New title]] and plain Old title."
    assert _get_note(client, ids["b"])["content"] == "Twice: [[New title]] then [[New title]]."
    # The renamed note gets its backlinks back.
    neighbors = client.get(
        f"/api/v1/notes/{ids['renamed']}/neighbors",
        params={"edge_types": "wikilink,backlink"},
        headers=_headers(),
    )
    assert neighbors.status_code == 200, neighbors.text
    edges = {(edge["type"], edge["source"], edge["target"]) for edge in neighbors.json()["edges"]}
    assert ("wikilink", ids["a"], ids["renamed"]) in edges
    assert ("wikilink", ids["b"], ids["renamed"]) in edges
    assert _referrers(client, title="Old title", exclude_note_id=ids["renamed"])["count"] == 0


def test_rewrite_skips_a_note_edited_since_the_count(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    referrers = _referrers(client, title="Old title", exclude_note_id=ids["renamed"], unresolved_only=True)
    edited = "Edited after the count: [[Old title]]."
    _put_note(client, ids["a"], {"content": edited}, expected_version=1)

    resp = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": _rewrite_targets(referrers)},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    by_id = {result["id"]: result for result in payload["results"]}
    assert (by_id[ids["a"]]["status"], by_id[ids["a"]]["version"]) == ("skipped_conflict", 2)
    assert by_id[ids["a"]]["replacements"] == []
    assert by_id[ids["b"]]["status"] == "updated"
    assert (payload["updated_count"], payload["skipped_count"]) == (1, 1)
    assert _get_note(client, ids["a"])["content"] == edited


def test_rewrite_links_by_id_when_another_note_shares_the_new_title(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    duplicate = _create_note(client, "new title", "another note with the new title")
    referrers = _referrers(client, title="Old title", exclude_note_id=ids["renamed"], unresolved_only=True)

    resp = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": _rewrite_targets(referrers)},
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    id_link = f"[[id:{ids['renamed']}]]"
    assert (payload["link_form"], payload["replacement"], payload["new_title_shared"]) == ("id", id_link, True)
    assert _get_note(client, ids["a"])["content"] == f"Intro {id_link} and plain Old title."
    neighbors = client.get(
        f"/api/v1/notes/{ids['a']}/neighbors", params={"edge_types": "wikilink"}, headers=_headers()
    )
    targets = {edge["target"] for edge in neighbors.json()["edges"] if edge["source"] == ids["a"]}
    assert targets == {ids["renamed"]}
    assert duplicate not in targets


def test_undo_restores_the_previous_text(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    referrers = _referrers(client, title="Old title", exclude_note_id=ids["renamed"], unresolved_only=True)
    rewrite = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": _rewrite_targets(referrers)},
        headers=_headers(),
    ).json()

    resp = client.post(
        UNDO,
        json={
            "old_title": "Old title",
            "replacement": rewrite["replacement"],
            "notes": _undo_targets(rewrite),
        },
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert (payload["restored_count"], payload["skipped_count"]) == (2, 0)
    assert {result["id"]: (result["status"], result["version"]) for result in payload["results"]} == {
        ids["a"]: ("restored", 3),
        ids["b"]: ("restored", 3),
    }
    assert _get_note(client, ids["a"])["content"] == "Intro [[Old title]] and plain Old title."
    assert _get_note(client, ids["b"])["content"] == "Twice: [[old  TITLE]] then [[ Old title ]]."


def test_undo_skips_a_note_edited_since_the_rewrite(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    referrers = _referrers(client, title="Old title", exclude_note_id=ids["renamed"], unresolved_only=True)
    rewrite = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": _rewrite_targets(referrers)},
        headers=_headers(),
    ).json()
    edited = "Edited after the rewrite: [[New title]]!"
    _put_note(client, ids["a"], {"content": edited}, expected_version=2)

    resp = client.post(
        UNDO,
        json={
            "old_title": "Old title",
            "replacement": rewrite["replacement"],
            "notes": _undo_targets(rewrite),
        },
        headers=_headers(),
    )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    by_id = {result["id"]: result for result in payload["results"]}
    assert (by_id[ids["a"]]["status"], by_id[ids["a"]]["title"]) == ("skipped_conflict", "Linker A")
    assert by_id[ids["b"]]["status"] == "restored"
    assert (payload["restored_count"], payload["skipped_count"]) == (1, 1)
    assert _get_note(client, ids["a"])["content"] == edited


def test_rewrite_needs_a_live_renamed_note_with_a_different_title(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    targets = [{"id": ids["a"], "expected_version": 1}]

    missing = client.post(
        REWRITE,
        json={
            "note_id": "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
            "old_title": "Old title",
            "notes": targets,
        },
        headers=_headers(),
    )
    same_title = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "new   TITLE", "notes": targets},
        headers=_headers(),
    )

    assert missing.status_code == 404, missing.text
    assert same_title.status_code == 400, same_title.text
    assert _get_note(client, ids["a"])["version"] == 1


def test_rewrite_and_undo_bound_their_batches(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)
    too_many = [{"id": f"note-{index}", "expected_version": 1} for index in range(201)]
    replacement = {"token_index": 0, "original": "[[Old title]]"}

    rewrite = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": too_many},
        headers=_headers(),
    )
    empty = client.post(
        REWRITE,
        json={"note_id": ids["renamed"], "old_title": "Old title", "notes": []},
        headers=_headers(),
    )
    undo = client.post(
        UNDO,
        json={
            "old_title": "Old title",
            "replacement": "[[New title]]",
            "notes": [{**note, "replacements": [replacement]} for note in too_many],
        },
        headers=_headers(),
    )

    assert (rewrite.status_code, empty.status_code, undo.status_code) == (422, 422, 422)


def test_undo_rejects_a_replacement_that_is_not_one_link(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)

    resp = client.post(
        UNDO,
        json={
            "old_title": "Old title",
            "replacement": "not a link",
            "notes": [
                {
                    "id": ids["a"],
                    "expected_version": 1,
                    "replacements": [{"token_index": 0, "original": "[[Old title]]"}],
                }
            ],
        },
        headers=_headers(),
    )

    assert resp.status_code == 422, resp.text
    assert _get_note(client, ids["a"])["version"] == 1


def test_rewrite_rejects_a_stale_account_assertion(client_and_db) -> None:
    client, _db = client_and_db
    ids = _renamed_library(client)

    resp = client.post(
        REWRITE,
        json={
            "note_id": ids["renamed"],
            "old_title": "Old title",
            "notes": [{"id": ids["a"], "expected_version": 1}],
        },
        headers=_headers(**{"X-TLDW-Expected-User-ID": "2"}),
    )

    assert resp.status_code == 412, resp.text
    assert _get_note(client, ids["a"])["content"] == "Intro [[Old title]] and plain Old title."
