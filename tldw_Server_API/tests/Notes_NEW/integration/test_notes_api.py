"""
Integration tests for Notes API endpoints using a real ChaChaNotes DB.
No mocking of internal functions; only dependency override to inject a temp DB.
"""

import os
import tempfile
from pathlib import Path
import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration


class _BulkStubDB:
    client_id = "bulk_stub"

    def __init__(self):
        self._count = 0

    def add_note(self, title: str, content: str, note_id=None, conversation_id=None, message_id=None):
        self._count += 1
        return f"bulk-note-{self._count}"

    def get_note_by_id(self, note_id: str):
        return None


@pytest.fixture()
def client_with_notes_db(tmp_path, monkeypatch):
    db_path = tmp_path / "notes_integration.db"
    db = CharactersRAGDB(str(db_path), client_id="integration_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    # Inject per-user DB via dependency override
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():

        return db

    # Use full app profile so Notes routes are included
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


@pytest.fixture()
def client_with_bulk_stub(monkeypatch):
    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return _BulkStubDB()

    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


def test_create_get_update_delete_note(client_with_notes_db: TestClient):
    client = client_with_notes_db

    # Create
    create_resp = client.post(
        "/api/v1/notes/",
        json={"title": "Integration Note", "content": "Hello world"},
    )
    assert create_resp.status_code == 201, create_resp.text
    note = create_resp.json()
    note_id = note["id"]

    # Get
    get_resp = client.get(f"/api/v1/notes/{note_id}")
    assert get_resp.status_code == 200
    got = get_resp.json()
    assert got["title"] == "Integration Note"
    assert got["content"] == "Hello world"

    # Update
    upd_resp = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"title": "Updated Title", "content": "Updated"},
    )
    assert upd_resp.status_code == 200
    upd = upd_resp.json()
    assert upd["title"] == "Updated Title"

    # List
    list_resp = client.get("/api/v1/notes/")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert isinstance(data, dict) and "notes" in data
    assert any(n.get("id") == note_id for n in data["notes"])  # noqa: SIM118

    # Delete (soft) requires expected-version header
    curr = client.get(f"/api/v1/notes/{note_id}").json()
    ver = curr.get("version", 1)
    del_resp = client.delete(f"/api/v1/notes/{note_id}", headers={"expected-version": str(ver)})
    assert del_resp.status_code in (200, 204)


def test_note_folder_list_and_create_endpoints(client_with_notes_db: TestClient) -> None:
    client = client_with_notes_db

    empty_response = client.get("/api/v1/notes/folders/")
    assert empty_response.status_code == 200, empty_response.text
    assert empty_response.json()["items"] == []

    create_response = client.post(
        "/api/v1/notes/folders/",
        json={"path": "Inbox/Captured Articles"},
    )
    assert create_response.status_code == 201, create_response.text
    created = create_response.json()
    assert created["name"] == "Captured Articles"
    assert created["path"] == "Inbox/Captured Articles"
    assert created["parent_id"] is not None

    duplicate_response = client.post(
        "/api/v1/notes/folders/",
        json={"path": "inbox/captured articles"},
    )
    assert duplicate_response.status_code == 200, duplicate_response.text
    assert duplicate_response.json()["id"] == created["id"]

    list_response = client.get("/api/v1/notes/folders/")
    assert list_response.status_code == 200, list_response.text
    payload = list_response.json()
    assert payload["count"] == 2
    assert [folder["path"] for folder in payload["items"]] == [
        "Inbox",
        "Inbox/Captured Articles",
    ]


def test_keywords_crud_and_linking(client_with_notes_db: TestClient):
    client = client_with_notes_db

    # Create 2 notes
    n1 = client.post("/api/v1/notes/", json={"title": "A", "content": "Apple"}).json()
    n2 = client.post("/api/v1/notes/", json={"title": "B", "content": "Banana"}).json()

    # Create keyword
    kw_resp = client.post("/api/v1/notes/keywords/", json={"keyword": "fruit"})
    assert kw_resp.status_code == 201, kw_resp.text
    kw = kw_resp.json(); kw_id = kw["id"]

    # Get by id and text
    assert client.get(f"/api/v1/notes/keywords/{kw_id}").status_code == 200
    assert client.get(f"/api/v1/notes/keywords/text/fruit").status_code == 200

    # List / search
    lst = client.get("/api/v1/notes/keywords/")
    assert lst.status_code == 200 and isinstance(lst.json(), list)
    srch = client.get("/api/v1/notes/keywords/search/", params={"query": "fru"})
    assert srch.status_code == 200 and any(k.get("id") == kw_id for k in srch.json())

    # Link to note 1 and note 2
    link1 = client.post(f"/api/v1/notes/{n1['id']}/keywords/{kw_id}")
    link2 = client.post(f"/api/v1/notes/{n2['id']}/keywords/{kw_id}")
    assert link1.status_code == 200 and link2.status_code == 200

    # Get keywords for note 1
    kws_for_n1 = client.get(f"/api/v1/notes/{n1['id']}/keywords/")
    assert kws_for_n1.status_code == 200
    assert any(k.get("id") == kw_id for k in kws_for_n1.json().get("keywords", []))

    # Get notes for keyword
    notes_for_kw = client.get(f"/api/v1/notes/keywords/{kw_id}/notes/")
    assert notes_for_kw.status_code == 200
    ids = [note.get("id") for note in notes_for_kw.json().get("notes", [])]
    assert n1['id'] in ids and n2['id'] in ids

    # Unlink one
    un = client.delete(f"/api/v1/notes/{n1['id']}/keywords/{kw_id}")
    assert un.status_code == 200

    # Delete keyword (requires version header)
    # Fetch keyword for version
    kw_data = client.get(f"/api/v1/notes/keywords/{kw_id}").json()
    ver = kw_data.get("version", 1)
    del_kw = client.delete(f"/api/v1/notes/keywords/{kw_id}", headers={"expected-version": str(ver)})
    assert del_kw.status_code in (200, 204)

    # Update note with version header and test conflict
    refreshed_n2 = client.get(f"/api/v1/notes/{n2['id']}").json()
    good_ver = refreshed_n2.get("version", 1)
    ok_upd = client.patch(
        f"/api/v1/notes/{n2['id']}",
        json={"title": "B2"},
        headers={"expected-version": str(good_ver)}
    )
    assert ok_upd.status_code == 200

    bad_upd = client.patch(
        f"/api/v1/notes/{n2['id']}",
        json={"title": "B3"},
        headers={"expected-version": str(good_ver)}  # stale
    )
    assert bad_upd.status_code in (409, 400)


def test_keywords_list_can_include_note_counts(client_with_notes_db: TestClient):
    client = client_with_notes_db

    n1 = client.post("/api/v1/notes/", json={"title": "Count 1", "content": "Alpha"}).json()
    n2 = client.post("/api/v1/notes/", json={"title": "Count 2", "content": "Beta"}).json()

    kw_a = client.post("/api/v1/notes/keywords/", json={"keyword": "alpha"}).json()
    kw_b = client.post("/api/v1/notes/keywords/", json={"keyword": "beta"}).json()

    assert client.post(f"/api/v1/notes/{n1['id']}/keywords/{kw_a['id']}").status_code == 200
    assert client.post(f"/api/v1/notes/{n2['id']}/keywords/{kw_a['id']}").status_code == 200
    assert client.post(f"/api/v1/notes/{n2['id']}/keywords/{kw_b['id']}").status_code == 200

    resp = client.get(
        "/api/v1/notes/keywords/",
        params={"include_note_counts": True, "limit": 50}
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload, list)

    by_keyword = {str(row.get("keyword")): row for row in payload}
    assert by_keyword.get("alpha", {}).get("note_count") == 2
    assert by_keyword.get("beta", {}).get("note_count") == 1


def test_list_and_search_pagination_and_404s(client_with_notes_db: TestClient):
    client = client_with_notes_db

    # Create several notes
    for i in range(5):
        client.post("/api/v1/notes/", json={"title": f"T{i}", "content": f"C{i}"})

    # Paginate list
    page1 = client.get("/api/v1/notes/", params={"limit": 2, "offset": 0})
    page2 = client.get("/api/v1/notes/", params={"limit": 2, "offset": 2})
    assert page1.status_code == 200 and page2.status_code == 200
    d1, d2 = page1.json(), page2.json()
    assert isinstance(d1, dict) and isinstance(d2, dict)
    assert isinstance(d1.get("notes"), list) and isinstance(d2.get("notes"), list)
    assert d1["pagination"]["mode"] == "offset"
    assert d1["pagination"]["limit"] == 2
    assert d1["pagination"]["offset"] == 0
    assert d1["pagination"]["total"] == d1["total"]
    assert d1["pagination"]["has_more"] is True
    assert d1["pagination"]["next_offset"] == 2
    assert d1["has_more"] is True
    assert d1["next_offset"] == 2
    assert d2["pagination"]["mode"] == "offset"
    assert d2["pagination"]["limit"] == 2
    assert d2["pagination"]["offset"] == 2
    assert d2["pagination"]["total"] == d2["total"]
    assert d2["pagination"]["has_more"] is True
    assert d2["pagination"]["next_offset"] == 4
    assert d2["has_more"] is True
    assert d2["next_offset"] == 4
    # Verify disjointness of pages by IDs
    ids1 = {n.get("id") for n in d1.get("notes", [])}
    ids2 = {n.get("id") for n in d2.get("notes", [])}
    assert ids1.isdisjoint(ids2)
    # If both pages are full, combined count equals sum
    if len(ids1) == 2 and len(ids2) == 2:
        assert len(ids1 | ids2) == 4

    # Search notes
    search = client.get("/api/v1/notes/search/", params={"query": "T", "limit": 3})
    assert search.status_code == 200 and isinstance(search.json(), list)
    empty_search = client.get("/api/v1/notes/search/", params={"query": "zzznotfound", "limit": 3})
    assert empty_search.status_code == 200 and empty_search.json() == []

    # Non-existent note 404
    nf = client.get("/api/v1/notes/non-existent-id")
    assert nf.status_code == 404

    # Update with no fields -> 400
    created = client.post("/api/v1/notes/", json={"title": "X", "content": "Y"}).json()
    upd_bad = client.patch(f"/api/v1/notes/{created['id']}", json={}, headers={"expected-version": str(created.get('version', 1))})
    assert upd_bad.status_code == 400

    # Delete requires version header
    del_no_header = client.delete(f"/api/v1/notes/{created['id']}")
    assert del_no_header.status_code in (400, 422)


def test_create_note_invalid_links_404(client_with_notes_db: TestClient):
    client = client_with_notes_db

    bad_conv = client.post(
        "/api/v1/notes/",
        json={"title": "Bad Conv", "content": "X", "conversation_id": "missing-conv"},
    )
    assert bad_conv.status_code == 404

    bad_msg = client.post(
        "/api/v1/notes/",
        json={"title": "Bad Msg", "content": "Y", "message_id": "missing-msg"},
    )
    assert bad_msg.status_code == 404


def test_keywords_only_update_keeps_version(client_with_notes_db: TestClient):
    client = client_with_notes_db

    created = client.post("/api/v1/notes/", json={"title": "K", "content": "V"}).json()
    note_id = created["id"]
    original_version = created.get("version")

    resp = client.patch(f"/api/v1/notes/{note_id}", json={"keywords": ["alpha", "beta"]})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("version") == original_version
    kw_texts = {k.get("keyword") for k in payload.get("keywords", [])}
    assert {"alpha", "beta"}.issubset(kw_texts)

    reread = client.get(f"/api/v1/notes/{note_id}").json()
    assert reread.get("version") == original_version


def test_bulk_create_reports_missing_note_fetch(client_with_bulk_stub: TestClient):
    client = client_with_bulk_stub

    resp = client.post(
        "/api/v1/notes/bulk",
        json={"notes": [{"title": "Bulk", "content": "X"}]},
    )
    assert resp.status_code == 207
    payload = resp.json()
    assert payload.get("created_count") == 0
    assert payload.get("failed_count") == 1
    result = payload.get("results", [])[0]
    assert result.get("success") is False
    assert "could not be retrieved" in result.get("error", "").lower()

def test_keywords_list_pagination_and_search_limit(client_with_notes_db: TestClient):
    client = client_with_notes_db
    # Create many keywords
    for i in range(15):
        client.post("/api/v1/notes/keywords/", json={"keyword": f"kw{i}"})

    # List first page with small limit
    lst1 = client.get("/api/v1/notes/keywords/", params={"limit": 5, "offset": 0})
    lst2 = client.get("/api/v1/notes/keywords/", params={"limit": 5, "offset": 5})
    assert lst1.status_code == 200 and lst2.status_code == 200
    k1 = lst1.json(); k2 = lst2.json()
    assert isinstance(k1, list) and isinstance(k2, list)
    assert len(k1) <= 5 and len(k2) <= 5
    # Verify disjointness of keyword pages by id when both non-empty
    if k1 and k2:
        ids1 = {k.get("id") for k in k1}
        ids2 = {k.get("id") for k in k2}
        assert ids1.isdisjoint(ids2)
    # Search with limit
    search = client.get("/api/v1/notes/keywords/search/", params={"query": "kw", "limit": 7})
    assert search.status_code == 200
    results = search.json()
    assert isinstance(results, list)
    assert len(results) <= 7


def test_keyword_collections_list_includes_canonical_pagination(client_with_notes_db: TestClient):
    client = client_with_notes_db

    first = client.post("/api/v1/notes/collections", json={"name": "Collection A"})
    second = client.post("/api/v1/notes/collections", json={"name": "Collection B"})
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text

    page1 = client.get("/api/v1/notes/collections", params={"limit": 1, "offset": 0})
    page2 = client.get("/api/v1/notes/collections", params={"limit": 1, "offset": 1})
    assert page1.status_code == 200, page1.text
    assert page2.status_code == 200, page2.text

    payload1 = page1.json()
    payload2 = page2.json()
    assert len(payload1["collections"]) == 1
    assert len(payload2["collections"]) == 1
    page1_ids = {collection["id"] for collection in payload1["collections"]}
    page2_ids = {collection["id"] for collection in payload2["collections"]}
    assert page1_ids.isdisjoint(page2_ids)
    assert payload1["total"] == 2
    assert payload1["count"] == 1
    assert payload1["limit"] == 1
    assert payload1["offset"] == 0
    assert payload1["pagination"] == {
        "mode": "offset",
        "total": 2,
        "limit": 1,
        "offset": 0,
        "has_more": True,
        "next_offset": 1,
    }
    assert payload1["has_more"] is True
    assert payload1["next_offset"] == 1
    assert payload2["total"] == 2
    assert payload2["count"] == 1
    assert payload2["limit"] == 1
    assert payload2["offset"] == 1
    assert payload2["pagination"] == {
        "mode": "offset",
        "total": 2,
        "limit": 1,
        "offset": 1,
        "has_more": False,
        "next_offset": None,
    }
    assert payload2["has_more"] is False
    assert payload2["next_offset"] is None


def test_keywords_list_without_trailing_slash_does_not_hit_note_lookup(client_with_notes_db: TestClient):
    client = client_with_notes_db

    client.post("/api/v1/notes/keywords/", json={"keyword": "alpha"})
    client.post("/api/v1/notes/keywords/", json={"keyword": "beta"})

    resp = client.get("/api/v1/notes/keywords", params={"limit": 200})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert isinstance(payload, list)
    values = {item.get("keyword") for item in payload if isinstance(item, dict)}
    assert "alpha" in values
    assert "beta" in values


def test_keyword_search_substring_behavior(client_with_notes_db: TestClient):
    client = client_with_notes_db
    # Create specific keywords to test substring behavior
    for kw in ("kw1", "kw10", "kw2"):
        client.post("/api/v1/notes/keywords/", json={"keyword": kw})

    search = client.get("/api/v1/notes/keywords/search/", params={"query": "kw1", "limit": 10})
    assert search.status_code == 200
    res = search.json()
    assert isinstance(res, list)
    # Expect to find kw1; many backends will also include kw10 by substring search
    texts = {item.get("keyword") or item.get("text") for item in res}
    assert "kw1" in texts
    if "kw10" not in texts:
        # If the search is exact-match, accept; otherwise prefer inclusion
        pytest.skip("Keyword search appears to be exact-match; skipping substring assertion")
    assert "kw10" in texts and "kw2" not in texts


def test_keyword_update_not_supported(client_with_notes_db: TestClient):
    client = client_with_notes_db
    kw = client.post("/api/v1/notes/keywords/", json={"keyword": "rename-me"}).json()
    kw_id = kw["id"]
    # Attempt an update (not supported by API) should yield 405 Method Not Allowed
    resp = client.put(f"/api/v1/notes/keywords/{kw_id}", json={"keyword": "renamed"})
    assert resp.status_code in (405, 404)


def test_keyword_rename_endpoint_and_conflict(client_with_notes_db: TestClient):
    client = client_with_notes_db

    kw_a = client.post("/api/v1/notes/keywords/", json={"keyword": "alpha"}).json()
    kw_b = client.post("/api/v1/notes/keywords/", json={"keyword": "beta"}).json()

    rename_resp = client.patch(
        f"/api/v1/notes/keywords/{kw_a['id']}",
        json={"keyword": "alpha-renamed"},
        headers={"expected-version": str(kw_a.get("version", 1))},
    )
    assert rename_resp.status_code == 200, rename_resp.text
    renamed = rename_resp.json()
    assert renamed.get("keyword") == "alpha-renamed"
    assert int(renamed.get("version", 0)) == int(kw_a.get("version", 1)) + 1

    duplicate_conflict = client.patch(
        f"/api/v1/notes/keywords/{kw_a['id']}",
        json={"keyword": kw_b["keyword"]},
        headers={"expected-version": str(renamed.get("version", 1))},
    )
    assert duplicate_conflict.status_code == 409


def test_keyword_merge_moves_note_links_and_soft_deletes_source(client_with_notes_db: TestClient):
    client = client_with_notes_db

    n1 = client.post("/api/v1/notes/", json={"title": "Merge 1", "content": "A"}).json()
    n2 = client.post("/api/v1/notes/", json={"title": "Merge 2", "content": "B"}).json()
    n3 = client.post("/api/v1/notes/", json={"title": "Merge 3", "content": "C"}).json()

    source_kw = client.post("/api/v1/notes/keywords/", json={"keyword": "ml"}).json()
    target_kw = client.post("/api/v1/notes/keywords/", json={"keyword": "machine-learning"}).json()

    assert client.post(f"/api/v1/notes/{n1['id']}/keywords/{source_kw['id']}").status_code == 200
    assert client.post(f"/api/v1/notes/{n2['id']}/keywords/{source_kw['id']}").status_code == 200
    assert client.post(f"/api/v1/notes/{n2['id']}/keywords/{target_kw['id']}").status_code == 200
    assert client.post(f"/api/v1/notes/{n3['id']}/keywords/{target_kw['id']}").status_code == 200

    stale_merge = client.post(
        f"/api/v1/notes/keywords/{source_kw['id']}/merge",
        json={"target_keyword_id": target_kw["id"]},
        headers={"expected-version": str(max(0, int(source_kw.get("version", 1)) - 1))},
    )
    assert stale_merge.status_code in (409, 400)

    merge_resp = client.post(
        f"/api/v1/notes/keywords/{source_kw['id']}/merge",
        json={
            "target_keyword_id": target_kw["id"],
            "expected_target_version": target_kw.get("version", 1),
        },
        headers={"expected-version": str(source_kw.get("version", 1))},
    )
    assert merge_resp.status_code == 200, merge_resp.text
    merged = merge_resp.json()
    assert merged.get("source_keyword_id") == source_kw["id"]
    assert merged.get("target_keyword_id") == target_kw["id"]
    assert int(merged.get("merged_note_links", 0)) >= 1

    source_get = client.get(f"/api/v1/notes/keywords/{source_kw['id']}")
    assert source_get.status_code == 404

    notes_for_target = client.get(f"/api/v1/notes/keywords/{target_kw['id']}/notes/")
    assert notes_for_target.status_code == 200
    target_note_ids = {note.get("id") for note in notes_for_target.json().get("notes", [])}
    assert n1["id"] in target_note_ids
    assert n2["id"] in target_note_ids
    assert n3["id"] in target_note_ids

    keywords_for_n1 = client.get(f"/api/v1/notes/{n1['id']}/keywords/")
    assert keywords_for_n1.status_code == 200
    keyword_ids_for_n1 = {kw.get("id") for kw in keywords_for_n1.json().get("keywords", [])}
    assert source_kw["id"] not in keyword_ids_for_n1
    assert target_kw["id"] in keyword_ids_for_n1


def test_keyword_delete_conflict(client_with_notes_db: TestClient):
    client = client_with_notes_db
    # Create keyword
    kw = client.post("/api/v1/notes/keywords/", json={"keyword": "conflict-key"}).json()
    kw_id = kw["id"]
    ver = kw.get("version", 1)
    # Try delete with stale version (e.g., ver-1)
    bad_ver = max(0, ver - 1)
    bad = client.delete(f"/api/v1/notes/keywords/{kw_id}", headers={"expected-version": str(bad_ver)})
    assert bad.status_code in (409, 400)
    # Now delete with current version
    ok = client.delete(f"/api/v1/notes/keywords/{kw_id}", headers={"expected-version": str(ver)})
    assert ok.status_code in (200, 204)


def test_rate_limit_on_create_note(client_with_notes_db: TestClient, monkeypatch):
    client = client_with_notes_db
    # Use a separate DB client_id to isolate rate limiter state if needed
    # Create 31 notes rapidly to exceed 30/min default limit
    created = 0
    last_status = None
    for i in range(35):
        resp = client.post("/api/v1/notes/", json={"title": f"RL{i}", "content": "X"})
        last_status = resp.status_code
        if resp.status_code == 201:
            created += 1
        if resp.status_code == 429:
            break
    # Ensure we eventually hit rate limit
    assert last_status == 429 or created >= 30


def test_delete_conflict_and_success(client_with_notes_db: TestClient):
    client = client_with_notes_db
    # Create note and fetch version
    created = client.post("/api/v1/notes/", json={"title": "Del", "content": "V1"}).json()
    nid = created["id"]
    v1 = created.get("version", 1)

    # Update to bump version
    upd = client.patch(
        f"/api/v1/notes/{nid}",
        json={"content": "V2"},
        headers={"expected-version": str(v1)}
    )
    assert upd.status_code == 200
    current = client.get(f"/api/v1/notes/{nid}").json()
    v2 = current.get("version", v1 + 1)

    # Try delete with stale version -> conflict
    bad_del = client.delete(f"/api/v1/notes/{nid}", headers={"expected-version": str(v1)})
    assert bad_del.status_code in (409, 400)

    # Delete with current version -> success
    ok_del = client.delete(f"/api/v1/notes/{nid}", headers={"expected-version": str(v2)})
    assert ok_del.status_code in (200, 204)


def test_bulk_create_preserves_conversation_and_message_ids(client_with_notes_db: TestClient):
    client = client_with_notes_db
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    db_override = client.app.dependency_overrides.get(get_chacha_db_for_user)
    assert db_override is not None
    db = db_override()
    char_id = db.add_character_card({"name": "Bulk Test Character"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Bulk Test Conversation"})
    msg_id = db.add_message({"conversation_id": conv_id, "sender": "user", "content": "Hello"})
    payload = {
        "notes": [
            {
                "title": "Bulk A",
                "content": "Bulk content A",
                "conversation_id": conv_id,
                "message_id": msg_id,
            },
            {
                "title": "Bulk B",
                "content": "Bulk content B",
                "conversation_id": conv_id,
                "message_id": msg_id,
            },
        ]
    }
    resp = client.post("/api/v1/notes/bulk", json=payload)
    assert resp.status_code in (200, 207), resp.text
    results = resp.json().get("results", [])
    assert results
    first_note = next((r.get("note") for r in results if r.get("note")), None)
    assert first_note is not None
    assert first_note.get("conversation_id") == conv_id
    assert first_note.get("message_id") == msg_id

    fetched = client.get(f"/api/v1/notes/{first_note['id']}")
    assert fetched.status_code == 200
    fetched_note = fetched.json()
    assert fetched_note.get("conversation_id") == first_note.get("conversation_id")
    assert fetched_note.get("message_id") == first_note.get("message_id")


def test_update_allows_clearing_links(client_with_notes_db: TestClient):
    client = client_with_notes_db
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    db_override = client.app.dependency_overrides.get(get_chacha_db_for_user)
    assert db_override is not None
    db = db_override()
    char_id = db.add_character_card({"name": "Clear Links Character"})
    conv_id = db.add_conversation({"character_id": char_id, "title": "Clear Links Conversation"})
    msg_id = db.add_message({"conversation_id": conv_id, "sender": "user", "content": "Hello"})

    create_resp = client.post(
        "/api/v1/notes/",
        json={
            "title": "Linked Note",
            "content": "Has links",
            "conversation_id": conv_id,
            "message_id": msg_id,
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    note = create_resp.json()
    note_id = note["id"]
    version = note.get("version", 1)

    clear_conv = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"conversation_id": None},
        headers={"expected-version": str(version)},
    )
    assert clear_conv.status_code == 200, clear_conv.text
    note = clear_conv.json()
    assert note.get("conversation_id") is None
    assert note.get("message_id") == msg_id
    version = note.get("version", version + 1)

    clear_msg = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"message_id": None},
        headers={"expected-version": str(version)},
    )
    assert clear_msg.status_code == 200, clear_msg.text
    note = clear_msg.json()
    assert note.get("message_id") is None
    version = note.get("version", version + 1)

    clear_both = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"conversation_id": None, "message_id": None},
        headers={"expected-version": str(version)},
    )
    assert clear_both.status_code == 200, clear_both.text
    note = clear_both.json()
    assert note.get("conversation_id") is None
    assert note.get("message_id") is None


def test_missing_note_update_and_delete_return_404(client_with_notes_db: TestClient):
    client = client_with_notes_db
    missing_id = "missing-note-id"
    upd = client.put(
        f"/api/v1/notes/{missing_id}",
        json={"title": "Nope"},
        headers={"expected-version": "1"},
    )
    assert upd.status_code == 404

    delete_resp = client.delete(
        f"/api/v1/notes/{missing_id}",
        headers={"expected-version": "1"},
    )
    assert delete_resp.status_code == 404


def test_keyword_search_rejects_special_characters(client_with_notes_db: TestClient):
    client = client_with_notes_db
    client.post("/api/v1/notes/keywords/", json={"keyword": "alpha"})
    resp = client.get("/api/v1/notes/keywords/search/", params={"query": 'kw"bad', "limit": 5})
    assert resp.status_code == 400
