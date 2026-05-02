"""
Integration tests for /api/v1/media/add using real flow with direct content.
No internal mocking; exercises the endpoint and Media DB.
"""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_auth():
    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    async def override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="single-user",
            token_type="single_user",
            jti=None,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    client = TestClient(app, headers=headers)
    try:
        yield client
    finally:
        client.close()
        app.dependency_overrides.clear()


def test_add_document_with_content_real(client_with_auth: TestClient, tmp_path):
    client = client_with_auth

    # Create a small temp text file to upload
    p = tmp_path / "integration_doc.txt"
    p.write_text("This is integration test content to be stored.")

    with p.open('rb') as f:
        resp = client.post(
            "/api/v1/media/add",
            data={
                "title": "Integration Doc",
                "media_type": "document",
                "author": "Integration Suite",
                "chunk_method": "words",
                "chunk_size": "50",
                "chunk_overlap": "10",
            },
            files=[("files", ("integration_doc.txt", f, "text/plain"))],
        )
    assert resp.status_code in (200, 207), resp.text
    data = resp.json()
    assert isinstance(data, dict) and "results" in data
    if not any(item.get("db_id") for item in data.get("results", [])):
        lst = client.get("/api/v1/media", params={"page": 1, "results_per_page": 50})
        assert lst.status_code == 200
        assert any(i.get("title") == "Integration Doc" for i in lst.json().get("items", []))


def test_add_document_with_sentences_chunking(client_with_auth: TestClient, tmp_path):
    client = client_with_auth
    p = tmp_path / "sentences.txt"
    p.write_text("Sentence one. Sentence two. Sentence three.")
    with p.open('rb') as f:
        resp = client.post(
            "/api/v1/media/add",
            data={
                "title": "Integration Doc 2",
                "media_type": "document",
                "chunk_method": "sentences",
                "chunk_size": "200",
                "chunk_overlap": "10",
            },
            files=[("files", ("sentences.txt", f, "text/plain"))],
        )
    assert resp.status_code in (200, 207), resp.text


def test_list_and_search_media_after_add(client_with_auth: TestClient, tmp_path):
    client = client_with_auth
    # Add two documents
    for title in ("Alpha Doc", "Beta Doc"):
        p = tmp_path / f"{title.replace(' ', '_').lower()}.txt"
        p.write_text(f"{title} content.")
        with p.open('rb') as f:
            client.post(
                "/api/v1/media/add",
                data={
                    "title": title,
                    "media_type": "document",
                    "chunk_method": "words",
                    "chunk_size": "20",
                    "chunk_overlap": "5",
                },
                files=[("files", (p.name, f, "text/plain"))],
            )

    # List media, requesting keywords in the listing payload
    lst = client.get(
        "/api/v1/media",
        params={"page": 1, "results_per_page": 5, "include_keywords": True},
    )
    assert lst.status_code == 200
    data = lst.json()
    # Strict schema checks for list response
    assert isinstance(data, dict)
    assert "items" in data and isinstance(data["items"], list)
    assert "pagination" in data and isinstance(data["pagination"], dict)
    for key in ("page", "results_per_page", "total_pages", "total_items"):
        assert key in data["pagination"]
    assert data["pagination"]["mode"] == "page"
    assert data["pagination"]["per_page"] == 5
    assert data["pagination"]["total"] >= len(data["items"])
    assert isinstance(data["pagination"]["has_more"], bool)
    # Validate an item shape if present (with keywords)
    if data["items"]:
        item = data["items"][0]
        for k in ("id", "title", "type", "url", "keywords"):
            assert k in item
    # When not requesting keywords explicitly, items should not include the field
    lst_default = client.get(
        "/api/v1/media",
        params={"page": 1, "results_per_page": 5},
    )
    assert lst_default.status_code == 200
    default_items = lst_default.json().get("items", [])
    if default_items:
        assert "keywords" not in default_items[0]
    # Search media using POST /search
    search = client.post(
        "/api/v1/media/search",
        json={"query": "Alpha", "fields": ["title"], "media_types": ["document"]},
        params={"page": 1, "results_per_page": 5}
    )
    assert search.status_code == 200
    sdata = search.json()
    # Strict schema checks for search response
    assert isinstance(sdata, dict)
    assert "items" in sdata and isinstance(sdata["items"], list)
    assert "pagination" in sdata and isinstance(sdata["pagination"], dict)
    for key in ("page", "results_per_page", "total_pages", "total_items"):
        assert key in sdata["pagination"]
    assert sdata["pagination"]["mode"] == "page"
    assert sdata["pagination"]["per_page"] == 5
    assert sdata["pagination"]["total"] >= len(sdata["items"])
    assert isinstance(sdata["pagination"]["has_more"], bool)
    assert any("Alpha" in item.get("title", "") for item in sdata.get("items", []))


def test_add_various_chunk_methods_persist(client_with_auth: TestClient, tmp_path):
    client = client_with_auth
    titles = [
        ("Words Doc", "words"),
        ("Paragraphs Doc", "paragraphs"),
    ]
    for title, method in titles:
        p = tmp_path / f"{title.replace(' ', '_').lower()}.txt"
        p.write_text(f"{title} content.\n\nNew paragraph here.")
        with p.open('rb') as f:
            r = client.post(
                "/api/v1/media/add",
                data={
                    "title": title,
                    "media_type": "document",
                    "keywords": ["test-keyword-1", "test-keyword-2"],
                    "chunk_method": method,
                },
                files=[("files", (p.name, f, "text/plain"))],
            )
        assert r.status_code in (200, 207), r.text

    # Verify both present in list and expose keywords when requested
    lst = client.get(
        "/api/v1/media",
        params={"page": 1, "results_per_page": 20, "include_keywords": True},
    )
    assert lst.status_code == 200
    items = lst.json().get("items", [])
    names = set(i.get("title") for i in items)
    assert "Words Doc" in names and "Paragraphs Doc" in names
    # Locate one of the docs and ensure keywords are attached
    words_doc = next((i for i in items if i.get("title") == "Words Doc"), None)
    assert words_doc is not None
    assert "keywords" in words_doc
    assert isinstance(words_doc["keywords"], list)


def test_add_document_with_invalid_payload_returns_422_or_400(client_with_auth: TestClient):
    client = client_with_auth
    # Missing any files/urls
    resp = client.post(
        "/api/v1/media/add",
        data={"media_type": "document"},
    )
    assert resp.status_code in (400, 422)
