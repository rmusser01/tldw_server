import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=912, username="wluser", email=None, is_active=True)

    base_dir = tmp_path / "test_user_dbs_preview_more"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")

    from fastapi import FastAPI
    from tldw_Server_API.app.core.config import API_V1_PREFIX
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_preview_empty_filters_has_ingestable_items(client_with_user: TestClient):
    c = client_with_user

    # Create an RSS source
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Feed", "url": "https://example.com/rss.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    # Create a job with empty filters payload
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "No Filters", "scope": {"sources": [sid]}, "job_filters": {"filters": []}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    # With no filters, preview should show ingestable items (TEST_MODE stubs)
    assert data["total"] >= 1
    assert data["ingestable"] >= 1


def test_preview_empty_filters_site_has_ingestable_items(client_with_user: TestClient):
    c = client_with_user

    # Create a site source
    s = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Site",
            "url": "https://example.com/",
            "source_type": "site",
            "settings": {"scrape_rules": {"list_url": "https://example.com/list", "skip_article_fetch": True}},
        },
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    # Create a job with empty filters payload
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "No Filters Site", "scope": {"sources": [sid]}, "job_filters": {"filters": []}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] >= 1


def test_preview_invalid_regex_filter_is_safe_and_gates_when_required(client_with_user: TestClient):
    c = client_with_user

    # Create a site source
    s = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Site",
            "url": "https://example.com/",
            "source_type": "site",
            "settings": {"scrape_rules": {"list_url": "https://example.com/list", "skip_article_fetch": True}},
        },
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    # Job with an invalid regex include rule and require_include=true
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Invalid Regex Include",
            "scope": {"sources": [sid]},
            "job_filters": {
                "filters": [
                    {"type": "regex", "action": "include", "value": {"pattern": "[unclosed", "flags": "i"}}
                ],
                "require_include": True,
            },
        },
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    # Preview should not error; with include-only gating, invalid regex won't match → all filtered
    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] == 0
    assert data["filtered"] >= 1


def test_preview_invalid_regex_without_require_include_keeps_ingestable_items(client_with_user: TestClient):
    c = client_with_user

    # Create an RSS source
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Feed", "url": "https://example.com/rss.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    # Invalid regex include filter with require_include disabled should not gate all items.
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Invalid Regex Include No Gating",
            "scope": {"sources": [sid]},
            "job_filters": {
                "filters": [
                    {"type": "regex", "action": "include", "value": {"pattern": "[broken", "flags": "i"}}
                ],
                "require_include": False,
            },
        },
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] >= 1


def test_preview_accepts_tldw_test_mode_y(client_with_user: TestClient, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    c = client_with_user

    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Feed TLDW", "url": "https://example.com/rss.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "Preview TLDW", "scope": {"sources": [sid]}, "job_filters": {"filters": []}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] >= 1


def _assert_preview_offset_pagination(data: dict, *, limit: int) -> None:
    assert "items" in data
    assert "total" in data
    assert "ingestable" in data
    assert "filtered" in data
    offset = int(data["pagination"]["offset"])
    total = int(data["total"])
    expected_has_more = offset + limit < total
    expected_next_offset = offset + limit if expected_has_more else None
    assert data["pagination"] == {
        "mode": "offset",
        "limit": limit,
        "offset": offset,
        "total": data["total"],
        "has_more": expected_has_more,
        "next_offset": expected_next_offset,
    }


def test_preview_job_includes_canonical_pagination(client_with_user: TestClient) -> None:
    """Job preview keeps legacy counters and adds a canonical pagination object."""
    c = client_with_user
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Feed Paginated", "url": "https://example.com/rss.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "Preview Pagination", "scope": {"sources": [sid]}, "job_filters": {"filters": []}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    _assert_preview_offset_pagination(r.json(), limit=5)


def test_preview_source_test_includes_canonical_pagination(client_with_user: TestClient) -> None:
    """Stored-source preview exposes canonical metadata without changing legacy fields."""
    c = client_with_user
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Stored Source", "url": "https://example.com/", "source_type": "site"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    r = c.post(f"/api/v1/watchlists/sources/{sid}/test", params={"limit": 4})
    assert r.status_code == 200, r.text
    _assert_preview_offset_pagination(r.json(), limit=4)


def test_preview_source_draft_includes_canonical_pagination(client_with_user: TestClient) -> None:
    """Draft-source preview exposes canonical metadata without changing legacy fields."""
    c = client_with_user
    r = c.post(
        "/api/v1/watchlists/sources/test",
        params={"limit": 3},
        json={"url": "https://example.com/", "source_type": "site"},
    )
    assert r.status_code == 200, r.text
    _assert_preview_offset_pagination(r.json(), limit=3)
