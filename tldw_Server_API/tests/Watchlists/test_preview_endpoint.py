import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=909, username="wluser", email=None, is_active=True)

    base_dir = tmp_path / "test_user_dbs_preview"
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


def test_preview_rss_include_only_toggle(client_with_user: TestClient):
    c = client_with_user

    # Create an RSS source (no network used in TEST_MODE)
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Feed", "url": "https://example.com/rss.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]

    # Job with include rule matching 'Test' and include-only gating ON
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Preview RSS",
            "scope": {"sources": [sid]},
            "job_filters": {
                "filters": [
                    {"type": "keyword", "action": "include", "value": {"keywords": ["Test"], "match": "any"}}
                ],
                "require_include": True,
            },
        },
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    # Preview should ingest some items
    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["total"] >= 1
    assert data["ingestable"] >= 1
    # Breakdown: at least one ingest; matched_action should not contain 'exclude'
    ingest_items = [it for it in data.get("items", []) if it.get("decision") == "ingest"]
    assert len(ingest_items) >= 1
    assert all(it.get("matched_action") in (None, "include", "flag") for it in ingest_items)
    include_matches = [it for it in ingest_items if it.get("matched_action") == "include"]
    assert len(include_matches) >= 1
    first_match = include_matches[0]
    assert first_match.get("matched_filter_type") == "keyword"
    assert (first_match.get("matched_filter_key") or "").startswith(("id:", "idx:"))
    assert first_match.get("matched_filter_id") is None

    # Now change include rule to not match and keep gating ON
    r2 = c.patch(
        f"/api/v1/watchlists/jobs/{jid}/filters",
        json={
            "filters": [
                {"type": "keyword", "action": "include", "value": {"keywords": ["NoMatch"], "match": "any"}}
            ],
            "require_include": True,
        },
    )
    assert r2.status_code == 200, r2.text

    r3 = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r3.status_code == 200, r3.text
    data2 = r3.json()
    # With include-only gating and no include match, nothing ingestable
    assert data2["total"] >= 1
    assert data2["ingestable"] == 0
    assert data2["filtered"] >= 1
    # Detailed breakdown: no item should have matched_action == 'include'
    assert all((it.get("matched_action") or "") != "include" for it in data2.get("items", []))


def test_preview_site_without_include_only_gating(client_with_user: TestClient):
    c = client_with_user

    # Create a site source (TEST_MODE yields synthetic items)
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

    # Include rule that does not match, and require_include is False/omitted
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Preview Site",
            "scope": {"sources": [sid]},
            "job_filters": {
                "filters": [
                    {"type": "keyword", "action": "include", "value": {"keywords": ["NoMatch"], "match": "any"}}
                ]
            },
        },
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    r = c.post(f"/api/v1/watchlists/jobs/{jid}/preview", params={"limit": 5, "per_source": 5})
    assert r.status_code == 200, r.text
    data = r.json()
    # Since include-only gating is OFF and include did not match, items should still be ingestable
    assert data["total"] >= 1
    assert data["ingestable"] >= 1
    # No enforced include-only; ensure some items can be 'ingest' even without include match
    assert any(it.get("decision") == "ingest" for it in data.get("items", []))
    # Some may be marked matched_action=None or include/flag depending on synthetic items


def test_draft_source_test_returns_scrape_rule_diagnostics(client_with_user: TestClient):
    c = client_with_user

    r = c.post(
        "/api/v1/watchlists/sources/test",
        json={
            "name": "Draft Site",
            "url": "https://example.com/news",
            "source_type": "site",
            "settings": {
                "scrape_rules": {
                    "list_url": "https://example.com/news",
                    "item_selector": "css:article",
                    "link_xpath": "//*",
                    "title_selector": "css:h2",
                    "guid_xpath": ".//a/@href",
                    "skip_article_fetch": True,
                }
            },
        },
    )

    assert r.status_code == 200, r.text
    diagnostics = r.json().get("diagnostics") or {}
    assert diagnostics.get("fetch_mode") == "scrape_rules"
    assert diagnostics.get("dedupe_preview_key") == "guid_xpath"
    assert any("link_xpath" in item for item in diagnostics.get("selector_errors", []))
    assert "selector_warnings" in diagnostics


def test_draft_source_test_returns_fetch_failure_diagnostics(client_with_user: TestClient, monkeypatch):
    c = client_with_user

    async def fail_fetch(*args, **kwargs):
        raise RuntimeError("upstream timeout")

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.fetch_site_items_with_rules",
        fail_fetch,
        raising=True,
    )

    r = c.post(
        "/api/v1/watchlists/sources/test",
        json={
            "name": "Draft Site",
            "url": "https://example.com/news",
            "source_type": "site",
            "settings": {
                "scrape_rules": {
                    "list_url": "https://example.com/news",
                    "item_selector": "css:article",
                    "link_xpath": ".//a/@href",
                }
            },
        },
    )

    assert r.status_code == 200, r.text
    diagnostics = r.json().get("diagnostics") or {}
    assert diagnostics.get("fetch_mode") == "scrape_rules"
    assert diagnostics.get("fetch_error") == "upstream timeout"


def test_draft_source_test_returns_fetch_status_diagnostics(client_with_user: TestClient, monkeypatch):
    c = client_with_user

    async def status_fetch(*args, fetch_diagnostics=None, **kwargs):
        if fetch_diagnostics:
            fetch_diagnostics({"url": "https://example.com/news", "status": 503})
        return []

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.watchlists.fetch_site_items_with_rules",
        status_fetch,
        raising=True,
    )

    r = c.post(
        "/api/v1/watchlists/sources/test",
        json={
            "name": "Draft Site",
            "url": "https://example.com/news",
            "source_type": "site",
            "settings": {
                "scrape_rules": {
                    "list_url": "https://example.com/news",
                    "item_selector": "css:article",
                    "link_xpath": ".//a/@href",
                }
            },
        },
    )

    assert r.status_code == 200, r.text
    diagnostics = r.json().get("diagnostics") or {}
    assert diagnostics.get("fetch_status") == 503


def test_source_diagnostics_preserve_warning_detail_and_selector_identity():
    from tldw_Server_API.app.api.v1.endpoints.watchlists import (
        _format_selector_diagnostic,
        _infer_source_dedupe_preview_key,
    )

    formatted = _format_selector_diagnostic(
        {
            "key": "title_selector",
            "selector": "css:.xYz123abc",
            "warning": "fragile_selector",
            "detail": "fragile class 'xYz123abc'",
        }
    )

    assert "fragile_selector" in formatted
    assert "fragile class 'xYz123abc'" in formatted
    assert _infer_source_dedupe_preview_key({"guid_selector": "css:.entry-id"}) == "guid_selector"
    assert (
        _infer_source_dedupe_preview_key(
            {"alternates": [{"url_selector": "css:a::attr(href)"}]}
        )
        == "alternates.url_selector"
    )
