from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        return User(id=922, username="wluser", email=None, is_active=True)

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_opml_tag_case"
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


def _extract_opml_urls(xml_text: str) -> list[str]:
    urls = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)  # nosec B314
        for outline in root.findall('.//outline'):
            url = outline.attrib.get('xmlUrl') or outline.attrib.get('xmlurl')
            if url:
                urls.append(url)
    except Exception:
        _ = None
    return urls


def test_opml_export_group_tag_filter_matches_tag_identity(client_with_user: TestClient):
    c = client_with_user
    group = c.post("/api/v1/watchlists/groups", json={"name": "Identity Group"})
    assert group.status_code == 200, group.text
    group_id = group.json()["id"]
    source = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Sharp",
            "url": "https://example.com/sharp-s.xml",
            "source_type": "rss",
            "tags": ["Straße"],
            "group_ids": [group_id],
        },
    )
    assert source.status_code == 200, source.text

    r = c.get("/api/v1/watchlists/sources/export", params={"type": "rss", "group": [group_id], "tag": ["STRASSE"]})
    assert r.status_code == 200, r.text
    assert "https://example.com/sharp-s.xml" in _extract_opml_urls(r.text)


def test_opml_export_tag_filter_case_insensitive(client_with_user: TestClient):
    c = client_with_user
    # Create two RSS sources with tags 'News' and 'news'
    s1 = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "TagA", "url": "https://example.com/a.xml", "source_type": "rss", "tags": ["News"]},
    )
    assert s1.status_code == 200, s1.text
    s2 = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "TagB", "url": "https://example.com/b.xml", "source_type": "rss", "tags": ["news"]},
    )
    assert s2.status_code == 200, s2.text

    # Export with tag=NEWS (upper) should include both
    r = c.get("/api/v1/watchlists/sources/export", params={"type": "rss", "tag": "NEWS"})
    assert r.status_code == 200, r.text
    xml = r.text
    urls = _extract_opml_urls(xml)
    assert "https://example.com/a.xml" in urls
    assert "https://example.com/b.xml" in urls
