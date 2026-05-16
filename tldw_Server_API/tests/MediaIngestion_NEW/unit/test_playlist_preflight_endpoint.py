import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.unit


@pytest.fixture
def playlist_preflight_client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-12345")

    from tldw_Server_API.app.api.v1.endpoints.media import playlist_preflight

    app = FastAPI()
    app.include_router(playlist_preflight.router, prefix="/api/v1/media", tags=["media"])
    with TestClient(app) as client:
        yield client, playlist_preflight


def test_playlist_preflight_endpoint_returns_metadata_only_items(
    playlist_preflight_client,
    monkeypatch,
):
    client, playlist_preflight = playlist_preflight_client

    from tldw_Server_API.app.api.v1.schemas.media_playlist_preflight import (
        PlaylistPreflightItem,
        PlaylistPreflightResponse,
    )

    def fake_preflight_playlist_url(url: str, *, max_items: int):
        assert url == "https://www.youtube.com/playlist?list=PLtest"
        assert max_items == 34
        return PlaylistPreflightResponse(
            source_url=url,
            source_kind="youtube_playlist",
            playlist_id="PLtest",
            playlist_title="Conference 2010",
            video_id=None,
            item_count=1,
            selected_count=1,
            duplicate_count=0,
            warnings=[],
            items=[
                PlaylistPreflightItem(
                    ordinal=1,
                    source_url="https://www.youtube.com/watch?v=abc123",
                    normalized_source_id="youtube:video:abc123",
                    source_kind="youtube_video",
                    title="Opening Keynote",
                    speaker="Conference Org",
                    duration_seconds=120,
                    published_at=None,
                    thumbnail_url=None,
                    duplicate_status="new",
                    duplicate_of_ordinal=None,
                    selected=True,
                )
            ],
        )

    monkeypatch.setattr(
        playlist_preflight,
        "preflight_playlist_url",
        fake_preflight_playlist_url,
        raising=True,
    )

    resp = client.post(
        "/api/v1/media/playlists/preflight",
        json={
            "url": "https://www.youtube.com/playlist?list=PLtest",
            "max_items": 34,
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["playlist_title"] == "Conference 2010"
    assert body["items"][0]["source_url"] == "https://www.youtube.com/watch?v=abc123"
    assert body["items"][0]["duplicate_status"] == "new"
    assert "cookies" not in resp.text.lower()
    assert "authorization" not in resp.text.lower()


def test_playlist_preflight_endpoint_returns_timeout_status(
    playlist_preflight_client,
    monkeypatch,
):
    client, playlist_preflight = playlist_preflight_client

    def slow_preflight_playlist_url(url: str, *, max_items: int):
        assert url == "https://www.youtube.com/playlist?list=PLtest"
        assert max_items == 34
        time.sleep(1.2)
        return {
            "source_url": url,
            "source_kind": "youtube_playlist",
            "item_count": 0,
            "selected_count": 0,
            "duplicate_count": 0,
        }

    monkeypatch.setattr(
        playlist_preflight,
        "preflight_playlist_url",
        slow_preflight_playlist_url,
        raising=True,
    )

    resp = client.post(
        "/api/v1/media/playlists/preflight",
        json={
            "url": "https://www.youtube.com/playlist?list=PLtest",
            "max_items": 34,
            "timeout_seconds": 1,
        },
        headers={"X-API-KEY": "test-api-key-12345"},
    )

    assert resp.status_code == 504
    assert resp.json()["detail"] == "playlist_preflight_timeout"
