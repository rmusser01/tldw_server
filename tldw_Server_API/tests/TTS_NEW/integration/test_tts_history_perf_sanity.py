import pytest
from fastapi import status

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.config import settings


pytestmark = [pytest.mark.integration]


def test_tts_history_cursor_pagination_sanity(test_client, auth_headers, monkeypatch, tmp_path):
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    monkeypatch.setattr(settings, "USER_DB_BASE_DIR", str(user_db_base), raising=False)

    media_db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(1)), client_id="tts_history_perf_sanity")

    # Keep the dataset large enough to exercise pagination without making tests too slow.
    total_rows = 2000
    for idx in range(total_rows):
        media_db.create_tts_history_entry(
            user_id="1",
            text_hash=f"hash-{idx}",
            provider="openai",
            model="tts-1",
            voice_name="alloy",
            format="mp3",
            status="success",
        )
    media_db.close_connection()

    resp = test_client.get("/api/v1/audio/history?limit=100", headers=auth_headers)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert len(payload["items"]) == 100
    assert payload["next_cursor"]
    assert payload["pagination"] == {
        "mode": "cursor",
        "limit": 100,
        "cursor": None,
        "next_cursor": payload["next_cursor"],
        "has_more": True,
    }

    first_page_ids = {item["id"] for item in payload["items"]}

    resp_page2 = test_client.get(
        f"/api/v1/audio/history?limit=100&cursor={payload['next_cursor']}",
        headers=auth_headers,
    )
    assert resp_page2.status_code == status.HTTP_200_OK
    payload_page2 = resp_page2.json()
    assert len(payload_page2["items"]) == 100
    assert payload_page2["pagination"] == {
        "mode": "cursor",
        "limit": 100,
        "cursor": payload["next_cursor"],
        "next_cursor": payload_page2["next_cursor"],
        "has_more": True,
    }

    second_page_ids = {item["id"] for item in payload_page2["items"]}
    assert first_page_ids.isdisjoint(second_page_ids)


def test_tts_history_include_total_sanity(test_client, auth_headers, monkeypatch, tmp_path):
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    monkeypatch.setattr(settings, "USER_DB_BASE_DIR", str(user_db_base), raising=False)

    media_db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(1)), client_id="tts_history_include_total")

    total_rows = 250
    for idx in range(total_rows):
        media_db.create_tts_history_entry(
            user_id="1",
            text_hash=f"include-total-{idx}",
            provider="openai",
            model="tts-1",
            voice_name="alloy",
            format="mp3",
            status="success",
        )
    media_db.close_connection()

    resp_default = test_client.get("/api/v1/audio/history?limit=25", headers=auth_headers)
    assert resp_default.status_code == status.HTTP_200_OK
    default_payload = resp_default.json()
    assert default_payload["total"] is None

    resp_total = test_client.get("/api/v1/audio/history?limit=25&include_total=true", headers=auth_headers)
    assert resp_total.status_code == status.HTTP_200_OK
    total_payload = resp_total.json()
    assert len(total_payload["items"]) == 25
    assert int(total_payload["total"]) == total_rows
    assert total_payload["pagination"] == {
        "mode": "cursor",
        "limit": 25,
        "cursor": None,
        "next_cursor": total_payload["next_cursor"],
        "has_more": True,
    }
