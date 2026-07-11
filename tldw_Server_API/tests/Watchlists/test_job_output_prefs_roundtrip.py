import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=946, username="wluser", email=None, is_active=True)

    base_dir = tmp_path / "test_user_dbs_job_output_prefs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")

    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    from tldw_Server_API.app.core.config import API_V1_PREFIX

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_job_output_prefs_template_and_delivery_roundtrip(client_with_user: TestClient):
    c = client_with_user

    source = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Output Prefs Feed",
            "url": "https://example.com/output-prefs-feed.xml",
            "source_type": "rss",
        },
    )
    assert source.status_code == 200, source.text
    source_id = source.json()["id"]

    create_payload = {
        "name": "Output Prefs Job",
        "scope": {"sources": [source_id]},
        "active": True,
        "output_prefs": {
            "template": {
                "default_name": "daily_md",
                "default_version": 2,
                "default_format": "md",
            },
            "deliveries": {
                "email": {
                    "enabled": True,
                    "recipients": ["digest@example.com"],
                    "body_format": "html",
                    "attach_file": False,
                },
                "chatbook": {
                    "enabled": True,
                    "title": "Digest Doc",
                    "description": "Daily digest export",
                    "conversation_id": 42,
                },
            },
            "ingest": {
                "persist_to_media_db": True,
            },
            "custom_future_key": {"keep": True},
        },
    }

    created = c.post("/api/v1/watchlists/jobs", json=create_payload)
    assert created.status_code == 200, created.text
    created_job = created.json()
    job_id = created_job["id"]

    created_prefs = created_job.get("output_prefs") or {}
    assert "template" not in created_prefs
    assert "deliveries" not in created_prefs
    created_contract = created_prefs["briefing_pipeline"]
    assert created_contract["text"]["template_name"] == "daily_md"
    assert created_contract["text"]["template_version"] == 2
    assert created_contract["delivery"]["email"]["recipients"] == ["digest@example.com"]
    assert created_contract["delivery"]["chatbook"]["conversation_id"] == 42
    assert created_prefs.get("ingest", {}).get("persist_to_media_db") is True
    assert created_prefs["custom_future_key"] == {"keep": True}

    fetched = c.get(f"/api/v1/watchlists/jobs/{job_id}")
    assert fetched.status_code == 200, fetched.text
    fetched_prefs = fetched.json().get("output_prefs") or {}
    assert fetched_prefs == created_prefs

    update_payload = {
        "output_prefs": {
            "template": {"default_name": "daily_md", "default_version": 3},
            "deliveries": {
                "email": {"enabled": False},
                "chatbook": {"enabled": True, "conversation_id": 99},
            },
            "custom_future_key": {"updated": True},
        }
    }
    updated = c.patch(f"/api/v1/watchlists/jobs/{job_id}", json=update_payload)
    assert updated.status_code == 200, updated.text
    updated_prefs = updated.json().get("output_prefs") or {}
    assert "template" not in updated_prefs
    assert "deliveries" not in updated_prefs
    updated_contract = updated_prefs["briefing_pipeline"]
    assert updated_contract["text"]["template_version"] == 3
    assert updated_contract["delivery"]["email"]["enabled"] is False
    assert updated_contract["delivery"]["chatbook"]["conversation_id"] == 99
    assert updated_prefs["custom_future_key"] == {"updated": True}

    ingest_merge = c.patch(
        f"/api/v1/watchlists/jobs/{job_id}",
        json={"ingest_prefs": {"persist_to_media_db": False}},
    )
    assert ingest_merge.status_code == 200, ingest_merge.text
    merged_prefs = ingest_merge.json().get("output_prefs") or {}
    merged_contract = merged_prefs["briefing_pipeline"]
    assert merged_contract["text"]["template_version"] == 3
    assert merged_contract["delivery"]["chatbook"]["conversation_id"] == 99
    assert merged_prefs.get("ingest", {}).get("persist_to_media_db") is False
    assert merged_prefs["custom_future_key"] == {"updated": True}


def test_job_output_prefs_can_be_explicitly_cleared(client_with_user: TestClient):
    c = client_with_user

    source = c.post(
        "/api/v1/watchlists/sources",
        json={
            "name": "Output Prefs Clear Feed",
            "url": "https://example.com/output-prefs-clear.xml",
            "source_type": "rss",
        },
    )
    assert source.status_code == 200, source.text
    source_id = source.json()["id"]

    created = c.post(
        "/api/v1/watchlists/jobs",
        json={
            "name": "Output Prefs Clear Job",
            "scope": {"sources": [source_id]},
            "active": True,
            "output_prefs": {
                "template": {"default_name": "daily_md", "default_version": 2},
                "deliveries": {
                    "email": {"enabled": True, "recipients": ["digest@example.com"]},
                },
                "ingest": {"persist_to_media_db": True},
            },
        },
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["id"]

    # Explicitly clearing output_prefs should replace persisted prefs with the safe canonical defaults.
    cleared = c.patch(f"/api/v1/watchlists/jobs/{job_id}", json={"output_prefs": {}})
    assert cleared.status_code == 200, cleared.text
    cleared_prefs = cleared.json()["output_prefs"]
    assert set(cleared_prefs) == {"briefing_pipeline"}
    cleared_contract = cleared_prefs["briefing_pipeline"]
    assert cleared_contract["text"]["enabled"] is False
    assert cleared_contract["audio"]["enabled"] is False
    assert cleared_contract["delivery"]["reports"]["enabled"] is True
    assert cleared_contract["delivery"]["email"]["enabled"] is False
    assert cleared_contract["delivery"]["chatbook"]["enabled"] is False
    assert cleared_contract["test"]["external_delivery"] is False
    assert cleared.json().get("ingest_prefs") is None

    fetched = c.get(f"/api/v1/watchlists/jobs/{job_id}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json().get("output_prefs") == cleared_prefs
    assert fetched.json().get("ingest_prefs") is None
