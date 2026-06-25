from pathlib import Path

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import rpg as rpg_endpoint
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.PrivilegeCatalog.test_endpoint_scope_catalog_sync import load_catalog_scope_ids


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _headers(**extra):
    return {"X-API-KEY": get_settings().SINGLE_USER_API_KEY, **extra}


def _create_campaign_and_session(client: TestClient, prefix: str) -> tuple[int, int]:
    campaign = client.post(
        "/api/v1/rpg/campaigns",
        headers=_headers(**{"Idempotency-Key": f"{prefix}-campaign"}),
        json={"title": f"{prefix} Campaign", "default_adapter_key": "fate"},
    )
    assert campaign.status_code == 201  # nosec B101
    campaign_id = campaign.json()["id"]

    session = client.post(
        f"/api/v1/rpg/campaigns/{campaign_id}/sessions",
        headers=_headers(**{"Idempotency-Key": f"{prefix}-session"}),
        json={"title": f"{prefix} Opening", "adapter_key": "fate"},
    )
    assert session.status_code == 201  # nosec B101
    return campaign_id, session.json()["id"]


def test_rpg_endpoint_scopes_are_cataloged():
    catalog_path = _REPO_ROOT / "tldw_Server_API" / "Config_Files" / "privilege_catalog.yaml"
    catalog_ids = load_catalog_scope_ids(catalog_path)
    expected_ids = {
        rpg_endpoint.RPG_RULES_READ,
        rpg_endpoint.RPG_CAMPAIGNS_MANAGE,
        rpg_endpoint.RPG_SESSIONS_MANAGE,
        rpg_endpoint.RPG_PROPOSALS_REVIEW,
    }

    assert expected_ids <= catalog_ids  # nosec B101


def test_rpg_adapters_endpoint_lists_default_adapters():
    client = TestClient(app)

    response = client.get("/api/v1/rpg/rules/adapters", headers=_headers())

    assert response.status_code == 200  # nosec B101
    keys = [item["adapter_key"] for item in response.json()["adapters"]]
    assert keys == ["dnd5e_srd", "fate", "pf2e"]  # nosec B101


def test_create_campaign_session_and_record_user_event():
    client = TestClient(app)

    _, session_id = _create_campaign_and_session(client, "api-main")

    event_response = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-main-event"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "note.added",
                    "event_payload": {"note_id": "n1", "text": "At the docks"},
                }
            ],
        },
    )

    assert event_response.status_code == 200  # nosec B101
    payload = event_response.json()
    assert payload["committed_events"][0]["sequence_number"] == 1  # nosec B101
    assert payload["proposal"] is None  # nosec B101


def test_create_campaign_requires_idempotency_key():
    client = TestClient(app)

    response = client.post(
        "/api/v1/rpg/campaigns",
        headers=_headers(),
        json={"title": "Missing Header", "default_adapter_key": "fate"},
    )

    assert response.status_code == 422  # nosec B101


def test_record_events_rejects_stale_expected_sequence():
    client = TestClient(app)
    _, session_id = _create_campaign_and_session(client, "api-stale")

    first = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-stale-event-1"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "note.added",
                    "event_payload": {"note_id": "n1", "text": "First note"},
                }
            ],
        },
    )
    assert first.status_code == 200  # nosec B101

    stale = client.post(
        f"/api/v1/rpg/sessions/{session_id}/events",
        headers=_headers(**{"Idempotency-Key": "api-stale-event-2"}),
        json={
            "expected_last_event_sequence": 0,
            "events": [
                {
                    "event_type": "note.added",
                    "event_payload": {"note_id": "n2", "text": "Second note"},
                }
            ],
        },
    )

    assert stale.status_code == 409  # nosec B101
    assert stale.json()["detail"] == "stale_event_sequence"  # nosec B101
