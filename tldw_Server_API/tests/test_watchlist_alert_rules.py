from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.watchlist_alert_rules import router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.watchlist_alert_rules_db import (
    create_watchlist_alert_rule,
    delete_watchlist_alert_rule,
    get_watchlist_alert_rule,
    list_watchlist_alert_rules,
    update_watchlist_alert_rule,
)
from tldw_Server_API.app.core.Watchlists.alert_rules import (
    create_alert_rule,
    ensure_alert_rules_table,
    evaluate_rules_for_run,
    update_alert_rule,
)


pytestmark = pytest.mark.unit


@pytest.fixture()
def alert_rules_db_path(tmp_path) -> str:
    db_path = tmp_path / "ChaChaNotes.db"
    ensure_alert_rules_table(str(db_path))
    return str(db_path)


@pytest.fixture()
def alert_rules_client(monkeypatch, tmp_path):
    app = FastAPI()
    app.include_router(router)

    async def override_user():
        return User(id=555, username="alert-user", email=None, is_active=True)

    app.dependency_overrides[get_request_user] = override_user
    monkeypatch.setenv("TLDW_USER_DB_DIR", str(tmp_path / "user_dbs"))

    with TestClient(app) as client:
        yield client


def test_update_alert_rule_returns_updated_rule(alert_rules_db_path: str) -> None:
    created = create_alert_rule(
        alert_rules_db_path,
        user_id="user-1",
        name="Items below",
        condition_type="items_below",
        condition_value={"threshold": 2},
    )

    updated = update_alert_rule(
        alert_rules_db_path,
        created.id,
        "user-1",
        name="Items below updated",
        enabled=False,
        condition_value={"threshold": 5},
    )

    assert updated is not None
    assert updated.id == created.id
    assert updated.name == "Items below updated"
    assert updated.enabled is False
    assert json.loads(updated.condition_value) == {"threshold": 5}


def test_watchlist_alert_rule_db_helpers_round_trip(alert_rules_db_path: str) -> None:
    created = create_watchlist_alert_rule(
        alert_rules_db_path,
        user_id="user-1",
        name="DB helper rule",
        condition_type="items_below",
        condition_value={"threshold": 3},
        job_id=12,
        severity="warning",
    )

    listed = list_watchlist_alert_rules(alert_rules_db_path, "user-1", job_id=12)
    updated = update_watchlist_alert_rule(
        alert_rules_db_path,
        created.id,
        "user-1",
        enabled=False,
        condition_value={"threshold": 7},
    )
    fetched = get_watchlist_alert_rule(alert_rules_db_path, created.id, "user-1")
    deleted = delete_watchlist_alert_rule(alert_rules_db_path, created.id, "user-1")

    assert [rule.id for rule in listed] == [created.id]
    assert updated is not None
    assert updated.enabled is False
    assert json.loads(updated.condition_value) == {"threshold": 7}
    assert fetched is not None
    assert fetched.id == created.id
    assert deleted is True


def test_update_alert_rule_rejects_invalid_condition_type(alert_rules_db_path: str) -> None:
    created = create_alert_rule(
        alert_rules_db_path,
        user_id="user-1",
        name="No items",
        condition_type="no_items",
    )

    with pytest.raises(ValueError, match="Invalid condition_type"):
        update_alert_rule(
            alert_rules_db_path,
            created.id,
            "user-1",
            condition_type="not-a-real-condition",
        )


def test_evaluate_rules_for_run_skips_invalid_threshold_values(alert_rules_db_path: str) -> None:
    create_alert_rule(
        alert_rules_db_path,
        user_id="user-1",
        job_id=11,
        name="Bad threshold",
        condition_type="items_above",
        condition_value={"threshold": "abc"},
    )
    create_alert_rule(
        alert_rules_db_path,
        user_id="user-1",
        job_id=11,
        name="Run failed",
        condition_type="run_failed",
    )

    triggered = evaluate_rules_for_run(
        alert_rules_db_path,
        user_id="user-1",
        job_id=11,
        run_id=77,
        stats={"items_found": 4, "items_ingested": 1, "error_msg": "boom"},
        status="failed",
    )

    assert len(triggered) == 1
    assert triggered[0]["rule"].condition_type == "run_failed"
    assert "Run failed" in triggered[0]["notification_kwargs"]["message"]


def test_evaluate_rules_for_run_emits_health_issue_notification_payload(alert_rules_db_path: str) -> None:
    create_alert_rule(
        alert_rules_db_path,
        user_id="user-1",
        job_id=11,
        name="Run failed",
        condition_type="run_failed",
    )

    triggered = evaluate_rules_for_run(
        alert_rules_db_path,
        user_id="user-1",
        job_id=11,
        run_id=77,
        stats={"items_found": 4, "items_ingested": 1, "error_msg": "boom"},
        status="failed",
    )

    payload = triggered[0]["notification_kwargs"]
    assert payload["kind"] == "watchlist_health_issue"
    assert payload["type"] == "watchlist_health_issue"
    assert payload["legacy_kind"] == "watchlist_alert"
    assert payload["category"] == "health"
    assert payload["link_type"] == "watchlist_run"
    assert payload["link_id"] == "77"
    assert "item_id" not in payload


def test_update_rule_rejects_invalid_condition_type(alert_rules_client: TestClient) -> None:
    create_response = alert_rules_client.post(
        "/watchlists/alert-rules",
        json={
            "name": "API rule",
            "condition_type": "no_items",
        },
    )
    assert create_response.status_code == 201, create_response.text
    rule_id = create_response.json()["id"]

    update_response = alert_rules_client.patch(
        f"/watchlists/alert-rules/{rule_id}",
        json={"condition_type": "invalid-condition"},
    )

    assert update_response.status_code == 400
    assert "Invalid condition_type" in update_response.text
