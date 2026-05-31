"""Integration tests for Kanban workflow endpoints."""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints.kanban import kanban_workflow
from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB, KanbanDBError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

pytestmark = pytest.mark.integration


def test_workflow_http_error_sanitizes_kanban_db_error_message():
    http_error = kanban_workflow._workflow_http_error(
        KanbanDBError("kanban backend exploded at /private/kanban.db"),
        operation="test workflow operation",
    )

    assert http_error.status_code == 500
    assert http_error.detail == {
        "code": "kanban_db_error",
        "message": "Workflow database error",
    }


@pytest.fixture()
def workflow_client_with_kanban_db(tmp_path, monkeypatch):
    """Create a workflow-aware test client with a temporary Kanban database."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    auth_db_path = (tmp_path / "auth_users.db").resolve()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:////{auth_db_path}")
    db_path = DatabasePaths.get_kanban_db_path("integration_workflow_user")
    db = KanbanDB(str(db_path), user_id="integration_workflow_user")

    async def override_user():
        return User(id=1, username="workflow-tester", email="workflow@test.local", is_active=True, is_admin=True)

    from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import get_kanban_db_for_user

    def override_db_dep():
        return db

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app
    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_kanban_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client, db

    fastapi_app.dependency_overrides.clear()


def _create_board_list_card(client: TestClient) -> tuple[int, int, int]:
    board_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Workflow API Board", "client_id": "workflow-board-1"},
    )
    assert board_resp.status_code == 201, board_resp.text
    board_id = board_resp.json()["id"]

    list_resp = client.post(
        f"/api/v1/kanban/boards/{board_id}/lists",
        json={"name": "Workflow API List", "client_id": "workflow-list-1"},
    )
    assert list_resp.status_code == 201, list_resp.text
    list_id = list_resp.json()["id"]

    card_resp = client.post(
        f"/api/v1/kanban/lists/{list_id}/cards",
        json={"title": "Workflow API Card", "client_id": "workflow-card-1"},
    )
    assert card_resp.status_code == 201, card_resp.text
    card_id = card_resp.json()["id"]

    return board_id, list_id, card_id


def _workflow_event(index: int, *, card_id: int) -> dict[str, Any]:
    return {
        "id": index + 1,
        "card_id": card_id,
        "event_type": "state_changed",
        "from_status_key": "todo",
        "to_status_key": "done",
        "actor": "tester",
        "reason": None,
        "idempotency_key": f"event-{index}",
        "correlation_id": f"corr-{index}",
        "before_snapshot": None,
        "after_snapshot": None,
        "created_at": datetime.now(timezone.utc),
    }


def _stale_claim(index: int) -> dict[str, Any]:
    return {
        "card_id": index + 1,
        "board_id": 100 + index,
        "list_id": 200 + index,
        "title": f"Stale card {index}",
        "workflow_status_key": "todo",
        "lease_owner": "stale-worker",
        "lease_expires_at": datetime.now(timezone.utc),
        "version": 3,
        "updated_at": datetime.now(timezone.utc),
    }


class _FakeWorkflowPaginationDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_card_workflow_events(self, *, card_id: int, limit: int, offset: int) -> list[dict[str, Any]]:
        self.calls.append(("events", {"card_id": card_id, "limit": limit, "offset": offset}))
        return [_workflow_event(index, card_id=card_id) for index in range(limit)]

    def list_stale_workflow_claims(
        self,
        *,
        board_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.calls.append(("stale_claims", {"board_id": board_id, "limit": limit, "offset": offset}))
        return [_stale_claim(index) for index in range(limit)]


async def test_workflow_events_response_includes_canonical_pagination() -> None:
    """Workflow event list overfetches and returns canonical offset metadata."""
    db = _FakeWorkflowPaginationDB()

    response = await kanban_workflow.list_card_workflow_events(card_id=7, limit=2, offset=4, db=db)  # type: ignore[arg-type]

    assert db.calls == [("events", {"card_id": 7, "limit": 3, "offset": 4})]
    assert len(response.events) == 2
    assert response.pagination.model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 4,
        "total": None,
        "has_more": True,
        "next_offset": 6,
    }


async def test_stale_workflow_claims_response_includes_canonical_pagination() -> None:
    """Stale-claim recovery list accepts offset and returns canonical metadata."""
    db = _FakeWorkflowPaginationDB()

    response = await kanban_workflow.list_stale_workflow_claims(board_id=42, limit=2, offset=5, db=db)  # type: ignore[arg-type]

    assert db.calls == [("stale_claims", {"board_id": 42, "limit": 3, "offset": 5})]
    assert len(response.stale_claims) == 2
    assert response.pagination.model_dump(mode="json") == {
        "mode": "offset",
        "limit": 2,
        "offset": 5,
        "total": None,
        "has_more": True,
        "next_offset": 7,
    }


def test_transition_endpoint_enforces_policy_and_returns_structured_error(workflow_client_with_kanban_db):
    """Transition endpoint should return stable structured error for lease failures."""
    client, db = workflow_client_with_kanban_db
    board_id, _list_id, card_id = _create_board_list_card(client)

    db.upsert_workflow_policy(
        board_id=board_id,
        statuses=[
            {"status_key": "todo", "display_name": "To Do", "sort_order": 0},
            {"status_key": "impl", "display_name": "Implement", "sort_order": 1},
        ],
        transitions=[
            {
                "from_status_key": "todo",
                "to_status_key": "impl",
                "requires_claim": True,
                "requires_approval": False,
            }
        ],
    )
    state = db.get_card_workflow_state(card_id)

    transition_resp = client.post(
        f"/api/v1/kanban/workflow/cards/{card_id}/transition",
        json={
            "to_status_key": "impl",
            "actor": "builder",
            "expected_version": state["version"],
            "idempotency_key": "api-transition-no-lease",
            "correlation_id": "corr-api-transition-no-lease",
            "reason": "start implementation",
        },
    )

    assert transition_resp.status_code == 409, transition_resp.text
    payload: dict[str, Any] = transition_resp.json()
    assert isinstance(payload.get("detail"), dict)
    assert payload["detail"]["code"] == "lease_required"
    assert "lease_required" in payload["detail"]["message"]


def test_transition_endpoint_returns_policy_paused_code(workflow_client_with_kanban_db):
    """Transition endpoint should return policy_paused when board policy is paused."""
    client, db = workflow_client_with_kanban_db
    board_id, _list_id, card_id = _create_board_list_card(client)

    db.upsert_workflow_policy(
        board_id=board_id,
        statuses=[
            {"status_key": "todo", "display_name": "To Do", "sort_order": 0},
            {"status_key": "impl", "display_name": "Implement", "sort_order": 1},
        ],
        transitions=[
            {
                "from_status_key": "todo",
                "to_status_key": "impl",
                "requires_claim": False,
                "requires_approval": False,
            }
        ],
        is_paused=True,
    )
    state = db.get_card_workflow_state(card_id)

    transition_resp = client.post(
        f"/api/v1/kanban/workflow/cards/{card_id}/transition",
        json={
            "to_status_key": "impl",
            "actor": "builder",
            "expected_version": state["version"],
            "idempotency_key": "api-transition-paused",
            "correlation_id": "corr-api-transition-paused",
            "reason": "attempt while paused",
        },
    )

    assert transition_resp.status_code == 409, transition_resp.text
    payload: dict[str, Any] = transition_resp.json()
    assert isinstance(payload.get("detail"), dict)
    assert payload["detail"]["code"] == "policy_paused"
    assert "policy_paused" in payload["detail"]["message"]


def test_patch_state_version_conflict_returns_stable_code(workflow_client_with_kanban_db):
    """State patch should return stable version_conflict code on CAS mismatch."""
    client, _db = workflow_client_with_kanban_db
    _board_id, _list_id, card_id = _create_board_list_card(client)

    state_resp = client.get(f"/api/v1/kanban/workflow/cards/{card_id}/state")
    assert state_resp.status_code == 200, state_resp.text
    state = state_resp.json()

    patch_resp = client.patch(
        f"/api/v1/kanban/workflow/cards/{card_id}/state",
        json={
            "workflow_status_key": state["workflow_status_key"],
            "expected_version": state["version"] + 10,
            "idempotency_key": "api-patch-version-conflict",
            "correlation_id": "corr-api-patch-version-conflict",
            "actor": "repair-admin",
        },
    )

    assert patch_resp.status_code == 409, patch_resp.text
    payload: dict[str, Any] = patch_resp.json()
    assert isinstance(payload.get("detail"), dict)
    assert payload["detail"]["code"] == "version_conflict"


def test_policy_upsert_preserves_metadata_when_omitted(workflow_client_with_kanban_db):
    """Policy metadata should be preserved when omitted from upsert payload."""
    client, _db = workflow_client_with_kanban_db
    board_id, _list_id, _card_id = _create_board_list_card(client)

    first = client.put(
        f"/api/v1/kanban/workflow/boards/{board_id}/policy",
        json={
            "default_lease_ttl_sec": 900,
            "metadata": {"owner": "workflow-team"},
        },
    )
    assert first.status_code == 200, first.text
    assert first.json()["metadata"] == {"owner": "workflow-team"}

    second = client.put(
        f"/api/v1/kanban/workflow/boards/{board_id}/policy",
        json={
            "default_lease_ttl_sec": 1200,
        },
    )
    assert second.status_code == 200, second.text
    assert second.json()["default_lease_ttl_sec"] == 1200
    assert second.json()["metadata"] == {"owner": "workflow-team"}
