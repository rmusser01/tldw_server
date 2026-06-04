from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Workspaces.operations import (
    fingerprint_workspace_command,
    operation_poll_href,
)


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    database.upsert_workspace("ws-ops", "Operation Workspace")
    return database


def test_workspace_operation_create_is_idempotent_and_conflicts_on_changed_request(db):
    fingerprint = fingerprint_workspace_command(
        {
            "display_name": "Project root",
            "requested_runtime": "docker",
            "raw_host_path": "/Users/alice/private/project",
        }
    )
    operation = db.create_workspace_operation(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="root-key",
        request_fingerprint=fingerprint,
        linked_idempotency_key="sandbox-key",
        status="running",
        diagnostics={
            "message": "cannot use /Users/alice/private/project",
            "api_key": "sk-secret",
        },
    )

    retry = db.create_workspace_operation(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="root-key",
        request_fingerprint=fingerprint,
        status="queued",
    )

    assert retry["id"] == operation["id"]
    assert retry["status"] == "running"
    assert db.get_workspace_operation("ws-ops", operation["id"])["id"] == operation["id"]
    assert db.get_workspace_operation_by_idempotency(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="root-key",
    )["id"] == operation["id"]
    assert "/Users/alice" not in repr(operation["diagnostics"])
    assert "sk-secret" not in repr(operation["diagnostics"])
    assert operation_poll_href("ws-ops", operation["id"]).endswith(f"/ws-ops/operations/{operation['id']}")

    with pytest.raises(ConflictError):
        db.create_workspace_operation(
            workspace_id="ws-ops",
            user_id="user-1",
            command="provision_sandbox_root",
            idempotency_key="root-key",
            request_fingerprint=fingerprint_workspace_command({"requested_runtime": "vz_linux"}),
        )


def test_workspace_operation_cleanup_preserves_attached_root(db):
    db.upsert_workspace_primary_root(
        "ws-ops",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
        },
    )
    expired = datetime.now(timezone.utc) - timedelta(minutes=1)
    operation = db.create_workspace_operation(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="expired-key",
        request_fingerprint=fingerprint_workspace_command({"requested_runtime": "docker"}),
        status="expired",
        result_ref={"root_id": "primary", "sandbox_volume_id": "volume-1"},
        expires_at=expired,
    )

    assert db.cleanup_expired_workspace_operations(now=datetime.now(timezone.utc)) == 1
    assert db.get_workspace_operation("ws-ops", operation["id"]) is None
    assert db.get_workspace_primary_root("ws-ops")["sandbox_volume_id"] == "volume-1"


def test_list_active_workspace_operations_filters_terminal_and_expired(db):
    active = db.create_workspace_operation(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="active-key",
        request_fingerprint=fingerprint_workspace_command({"requested_runtime": "docker"}),
        status="running",
    )
    db.create_workspace_operation(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="done-key",
        request_fingerprint=fingerprint_workspace_command({"requested_runtime": "docker", "done": True}),
        status="succeeded",
    )
    db.create_workspace_operation(
        workspace_id="ws-ops",
        user_id="user-1",
        command="provision_sandbox_root",
        idempotency_key="expired-active-key",
        request_fingerprint=fingerprint_workspace_command({"requested_runtime": "docker", "expired": True}),
        status="running",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    operations = db.list_active_workspace_operations("ws-ops")

    assert [operation["id"] for operation in operations] == [active["id"]]
