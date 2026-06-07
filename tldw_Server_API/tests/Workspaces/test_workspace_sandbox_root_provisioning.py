from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Sandbox.store import InMemoryStore
from tldw_Server_API.app.core.Sandbox.workspace_volumes import SandboxWorkspaceVolumeService
from tldw_Server_API.app.core.Workspaces.sandbox_root_provisioning import (
    WorkspaceSandboxRootProvisionRequestData,
    provision_and_attach_sandbox_root,
)
from tldw_Server_API.app.core.Workspaces.root_binding_service import WorkspaceRootConflictError


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    database.upsert_workspace("ws-project", "Project Workspace")
    return database


@pytest.fixture
def volume_service():
    return SandboxWorkspaceVolumeService(store=InMemoryStore())


def test_provision_and_attach_sandbox_root_creates_active_operation_and_project_root(db, volume_service):
    result = provision_and_attach_sandbox_root(
        db=db,
        sandbox_volume_service=volume_service,
        workspace_id="ws-project",
        user_id="user-1",
        request=WorkspaceSandboxRootProvisionRequestData(
            display_name="Project root",
            requested_runtime="docker",
        ),
        idempotency_key="root-key",
    )

    assert result.http_status_code == 202
    assert result.operation["status"] == "running"
    assert result.operation["retryable"] is True
    assert result.primary_root is not None
    assert result.primary_root["backend"] == "sandbox_volume"
    assert result.primary_root["sandbox_mount_state"] == "not_configured"
    assert db.get_workspace("ws-project")["workspace_profile"] == "project"
    assert db.list_active_workspace_operations("ws-project")[0]["id"] == result.operation["operation_id"]


def test_provision_and_attach_sandbox_root_retries_return_same_operation(db, volume_service):
    request = WorkspaceSandboxRootProvisionRequestData(
        display_name="Project root",
        requested_runtime="docker",
    )
    first = provision_and_attach_sandbox_root(
        db=db,
        sandbox_volume_service=volume_service,
        workspace_id="ws-project",
        user_id="user-1",
        request=request,
        idempotency_key="root-key",
    )
    retry = provision_and_attach_sandbox_root(
        db=db,
        sandbox_volume_service=volume_service,
        workspace_id="ws-project",
        user_id="user-1",
        request=request,
        idempotency_key="root-key",
    )

    assert retry.operation["operation_id"] == first.operation["operation_id"]
    assert retry.primary_root["sandbox_mount_state"] == "not_configured"
    volumes = volume_service.store.list_workspace_volumes_for_workspace("ws-project")
    assert len(volumes) == 1
    assert volumes[0].root_id == retry.primary_root["root_id"]


def test_provision_and_attach_sandbox_root_rejects_changed_request_for_same_key(db, volume_service):
    provision_and_attach_sandbox_root(
        db=db,
        sandbox_volume_service=volume_service,
        workspace_id="ws-project",
        user_id="user-1",
        request=WorkspaceSandboxRootProvisionRequestData(
            display_name="Project root",
            requested_runtime="docker",
        ),
        idempotency_key="root-key",
    )

    with pytest.raises(ConflictError):
        provision_and_attach_sandbox_root(
            db=db,
            sandbox_volume_service=volume_service,
            workspace_id="ws-project",
            user_id="user-1",
            request=WorkspaceSandboxRootProvisionRequestData(
                display_name="Project root",
                requested_runtime="vz_linux",
            ),
            idempotency_key="root-key",
        )


def test_provision_and_attach_sandbox_root_parses_string_false_replace_existing(db, volume_service):
    first = provision_and_attach_sandbox_root(
        db=db,
        sandbox_volume_service=volume_service,
        workspace_id="ws-project",
        user_id="user-1",
        request=WorkspaceSandboxRootProvisionRequestData(
            display_name="Project root",
            requested_runtime="docker",
        ),
        idempotency_key="root-key",
    )

    with pytest.raises(WorkspaceRootConflictError):
        provision_and_attach_sandbox_root(
            db=db,
            sandbox_volume_service=volume_service,
            workspace_id="ws-project",
            user_id="user-1",
            request={
                "display_name": "Replacement",
                "requested_runtime": "docker",
                "root_id": "replacement",
                "replace_existing": "false",
                "expected_workspace_version": first.primary_root["version"],
            },
            idempotency_key="replacement-key",
        )
