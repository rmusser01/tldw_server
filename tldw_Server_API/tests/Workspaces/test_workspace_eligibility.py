from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Workspaces.eligibility import (
    WorkspaceEligibilityOperation,
    WorkspaceEligibilityRequest,
    WorkspaceEligibilityService,
    WORKSPACE_ACTIVE_CONTEXT_OPERATIONS,
    WORKSPACE_VISIBILITY_OPERATIONS,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(
        db_path=str(tmp_path / "workspace_eligibility.sqlite"),
        client_id="test-client",
    )
    database.upsert_workspace("ws-active", "Active Workspace")
    database.upsert_workspace("ws-other", "Other Workspace")
    database.upsert_workspace("ws-archived", "Archived Workspace")
    database.update_workspace("ws-archived", {"archived": True}, expected_version=1)
    database.add_workspace_resource_membership(
        "ws-active",
        {"resource_type": "chat", "resource_id": "chat-1", "role": "conversation"},
    )
    database.add_workspace_resource_membership(
        "ws-other",
        {"resource_type": "chat", "resource_id": "chat-other", "role": "conversation"},
    )
    return database


@pytest.fixture
def service(db: CharactersRAGDB) -> WorkspaceEligibilityService:
    return WorkspaceEligibilityService(db)


def _request(
    operation: WorkspaceEligibilityOperation,
    *,
    active_workspace_id: str | None = "ws-active",
    resource_type: str = "chat",
    resource_id: str = "chat-1",
    runtime_state: str = "not_required",
    permission_state: str = "granted",
) -> WorkspaceEligibilityRequest:
    return WorkspaceEligibilityRequest(
        operation=operation,
        active_workspace_id=active_workspace_id,
        resource_type=resource_type,
        resource_id=resource_id,
        runtime_state=runtime_state,
        permission_state=permission_state,
    )


def test_operation_matrix_separates_visibility_from_active_context() -> None:
    assert WORKSPACE_VISIBILITY_OPERATIONS == frozenset({"browse", "search", "open", "edit"})
    assert WORKSPACE_ACTIVE_CONTEXT_OPERATIONS == frozenset(
        {
            "stage",
            "rag_ground",
            "prompt_use",
            "tool_use",
            "agent_manipulate",
            "acp_run",
            "sandbox_operation",
            "workflow_launch",
            "watchlist_run",
        }
    )
    assert WORKSPACE_VISIBILITY_OPERATIONS.isdisjoint(WORKSPACE_ACTIVE_CONTEXT_OPERATIONS)


@pytest.mark.parametrize("operation", ["browse", "search", "open", "edit"])
def test_visibility_operations_do_not_require_active_workspace(
    service: WorkspaceEligibilityService,
    operation: WorkspaceEligibilityOperation,
) -> None:
    result = service.check(
        _request(
            operation,
            active_workspace_id=None,
            resource_type="prompt",
            resource_id="prompt-1",
        )
    )

    assert result.allowed is True
    assert result.operation_category == "visibility"
    assert result.reason_code == "visibility_allowed"
    assert result.global_visibility_preserved is True
    assert result.recovery_actions == []


def test_permission_denial_applies_before_visibility_allowance(
    service: WorkspaceEligibilityService,
) -> None:
    result = service.check(
        _request(
            "open",
            active_workspace_id=None,
            resource_type="chat",
            resource_id="chat-1",
            permission_state="denied",
        )
    )

    assert result.allowed is False
    assert result.reason_code == "permission_denied"
    assert result.operation_category == "visibility"
    assert result.global_visibility_preserved is True


def test_active_operation_requires_active_workspace(service: WorkspaceEligibilityService) -> None:
    result = service.check(_request("stage", active_workspace_id=None))

    assert result.allowed is False
    assert result.reason_code == "no_active_workspace"
    assert result.operation_category == "active_context"
    assert {action.action for action in result.recovery_actions} == {"select_workspace", "create_workspace"}


def test_active_operation_rejects_archived_workspace(service: WorkspaceEligibilityService) -> None:
    result = service.check(_request("rag_ground", active_workspace_id="ws-archived"))

    assert result.allowed is False
    assert result.reason_code == "workspace_archived"
    assert {action.action for action in result.recovery_actions} == {"select_workspace", "unarchive_workspace"}


def test_active_operation_rejects_nonexistent_workspace(service: WorkspaceEligibilityService) -> None:
    result = service.check(_request("stage", active_workspace_id="ws-nonexistent"))

    assert result.allowed is False
    assert result.reason_code == "workspace_not_found"
    assert {action.action for action in result.recovery_actions} == {"select_workspace"}


def test_active_operation_allows_resource_linked_to_active_workspace(
    service: WorkspaceEligibilityService,
) -> None:
    result = service.check(_request("rag_ground"))

    assert result.allowed is True
    assert result.reason_code == "allowed"
    assert result.operation_category == "active_context"
    assert result.membership is not None
    assert result.membership.workspace_id == "ws-active"
    assert result.resource_workspace_ids == ["ws-active"]


def test_active_operation_rejects_resource_linked_to_other_workspace(
    service: WorkspaceEligibilityService,
) -> None:
    result = service.check(_request("rag_ground", resource_id="chat-other"))

    assert result.allowed is False
    assert result.reason_code == "cross_workspace_resource"
    assert result.resource_workspace_ids == ["ws-other"]
    assert {action.action for action in result.recovery_actions} == {"copy_to_active_workspace", "switch_workspace"}


def test_active_operation_rejects_unlinked_resource(service: WorkspaceEligibilityService) -> None:
    result = service.check(_request("rag_ground", resource_id="chat-missing"))

    assert result.allowed is False
    assert result.reason_code == "resource_not_linked"
    assert result.resource_workspace_ids == []
    assert {action.action for action in result.recovery_actions} == {"link_to_active_workspace"}


def test_active_operation_rejects_unsupported_resource_type(
    service: WorkspaceEligibilityService,
) -> None:
    result = service.check(
        _request(
            "acp_run",
            resource_type=" acp_session ",
            resource_id="session-1",
            runtime_state="ready",
        )
    )

    assert result.allowed is False
    assert result.reason_code == "unsupported_resource_type"
    assert result.resource_type == "acp_session"
    assert {action.action for action in result.recovery_actions} == {"wait_for_adapter"}


def test_runtime_required_operations_require_ready_runtime(
    service: WorkspaceEligibilityService,
) -> None:
    result = service.check(_request("acp_run", runtime_state="not_required"))

    assert result.allowed is False
    assert result.reason_code == "missing_runtime"
    assert {action.action for action in result.recovery_actions} == {"configure_workspace_runtime"}


def test_permission_denial_applies_before_active_context_membership(
    service: WorkspaceEligibilityService,
) -> None:
    result = service.check(_request("rag_ground", permission_state="denied"))

    assert result.allowed is False
    assert result.reason_code == "permission_denied"
    assert result.resource_workspace_ids == []
