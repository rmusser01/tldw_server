from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sandbox.workspace_volumes import SandboxWorkspaceVolumeService
from tldw_Server_API.app.core.Workspaces.operations import (
    fingerprint_workspace_command,
    workspace_operation_response_payload,
)
from tldw_Server_API.app.core.Workspaces.root_binding_service import (
    WorkspaceRootAttachRequest,
    WorkspaceRootServiceError,
    attach_primary_workspace_root,
)

WORKSPACE_SANDBOX_ROOT_COMMAND = "provision_sandbox_root"


@dataclass(frozen=True)
class WorkspaceSandboxRootProvisionRequestData:
    display_name: str | None = None
    requested_runtime: str | None = None
    root_id: str | None = None
    replace_existing: bool = False
    expected_workspace_version: int | None = None


@dataclass(frozen=True)
class WorkspaceSandboxRootProvisionResult:
    workspace_id: str
    workspace_profile: str
    operation: dict[str, Any]
    primary_root: dict[str, Any] | None
    http_status_code: int


def provision_and_attach_sandbox_root(
    *,
    db: CharactersRAGDB,
    sandbox_volume_service: SandboxWorkspaceVolumeService,
    workspace_id: str,
    user_id: str,
    request: WorkspaceSandboxRootProvisionRequestData | Mapping[str, Any] | Any,
    idempotency_key: str,
) -> WorkspaceSandboxRootProvisionResult:
    """Provision a Sandbox-owned Workspace volume and attach it as the primary project root."""
    request_data = _coerce_request_data(request)
    workspace_key = str(workspace_id or "").strip()
    user_key = str(user_id or "").strip()
    key = str(idempotency_key or "").strip()
    if not workspace_key or not user_key or not key:
        raise InputError("workspace_id, user_id, and idempotency_key are required.")  # noqa: TRY003

    fingerprint = fingerprint_workspace_command(
        {
            "workspace_id": workspace_key,
            "user_id": user_key,
            "display_name": request_data.display_name,
            "requested_runtime": request_data.requested_runtime,
            "root_id": request_data.root_id,
            "replace_existing": request_data.replace_existing,
            "expected_workspace_version": request_data.expected_workspace_version,
        }
    )
    operation = db.create_workspace_operation(
        workspace_id=workspace_key,
        user_id=user_key,
        command=WORKSPACE_SANDBOX_ROOT_COMMAND,
        idempotency_key=key,
        request_fingerprint=fingerprint,
        linked_idempotency_key=f"sandbox-volume:{key}",
        status="running",
        diagnostics={"message": "Sandbox project root provisioning is active.", "retryable": True},
    )
    if operation.get("request_fingerprint") != fingerprint:
        raise ConflictError(
            "Workspace operation idempotency key was reused with a different request.",
            entity="workspace_operations",
            entity_id=str(operation.get("id") or ""),
        )
    primary_root = db.get_workspace_primary_root(workspace_key)
    if operation.get("status") == "succeeded":
        return _result(
            db=db,
            workspace_id=workspace_key,
            operation=operation,
            primary_root=primary_root,
            http_status_code=200,
        )
    if operation.get("result_ref", {}).get("sandbox_volume_id") and primary_root is not None:
        return _result(
            db=db,
            workspace_id=workspace_key,
            operation=operation,
            primary_root=primary_root,
            http_status_code=202,
        )

    try:
        volume = sandbox_volume_service.provision_workspace_volume(
            workspace_id=workspace_key,
            user_id=user_key,
            display_name=request_data.display_name,
            idempotency_key=f"workspace-root:{operation['id']}",
            requested_runtime=request_data.requested_runtime,
        )
        primary_root = attach_primary_workspace_root(
            db=db,
            workspace_id=workspace_key,
            user_id=user_key,
            request=WorkspaceRootAttachRequest(
                backend="sandbox_volume",
                root_id=request_data.root_id,
                sandbox_volume_id=volume.id,
                display_name=request_data.display_name or volume.display_name,
                replace_existing=request_data.replace_existing,
                expected_workspace_version=request_data.expected_workspace_version,
                strict_sandbox_validation=False,
            ),
            sandbox_resolver=sandbox_volume_service,
        )
    except WorkspaceRootServiceError as exc:
        db.update_workspace_operation(
            workspace_key,
            str(operation["id"]),
            status="failed",
            diagnostics={"message": str(exc), "code": exc.code, "retryable": True},
        )
        raise
    except (ConflictError, InputError, CharactersRAGDBError):
        raise

    if str(volume.state.value) == "ready":
        operation = db.update_workspace_operation(
            workspace_key,
            str(operation["id"]),
            status="succeeded",
            result_ref={"root_id": primary_root.get("root_id"), "sandbox_volume_id": volume.id},
            diagnostics={"message": "Sandbox project root is ready.", "retryable": False},
        )
        return _result(
            db=db,
            workspace_id=workspace_key,
            operation=operation,
            primary_root=primary_root,
            http_status_code=200,
        )

    operation = db.update_workspace_operation(
        workspace_key,
        str(operation["id"]),
        status="running",
        result_ref={"root_id": primary_root.get("root_id"), "sandbox_volume_id": volume.id},
        diagnostics={
            "message": "Sandbox project root is attached but the durable runtime mount is not ready.",
            "sandbox_volume_state": volume.state.value,
            "reason_code": volume.diagnostics.get("reason_code") if isinstance(volume.diagnostics, dict) else None,
            "retryable": True,
        },
    )
    return _result(
        db=db,
        workspace_id=workspace_key,
        operation=operation,
        primary_root=primary_root,
        http_status_code=202,
    )


def _coerce_request_data(
    request: WorkspaceSandboxRootProvisionRequestData | Mapping[str, Any] | Any,
) -> WorkspaceSandboxRootProvisionRequestData:
    if isinstance(request, WorkspaceSandboxRootProvisionRequestData):
        return request
    if hasattr(request, "model_dump"):
        return _coerce_request_data(request.model_dump())
    if isinstance(request, Mapping):
        return WorkspaceSandboxRootProvisionRequestData(
            display_name=request.get("display_name"),
            requested_runtime=request.get("requested_runtime"),
            root_id=request.get("root_id"),
            replace_existing=bool(request.get("replace_existing", False)),
            expected_workspace_version=request.get("expected_workspace_version"),
        )
    return WorkspaceSandboxRootProvisionRequestData(
        display_name=getattr(request, "display_name", None),
        requested_runtime=getattr(request, "requested_runtime", None),
        root_id=getattr(request, "root_id", None),
        replace_existing=bool(getattr(request, "replace_existing", False)),
        expected_workspace_version=getattr(request, "expected_workspace_version", None),
    )


def _result(
    *,
    db: CharactersRAGDB,
    workspace_id: str,
    operation: Mapping[str, Any],
    primary_root: dict[str, Any] | None,
    http_status_code: int,
) -> WorkspaceSandboxRootProvisionResult:
    workspace = db.get_workspace(workspace_id) or {}
    return WorkspaceSandboxRootProvisionResult(
        workspace_id=workspace_id,
        workspace_profile=str(workspace.get("workspace_profile") or "research"),
        operation=workspace_operation_response_payload(operation),
        primary_root=primary_root,
        http_status_code=http_status_code,
    )
