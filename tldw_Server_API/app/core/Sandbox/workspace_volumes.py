from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .models import WorkspaceVolume, WorkspaceVolumeState
from .store import (
    SandboxStore,
    sanitize_workspace_volume_diagnostics,
    sanitize_workspace_volume_mount_path,
)


@dataclass(frozen=True)
class WorkspaceVolumeBinding:
    sandbox_volume_id: str
    state: str
    display_name: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class WorkspaceVolumeMount:
    sandbox_volume_id: str
    state: str
    local_path: str | None = None
    reason_code: str | None = None


class SandboxWorkspaceVolumeService:
    """Sandbox-owned registry for durable Workspace volume records."""

    def __init__(self, *, store: SandboxStore) -> None:
        self.store = store

    def provision_workspace_volume(
        self,
        *,
        workspace_id: str,
        user_id: str,
        display_name: str | None,
        idempotency_key: str | None,
        requested_runtime: str | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> WorkspaceVolume:
        workspace_key = _required_text(workspace_id, "workspace_id")
        user_key = _required_text(user_id, "user_id")
        runtime = _optional_text(requested_runtime)
        key = _optional_text(idempotency_key)
        fingerprint = _provision_fingerprint(
            workspace_id=workspace_key,
            user_id=user_key,
            display_name=_optional_text(display_name),
            requested_runtime=runtime,
        )
        if key:
            existing = self.store.find_workspace_volume_by_idempotency(
                user_id=user_key,
                workspace_id=workspace_key,
                idempotency_key=key,
                idempotency_fingerprint=fingerprint,
            )
            if existing is not None:
                return existing

        volume = WorkspaceVolume(
            id=f"wsv_{uuid.uuid4().hex}",
            workspace_id=workspace_key,
            user_id=user_key,
            state=WorkspaceVolumeState.not_configured,
            root_id=None,
            runtime=runtime,
            display_name=_bounded_text(display_name, 120) or "Workspace volume",
            mount_path=None,
            diagnostics=sanitize_workspace_volume_diagnostics(
                diagnostics
                or {
                    "reason_code": "workspace_sandbox_volume_runtime_not_configured",
                    "message": "No durable workspace volume runtime is configured.",
                }
            ),
            idempotency_key=key,
            idempotency_fingerprint=fingerprint if key else None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.store.put_workspace_volume(volume)
        return volume

    def validate_workspace_volume(
        self,
        *,
        workspace_id: str,
        user_id: str,
        sandbox_volume_id: str,
    ) -> WorkspaceVolumeBinding:
        volume = self.store.get_workspace_volume(str(sandbox_volume_id or "").strip())
        if volume is None:
            return WorkspaceVolumeBinding(
                sandbox_volume_id=str(sandbox_volume_id or "").strip(),
                state=WorkspaceVolumeState.unavailable.value,
                reason_code="workspace_sandbox_volume_not_found",
            )
        if volume.workspace_id != str(workspace_id or "").strip() or volume.user_id != str(user_id or "").strip():
            return WorkspaceVolumeBinding(
                sandbox_volume_id=volume.id,
                state=WorkspaceVolumeState.unavailable.value,
                reason_code="workspace_sandbox_volume_owner_mismatch",
            )
        return WorkspaceVolumeBinding(
            sandbox_volume_id=volume.id,
            state=volume.state.value,
            display_name=volume.display_name,
            reason_code=_reason_code(volume),
        )

    def resolve_workspace_volume_mount(
        self,
        *,
        workspace_id: str,
        root_id: str,
        sandbox_volume_id: str,
    ) -> WorkspaceVolumeMount:
        root_key = str(root_id or "").strip()
        volume = self.store.get_workspace_volume(str(sandbox_volume_id or "").strip())
        if volume is None:
            return WorkspaceVolumeMount(
                sandbox_volume_id=str(sandbox_volume_id or "").strip(),
                state=WorkspaceVolumeState.unavailable.value,
                reason_code="workspace_sandbox_volume_not_found",
            )
        if volume.workspace_id != str(workspace_id or "").strip():
            return WorkspaceVolumeMount(
                sandbox_volume_id=volume.id,
                state=WorkspaceVolumeState.unavailable.value,
                reason_code="workspace_sandbox_volume_owner_mismatch",
            )
        if volume.root_id and volume.root_id != root_key:
            return WorkspaceVolumeMount(
                sandbox_volume_id=volume.id,
                state=WorkspaceVolumeState.unavailable.value,
                reason_code="workspace_sandbox_volume_root_mismatch",
            )
        mount_path = sanitize_workspace_volume_mount_path(volume.mount_path)
        if volume.state is WorkspaceVolumeState.ready and mount_path:
            return WorkspaceVolumeMount(
                sandbox_volume_id=volume.id,
                state=WorkspaceVolumeState.ready.value,
                local_path=mount_path,
            )
        return WorkspaceVolumeMount(
            sandbox_volume_id=volume.id,
            state=volume.state.value,
            local_path=None,
            reason_code=_reason_code(volume) or "workspace_sandbox_volume_runtime_not_configured",
        )

    def bind_workspace_volume_root(
        self,
        *,
        sandbox_volume_id: str,
        workspace_id: str,
        root_id: str,
    ) -> WorkspaceVolume | None:
        """Record the Workspace project root that owns a durable Sandbox volume."""
        workspace_key = _required_text(workspace_id, "workspace_id")
        root_key = _required_text(root_id, "root_id")
        volume = self.store.get_workspace_volume(str(sandbox_volume_id or "").strip())
        if volume is None or volume.workspace_id != workspace_key:
            return None
        return self.store.bind_workspace_volume_root(volume.id, root_id=root_key)


def _provision_fingerprint(
    *,
    workspace_id: str,
    user_id: str,
    display_name: str | None,
    requested_runtime: str | None,
) -> str:
    body = {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "display_name": display_name,
        "requested_runtime": requested_runtime,
    }
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reason_code(volume: WorkspaceVolume) -> str | None:
    raw = volume.diagnostics.get("reason_code") if isinstance(volume.diagnostics, dict) else None
    return _bounded_text(raw, 96)


def _required_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _bounded_text(value: Any, limit: int) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    return text[: max(1, int(limit))]
