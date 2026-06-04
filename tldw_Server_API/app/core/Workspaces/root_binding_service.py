from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path

_DEFAULT_ROOT_ID = "primary"
_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SANDBOX_VOLUME_ID_RE = _ROOT_ID_RE
_DISPLAY_NAME_MAX_LENGTH = 120
_SANDBOX_VOLUME_STATES = frozenset({"ready", "not_configured", "unavailable", "failed"})


class WorkspaceRootServiceError(Exception):
    """Base service error for Workspace primary-root binding failures."""

    status_code = 400
    code = "workspace_root_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class WorkspaceRootInputError(WorkspaceRootServiceError):
    """Raised when the root attach request is malformed."""

    status_code = 400
    code = "workspace_root_invalid_request"


class WorkspaceRootValidationError(WorkspaceRootServiceError):
    """Raised when a syntactically valid root is not allowed for this host."""

    status_code = 403
    code = "workspace_project_root_not_allowed"


class WorkspaceRootConflictError(WorkspaceRootServiceError):
    """Raised when a different primary root already exists or the DB write conflicts."""

    status_code = 409
    code = "workspace_primary_root_exists"


class WorkspaceRootNotFoundError(WorkspaceRootServiceError):
    """Raised when the target workspace cannot be found."""

    status_code = 404
    code = "workspace_not_found"


class WorkspaceRootConfigurationError(WorkspaceRootServiceError):
    """Raised when root validation cannot proceed because required config is missing."""

    status_code = 503
    code = "workspace_project_roots_not_configured"


@dataclass(frozen=True)
class WorkspaceRootAttachRequest:
    backend: Literal["host_local", "sandbox_volume"]
    root_id: str | None = None
    absolute_root: str | None = None
    sandbox_volume_id: str | None = None
    display_name: str | None = None
    replace_existing: bool = False
    expected_workspace_version: int | None = None
    strict_sandbox_validation: bool = False


@dataclass(frozen=True)
class SandboxVolumeBinding:
    sandbox_volume_id: str
    state: Literal["ready", "not_configured", "unavailable", "failed"]
    display_name: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class SandboxInventoryMount:
    sandbox_volume_id: str
    state: Literal["ready", "not_configured", "unavailable", "failed"]
    local_path: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class ResolvedWorkspaceInventoryRoot:
    ok: bool
    backend: str
    local_path: Path | None = None
    root_snapshot_token: str | None = None
    failure_code: str | None = None
    message: str | None = None


class SandboxVolumeResolver(Protocol):
    def validate_workspace_volume(
        self,
        *,
        workspace_id: str,
        user_id: str,
        sandbox_volume_id: str,
    ) -> SandboxVolumeBinding:
        """Return the current binding state for a Workspace-owned sandbox volume."""


class SandboxInventoryMountResolver(Protocol):
    def resolve_workspace_volume_mount(
        self,
        *,
        workspace_id: str,
        root_id: str,
        sandbox_volume_id: str,
    ) -> SandboxInventoryMount:
        """Return the current local mount state for a Workspace-owned sandbox volume."""


class DefaultSandboxVolumeResolver:
    """Conservative resolver used until a persistent Sandbox volume registry exists."""

    def validate_workspace_volume(
        self,
        *,
        workspace_id: str,
        user_id: str,
        sandbox_volume_id: str,
    ) -> SandboxVolumeBinding:
        return SandboxVolumeBinding(
            sandbox_volume_id=sandbox_volume_id,
            state="not_configured",
            reason_code="workspace_sandbox_volume_resolver_not_configured",
        )


@dataclass(frozen=True)
class _NormalizedRootBinding:
    backend: Literal["host_local", "sandbox_volume"]
    target: str
    display_name: str
    absolute_root: str | None = None
    sandbox_volume_id: str | None = None
    sandbox_mount_state: str | None = None


def attach_primary_workspace_root(
    *,
    db: Any,
    workspace_id: str,
    user_id: str,
    request: WorkspaceRootAttachRequest,
    allowed_roots: Sequence[Path | str] | None = None,
    sandbox_resolver: SandboxVolumeResolver | None = None,
) -> dict[str, Any]:
    """Validate and persist a Workspace primary project-root binding."""

    workspace_version = _load_workspace_version_for_attach(db, workspace_id)
    current_primary = _load_current_primary_root(db, workspace_id)
    normalized = _normalize_request(
        workspace_id=workspace_id,
        user_id=user_id,
        request=request,
        allowed_roots=allowed_roots,
        sandbox_resolver=sandbox_resolver,
    )
    same_binding = _matches_current_binding(current_primary, normalized)
    root_id = _resolve_root_id(request.root_id, current_primary, same_binding)

    if current_primary is not None and not same_binding and not request.replace_existing:
        raise WorkspaceRootConflictError(
            "Workspace already has a different primary project root.",
            code="workspace_primary_root_exists",
        )

    payload: dict[str, Any] = {
        "root_id": root_id,
        "backend": normalized.backend,
        "absolute_root": normalized.absolute_root,
        "sandbox_volume_id": normalized.sandbox_volume_id,
        "display_name": normalized.display_name,
        "root_state": "attached",
        "expected_workspace_version": (
            request.expected_workspace_version
            if request.expected_workspace_version is not None
            else workspace_version
        ),
        "replace_existing": request.replace_existing,
    }
    if normalized.sandbox_mount_state is not None:
        payload["sandbox_mount_state"] = normalized.sandbox_mount_state

    try:
        return db.upsert_workspace_primary_root(workspace_id, payload)
    except ConflictError as exc:
        raise _wrap_db_conflict(exc) from exc
    except InputError as exc:
        raise WorkspaceRootInputError(str(exc), code="workspace_root_invalid_request") from exc
    except CharactersRAGDBError:
        raise


def _load_workspace_version_for_attach(db: Any, workspace_id: str) -> int:
    try:
        workspace = db.get_workspace(workspace_id)
    except ConflictError as exc:
        raise _wrap_db_conflict(exc) from exc
    except InputError as exc:
        raise WorkspaceRootInputError(str(exc), code="workspace_root_invalid_request") from exc
    except CharactersRAGDBError:
        raise

    if workspace is None:
        raise WorkspaceRootNotFoundError(
            "Workspace was not found.",
            code="workspace_not_found",
        )
    try:
        return int(workspace["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkspaceRootInputError(
            "Workspace version is invalid.",
            code="workspace_root_invalid_request",
        ) from exc


def _load_current_primary_root(db: Any, workspace_id: str) -> dict[str, Any] | None:
    try:
        return db.get_workspace_primary_root(workspace_id)
    except ConflictError as exc:
        raise _wrap_db_conflict(exc) from exc
    except InputError as exc:
        raise WorkspaceRootInputError(str(exc), code="workspace_root_invalid_request") from exc
    except CharactersRAGDBError:
        raise


def _wrap_db_conflict(exc: ConflictError) -> WorkspaceRootConflictError:
    message = str(exc)
    code = (
        "workspace_version_mismatch"
        if "version" in message.lower()
        else "workspace_primary_root_write_conflict"
    )
    return WorkspaceRootConflictError(message, code=code)


def _normalize_request(
    *,
    workspace_id: str,
    user_id: str,
    request: WorkspaceRootAttachRequest,
    allowed_roots: Sequence[Path | str] | None,
    sandbox_resolver: SandboxVolumeResolver | None,
) -> _NormalizedRootBinding:
    if request.backend == "host_local":
        return _normalize_host_local_request(request, allowed_roots)
    if request.backend == "sandbox_volume":
        return _normalize_sandbox_volume_request(
            workspace_id=workspace_id,
            user_id=user_id,
            request=request,
            sandbox_resolver=sandbox_resolver,
        )
    raise WorkspaceRootInputError(
        "Unsupported workspace root backend.",
        code="workspace_root_invalid_request",
    )


def _normalize_host_local_request(
    request: WorkspaceRootAttachRequest,
    allowed_roots: Sequence[Path | str] | None,
) -> _NormalizedRootBinding:
    if not request.absolute_root or not request.absolute_root.strip():
        raise WorkspaceRootInputError(
            "absolute_root is required for host_local roots.",
            code="workspace_project_root_path_required",
        )
    if request.sandbox_volume_id is not None:
        raise WorkspaceRootInputError(
            "sandbox_volume_id is not valid for host_local roots.",
            code="workspace_root_invalid_request",
        )

    candidate = Path(request.absolute_root.strip()).expanduser()
    if not candidate.is_absolute():
        raise WorkspaceRootInputError(
            "absolute_root must be absolute after user expansion.",
            code="workspace_project_root_not_absolute",
        )

    configured_allowed_roots = (
        tuple(Path(root) for root in allowed_roots)
        if allowed_roots is not None
        else config.get_workspace_project_root_allowed_roots()
    )
    if not configured_allowed_roots:
        raise WorkspaceRootConfigurationError(
            "Workspace project root allowed roots are not configured.",
            code="workspace_project_roots_not_configured",
        )
    raw_containing_roots = _raw_containing_allowed_roots(candidate, configured_allowed_roots)
    if not raw_containing_roots:
        raise WorkspaceRootValidationError(
            "Workspace project root is outside the configured allowed roots.",
            code="workspace_project_root_outside_allowed_roots",
        )
    if candidate.is_symlink():
        raise WorkspaceRootInputError(
            "Workspace project root cannot be a symlink.",
            code="workspace_project_root_symlink",
        )

    resolved = candidate.resolve(strict=False)
    if not any(
        resolve_safe_local_path(resolved, allowed_root) is not None
        for allowed_root in raw_containing_roots
    ):
        raise WorkspaceRootValidationError(
            "Workspace project root is outside the configured allowed roots.",
            code="workspace_project_root_outside_allowed_roots",
        )
    if not resolved.exists():
        raise WorkspaceRootInputError(
            "Workspace project root does not exist.",
            code="workspace_project_root_missing",
        )
    if not resolved.is_dir():
        raise WorkspaceRootInputError(
            "Workspace project root is not a directory.",
            code="workspace_project_root_not_directory",
        )

    display_name = _normalize_display_name(
        request.display_name,
        fallback=resolved.name or str(resolved),
    )
    return _NormalizedRootBinding(
        backend="host_local",
        target=str(resolved),
        absolute_root=str(resolved),
        display_name=display_name,
    )


def _raw_containing_allowed_roots(candidate: Path, allowed_roots: Sequence[Path]) -> tuple[Path, ...]:
    raw_candidate = Path(os.path.abspath(str(candidate)))
    containing_roots: list[Path] = []
    for allowed_root in allowed_roots:
        base = Path(allowed_root).expanduser().resolve(strict=False)
        try:
            common_path = os.path.commonpath([str(base), str(raw_candidate)])
        except ValueError:
            continue
        if common_path == str(base):
            containing_roots.append(base)
    return tuple(containing_roots)


def resolve_workspace_root_for_inventory_scan(
    *,
    root: Mapping[str, Any],
    allowed_roots: Sequence[Path | str] | None = None,
    sandbox_mount_resolver: SandboxInventoryMountResolver | None = None,
) -> ResolvedWorkspaceInventoryRoot:
    backend = str(root.get("backend") or "").strip()
    if backend == "host_local":
        return _resolve_host_local_root_for_inventory_scan(root, allowed_roots)
    if backend == "sandbox_volume":
        return _resolve_sandbox_root_for_inventory_scan(root, sandbox_mount_resolver)
    return _inventory_root_failure(
        backend=backend or "unknown",
        code="workspace_project_root_backend_unsupported",
        message="Workspace project root backend is not supported for inventory scans.",
    )


def _resolve_host_local_root_for_inventory_scan(
    root: Mapping[str, Any],
    allowed_roots: Sequence[Path | str] | None,
) -> ResolvedWorkspaceInventoryRoot:
    absolute_root = str(root.get("absolute_root") or "").strip()
    if not absolute_root:
        return _inventory_root_failure(
            backend="host_local",
            code="workspace_project_root_path_required",
            message="Workspace project root path is required.",
        )
    candidate = Path(absolute_root).expanduser()
    if not candidate.is_absolute():
        return _inventory_root_failure(
            backend="host_local",
            code="workspace_project_root_not_absolute",
            message="Workspace project root path must be absolute.",
        )
    configured_allowed_roots = (
        tuple(Path(root_path) for root_path in allowed_roots)
        if allowed_roots is not None
        else config.get_workspace_project_root_allowed_roots()
    )
    if not configured_allowed_roots:
        return _inventory_root_failure(
            backend="host_local",
            code="workspace_project_roots_not_configured",
            message="Workspace project root allowed roots are not configured.",
        )
    raw_containing_roots = _raw_containing_allowed_roots(candidate, configured_allowed_roots)
    if not raw_containing_roots:
        return _inventory_root_failure(
            backend="host_local",
            code="workspace_project_root_outside_allowed_roots",
            message="Workspace project root is outside the configured allowed roots.",
        )
    if candidate.is_symlink():
        return _inventory_root_failure(
            backend="host_local",
            code="workspace_project_root_symlink",
            message="Workspace project root cannot be a symlink.",
        )

    resolved = candidate.resolve(strict=False)
    if not any(resolve_safe_local_path(resolved, allowed_root) is not None for allowed_root in raw_containing_roots):
        return _inventory_root_failure(
            backend="host_local",
            code="workspace_project_root_outside_allowed_roots",
            message="Workspace project root is outside the configured allowed roots.",
        )
    return _resolve_existing_inventory_directory(
        backend="host_local",
        root=root,
        path=resolved,
    )


def _resolve_sandbox_root_for_inventory_scan(
    root: Mapping[str, Any],
    sandbox_mount_resolver: SandboxInventoryMountResolver | None,
) -> ResolvedWorkspaceInventoryRoot:
    sandbox_volume_id = str(root.get("sandbox_volume_id") or "").strip()
    if not sandbox_volume_id or sandbox_mount_resolver is None:
        return _inventory_root_failure(
            backend="sandbox_volume",
            code="sandbox_mount_not_ready",
            message="Workspace sandbox volume is not mounted.",
        )
    try:
        mount = sandbox_mount_resolver.resolve_workspace_volume_mount(
            workspace_id=str(root.get("workspace_id") or "").strip(),
            root_id=str(root.get("root_id") or "").strip(),
            sandbox_volume_id=sandbox_volume_id,
        )
    except Exception:
        return _inventory_root_failure(
            backend="sandbox_volume",
            code="sandbox_mount_resolution_failed",
            message="Workspace sandbox volume mount could not be resolved.",
        )
    if mount.state != "ready" or not mount.local_path:
        return _inventory_root_failure(
            backend="sandbox_volume",
            code=mount.reason_code or "sandbox_mount_not_ready",
            message="Workspace sandbox volume is not ready.",
        )
    candidate = Path(mount.local_path).expanduser()
    if candidate.is_symlink():
        return _inventory_root_failure(
            backend="sandbox_volume",
            code="workspace_project_root_symlink",
            message="Workspace sandbox mount root cannot be a symlink.",
        )
    return _resolve_existing_inventory_directory(
        backend="sandbox_volume",
        root=root,
        path=candidate.resolve(strict=False),
    )


def _resolve_existing_inventory_directory(
    *,
    backend: str,
    root: Mapping[str, Any],
    path: Path,
) -> ResolvedWorkspaceInventoryRoot:
    if not path.exists():
        return _inventory_root_failure(
            backend=backend,
            code="workspace_project_root_missing",
            message="Workspace project root does not exist.",
        )
    if not path.is_dir():
        return _inventory_root_failure(
            backend=backend,
            code="workspace_project_root_not_directory",
            message="Workspace project root is not a directory.",
        )
    return ResolvedWorkspaceInventoryRoot(
        ok=True,
        backend=backend,
        local_path=path,
        root_snapshot_token=_inventory_root_snapshot_token(root, path),
    )


def _inventory_root_snapshot_token(root: Mapping[str, Any], path: Path) -> str:
    root_id = str(root.get("root_id") or "").strip() or "root"
    version = str(root.get("version") or "").strip() or "0"
    try:
        stat_result = path.stat()
        return f"{root_id}:{version}:{stat_result.st_mtime_ns}:{stat_result.st_ino}"
    except OSError:
        return f"{root_id}:{version}"


def _inventory_root_failure(*, backend: str, code: str, message: str) -> ResolvedWorkspaceInventoryRoot:
    return ResolvedWorkspaceInventoryRoot(
        ok=False,
        backend=backend,
        failure_code=code,
        message=message,
    )


def _normalize_sandbox_volume_request(
    *,
    workspace_id: str,
    user_id: str,
    request: WorkspaceRootAttachRequest,
    sandbox_resolver: SandboxVolumeResolver | None,
) -> _NormalizedRootBinding:
    if request.absolute_root is not None:
        raise WorkspaceRootInputError(
            "absolute_root is not valid for sandbox_volume roots.",
            code="workspace_root_invalid_request",
        )
    sandbox_volume_id = (request.sandbox_volume_id or "").strip()
    if not sandbox_volume_id:
        raise WorkspaceRootInputError(
            "sandbox_volume_id is required for sandbox_volume roots.",
            code="workspace_sandbox_volume_id_required",
        )
    if not _SANDBOX_VOLUME_ID_RE.fullmatch(sandbox_volume_id):
        raise WorkspaceRootInputError(
            "sandbox_volume_id contains invalid characters.",
            code="workspace_root_invalid_request",
        )

    resolver = sandbox_resolver or DefaultSandboxVolumeResolver()
    try:
        binding = resolver.validate_workspace_volume(
            workspace_id=workspace_id,
            user_id=user_id,
            sandbox_volume_id=sandbox_volume_id,
        )
    except WorkspaceRootServiceError:
        raise
    except Exception as exc:
        raise WorkspaceRootConfigurationError(
            "Workspace sandbox volume resolver failed.",
            code="workspace_sandbox_volume_resolver_failed",
        ) from exc
    if str(binding.sandbox_volume_id or "").strip() != sandbox_volume_id:
        raise WorkspaceRootConfigurationError(
            "Workspace sandbox volume resolver returned a mismatched volume id.",
            code="workspace_sandbox_volume_id_mismatch",
        )
    if binding.state not in _SANDBOX_VOLUME_STATES:
        raise WorkspaceRootConfigurationError(
            "Workspace sandbox volume resolver returned an invalid state.",
            code="workspace_sandbox_volume_state_invalid",
        )
    if request.strict_sandbox_validation and binding.state != "ready":
        raise WorkspaceRootConfigurationError(
            "Workspace sandbox volume resolver is unavailable.",
            code="workspace_sandbox_volume_resolver_unavailable",
        )

    display_name = _normalize_display_name(
        request.display_name,
        fallback=binding.display_name or sandbox_volume_id,
    )
    return _NormalizedRootBinding(
        backend="sandbox_volume",
        target=sandbox_volume_id,
        sandbox_volume_id=sandbox_volume_id,
        sandbox_mount_state=binding.state,
        display_name=display_name,
    )


def _normalize_display_name(display_name: str | None, *, fallback: str) -> str:
    normalized = display_name.strip() if display_name is not None else ""
    if len(normalized) > _DISPLAY_NAME_MAX_LENGTH:
        raise WorkspaceRootInputError(
            "display_name must be 120 characters or fewer.",
            code="workspace_root_invalid_request",
        )
    return normalized or fallback


def _resolve_root_id(
    requested_root_id: str | None,
    current_primary: dict[str, Any] | None,
    same_binding: bool,
) -> str:
    if requested_root_id is not None:
        root_id = requested_root_id.strip()
        if not _ROOT_ID_RE.fullmatch(root_id):
            raise WorkspaceRootInputError(
                "root_id contains invalid characters.",
                code="workspace_root_invalid_request",
            )
        return root_id
    if same_binding and current_primary is not None:
        existing_root_id = str(current_primary.get("root_id") or "").strip()
        if existing_root_id:
            return existing_root_id
    return _DEFAULT_ROOT_ID


def _matches_current_binding(
    current_primary: dict[str, Any] | None,
    normalized: _NormalizedRootBinding,
) -> bool:
    if current_primary is None or current_primary.get("backend") != normalized.backend:
        return False
    if normalized.backend == "host_local":
        return str(current_primary.get("absolute_root") or "") == normalized.target
    return str(current_primary.get("sandbox_volume_id") or "") == normalized.target
