"""Agent Orchestration API endpoints.

Provides project/task management, run dispatch, reviewer gate,
and workspace CRUD with discovery and health monitoring.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_token_scope
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Agent_Orchestration.models import TaskStatus
from tldw_Server_API.app.core.Agent_Orchestration.orchestration_service import (
    CycleDependencyError,
    get_orchestration_db,
)
from tldw_Server_API.app.core.DB_Management.Orchestration_DB import (
    InvalidTransitionError,
    OrchestrationNotFoundError,
)

router = APIRouter(prefix="/agent-orchestration", tags=["agent-orchestration"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_id_int(user: User) -> int:
    """Safely extract integer user ID, raising 400 for non-numeric IDs."""
    uid = getattr(user, "id_int", None)
    if uid is not None:
        return uid
    try:
        return int(user.id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Non-numeric user ID not supported for orchestration",
        ) from exc


async def _run_sync(fn: Any) -> Any:
    """Run a synchronous callable in a threadpool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


def _allowed_workspace_roots() -> tuple[Path, ...]:
    """Return the configured ACP workspace allowlist."""
    from tldw_Server_API.app.core.config import get_config_value

    raw_values: list[str] = []
    raw_values.extend(
        entry.strip()
        for entry in str(get_config_value("ACP-WORKSPACE", "allowed_base_paths", "") or "").replace(
            os.pathsep,
            ",",
        ).split(",")
        if entry.strip()
    )
    raw_values.extend(
        entry.strip()
        for entry in str(os.getenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", "") or "").replace(
            os.pathsep,
            ",",
        ).split(",")
        if entry.strip()
    )

    roots: list[Path] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            logger.warning("Ignoring non-absolute ACP workspace allowlist entry: {}", raw_value)
            continue
        resolved = candidate.resolve()
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        roots.append(resolved)
    return tuple(roots)


def _validate_workspace_root(root_path: str) -> str:
    """Validate and normalize root_path within configured ACP workspace roots."""
    candidate = Path(root_path).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(400, "root_path must be absolute")
    path = candidate.resolve()

    bases = _allowed_workspace_roots()
    if not bases:
        raise HTTPException(
            503,
            "ACP workspace roots are not configured. Set ACP-WORKSPACE.allowed_base_paths or ACP_WORKSPACE_ALLOWED_BASE_PATHS.",
        )
    if not any(path == b or path.is_relative_to(b) for b in bases):
        raise HTTPException(
            403,
            f"root_path not under allowed base paths: {', '.join(str(b) for b in bases)}",
        )
    return str(path)


def _resolve_dispatch_cwd(raw_cwd: str, *, workspace_root: str | None = None) -> str:
    """Resolve a run cwd, confining workspace-relative paths to the workspace root."""
    candidate = (raw_cwd or ".").strip() or "."
    if not workspace_root:
        if candidate == ".":
            return "."
        return _validate_workspace_root(candidate)

    workspace_root_text = _validate_workspace_root(workspace_root)
    if candidate == ".":
        return workspace_root_text

    expanded_candidate = os.path.expanduser(candidate)
    if os.path.isabs(expanded_candidate):
        raise HTTPException(403, "cwd must be relative to the workspace root")

    resolved_path = os.path.realpath(os.path.join(workspace_root_text, expanded_candidate))
    if os.path.commonpath([workspace_root_text, resolved_path]) != workspace_root_text:
        raise HTTPException(403, "cwd must stay within the workspace root")
    return resolved_path


# ---------------------------------------------------------------------------
# Schemas — Workspaces
# ---------------------------------------------------------------------------


_VALID_WORKSPACE_TYPES = {"manual", "discovered", "monorepo_child"}


class ACPWorkspaceCreateRequest(BaseModel):
    name: str = Field(..., description="Workspace name")
    root_path: str = Field(..., description="Absolute filesystem path")
    description: str = Field(default="", description="Workspace description")
    workspace_type: str = Field(default="manual", description="manual | discovered | monorepo_child")
    parent_workspace_id: int | None = Field(default=None, description="Parent workspace ID for monorepo children")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables for sessions (stored as plaintext)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("workspace_type")
    @classmethod
    def check_workspace_type(cls, v: str) -> str:
        if v not in _VALID_WORKSPACE_TYPES:
            raise ValueError(f"workspace_type must be one of {_VALID_WORKSPACE_TYPES}")
        return v


class ACPWorkspaceUpdateRequest(BaseModel):
    name: str | None = None
    root_path: str | None = None
    description: str | None = None
    env_vars: dict[str, str] | None = None
    metadata: dict[str, Any] | None = None


class ACPWorkspaceResponse(BaseModel):
    id: int
    name: str
    root_path: str
    description: str = ""
    workspace_type: str = "manual"
    parent_workspace_id: int | None = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    git_remote_url: str | None = None
    git_default_branch: str | None = None
    git_current_branch: str | None = None
    git_is_dirty: bool | None = None
    last_health_check: str | None = None
    health_status: str = "unknown"
    user_id: int = 0
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[ACPWorkspaceResponse] | None = None
    mcp_servers: list[dict[str, Any]] | None = None


class ACPWorkspaceMCPServerCreateRequest(BaseModel):
    server_name: str = Field(..., description="Unique server name within workspace")
    server_type: str = Field(default="stdio", description="stdio | sse")
    command: str | None = Field(default=None, description="Command to run (stdio type)")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    url: str | None = Field(default=None, description="Server URL (sse type)")
    enabled: bool = Field(default=True)


class WorkspaceDiscoverRequest(BaseModel):
    base_path: str = Field(..., description="Absolute path to scan")
    max_depth: int = Field(default=3, ge=1, le=10, description="Max directory depth")
    patterns: list[str] | None = Field(default=None, description="Marker files to look for")


# ---------------------------------------------------------------------------
# Schemas — Projects
# ---------------------------------------------------------------------------


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., description="Project name")
    description: str = Field(default="", description="Project description")
    workspace_id: int | None = Field(default=None, description="Bind project to a workspace")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: str = ""
    workspace_id: int | None = None
    user_id: int = 0
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    task_summary: dict[str, Any] | None = None
    workspace: ACPWorkspaceResponse | None = None


# ---------------------------------------------------------------------------
# Schemas — Tasks
# ---------------------------------------------------------------------------


class TaskCreateRequest(BaseModel):
    title: str = Field(..., description="Task title")
    description: str = Field(default="", description="Task description")
    agent_type: str | None = Field(default=None, description="Agent type to use for this task")
    dependency_id: int | None = Field(default=None, description="Task ID this depends on")
    reviewer_agent_type: str | None = Field(default=None, description="Agent type for review gate")
    max_review_attempts: int = Field(default=3, ge=1, le=10, description="Max review attempts before triage")
    success_criteria: str = Field(default="", description="Success criteria for the task")
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    id: int
    project_id: int
    title: str
    description: str = ""
    status: str = "todo"
    agent_type: str | None = None
    dependency_id: int | None = None
    reviewer_agent_type: str | None = None
    max_review_attempts: int = 3
    review_count: int = 0
    success_criteria: str = ""
    user_id: int = 0
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    runs: list[dict[str, Any]] | None = None


class RunDispatchRequest(BaseModel):
    agent_type: str | None = Field(default=None, description="Override agent type for this run")
    cwd: str = Field(default=".", description="Working directory for the ACP session")


class ReviewRequest(BaseModel):
    approved: bool = Field(..., description="Whether the review is approved")
    feedback: str = Field(default="", description="Review feedback")


# ---------------------------------------------------------------------------
# Workspace endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/workspaces",
    response_model=ACPWorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def create_workspace(
    payload: ACPWorkspaceCreateRequest,
    user: User = Depends(get_request_user),
) -> ACPWorkspaceResponse:
    """Create a new ACP workspace."""
    validated_path = _validate_workspace_root(payload.root_path)
    db = get_orchestration_db(_user_id_int(user))
    try:
        ws = await _run_sync(lambda: db.create_workspace(
            name=payload.name,
            root_path=validated_path,
            description=payload.description,
            workspace_type=payload.workspace_type,
            parent_workspace_id=payload.parent_workspace_id,
            env_vars=payload.env_vars,
            metadata=payload.metadata,
        ))
    except OrchestrationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ACPWorkspaceResponse(**ws.to_dict())


@router.get(
    "/workspaces",
    response_model=list[ACPWorkspaceResponse],
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.read"))],
)
async def list_workspaces(
    workspace_type: str | None = Query(default=None, description="Filter by type"),
    health_status: str | None = Query(default=None, description="Filter by health"),
    user: User = Depends(get_request_user),
) -> list[ACPWorkspaceResponse]:
    """List all workspaces for the current user."""
    db = get_orchestration_db(_user_id_int(user))
    workspaces = await _run_sync(lambda: db.list_workspaces(
        workspace_type=workspace_type,
        health_status=health_status,
    ))
    return [ACPWorkspaceResponse(**ws.to_dict()) for ws in workspaces]


@router.get(
    "/workspaces/{workspace_id}",
    response_model=ACPWorkspaceResponse,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.read"))],
)
async def get_workspace(
    workspace_id: int,
    user: User = Depends(get_request_user),
) -> ACPWorkspaceResponse:
    """Get a workspace with children and MCP servers."""
    db = get_orchestration_db(_user_id_int(user))
    ws = await _run_sync(lambda: db.get_workspace(workspace_id))
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    children = await _run_sync(lambda: db.list_workspace_children(workspace_id))
    mcp_servers = await _run_sync(lambda: db.list_workspace_mcp_servers(workspace_id))

    d = ws.to_dict()
    d["children"] = [ACPWorkspaceResponse(**c.to_dict()).model_dump() for c in children]
    d["mcp_servers"] = mcp_servers
    return ACPWorkspaceResponse(**d)


@router.put(
    "/workspaces/{workspace_id}",
    response_model=ACPWorkspaceResponse,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def update_workspace(
    workspace_id: int,
    payload: ACPWorkspaceUpdateRequest,
    user: User = Depends(get_request_user),
) -> ACPWorkspaceResponse:
    """Update a workspace."""
    db = get_orchestration_db(_user_id_int(user))
    update_fields = payload.model_dump(exclude_unset=True)

    # Validate new root_path if provided
    if "root_path" in update_fields and update_fields["root_path"] is not None:
        update_fields["root_path"] = _validate_workspace_root(update_fields["root_path"])

    try:
        ws = await _run_sync(lambda: db.update_workspace(workspace_id, **update_fields))
    except OrchestrationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ACPWorkspaceResponse(**ws.to_dict())


@router.delete(
    "/workspaces/{workspace_id}",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def delete_workspace(
    workspace_id: int,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Delete a workspace. Projects are unbound (SET NULL), not deleted."""
    db = get_orchestration_db(_user_id_int(user))
    deleted = await _run_sync(lambda: db.delete_workspace(workspace_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"deleted": True, "workspace_id": workspace_id}


# ---------------------------------------------------------------------------
# Workspace health
# ---------------------------------------------------------------------------


@router.get(
    "/workspaces/{workspace_id}/health",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.read"))],
)
async def check_workspace_health(
    workspace_id: int,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """On-demand health check for a single workspace."""
    db = get_orchestration_db(_user_id_int(user))
    ws = await _run_sync(lambda: db.get_workspace(workspace_id))
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    from tldw_Server_API.app.services.workspace_health_service import WorkspaceHealthService
    svc = WorkspaceHealthService()
    result = await svc.check_health(ws)

    # Persist health update
    await _run_sync(lambda: db.update_workspace_health(
        workspace_id=ws.id,
        health_status=result.health_status,
        git_remote_url=result.git_remote_url,
        git_default_branch=result.git_default_branch,
        git_current_branch=result.git_current_branch,
        git_is_dirty=result.git_is_dirty,
        last_health_check=result.checked_at,
    ))

    return result.to_dict()


@router.post(
    "/workspaces/health/refresh-all",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def refresh_all_workspace_health(
    user: User = Depends(get_request_user),
) -> list[dict[str, Any]]:
    """Refresh health status for all workspaces of the current user."""
    db = get_orchestration_db(_user_id_int(user))

    from tldw_Server_API.app.services.workspace_health_service import WorkspaceHealthService
    svc = WorkspaceHealthService()
    results = await svc.refresh_all(db)
    return [r.to_dict() for r in results]


# ---------------------------------------------------------------------------
# Workspace MCP servers
# ---------------------------------------------------------------------------


@router.get(
    "/workspaces/{workspace_id}/mcp-servers",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.read"))],
)
async def list_workspace_mcp_servers(
    workspace_id: int,
    user: User = Depends(get_request_user),
) -> list[dict[str, Any]]:
    """List MCP servers configured for a workspace."""
    db = get_orchestration_db(_user_id_int(user))
    ws = await _run_sync(lambda: db.get_workspace(workspace_id))
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    servers = await _run_sync(lambda: db.list_workspace_mcp_servers(workspace_id))
    return servers


@router.post(
    "/workspaces/{workspace_id}/mcp-servers",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def create_workspace_mcp_server(
    workspace_id: int,
    payload: ACPWorkspaceMCPServerCreateRequest,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Add an MCP server configuration to a workspace."""
    db = get_orchestration_db(_user_id_int(user))
    try:
        server = await _run_sync(lambda: db.create_workspace_mcp_server(
            workspace_id=workspace_id,
            server_name=payload.server_name,
            server_type=payload.server_type,
            command=payload.command,
            args=payload.args,
            env=payload.env,
            url=payload.url,
            enabled=payload.enabled,
        ))
    except OrchestrationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return server


@router.delete(
    "/workspaces/{workspace_id}/mcp-servers/{server_id}",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def delete_workspace_mcp_server(
    workspace_id: int,
    server_id: int,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Remove an MCP server from a workspace."""
    db = get_orchestration_db(_user_id_int(user))
    # Single atomic delete that verifies workspace ownership
    deleted = await _run_sync(
        lambda: db.delete_workspace_mcp_server(workspace_id, server_id)
    )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="MCP server not found in this workspace, or workspace not found",
        )
    return {"deleted": True, "server_id": server_id}


# ---------------------------------------------------------------------------
# Workspace discovery
# ---------------------------------------------------------------------------


@router.post(
    "/workspaces/discover",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.workspaces.manage"))],
)
async def discover_workspaces(
    payload: WorkspaceDiscoverRequest,
    user: User = Depends(get_request_user),
) -> list[dict[str, Any]]:
    """Scan a directory tree to discover candidate workspaces."""
    validated_path = _validate_workspace_root(payload.base_path)

    db = get_orchestration_db(_user_id_int(user))

    # Gather existing registered paths for already_registered tagging
    existing = await _run_sync(lambda: db.list_workspaces())
    registered_paths = {ws.root_path for ws in existing}

    # Read config defaults for discovery
    from tldw_Server_API.app.core.config import get_config_value as _gcv
    max_depth = payload.max_depth
    patterns = payload.patterns
    if patterns is None:
        config_patterns = _gcv("ACP-WORKSPACE", "discovery_patterns", "")
        if config_patterns:
            patterns = [p.strip() for p in config_patterns.split(",") if p.strip()]

    from tldw_Server_API.app.services.workspace_discovery_service import WorkspaceDiscoveryService
    svc = WorkspaceDiscoveryService()
    candidates = await svc.discover(
        base_path=validated_path,
        max_depth=max_depth,
        patterns=patterns,
        registered_paths=registered_paths,
    )
    return [c.to_dict() for c in candidates]


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.projects.manage"))],
)
async def create_project(
    payload: ProjectCreateRequest,
    user: User = Depends(get_request_user),
) -> ProjectResponse:
    """Create a new agent project."""
    db = get_orchestration_db(_user_id_int(user))
    try:
        project = await _run_sync(lambda: db.create_project(
            name=payload.name,
            description=payload.description,
            workspace_id=payload.workspace_id,
            metadata=payload.metadata,
        ))
    except OrchestrationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProjectResponse(**project.to_dict())


@router.get(
    "/projects",
    response_model=list[ProjectResponse],
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.projects.read"))],
)
async def list_projects(
    workspace_id: int | None = Query(default=None, description="Filter by workspace ID (omit for all)"),
    unbound: bool = Query(default=False, description="If true, list only projects without a workspace"),
    user: User = Depends(get_request_user),
) -> list[ProjectResponse]:
    """List projects for the current user, optionally filtered by workspace."""
    db = get_orchestration_db(_user_id_int(user))

    def _list() -> list[dict[str, Any]]:
        if workspace_id is not None:
            projects = db.list_projects(workspace_id=workspace_id)
        elif unbound:
            projects = db.list_projects(workspace_id=None)
        else:
            projects = db.list_projects()
        results = []
        for p in projects:
            summary = db.get_project_summary(p.id)
            d = p.to_dict()
            d["task_summary"] = summary
            # Include workspace info if bound
            if p.workspace_id:
                ws = db.get_workspace(p.workspace_id)
                if ws:
                    d["workspace"] = ws.to_dict()
            results.append(d)
        return results

    rows = await _run_sync(_list)
    return [ProjectResponse(**d) for d in rows]


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.projects.read"))],
)
async def get_project(
    project_id: int,
    user: User = Depends(get_request_user),
) -> ProjectResponse:
    """Get a project by ID."""
    db = get_orchestration_db(_user_id_int(user))
    project = await _run_sync(lambda: db.get_project(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    summary = await _run_sync(lambda: db.get_project_summary(project_id))
    d = project.to_dict()
    d["task_summary"] = summary
    if project.workspace_id:
        ws = await _run_sync(lambda: db.get_workspace(project.workspace_id))
        if ws:
            d["workspace"] = ws.to_dict()
    return ProjectResponse(**d)


@router.delete(
    "/projects/{project_id}",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.projects.manage"))],
)
async def delete_project(
    project_id: int,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Delete a project and all associated tasks/runs."""
    db = get_orchestration_db(_user_id_int(user))
    project = await _run_sync(lambda: db.get_project(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    await _run_sync(lambda: db.delete_project(project_id))
    return {"deleted": True, "project_id": project_id}


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.tasks.manage"))],
)
async def create_task(
    project_id: int,
    payload: TaskCreateRequest,
    user: User = Depends(get_request_user),
) -> TaskResponse:
    """Create a new task in a project."""
    db = get_orchestration_db(_user_id_int(user))
    project = await _run_sync(lambda: db.get_project(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        task = await _run_sync(lambda: db.create_task(
            project_id=project_id,
            title=payload.title,
            description=payload.description,
            agent_type=payload.agent_type,
            dependency_id=payload.dependency_id,
            reviewer_agent_type=payload.reviewer_agent_type,
            max_review_attempts=payload.max_review_attempts,
            success_criteria=payload.success_criteria,
            metadata=payload.metadata,
        ))
    except CycleDependencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except OrchestrationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TaskResponse(**task.to_dict())


@router.get(
    "/projects/{project_id}/tasks",
    response_model=list[TaskResponse],
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.tasks.read"))],
)
async def list_tasks(
    project_id: int,
    status_filter: str | None = Query(default=None, alias="status"),
    user: User = Depends(get_request_user),
) -> list[TaskResponse]:
    """List tasks in a project with optional status filter."""
    db = get_orchestration_db(_user_id_int(user))
    project = await _run_sync(lambda: db.get_project(project_id))
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    task_status = None
    if status_filter:
        try:
            task_status = TaskStatus(status_filter)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_filter}")
    tasks = await _run_sync(lambda: db.list_tasks(project_id, status=task_status))
    return [TaskResponse(**t.to_dict()) for t in tasks]


@router.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.tasks.read"))],
)
async def get_task(
    task_id: int,
    user: User = Depends(get_request_user),
) -> TaskResponse:
    """Get task detail including run history."""
    db = get_orchestration_db(_user_id_int(user))
    task = await _run_sync(lambda: db.get_task(task_id))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    runs = await _run_sync(lambda: db.list_runs(task_id))
    d = task.to_dict()
    d["runs"] = [r.to_dict() for r in runs]
    return TaskResponse(**d)


# ---------------------------------------------------------------------------
# Run dispatch (with CWD inheritance from workspace)
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/{task_id}/run",
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.tasks.manage"))],
)
async def dispatch_run(
    task_id: int,
    payload: RunDispatchRequest,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Dispatch a task run to an ACP agent.

    CWD resolution: explicit cwd > workspace root_path > "."
    Workspace MCP servers and env_vars are merged into the session.
    """
    db = get_orchestration_db(_user_id_int(user))
    task = await _run_sync(lambda: db.get_task(task_id))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check dependency
    dep_ready = await _run_sync(lambda: db.check_dependency_ready(task_id))
    if not dep_ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task dependency {task.dependency_id} is not complete",
        )

    # Transition to in_progress
    try:
        await _run_sync(lambda: db.transition_task(task_id, TaskStatus.IN_PROGRESS))
    except (InvalidTransitionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- CWD resolution: explicit > workspace root > "." ---
    project = await _run_sync(lambda: db.get_project(task.project_id))
    workspace = None
    if project and project.workspace_id:
        workspace = await _run_sync(lambda: db.get_workspace(project.workspace_id))

    effective_cwd = _resolve_dispatch_cwd(
        payload.cwd,
        workspace_root=workspace.root_path if workspace else None,
    )

    # Gather workspace MCP servers for injection
    workspace_mcp_servers: list[dict[str, Any]] = []
    if workspace:
        workspace_mcp_servers = await _run_sync(
            lambda: db.list_workspace_mcp_servers(workspace.id)
        )

    # Convert workspace MCP servers to create_session format
    mcp_servers_param: list[dict[str, Any]] | None = None
    if workspace_mcp_servers:
        mcp_servers_param = [
            {
                "name": s["server_name"],
                "type": s["server_type"],
                **({"command": s["command"]} if s.get("command") else {}),
                **({"args": s["args"]} if s.get("args") else {}),
                **({"env": s["env"]} if s.get("env") else {}),
                **({"url": s["url"]} if s.get("url") else {}),
            }
            for s in workspace_mcp_servers
            if s.get("enabled", True)
        ]

    # Create ACP session
    session_id: str | None = None
    agent_type = payload.agent_type or task.agent_type
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import get_runner_client
        from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store

        # Quota check
        store = await get_acp_session_store()
        quota_error = await store.check_session_quota(_user_id_int(user))
        if quota_error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=quota_error,
            )

        client = await get_runner_client()
        session_id = await client.create_session(
            effective_cwd,
            mcp_servers=mcp_servers_param,
            agent_type=agent_type,
            user_id=user.id,
        )

        # Register in session store
        try:
            await store.register_session(
                session_id=session_id,
                user_id=_user_id_int(user),
                agent_type=agent_type or "custom",
                name=f"orchestration-task-{task_id}",
                cwd=effective_cwd,
            )
        except Exception as reg_exc:
            logger.warning("Failed to register orchestration ACP session")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create ACP session")
        # Create a failed run record
        run = await _run_sync(lambda: db.create_run(task_id, agent_type=agent_type))
        await _run_sync(lambda: db.fail_run(run.id, error=str(exc)))
        await _run_sync(lambda: db.transition_task(task_id, TaskStatus.TRIAGE))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create ACP session",
        ) from exc

    # Create run record
    run = await _run_sync(lambda: db.create_run(
        task_id,
        agent_type=payload.agent_type or task.agent_type,
        session_id=session_id,
    ))

    # Send initial prompt with task description
    prompt_text = f"Task: {task.title}\n\n{task.description}"
    if task.success_criteria:
        prompt_text += f"\n\nSuccess Criteria: {task.success_criteria}"

    try:
        result = await client.prompt(
            session_id,
            [{"role": "user", "content": prompt_text}],
        )
        stop_reason = result.get("stopReason", "")
        await _run_sync(lambda: db.complete_run(
            run.id,
            result_summary=stop_reason,
            token_usage=result.get("usage", {}),
        ))
        # Transition to review if reviewer is configured, else complete
        if task.reviewer_agent_type:
            await _run_sync(lambda: db.transition_task(task_id, TaskStatus.REVIEW))
        else:
            await _run_sync(lambda: db.transition_task(task_id, TaskStatus.COMPLETE))

    except Exception as exc:
        logger.error("ACP prompt failed")
        await _run_sync(lambda: db.fail_run(run.id, error=str(exc)))
        await _run_sync(lambda: db.transition_task(task_id, TaskStatus.TRIAGE))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ACP prompt failed",
        ) from exc

    # Refetch task to get post-transition status
    updated_task = await _run_sync(lambda: db.get_task(task_id))
    return {
        "task_id": task_id,
        "run_id": run.id,
        "session_id": session_id,
        "status": updated_task.status.value if updated_task else "unknown",
        "effective_cwd": effective_cwd,
    }


# ---------------------------------------------------------------------------
# Review gate
# ---------------------------------------------------------------------------


@router.post(
    "/tasks/{task_id}/review",
    response_model=TaskResponse,
    dependencies=[Depends(require_token_scope("any", require_if_present=True, endpoint_id="agent_orchestration.tasks.manage"))],
)
async def submit_review(
    task_id: int,
    payload: ReviewRequest,
    user: User = Depends(get_request_user),
) -> TaskResponse:
    """Submit a review result for a task.

    Approved -> complete. Rejected -> back to in_progress or triage (after max attempts).
    """
    db = get_orchestration_db(_user_id_int(user))
    task = await _run_sync(lambda: db.get_task(task_id))
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        updated = await _run_sync(lambda: db.submit_review(task_id, payload.approved, payload.feedback))
    except OrchestrationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (InvalidTransitionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TaskResponse(**updated.to_dict())
