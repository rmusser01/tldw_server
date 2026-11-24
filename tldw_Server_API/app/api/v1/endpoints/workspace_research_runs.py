"""Admit native Runs with a verified Research Workspace origin."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user, require_expected_user_if_present
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.research_runs import get_research_service
from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchWorkspaceRunResponse
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchWorkspaceLaunchConflict
from tldw_Server_API.app.core.Research.service import ResearchService
from tldw_Server_API.app.core.Research_Workspace.run_origin import (
    WorkspaceRunCapability,
    WorkspaceRunLaunch,
    WorkspaceRunOrigin,
    WorkspaceRunOriginConflict,
    require_workspace,
    validate_workspace_launch,
)

router = APIRouter(dependencies=[Depends(require_expected_user_if_present)])
Database = Annotated[CharactersRAGDB, Depends(get_chacha_db_for_user)]
RequestUser = Annotated[User, Depends(get_request_user)]


@router.get("/{workspace_id}/research-runs", dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)])
def get_workspace_run_capability(workspace_id: str, db: Database, user: RequestUser) -> WorkspaceRunCapability:
    """Advertise restart-safe Workspace launch for the authenticated owner."""
    try:
        require_workspace(db, workspace_id, str(user.id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Workspace not found") from None
    return WorkspaceRunCapability()


@router.post("/{workspace_id}/research-runs", dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)])
def launch_workspace_run(
    workspace_id: str,
    payload: WorkspaceRunLaunch,
    db: Database,
    user: RequestUser,
    service: Annotated[ResearchService, Depends(get_research_service)],
) -> ResearchWorkspaceRunResponse:
    """Create or recover a native run validated against the captured workspace."""
    try:
        owner_user_id = str(user.id)
        require_workspace(db, workspace_id, owner_user_id)
        origin = WorkspaceRunOrigin(workspace_id=workspace_id, owner_user_id=owner_user_id, launch=payload)
        run = service.create_workspace_session(
            origin,
            validate_workspace=lambda: validate_workspace_launch(db, origin),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Workspace not found") from None
    except (ResearchWorkspaceLaunchConflict, WorkspaceRunOriginConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError:
        raise HTTPException(status_code=400, detail="Workspace research launch could not be admitted") from None
    return ResearchWorkspaceRunResponse.model_validate(run)
