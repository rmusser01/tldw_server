"""Publish and discover native Study outputs from an owned Research Workspace."""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_expected_user_if_present
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError
from tldw_Server_API.app.core.Research_Workspace.study_output_models import (
    ResearchStudyKind,
    ResearchStudyOutput,
    ResearchStudyOutputsPage,
    ResearchStudyPublication,
)
from tldw_Server_API.app.core.Research_Workspace.study_publication import (
    ResearchStudyPublicationError,
    ResearchStudyPublisher,
)

router = APIRouter(dependencies=[Depends(require_expected_user_if_present)])
Database = Annotated[CharactersRAGDB, Depends(get_chacha_db_for_user)]


def _failure(exc: Exception) -> HTTPException:
    if isinstance(exc, ResearchStudyPublicationError):
        return HTTPException(status_code=exc.status, detail=str(exc))
    if isinstance(exc, ConflictError):
        return HTTPException(status_code=409, detail="The native Study record changed. Retry the save.")
    logger.error("Research Study output storage failed ({})", type(exc).__name__)
    return HTTPException(status_code=500, detail="Study output storage failed")


_STORAGE_ERRORS = (ResearchStudyPublicationError, CharactersRAGDBError, BackendDatabaseError, sqlite3.Error)


@router.get(
    "/{workspace_id}/study-outputs",
    response_model_exclude_none=True,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
)
def list_study_outputs(
    workspace_id: str,
    db: Database,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
) -> ResearchStudyOutputsPage:
    """List Study outputs and advertise the supported publication contract."""
    try:
        return ResearchStudyPublisher(db).page(workspace_id, limit=limit, offset=offset)
    except _STORAGE_ERRORS as exc:
        raise _failure(exc) from exc


@router.get(
    "/{workspace_id}/study-outputs/{kind}/{owner_id}",
    response_model_exclude_none=True,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
)
def get_study_output(
    workspace_id: str,
    kind: ResearchStudyKind,
    owner_id: Annotated[int, Path(ge=1, le=2**63 - 1)],
    db: Database,
) -> ResearchStudyOutput:
    """Resolve the exact generated deck or quiz within its owning Workspace."""
    try:
        return ResearchStudyPublisher(db).get(workspace_id, kind, owner_id)
    except _STORAGE_ERRORS as exc:
        raise _failure(exc) from exc


@router.post(
    "/{workspace_id}/study-outputs",
    response_model_exclude_none=True,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
)
def save_study_output(workspace_id: str, request: ResearchStudyPublication, db: Database) -> ResearchStudyOutput:
    """Publish a complete deck or quiz, or recover the original operation."""
    try:
        return ResearchStudyPublisher(db).save(workspace_id, request)
    except _STORAGE_ERRORS as exc:
        raise _failure(exc) from exc
