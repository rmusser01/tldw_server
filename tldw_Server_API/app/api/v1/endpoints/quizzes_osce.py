"""OSCE station authoring and self-assessed practice endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import Field, model_validator

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.osce import (
    OsceAttemptCreate,
    OsceAttemptPatch,
    OsceAttemptState,
    OsceAttemptSummary,
    OsceAttemptTransition,
    OsceCandidateAttemptResponse,
    OsceRevealedAttemptResponse,
    OsceStationAuthoringResponse,
    OsceStationCreateRequest,
    OsceStationStoredContent,
    OsceStationSummary,
    OsceStationUpdateContent,
    StrictModel,
)
from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.services.osce_practice import (
    OsceStationIdentityError,
    materialize_station_content,
    project_candidate_attempt,
    project_revealed_attempt,
    project_station_summary,
    reconcile_station_update,
)

router = APIRouter(tags=["quizzes"])


class OsceStationPatchRequest(StrictModel):
    """Optimistic station update envelope."""

    expected_version: int = Field(ge=1)
    content: OsceStationUpdateContent
    order_index: int | None = Field(default=None, ge=0)


class _OsceOffsetPage(StrictModel):
    count: int = Field(ge=0)
    has_more: bool | None = None
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def populate_pagination_aliases(self) -> _OsceOffsetPage:
        return default_offset_pagination_aliases(self)


class OsceStationSummaryPage(_OsceOffsetPage):
    """Paginated station summaries without authoring content."""

    items: list[OsceStationSummary]


class OsceAttemptSummaryPage(_OsceOffsetPage):
    """Paginated attempt summaries without notes or station guides."""

    items: list[OsceAttemptSummary]


OsceAttemptResponse = Annotated[
    OsceCandidateAttemptResponse | OsceRevealedAttemptResponse,
    Field(discriminator="state"),
]


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _require_osce_quiz(db: CharactersRAGDB, quiz_id: int) -> None:
    quiz = db.get_quiz(quiz_id)
    if quiz is None:
        raise _not_found("Quiz not found")
    if quiz["activity_type"] != "osce":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Quiz activity type is not OSCE",
        )


def _project_attempt(row: dict[str, Any]) -> OsceAttemptResponse:
    if row["state"] == OsceAttemptState.IN_PROGRESS.value:
        return project_candidate_attempt(row)
    return project_revealed_attempt(row)


def _project_station_authoring(row: dict[str, Any]) -> OsceStationAuthoringResponse:
    return OsceStationAuthoringResponse.model_validate(
        {
            field_name: row[field_name]
            for field_name in OsceStationAuthoringResponse.model_fields
        }
    )


def _raise_osce_error(exc: Exception, *, default_detail: str) -> None:
    if isinstance(exc, ConflictError):
        raise map_db_error_to_http(exc, conflict_status_code=409) from exc
    if isinstance(exc, (InputError, OsceStationIdentityError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc) or "Invalid OSCE request") from exc
    if isinstance(exc, CharactersRAGDBError):
        raise map_db_error_to_http(exc, default_detail=default_detail) from exc
    raise exc


@router.post(
    "/{quiz_id:int}/osce-stations",
    response_model=OsceStationAuthoringResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_osce_station(
    quiz_id: int,
    request: OsceStationCreateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceStationAuthoringResponse:
    """Create one manually authored station for an OSCE quiz."""
    try:
        _require_osce_quiz(db, quiz_id)
        stored = materialize_station_content(request.content)
        row = db.create_osce_station(
            quiz_id,
            stored,
            order_index=request.order_index,
            origin="manual",
        )
        return _project_station_authoring(row)
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to create OSCE station")


@router.get(
    "/{quiz_id:int}/osce-stations",
    response_model=OsceStationSummaryPage,
)
def list_osce_stations(
    quiz_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceStationSummaryPage:
    """List compact active station summaries in authoring order."""
    try:
        _require_osce_quiz(db, quiz_id)
        payload = db.list_osce_stations(quiz_id, limit=limit, offset=offset)
        items = [project_station_summary(row) for row in payload.get("items") or []]
        total = int(payload.get("count") or 0)
        return OsceStationSummaryPage(
            items=items,
            count=total,
            pagination=build_offset_pagination_meta(
                total=total,
                offset=offset,
                limit=limit,
                count=len(items),
            ),
        )
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to list OSCE stations")


@router.get(
    "/{quiz_id:int}/osce-stations/{station_id:int}",
    response_model=OsceStationAuthoringResponse,
)
def get_osce_station(
    quiz_id: int,
    station_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceStationAuthoringResponse:
    """Return authoring detail for one active station."""
    try:
        _require_osce_quiz(db, quiz_id)
        row = db.get_osce_station(quiz_id, station_id)
        if row is None:
            raise _not_found("OSCE station not found")
        return _project_station_authoring(row)
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to load OSCE station")


@router.patch(
    "/{quiz_id:int}/osce-stations/{station_id:int}",
    response_model=OsceStationAuthoringResponse,
)
def update_osce_station(
    quiz_id: int,
    station_id: int,
    request: OsceStationPatchRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceStationAuthoringResponse:
    """Update station content under optimistic locking."""
    try:
        _require_osce_quiz(db, quiz_id)
        current = db.get_osce_station(quiz_id, station_id)
        if current is None:
            raise _not_found("OSCE station not found")
        reconciled = reconcile_station_update(
            OsceStationStoredContent.model_validate(current["content"]),
            request.content,
            current["verification_state"],
        )
        row = db.update_osce_station(
            quiz_id,
            station_id,
            reconciled.content,
            expected_version=request.expected_version,
            order_index=request.order_index,
            verification_state=reconciled.verification_state,
        )
        if row is None:
            raise _not_found("OSCE station not found")
        return _project_station_authoring(row)
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to update OSCE station")


@router.delete("/{quiz_id:int}/osce-stations/{station_id:int}")
def delete_osce_station(
    quiz_id: int,
    station_id: int,
    expected_version: int | None = Query(default=None, ge=1),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> dict[str, str]:
    """Soft-delete one station."""
    try:
        _require_osce_quiz(db, quiz_id)
        if db.get_osce_station(quiz_id, station_id) is None:
            raise _not_found("OSCE station not found")
        deleted = db.delete_osce_station(
            quiz_id,
            station_id,
            expected_version=expected_version,
        )
        if not deleted:
            raise _not_found("OSCE station not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to delete OSCE station")


@router.post(
    "/osce-stations/{station_id:int}/attempts",
    response_model=OsceAttemptResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_osce_attempt(
    station_id: int,
    request: OsceAttemptCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceAttemptResponse:
    """Create or return a retry-safe attempt from the station snapshot."""
    try:
        row = db.start_osce_attempt(station_id, request.client_attempt_id)
        if row is None:
            raise _not_found("OSCE station not found")
        return _project_attempt(row)
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to start OSCE attempt")


@router.get("/osce-attempts", response_model=OsceAttemptSummaryPage)
def list_osce_attempts(
    quiz_id: int | None = Query(default=None, ge=1),
    station_id: int | None = Query(default=None, ge=1),
    state: list[OsceAttemptState] | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceAttemptSummaryPage:
    """List note-free attempt summaries by most recent activity."""
    try:
        payload = db.list_osce_attempts(
            quiz_id=quiz_id,
            station_id=station_id,
            states=state,
            limit=limit,
            offset=offset,
        )
        items = [OsceAttemptSummary.model_validate(item) for item in payload.get("items") or []]
        total = int(payload.get("count") or 0)
        return OsceAttemptSummaryPage(
            items=items,
            count=total,
            pagination=build_offset_pagination_meta(
                total=total,
                offset=offset,
                limit=limit,
                count=len(items),
            ),
        )
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to list OSCE attempts")


@router.get(
    "/osce-attempts/{attempt_id:int}",
    response_model=OsceAttemptResponse,
)
def get_osce_attempt(
    attempt_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceAttemptResponse:
    """Return candidate or revealed detail according to attempt state."""
    try:
        row = db.get_osce_attempt(attempt_id)
        if row is None:
            raise _not_found("OSCE attempt not found")
        return _project_attempt(row)
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to load OSCE attempt")


@router.patch(
    "/osce-attempts/{attempt_id:int}",
    response_model=OsceAttemptResponse,
)
def update_osce_attempt(
    attempt_id: int,
    request: OsceAttemptPatch,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceAttemptResponse:
    """Update phase-appropriate attempt fields under optimistic locking."""
    try:
        changes = request.model_dump(exclude={"expected_version"}, exclude_unset=True)
        row = db.patch_osce_attempt(
            attempt_id,
            expected_version=request.expected_version,
            **changes,
        )
        if row is None:
            raise _not_found("OSCE attempt not found")
        return _project_attempt(row)
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to update OSCE attempt")


def _transition_attempt(
    db: CharactersRAGDB,
    attempt_id: int,
    request: OsceAttemptTransition,
    target_state: OsceAttemptState,
) -> OsceAttemptResponse:
    row = db.transition_osce_attempt(
        attempt_id,
        target_state,
        expected_version=request.expected_version,
    )
    if row is None:
        raise _not_found("OSCE attempt not found")
    return _project_attempt(row)


@router.post(
    "/osce-attempts/{attempt_id:int}/begin-self-assessment",
    response_model=OsceAttemptResponse,
)
def begin_osce_self_assessment(
    attempt_id: int,
    request: OsceAttemptTransition,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceAttemptResponse:
    """Reveal the immutable marking guide and freeze elapsed time."""
    try:
        return _transition_attempt(
            db,
            attempt_id,
            request,
            OsceAttemptState.SELF_ASSESSMENT,
        )
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to reveal OSCE attempt")


@router.post(
    "/osce-attempts/{attempt_id:int}/complete",
    response_model=OsceAttemptResponse,
)
def complete_osce_attempt(
    attempt_id: int,
    request: OsceAttemptTransition,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> OsceAttemptResponse:
    """Complete an attempt after validating every assessment selection."""
    try:
        return _transition_attempt(
            db,
            attempt_id,
            request,
            OsceAttemptState.COMPLETED,
        )
    except HTTPException:
        raise
    except (ValueError, CharactersRAGDBError) as exc:
        _raise_osce_error(exc, default_detail="Failed to complete OSCE attempt")
