"""Nested transport routes for reviewable Notes graph suggestions."""

from __future__ import annotations

from typing import Annotated, Any, Callable, TypeVar

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from starlette.concurrency import run_in_threadpool

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    TokenScopeGuard,
    User,
    get_request_user,
    principal_has_admin_bypass_claims,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.api.v1.endpoints.notes_graph import _normalize_note_id
from tldw_Server_API.app.api.v1.schemas.notes_graph_suggestions import (
    SuggestionCapabilitiesResponse,
    SuggestionDecisionRequest,
    SuggestionHTTPErrorResponse,
    SuggestionListResponse,
    SuggestionMutationResponse,
    SuggestionResetRequest,
    SuggestionRunCancelRequest,
    SuggestionRunCreateRequest,
    SuggestionRunListResponse,
    SuggestionRunResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import NOTES_GRAPH_READ, NOTES_GRAPH_SUGGEST
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Notes_Graph.suggestion_api import (
    SuggestionAPIError,
    build_notes_graph_suggestions_api,
)
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkDatasetConflictError,
    NotesLinkSyncInactiveDatasetError,
    resolve_notes_link_dataset_authority,
)

_SUGGESTION_ERROR_RESPONSES = {code: {"model": SuggestionHTTPErrorResponse} for code in (404, 409, 412, 422, 429, 503)}
router = APIRouter(
    tags=["notes", "notes-graph-suggestions"],
    responses=_SUGGESTION_ERROR_RESPONSES,
)

SUGGESTION_ERROR_MESSAGES = {
    "notes_graph_active_run_conflict": "A matching suggestion run is already active.",
    "notes_graph_admission_rate_limited": "Suggestion generation is temporarily rate limited.",
    "notes_graph_capabilities_changed": "Suggestion capabilities changed; refresh and retry.",
    "notes_graph_cursor_invalid": "The suggestion cursor is invalid or no longer applicable.",
    "notes_graph_fingerprint_stale": "The note changed; refresh before retrying.",
    "notes_graph_fts_not_ready": "Notes search is not ready for suggestion generation.",
    "notes_graph_invalid_request": "The suggestion request is invalid.",
    "notes_graph_owner_active_run_conflict": "Another suggestion run is already active.",
    "notes_graph_provider_call_policy_unsupported": "The selected provider cannot safely generate suggestions.",
    "notes_graph_provider_disallowed": "The selected provider or model is not allowed.",
    "notes_graph_provider_not_configured": "The selected provider is not configured.",
    "notes_graph_provider_retry_policy_unsupported": "The selected provider retry policy is unsupported.",
    "notes_graph_provider_unavailable": "The selected provider is unavailable.",
    "notes_graph_source_too_large": "The selected note is too large for suggestion generation.",
    "notes_graph_suggestion_conflict": "The suggestion changed; refresh and retry.",
    "notes_graph_suggestion_idempotency_mismatch": "The idempotency key was reused for another request.",
    "notes_graph_suggestion_not_found": "The requested Notes graph resource was not found.",
    "notes_graph_suggestions_disabled": "Notes graph suggestions are disabled.",
    "notes_graph_suggestions_unavailable": "Notes graph suggestions are temporarily unavailable.",
    "notes_graph_suggestions_worker_unavailable": "The suggestion worker is unavailable.",
    "notes_graph_sync_not_ready": "Notes Sync is not ready for this decision.",
}

_RATE_LIMIT_DETAIL = {
    "error_code": "notes_graph_admission_rate_limited",
    "message": SUGGESTION_ERROR_MESSAGES["notes_graph_admission_rate_limited"],
}
_suggestion_rate_limit = rbac_rate_limit(
    "notes.graph.suggest",
    detail=_RATE_LIMIT_DETAIL,
)
IdempotencyKeyHeader = Annotated[
    str | None,
    Header(alias="Idempotency-Key", min_length=1, max_length=256),
]
IfMatchHeader = Annotated[
    str | None,
    Header(alias="If-Match", min_length=1, max_length=128),
]
_T = TypeVar("_T")


def _http_error(exc: SuggestionAPIError) -> HTTPException:
    code = exc.code if exc.code in SUGGESTION_ERROR_MESSAGES else "notes_graph_suggestions_unavailable"
    status_code = exc.status_code if code == exc.code else 503
    return HTTPException(
        status_code=status_code,
        detail={"error_code": code, "message": SUGGESTION_ERROR_MESSAGES[code]},
    )


def _dataset_key(*, owner_user_id: str, dataset_id: str | None) -> str:
    try:
        authority = resolve_notes_link_dataset_authority(
            user_id=owner_user_id,
            dataset_id=dataset_id,
        )
    except (NotesLinkDatasetConflictError, NotesLinkSyncInactiveDatasetError) as exc:
        raise SuggestionAPIError(404, "notes_graph_suggestion_not_found") from exc
    return authority[1].dataset_id if authority is not None else f"legacy:{owner_user_id}"


def _api(*, user: User, db: Any, jobs: Any, dataset_id: str | None) -> Any:
    owner = str(user.id_str)
    return build_notes_graph_suggestions_api(
        note_db=db,
        owner_user_id=owner,
        dataset_id=_dataset_key(owner_user_id=owner, dataset_id=dataset_id),
        jobs=jobs,
    )


async def _call_api(
    *,
    user: User,
    db: Any,
    jobs: Any,
    dataset_id: str | None,
    operation: Callable[[Any], _T],
) -> _T:
    def call() -> _T:
        try:
            return operation(_api(user=user, db=db, jobs=jobs, dataset_id=dataset_id))
        finally:
            release = getattr(db, "release_context_connection", None)
            close = release if callable(release) else getattr(db, "close_connection", None)
            if callable(close):
                close()

    return await run_in_threadpool(call)


def _required_idempotency_key(value: str | None) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized.encode("utf-8")) > 256:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "notes_graph_invalid_request",
                "message": SUGGESTION_ERROR_MESSAGES["notes_graph_invalid_request"],
            },
        )
    return normalized


def _required_if_match(value: str | None) -> str:
    raw = value.strip() if isinstance(value, str) else ""
    if (
        raw.startswith("W/")
        or "," in raw
        or raw == "*"
        or len(raw) < 2
        or not raw.startswith('"')
        or not raw.endswith('"')
    ):
        raw = ""
    else:
        raw = raw[1:-1]
    if (
        not raw.startswith("sha256:")
        or len(raw) != 71
        or any(character not in "0123456789abcdef" for character in raw[7:])
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "notes_graph_invalid_request",
                "message": SUGGESTION_ERROR_MESSAGES["notes_graph_invalid_request"],
            },
        )
    return raw


def _states(values: list[str] | None, *, default: tuple[str, ...] | None = None) -> tuple[str, ...] | None:
    if not values:
        return default
    selected = tuple(part.strip() for value in values for part in value.split(",") if part.strip())
    return selected or default


def _state_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _run_response(run: Any) -> SuggestionRunResponse:
    state_value = _state_value(run.state)
    return SuggestionRunResponse(
        id=str(run.id),
        provider=str(getattr(run, "provider", "")),
        model=str(getattr(run, "model", "")),
        state=state_value,
        revision=int(getattr(run, "revision", 1)),
        created_at=getattr(run, "created_at", None) or "1970-01-01T00:00:00+00:00",
        started_at=getattr(run, "started_at", None),
        completed_at=getattr(run, "completed_at", None),
        suggestion_count=int(getattr(run, "suggestion_count", 0)),
        related_note_count=int(getattr(run, "related_note_count", 0)),
        tag_count=int(getattr(run, "tag_count", 0)),
        invalid_item_count=int(getattr(run, "invalid_item_count", 0)),
        cancellation_available=state_value in {"admitting", "queued", "running"},
        error_code=getattr(run, "error_code", None),
        guidance_key=getattr(run, "guidance_key", None),
    )


def _run_replay_response(envelope: dict[str, Any]) -> SuggestionRunResponse:
    payload = dict(envelope)
    payload["id"] = payload.pop("run_id")
    return SuggestionRunResponse.model_validate(payload)


def _mutation_response(envelope: dict[str, Any]) -> SuggestionMutationResponse:
    resource_id = str(envelope.get("suggestion_id") or envelope.get("run_id") or envelope.get("source_note_id") or "")
    return SuggestionMutationResponse(
        resource_id=resource_id,
        state=str(envelope.get("state") or "completed"),
        revision=int(envelope.get("revision") or 0),
        cleared_count=envelope.get("cleared_count"),
    )


def _principal_allows(principal: AuthPrincipal, permissions: tuple[str, ...]) -> bool:
    claims = set(principal.permissions)
    return principal_has_admin_bypass_claims(principal) or all(permission in claims for permission in permissions)


_require_suggestion_permissions = RequirePermission(NOTES_GRAPH_READ, NOTES_GRAPH_SUGGEST)


@router.get("/{note_id}/graph/suggestions/capabilities", response_model=SuggestionCapabilitiesResponse)
async def get_suggestion_capabilities(
    note_id: str,
    response: Response,
    provider: str | None = Query(default=None, min_length=1, max_length=128),
    model: str | None = Query(default=None, min_length=1, max_length=256),
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionCapabilitiesResponse:
    try:
        capabilities = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.get_capabilities(
                note_id=_normalize_note_id(note_id),
                provider=provider,
                model=model,
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    response.headers["ETag"] = f'"{capabilities.revision}"'
    response.headers["Cache-Control"] = "no-store"
    return SuggestionCapabilitiesResponse.model_validate(capabilities, from_attributes=True)


@router.post(
    "/{note_id}/graph/suggestions/runs",
    response_model=SuggestionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_suggestion_run(
    note_id: str,
    body: SuggestionRunCreateRequest,
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    if_match: IfMatchHeader = None,
    idempotency_key: IdempotencyKeyHeader = None,
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionRunResponse:
    """Admit a source-grounded suggestion run for one note.

    Args:
        note_id: Source note identifier nested in the route.
        body: Optional provider and model overrides for generation.
        dataset_id: Optional Notes dataset scope.
        if_match: Required capability revision from the latest capabilities response.
        idempotency_key: Required key that makes admission retries replay-safe.
        user: Authenticated request user supplied by FastAPI.
        db: Per-user Notes database supplied by FastAPI.
        jobs: Optional Jobs manager used to queue generation.
        _principal: Result of the graph-suggestion permission guard.
        _rate: Result of the graph-suggestion rate-limit guard.
        _scope: Result of the Notes token-scope guard.

    Returns:
        The newly admitted run or the durable response from an exact retry.

    Raises:
        HTTPException: If headers, note scope, capabilities, or admission are invalid.
    """

    try:
        admitted = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.admit_run(
                note_id=_normalize_note_id(note_id),
                provider=body.provider,
                model=body.model,
                capability_revision=_required_if_match(if_match),
                idempotency_key=_required_idempotency_key(idempotency_key),
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    replay_envelope = getattr(admitted, "replay_envelope", None)
    if replay_envelope is not None:
        return _run_replay_response(replay_envelope)
    if admitted.run is None:
        raise _http_error(SuggestionAPIError(503, "notes_graph_suggestions_unavailable"))
    return _run_response(admitted.run)


@router.get("/{note_id}/graph/suggestions/runs", response_model=SuggestionRunListResponse)
async def list_suggestion_runs(
    note_id: str,
    state: list[str] | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=4096),
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionRunListResponse:
    try:
        page = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.list_runs(
                note_id=_normalize_note_id(note_id),
                states=_states(state),
                limit=limit,
                cursor=cursor,
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    return SuggestionRunListResponse(
        items=tuple(_run_response(run) for run in page.items),
        next_cursor=page.next_cursor,
    )


@router.get("/{note_id}/graph/suggestions/runs/{run_id}", response_model=SuggestionRunResponse)
async def get_suggestion_run(
    note_id: str,
    run_id: str,
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionRunResponse:
    try:
        run = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.get_run(
                note_id=_normalize_note_id(note_id),
                run_id=run_id,
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    return _run_response(run)


@router.post("/{note_id}/graph/suggestions/runs/{run_id}/cancel", response_model=SuggestionMutationResponse)
async def cancel_suggestion_run(
    note_id: str,
    run_id: str,
    body: SuggestionRunCancelRequest,
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    idempotency_key: IdempotencyKeyHeader = None,
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionMutationResponse:
    try:
        result = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.cancel_run(
                note_id=_normalize_note_id(note_id),
                run_id=run_id,
                expected_revision=body.expected_revision,
                idempotency_key=_required_idempotency_key(idempotency_key),
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    envelope = result.cancellation.replay_envelope
    if envelope is None:
        run = result.cancellation.run
        if run is None:
            raise _http_error(SuggestionAPIError(503, "notes_graph_suggestions_unavailable"))
        envelope = {
            "run_id": run.id,
            "state": _state_value(run.state),
            "revision": run.revision,
        }
    return _mutation_response(envelope)


@router.get("/{note_id}/graph/suggestions", response_model=SuggestionListResponse)
async def list_suggestions(
    note_id: str,
    state: list[str] | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None, max_length=4096),
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionListResponse:
    try:
        page = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.list_suggestions(
                note_id=_normalize_note_id(note_id),
                states=_states(state, default=("pending", "accepting")),
                limit=limit,
                cursor=cursor,
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    return SuggestionListResponse(
        items=tuple(
            {
                "id": item.suggestion.id,
                "run_id": item.suggestion.run_id,
                "kind": item.suggestion.kind.value,
                "state": item.suggestion.state.value,
                "revision": item.suggestion.revision,
                "source_note_id": item.suggestion.source_note_id,
                "source_fingerprint": item.suggestion.source_fingerprint,
                "target_note_id": item.suggestion.target_note_id,
                "target_fingerprint": item.suggestion.target_fingerprint,
                "target_title": item.target_title,
                "normalized_tag": item.suggestion.normalized_tag,
                "display_tag": item.suggestion.display_tag,
                "existing_tag": item.suggestion.keyword_sync_id is not None,
                "match_strength": item.suggestion.match_strength,
                "rationale": item.suggestion.rationale,
                "evidence": item.evidence,
                "updated_at": item.suggestion.updated_at,
            }
            for item in page.items
        ),
        next_cursor=page.next_cursor,
        current_source_fingerprint=page.current_source_fingerprint,
        rejection_set_revision=page.rejection_set_revision,
        rejection_count=page.rejection_count,
    )


# This static path must remain before the dynamic suggestion decision paths.
@router.post("/{note_id}/graph/suggestions/rejections/reset", response_model=SuggestionMutationResponse)
async def reset_suggestion_rejections(
    note_id: str,
    body: SuggestionResetRequest,
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    idempotency_key: IdempotencyKeyHeader = None,
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionMutationResponse:
    try:
        result = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.reset_rejections(
                note_id=_normalize_note_id(note_id),
                source_fingerprint=body.source_fingerprint,
                expected_revision=body.expected_rejection_revision,
                idempotency_key=_required_idempotency_key(idempotency_key),
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    return _mutation_response(result.envelope)


@router.post("/{note_id}/graph/suggestions/{suggestion_id}/accept", response_model=SuggestionMutationResponse)
async def accept_suggestion(
    note_id: str,
    suggestion_id: str,
    body: SuggestionDecisionRequest,
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    idempotency_key: IdempotencyKeyHeader = None,
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionMutationResponse:
    try:
        normalized_note_id = _normalize_note_id(note_id)
        normalized_idempotency_key = _required_idempotency_key(idempotency_key)

        def accept(api: Any) -> Any:
            required = api.accept_permission_requirements(
                note_id=normalized_note_id,
                suggestion_id=suggestion_id,
                expected_revision=body.expected_revision,
                expected_source_fingerprint=body.expected_source_fingerprint,
                expected_target_fingerprint=body.expected_target_fingerprint,
                idempotency_key=normalized_idempotency_key,
            )
            if principal is None or not _principal_allows(principal, required):
                raise HTTPException(status_code=403, detail=f"Permission denied: missing {', '.join(required)}")
            return api.accept_suggestion(
                note_id=normalized_note_id,
                suggestion_id=suggestion_id,
                expected_revision=body.expected_revision,
                expected_source_fingerprint=body.expected_source_fingerprint,
                expected_target_fingerprint=body.expected_target_fingerprint,
                idempotency_key=normalized_idempotency_key,
            )

        result = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=accept,
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    return _mutation_response(result.envelope)


@router.post("/{note_id}/graph/suggestions/{suggestion_id}/reject", response_model=SuggestionMutationResponse)
async def reject_suggestion(
    note_id: str,
    suggestion_id: str,
    body: SuggestionDecisionRequest,
    dataset_id: str | None = Query(default=None, min_length=1, max_length=256),
    idempotency_key: IdempotencyKeyHeader = None,
    user: User = Depends(get_request_user),
    db: Any = Depends(get_chacha_db_for_user),
    jobs: Any = Depends(try_get_job_manager),
    _principal: AuthPrincipal = Depends(_require_suggestion_permissions),
    _rate: None = Depends(_suggestion_rate_limit),
    _scope: None = Depends(TokenScopeGuard("notes", require_if_present=True, endpoint_id="notes.graph.suggest")),
) -> SuggestionMutationResponse:
    try:
        result = await _call_api(
            user=user,
            db=db,
            jobs=jobs,
            dataset_id=dataset_id,
            operation=lambda api: api.reject_suggestion(
                note_id=_normalize_note_id(note_id),
                suggestion_id=suggestion_id,
                expected_revision=body.expected_revision,
                expected_source_fingerprint=body.expected_source_fingerprint,
                expected_target_fingerprint=body.expected_target_fingerprint,
                idempotency_key=_required_idempotency_key(idempotency_key),
            ),
        )
    except SuggestionAPIError as exc:
        raise _http_error(exc) from exc
    return _mutation_response(result.envelope)


__all__ = ["SUGGESTION_ERROR_MESSAGES", "build_notes_graph_suggestions_api", "router"]
