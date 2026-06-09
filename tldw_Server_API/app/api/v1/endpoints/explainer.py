"""Explainer workspace CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from tldw_Server_API.app.api.v1.API_Deps.Explainer_DB_Deps import get_explainer_db
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.schemas.explainer import (
    ExplainerDeleteNodeResponse,
    ExplainerJobAcceptedResponse,
    ExplainerJobStatusResponse,
    ExplainerNodeExpandRequest,
    ExplainerNodeCreateRequest,
    ExplainerNodePatchRequest,
    ExplainerNodeResponse,
    ExplainerQuestionAnswerRequest,
    ExplainerSessionCreateRequest,
    ExplainerSessionListResponse,
    ExplainerSessionPatchRequest,
    ExplainerSessionResponse,
    ExplainerSessionSummaryResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.jobs import EXPLAINER_DOMAIN
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository
from tldw_Server_API.app.core.Explainer.service import (
    ExplainerService,
    map_explainer_service_error,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager

router = APIRouter(prefix="/explainer", tags=["explainer"])


def _owner_user_id(current_user: User) -> str:
    if current_user.id is None:
        raise HTTPException(status_code=500, detail="User identification failed")
    return str(current_user.id)


def _service(db: ExplainerDatabase, job_manager=None) -> ExplainerService:
    return ExplainerService(ExplainerRepository(db), job_manager=job_manager)


def _source_payloads(sources) -> list[dict]:
    return [source.model_dump(by_alias=False) for source in sources]


def _citation_payloads(citations) -> list[dict]:
    return [citation.model_dump(by_alias=False) for citation in citations]


def _raise_http(exc: Exception) -> None:
    status_code, detail = map_explainer_service_error(exc)
    raise HTTPException(status_code=status_code, detail=detail) from exc


def _job_payload(job: dict) -> dict:
    payload = job.get("payload") or {}
    return payload if isinstance(payload, dict) else {}


def _public_job_error(job: dict) -> str | None:
    raw_status = str(job.get("status") or "").strip().lower()
    if raw_status == "failed":
        return "Explainer job failed."
    if raw_status == "cancelled":
        return "Explainer job was cancelled."
    return None


@router.post(
    "/sessions",
    response_model=ExplainerSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Explainer session",
)
async def create_explainer_session(
    body: ExplainerSessionCreateRequest,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerSessionResponse:
    try:
        session = _service(db).create_session(
            owner_user_id=_owner_user_id(current_user),
            title=body.title,
            mode=body.mode,
            output_intent=body.output_intent,
            grounding=body.grounding,
            depth_preset=body.depth_preset,
            selected_sources=_source_payloads(body.selected_sources),
            root_prompt=body.root_prompt,
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerSessionResponse.from_domain(session)


@router.get(
    "/sessions",
    response_model=ExplainerSessionListResponse,
    summary="List Explainer sessions",
)
async def list_explainer_sessions(
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ExplainerSessionListResponse:
    summaries, total = _service(db).list_session_summaries(
        owner_user_id=_owner_user_id(current_user),
        limit=limit,
        offset=offset,
    )
    items = [ExplainerSessionSummaryResponse.from_domain(summary) for summary in summaries]
    return ExplainerSessionListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/sessions/{session_id}",
    response_model=ExplainerSessionResponse,
    summary="Get Explainer session",
)
async def get_explainer_session(
    session_id: str,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerSessionResponse:
    try:
        session = _service(db).get_session(session_id, owner_user_id=_owner_user_id(current_user))
    except Exception as exc:
        _raise_http(exc)
    return ExplainerSessionResponse.from_domain(session)


@router.patch(
    "/sessions/{session_id}",
    response_model=ExplainerSessionResponse,
    summary="Update Explainer session",
)
async def update_explainer_session(
    session_id: str,
    body: ExplainerSessionPatchRequest,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerSessionResponse:
    selected_sources = (
        _source_payloads(body.selected_sources)
        if body.selected_sources is not None
        else None
    )
    try:
        session = _service(db).update_session(
            session_id,
            owner_user_id=_owner_user_id(current_user),
            title=body.title,
            output_intent=body.output_intent,
            grounding=body.grounding,
            depth_preset=body.depth_preset,
            selected_sources=selected_sources,
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerSessionResponse.from_domain(session)


@router.delete(
    "/sessions/{session_id}",
    response_model=ExplainerSessionResponse,
    summary="Archive Explainer session",
)
async def archive_explainer_session(
    session_id: str,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerSessionResponse:
    try:
        session = _service(db).archive_session(session_id, owner_user_id=_owner_user_id(current_user))
    except Exception as exc:
        _raise_http(exc)
    return ExplainerSessionResponse.from_domain(session)


@router.post(
    "/sessions/{session_id}/nodes",
    response_model=ExplainerNodeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Explainer node",
)
async def create_explainer_node(
    session_id: str,
    body: ExplainerNodeCreateRequest,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerNodeResponse:
    try:
        node = _service(db).create_node(
            session_id,
            owner_user_id=_owner_user_id(current_user),
            title=body.title,
            parent_id=body.parent_id,
            body=body.body,
            kind=body.kind,
            intent=body.intent,
            status=body.status,
            evidence_state=body.evidence_state,
            outside_knowledge_used=body.outside_knowledge_used,
            citations=_citation_payloads(body.citations),
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerNodeResponse.from_domain(node)


@router.patch(
    "/sessions/{session_id}/nodes/{node_id}",
    response_model=ExplainerNodeResponse,
    summary="Update Explainer node",
)
async def update_explainer_node(
    session_id: str,
    node_id: str,
    body: ExplainerNodePatchRequest,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerNodeResponse:
    updates = body.model_dump(exclude_unset=True, by_alias=False)
    try:
        node = _service(db).update_node(
            session_id,
            node_id,
            owner_user_id=_owner_user_id(current_user),
            updates=updates,
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerNodeResponse.from_domain(node)


@router.delete(
    "/sessions/{session_id}/nodes/{node_id}",
    response_model=ExplainerDeleteNodeResponse,
    summary="Delete Explainer node",
)
async def delete_explainer_node(
    session_id: str,
    node_id: str,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerDeleteNodeResponse:
    try:
        result = _service(db).delete_node(
            session_id,
            node_id,
            owner_user_id=_owner_user_id(current_user),
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerDeleteNodeResponse(**result)


@router.post(
    "/sessions/{session_id}/nodes/{node_id}/expand",
    response_model=ExplainerJobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue Explainer node expansion",
)
async def expand_explainer_node(
    session_id: str,
    node_id: str,
    body: ExplainerNodeExpandRequest,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
    jm: JobManager = Depends(get_job_manager),
) -> ExplainerJobAcceptedResponse:
    try:
        accepted = _service(db, jm).enqueue_node_expansion(
            session_id=session_id,
            node_id=node_id,
            owner_user_id=_owner_user_id(current_user),
            intent=body.intent,
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerJobAcceptedResponse.from_domain(accepted)


@router.post(
    "/sessions/{session_id}/nodes/{node_id}/answer-question",
    response_model=ExplainerNodeResponse,
    summary="Persist an Explainer question answer",
)
async def answer_explainer_question(
    session_id: str,
    node_id: str,
    body: ExplainerQuestionAnswerRequest,
    current_user: User = Depends(get_request_user),
    db: ExplainerDatabase = Depends(get_explainer_db),
) -> ExplainerNodeResponse:
    try:
        node = _service(db).answer_question(
            session_id,
            node_id,
            owner_user_id=_owner_user_id(current_user),
            selected_option_id=body.selected_option_id,
            selected_custom_answer=body.selected_custom_answer,
        )
    except Exception as exc:
        _raise_http(exc)
    return ExplainerNodeResponse.from_domain(node)


@router.get(
    "/jobs/{job_id}",
    response_model=ExplainerJobStatusResponse,
    summary="Get an ownership-scoped Explainer job",
)
async def get_explainer_job_status(
    job_id: int,
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> ExplainerJobStatusResponse:
    job = jm.get_job(job_id)
    if (
        not job
        or str(job.get("domain") or "").strip().lower() != EXPLAINER_DOMAIN
        or str(job.get("owner_user_id") or "").strip() != _owner_user_id(current_user)
    ):
        raise HTTPException(status_code=404, detail="Explainer job not found")
    payload = _job_payload(job)
    return ExplainerJobStatusResponse(
        job_id=str(job.get("id")),
        session_id=payload.get("session_id"),
        node_id=payload.get("node_id"),
        status=str(job.get("status") or "queued"),
        progress_percent=job.get("progress_percent"),
        progress_message=job.get("progress_message"),
        error=_public_job_error(job),
    )
