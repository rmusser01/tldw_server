"""REST endpoints for managing chat workflow templates and runs."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission
from tldw_Server_API.app.api.v1.API_Deps.chat_workflows_deps import (
    get_chat_workflows_db,
    get_chat_workflows_user,
)
from tldw_Server_API.app.api.v1.schemas.chat_workflows import (
    ChatWorkflowRoundResponse,
    ChatWorkflowRunResponse,
    ChatWorkflowTranscriptResponse,
    ChatWorkflowTemplateCreate,
    ChatWorkflowTemplateDraft,
    ChatWorkflowTemplateResponse,
    ChatWorkflowTemplateUpdate,
    ContinueChatResponse,
    GenerateDraftRequest,
    GenerateDraftResponse,
    StartRunRequest,
    SubmitAnswerRequest,
    SubmitRoundRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    CHAT_WORKFLOWS_READ,
    CHAT_WORKFLOWS_RUN,
    CHAT_WORKFLOWS_WRITE,
)
from tldw_Server_API.app.core.Chat_Workflows.question_renderer import (
    ChatWorkflowQuestionRenderer,
)
from tldw_Server_API.app.core.Chat_Workflows.dialogue_orchestrator import (
    ChatWorkflowDialogueOrchestrator,
)
from tldw_Server_API.app.core.Chat_Workflows.service import (
    ChatWorkflowConflictError,
    ChatWorkflowService,
)
from tldw_Server_API.app.core.DB_Management.ChatWorkflows_DB import (
    ChatWorkflowsDatabase,
)


router = APIRouter(prefix="/api/v1/chat-workflows", tags=["chat-workflows"])


def _utcnow_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _json_loads(value: str | None, *, default: Any) -> Any:
    """Decode optional JSON text and fall back to a caller-provided default."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


async def _get_user_context(
    user_context: dict[str, Any] = Depends(get_chat_workflows_user),
) -> dict[str, Any]:
    """Expose the authenticated chat-workflows user context."""
    return user_context


async def _get_db(
    db: ChatWorkflowsDatabase = Depends(get_chat_workflows_db),
) -> ChatWorkflowsDatabase:
    """Expose the chat workflows persistence adapter."""
    return db


async def _get_service(
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> ChatWorkflowService:
    """Build the chat workflows service for the current request."""
    return ChatWorkflowService(
        db=db,
        question_renderer=ChatWorkflowQuestionRenderer(),
        dialogue_orchestrator=ChatWorkflowDialogueOrchestrator(),
    )


def _tenant_id_for(user_context: dict[str, Any]) -> str:
    """Normalize the caller tenant identifier for persistence filtering."""
    tenant_id = user_context.get("tenant_id")
    return str(tenant_id) if tenant_id is not None else "default"


def _is_admin(user_context: dict[str, Any]) -> bool:
    """Return whether the current caller bypasses per-user ownership checks."""
    return bool(user_context.get("is_admin", False))


def _require_template_access(
    template: dict[str, Any] | None,
    *,
    user_context: dict[str, Any],
) -> dict[str, Any]:
    """Enforce caller access to a workflow template or raise 404."""
    if template is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
    if _is_admin(user_context):
        return template
    if str(template.get("tenant_id") or "default") != _tenant_id_for(user_context):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
    if str(template.get("user_id")) != str(user_context.get("user_id")):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
    return template


def _require_run_access(
    run: dict[str, Any] | None,
    *,
    user_context: dict[str, Any],
) -> dict[str, Any]:
    """Enforce caller access to a workflow run or raise 404."""
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if _is_admin(user_context):
        return run
    if str(run.get("tenant_id") or "default") != _tenant_id_for(user_context):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if str(run.get("user_id")) != str(user_context.get("user_id")):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


def _db_call(db: ChatWorkflowsDatabase, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Invoke a DB adapter method or raise a 503 when storage is unavailable."""
    method = getattr(db, method_name, None)
    if not callable(method):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat workflows storage is unavailable",
        )
    return method(*args, **kwargs)


async def _db_call_async(
    db: ChatWorkflowsDatabase,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a synchronous chat workflows DB call on a worker thread."""
    return await asyncio.to_thread(_db_call, db, method_name, *args, **kwargs)


def _serialize_template(template: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw DB template row into the API response shape."""
    serialized_steps: list[dict[str, Any]] = []
    for step in sorted(template.get("steps", []), key=lambda row: int(row.get("step_index", 0))):
        step_type = str(step.get("step_type") or "question_step")
        serialized_steps.append(
            {
                "id": str(step.get("step_id") or step.get("id")),
                "step_index": int(step.get("step_index", 0)),
                "step_type": step_type,
                "label": step.get("label"),
                "base_question": step.get("base_question"),
                "question_mode": step.get("question_mode", "stock"),
                "phrasing_instructions": step.get("phrasing_instructions"),
                "context_refs": _json_loads(step.get("context_refs_json"), default=[]),
                "dialogue_config": (
                    _json_loads(step.get("dialogue_config_json"), default=None)
                    if step_type == "dialogue_round_step"
                    else None
                ),
            }
        )
    return {
        "id": int(template["id"]),
        "title": template["title"],
        "description": template.get("description"),
        "version": int(template.get("version", 1)),
        "status": template.get("status", "active"),
        "steps": serialized_steps,
        "created_at": template.get("created_at"),
        "updated_at": template.get("updated_at"),
    }


def _serialize_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw DB answer row into the API response shape."""
    return {
        "step_id": answer["step_id"],
        "step_index": int(answer["step_index"]),
        "displayed_question": answer["displayed_question"],
        "answer_text": answer["answer_text"],
        "question_generation_meta": _json_loads(
            answer.get("question_generation_meta_json"),
            default={},
        ),
        "answered_at": answer.get("answered_at"),
    }


def _serialize_round(round_row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw DB dialogue round row into the API response shape."""
    return ChatWorkflowRoundResponse.model_validate(
        {
            "round_index": int(round_row.get("round_index", 0)),
            "user_message": round_row.get("user_message", ""),
            "debate_llm_message": round_row.get("debate_llm_message"),
            "moderator_decision": round_row.get("moderator_decision"),
            "moderator_summary": round_row.get("moderator_summary"),
            "next_user_prompt": round_row.get("next_user_prompt"),
            "status": round_row.get("status", "completed"),
            "created_at": round_row.get("created_at"),
            "updated_at": round_row.get("updated_at"),
        }
    ).model_dump()


def _get_run_steps(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ordered step snapshot for a run."""
    template_snapshot = _json_loads(run.get("template_snapshot_json"), default={})
    steps = template_snapshot.get("steps", [])
    if not isinstance(steps, list):
        return []
    return steps


async def _project_transcript_messages(
    *,
    db: ChatWorkflowsDatabase,
    service: ChatWorkflowService,
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project the canonical workflow transcript from answers and dialogue rounds."""
    run_id = str(run["run_id"])
    answers_by_step = {
        int(answer["step_index"]): answer
        for answer in await _db_call_async(db, "list_answers", run_id)
    }
    current_step = await service.get_current_step(run_id) if run.get("status") == "active" else None
    current_step_index = int(run.get("current_step_index", 0))
    messages: list[dict[str, Any]] = []

    for step in _get_run_steps(run):
        step_index = int(step.get("step_index", 0))
        step_type = str(step.get("step_type") or "question_step")
        if step_type == "dialogue_round_step":
            rounds = [
                _serialize_round(round_row)
                for round_row in await _db_call_async(db, "list_rounds", run_id, step_index)
            ]
            for round_row in rounds:
                if round_row.get("user_message"):
                    messages.append(
                        {
                            "role": "user",
                            "content": round_row["user_message"],
                            "step_index": step_index,
                        }
                    )
                if round_row.get("debate_llm_message"):
                    messages.append(
                        {
                            "role": "debate_llm",
                            "content": round_row["debate_llm_message"],
                            "step_index": step_index,
                        }
                    )
                moderator_parts = [
                    str(round_row.get("moderator_summary") or "").strip(),
                    str(round_row.get("next_user_prompt") or "").strip(),
                ]
                moderator_content = "\n\n".join(part for part in moderator_parts if part)
                if moderator_content:
                    messages.append(
                        {
                            "role": "moderator",
                            "content": moderator_content,
                            "step_index": step_index,
                        }
                    )
            last_prompt = (
                rounds[-1].get("next_user_prompt")
                if rounds
                else None
            )
            if (
                current_step is not None
                and step_index == current_step_index
                and current_step.get("displayed_question")
                and current_step.get("displayed_question") != last_prompt
            ):
                messages.append(
                    {
                        "role": "moderator",
                        "content": current_step["displayed_question"],
                        "step_index": step_index,
                    }
                )
            continue

        answer = answers_by_step.get(step_index)
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": answer["displayed_question"],
                    "step_index": step_index,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": answer["answer_text"],
                    "step_index": step_index,
                }
            )
            continue

        if (
            current_step is not None
            and step_index == current_step_index
            and current_step.get("displayed_question")
        ):
            messages.append(
                {
                    "role": "assistant",
                    "content": current_step["displayed_question"],
                    "step_index": step_index,
                }
            )

    return messages


async def _build_run_response(
    *,
    db: ChatWorkflowsDatabase,
    service: ChatWorkflowService,
    run: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical run response, including current question state."""
    answers = [
        _serialize_answer(answer)
        for answer in await _db_call_async(db, "list_answers", run["run_id"])
    ]
    current_question: str | None = None
    current_step_kind: str | None = None
    current_prompt: str | None = None
    current_round_index: int | None = None
    rounds: list[dict[str, Any]] = []
    if run.get("status") == "active":
        current_step = await service.get_current_step(run["run_id"])
        if current_step is not None:
            current_question = current_step.get("displayed_question")
            current_step_kind = str(current_step.get("step_type") or "question_step")
            current_prompt = current_step.get("current_prompt") or current_question
            if current_step_kind == "dialogue_round_step":
                current_round_index = int(run.get("active_round_index", 0))
                rounds = [
                    _serialize_round(round_row)
                    for round_row in current_step.get("rounds", [])
                ]

    return {
        "run_id": run["run_id"],
        "template_id": run.get("template_id"),
        "template_version": int(run.get("template_version", 1)),
        "status": run["status"],
        "current_step_index": int(run.get("current_step_index", 0)),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "canceled_at": run.get("canceled_at"),
        "free_chat_conversation_id": run.get("free_chat_conversation_id"),
        "selected_context_refs": _json_loads(run.get("selected_context_refs_json"), default=[]),
        "current_question": current_question,
        "current_step_kind": current_step_kind,
        "current_prompt": current_prompt,
        "current_round_index": current_round_index,
        "rounds": rounds,
        "answers": answers,
    }


@router.post(
    "/templates",
    response_model=ChatWorkflowTemplateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_WRITE))],
)
async def create_template(
    payload: ChatWorkflowTemplateCreate,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> dict[str, Any]:
    """Create a workflow template owned by the authenticated user."""
    template_id = await _db_call_async(
        db,
        "create_template",
        tenant_id=_tenant_id_for(user_context),
        user_id=str(user_context["user_id"]),
        title=payload.title,
        description=payload.description,
        version=payload.version,
    )
    await _db_call_async(
        db,
        "replace_template_steps",
        template_id,
        [step.model_dump() for step in payload.steps],
    )
    return _serialize_template(await _db_call_async(db, "get_template", template_id) or {})


@router.get(
    "/templates",
    response_model=list[ChatWorkflowTemplateResponse],
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_READ))],
)
async def list_templates(
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> list[dict[str, Any]]:
    """List the caller's workflow templates in reverse update order."""
    templates = await _db_call_async(
        db,
        "list_templates",
        tenant_id=_tenant_id_for(user_context),
        user_id=str(user_context["user_id"]),
    )
    serialized: list[dict[str, Any]] = []
    for template in templates:
        full_template = await _db_call_async(db, "get_template", int(template["id"]))
        serialized.append(
            _serialize_template(
                _require_template_access(full_template, user_context=user_context)
            )
        )
    return serialized


@router.get(
    "/templates/{template_id}",
    response_model=ChatWorkflowTemplateResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_READ))],
)
async def get_template(
    template_id: int,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> dict[str, Any]:
    """Fetch a single workflow template visible to the caller."""
    template = _require_template_access(
        await _db_call_async(db, "get_template", template_id),
        user_context=user_context,
    )
    return _serialize_template(template)


@router.put(
    "/templates/{template_id}",
    response_model=ChatWorkflowTemplateResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_WRITE))],
)
async def update_template(
    template_id: int,
    payload: ChatWorkflowTemplateUpdate,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> dict[str, Any]:
    """Update mutable template fields and bump version on content changes."""
    existing = _require_template_access(
        await _db_call_async(db, "get_template", template_id),
        user_context=user_context,
    )
    content_changed = any(
        value is not None
        for value in (payload.title, payload.description, payload.steps)
    )
    version = int(existing.get("version", 1)) + 1 if content_changed else None
    await _db_call_async(
        db,
        "update_template",
        template_id,
        title=payload.title,
        description=payload.description,
        status=payload.status,
        version=version,
    )
    if payload.steps is not None:
        await _db_call_async(
            db,
            "replace_template_steps",
            template_id,
            [step.model_dump() for step in payload.steps],
        )
    updated = _require_template_access(
        await _db_call_async(db, "get_template", template_id),
        user_context=user_context,
    )
    return _serialize_template(updated)


@router.delete(
    "/templates/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_WRITE))],
)
async def delete_template(
    template_id: int,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> Response:
    """Delete a workflow template owned by the authenticated user."""
    _require_template_access(
        await _db_call_async(db, "get_template", template_id),
        user_context=user_context,
    )
    await _db_call_async(db, "delete_template", template_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/generate-draft",
    response_model=GenerateDraftResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_WRITE))],
)
async def generate_draft(
    payload: GenerateDraftRequest,
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Generate a temporary linear workflow draft from a goal statement."""
    try:
        draft = service.generate_draft(
            goal=payload.goal,
            base_question=payload.base_question,
            desired_step_count=payload.desired_step_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"template_draft": ChatWorkflowTemplateDraft.model_validate(draft)}


@router.post(
    "/runs",
    response_model=ChatWorkflowRunResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_RUN))],
)
async def start_run(
    payload: StartRunRequest,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Start a new workflow run from a saved template or generated draft."""
    template: dict[str, Any]
    source_mode: str
    if payload.template_id is not None:
        template = _require_template_access(
            await _db_call_async(db, "get_template", payload.template_id),
            user_context=user_context,
        )
        source_mode = "saved_template"
    else:
        template = payload.template_draft.model_dump() if payload.template_draft is not None else {}
        source_mode = "generated_draft"

    try:
        created_run = await asyncio.to_thread(
            service.start_run,
            tenant_id=_tenant_id_for(user_context),
            user_id=str(user_context["user_id"]),
            template=template,
            source_mode=source_mode,
            selected_context_refs=payload.selected_context_refs,
            question_renderer_model=payload.question_renderer_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    run = _require_run_access(
        await _db_call_async(db, "get_run", created_run["run_id"]),
        user_context=user_context,
    )
    return await _build_run_response(db=db, service=service, run=run)


@router.get(
    "/runs/{run_id}",
    response_model=ChatWorkflowRunResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_READ))],
)
async def get_run(
    run_id: str,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Return the current state of a workflow run."""
    run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    return await _build_run_response(db=db, service=service, run=run)


@router.get(
    "/runs/{run_id}/transcript",
    response_model=ChatWorkflowTranscriptResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_READ))],
)
async def get_run_transcript(
    run_id: str,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Render the structured run as assistant/user transcript messages."""
    run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    return {
        "run_id": run_id,
        "messages": await _project_transcript_messages(
            db=db,
            service=service,
            run=run,
        ),
    }


@router.post(
    "/runs/{run_id}/answer",
    response_model=ChatWorkflowRunResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_RUN))],
)
async def answer_run_step(
    run_id: str,
    payload: SubmitAnswerRequest,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Submit an answer for the run's current step."""
    run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )

    try:
        await service.record_answer(
            run_id=run_id,
            step_index=payload.step_index,
            answer_text=payload.answer_text,
            idempotency_key=payload.idempotency_key,
        )
    except ChatWorkflowConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    updated_run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    return await _build_run_response(db=db, service=service, run=updated_run)


@router.post(
    "/runs/{run_id}/rounds/{round_index}/respond",
    response_model=ChatWorkflowRunResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_RUN))],
)
async def respond_to_run_round(
    run_id: str,
    round_index: int,
    payload: SubmitRoundRequest,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Submit a user message for the current dialogue round."""
    _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )

    try:
        await service.respond_to_round(
            run_id=run_id,
            round_index=round_index,
            user_message=payload.user_message,
            idempotency_key=payload.idempotency_key,
        )
    except ChatWorkflowConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    updated_run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    return await _build_run_response(db=db, service=service, run=updated_run)


@router.post(
    "/runs/{run_id}/cancel",
    response_model=ChatWorkflowRunResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_RUN))],
)
async def cancel_run(
    run_id: str,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
    service: ChatWorkflowService = Depends(_get_service),
) -> dict[str, Any]:
    """Cancel an active or not-yet-finished workflow run."""
    run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    if run.get("status") == "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Completed runs cannot be canceled",
        )
    if run.get("status") != "canceled":
        await _db_call_async(
            db,
            "update_run",
            run_id,
            status="canceled",
            canceled_at=_utcnow_iso(),
        )
        await _db_call_async(db, "append_event", run_id, "run_canceled", {"status": "canceled"})

    updated_run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    return await _build_run_response(db=db, service=service, run=updated_run)


@router.post(
    "/runs/{run_id}/continue-chat",
    response_model=ContinueChatResponse,
    dependencies=[Depends(RequirePermission(CHAT_WORKFLOWS_RUN))],
)
async def continue_chat(
    run_id: str,
    user_context: dict[str, Any] = Depends(_get_user_context),
    db: ChatWorkflowsDatabase = Depends(_get_db),
) -> dict[str, Any]:
    """Create or return the free-chat handoff conversation for a completed run."""
    run = _require_run_access(
        await _db_call_async(db, "get_run", run_id),
        user_context=user_context,
    )
    if run.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow must be completed before continuing to free chat",
        )

    conversation_id = run.get("free_chat_conversation_id")
    if not conversation_id:
        conversation_id = str(uuid4())
        await _db_call_async(db, "update_run", run_id, free_chat_conversation_id=conversation_id)
        await _db_call_async(
            db,
            "append_event",
            run_id,
            "continued_to_free_chat",
            {"conversation_id": conversation_id},
        )

    return {"conversation_id": conversation_id}
