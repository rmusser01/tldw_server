"""Explicit feedback endpoints shared by chat and RAG."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, User

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.feedback_schemas import (
    ErrorDetail,
    ExplicitFeedbackRequest,
    ExplicitFeedbackResponse,
    FeedbackDeleteResponse,
    FeedbackListResponse,
    FeedbackRecord,
    FeedbackUpdateRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import UnifiedFeedbackSystem, UserFeedbackStore

router = APIRouter()

# Feedback idempotency is persisted in the user ChaCha DB so dedupe is shared
# across workers/processes.
_IDEMPOTENCY_WINDOW_SECONDS = 300


@dataclass
class _IdempotencyRecord:
    feedback_id: str | None
    issues: list[str]
    user_notes: str | None
    pending: bool = False


def _normalize_text_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _merge_issues(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    seen = set()
    for item in existing + incoming:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def _get_feedback_store(db: CharactersRAGDB) -> UserFeedbackStore:
    return UserFeedbackStore(db)


async def _reserve_idempotency_record(
    db: CharactersRAGDB,
    key: str,
    issues: list[str],
    user_notes: str | None,
) -> tuple[bool, _IdempotencyRecord]:
    store = _get_feedback_store(db)
    reserved, record = await store.claim_idempotency(
        key,
        issues,
        user_notes,
        ttl_seconds=_IDEMPOTENCY_WINDOW_SECONDS,
    )
    return reserved, _IdempotencyRecord(
        feedback_id=record.get("feedback_id"),
        issues=_normalize_text_list(record.get("issues")),
        user_notes=record.get("user_notes"),
        pending=bool(record.get("pending")),
    )


async def _finalize_idempotency_record(
    db: CharactersRAGDB,
    key: str,
    feedback_id: str | None,
    fallback_issues: list[str],
    fallback_user_notes: str | None,
) -> tuple[list[str], str | None, bool]:
    store = _get_feedback_store(db)
    return await store.finalize_idempotency(
        key,
        feedback_id,
        fallback_issues,
        fallback_user_notes,
    )


async def _clear_idempotency_record(db: CharactersRAGDB, key: str) -> None:
    store = _get_feedback_store(db)
    await store.clear_idempotency(key)


async def _update_idempotency_record(
    db: CharactersRAGDB,
    key: str,
    issues: list[str],
    user_notes: str | None,
) -> None:
    store = _get_feedback_store(db)
    await store.update_idempotency(key, issues, user_notes)


def _build_dedupe_key(
    *,
    user_key: str,
    request: ExplicitFeedbackRequest,
    conversation_id: str | None,
    query: str,
    document_ids: list[str],
    chunk_ids: list[str],
) -> str:
    if request.idempotency_key:
        return f"idem:{user_key}:{request.idempotency_key}"
    if request.message_id:
        return (
            f"chat:{user_key}:{conversation_id or ''}:{request.message_id}:"
            f"{request.feedback_type}:{request.helpful}:{request.relevance_score}"
        )
    normalized_query = str(query or "")
    if normalized_query.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required when message_id is not provided",
        )
    doc_key = ",".join(sorted(document_ids)) if document_ids else ""
    chunk_key = ",".join(sorted(chunk_ids)) if chunk_ids else ""
    corpus = request.corpus or ""
    return (
        f"rag:{user_key}:{normalized_query}:{request.feedback_type}:"
        f"{request.helpful}:{request.relevance_score}:{doc_key}:{chunk_key}:{corpus}"
    )


def _ensure_conversation_owner(conversation: dict, current_user: User) -> None:
    conv_client_id = conversation.get("client_id")
    user_id = current_user.id
    if conv_client_id is None or user_id is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this conversation")
    if str(conv_client_id) != str(user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this conversation")


@router.post(
    "/explicit",
    response_model=ExplicitFeedbackResponse,
    summary="Submit explicit feedback (chat + RAG)",
    dependencies=[Depends(check_rate_limit)],
    responses={
        400: {"model": ErrorDetail, "description": "Bad request (empty query, mismatched message)"},
        403: {"model": ErrorDetail, "description": "Forbidden – not the conversation/message owner"},
        404: {"model": ErrorDetail, "description": "Conversation or message not found"},
        422: {"model": ErrorDetail, "description": "Validation error (missing required fields)"},
    },
)
async def submit_explicit_feedback(
    payload: ExplicitFeedbackRequest,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ExplicitFeedbackResponse:
    resolved_conversation_id = payload.conversation_id
    message = None

    if payload.message_id:
        message = db.get_message_by_id(payload.message_id)
        if not message or message.get("deleted"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
        message_owner = message.get("user_id") or message.get("owner_id") or message.get("client_id")
        if message_owner is not None and str(message_owner) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not allowed to submit feedback for this message",
            )
        if not resolved_conversation_id:
            resolved_conversation_id = message.get("conversation_id")
        if resolved_conversation_id != message.get("conversation_id"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is not in conversation",
            )

    if resolved_conversation_id:
        conversation = db.get_conversation_by_id(resolved_conversation_id)
        if not conversation or conversation.get("deleted"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        _ensure_conversation_owner(conversation, current_user)

    resolved_query = payload.query
    if (resolved_query is None or str(resolved_query).strip() == "") and message is not None:
        resolved_query = message.get("content") or ""
    if resolved_query is None:
        resolved_query = ""
    resolved_query = str(resolved_query)

    issues = _normalize_text_list(payload.issues)
    document_ids = _normalize_text_list(payload.document_ids)
    chunk_ids = _normalize_text_list(payload.chunk_ids)
    user_key = str(current_user.id)

    dedupe_key = _build_dedupe_key(
        user_key=user_key,
        request=payload,
        conversation_id=resolved_conversation_id,
        query=resolved_query,
        document_ids=document_ids,
        chunk_ids=chunk_ids,
    )

    reserved, existing = await _reserve_idempotency_record(db, dedupe_key, issues, payload.user_notes)
    if not reserved:
        if issues or payload.user_notes is not None:
            merged_issues = _merge_issues(existing.issues, issues)
            updated_notes = existing.user_notes if payload.user_notes is None else payload.user_notes
            if existing.feedback_id and not existing.pending:
                collector = UnifiedFeedbackSystem(chacha_db=db)
                if collector.user_feedback:
                    try:
                        await collector.user_feedback.merge_feedback_update(
                            existing.feedback_id,
                            issues=merged_issues or None,
                            user_notes=updated_notes,
                        )
                    except (CharactersRAGDBError, HTTPException, ValueError) as e:
                        logger.exception("Failed to merge feedback update")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="merge_feedback_update_failed",
                        ) from e
            await _update_idempotency_record(db, dedupe_key, merged_issues, updated_notes)
        return ExplicitFeedbackResponse(ok=True, feedback_id=existing.feedback_id)

    collector = UnifiedFeedbackSystem(chacha_db=db)
    try:
        result = await collector.submit_feedback(
            conversation_id=resolved_conversation_id or "",
            query=resolved_query,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
            feedback_type=payload.feedback_type,
            relevance_score=payload.relevance_score,
            helpful=payload.helpful,
            issues=issues or None,
            user_notes=payload.user_notes,
            session_id=payload.session_id,
            _user_id=current_user.username if current_user else None,
            message_id=payload.message_id,
        )
    except (HTTPException, ValueError, RuntimeError):
        await _clear_idempotency_record(db, dedupe_key)
        raise
    except Exception as exc:
        await _clear_idempotency_record(db, dedupe_key)
        logger.exception("Unexpected error in submit_feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc

    if result.get("errors"):
        logger.warning("Explicit feedback completed with errors")

    feedback_id = result.get("feedback_id")
    if resolved_conversation_id and not feedback_id:
        await _clear_idempotency_record(db, dedupe_key)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not record feedback")

    final_issues, final_user_notes, has_pending_merge = await _finalize_idempotency_record(
        db,
        dedupe_key,
        feedback_id,
        issues,
        payload.user_notes,
    )

    # If duplicates arrived while the record was pending, merge those updates now
    # so the persisted row reflects the final idempotent payload.
    if feedback_id and has_pending_merge and collector.user_feedback:
        try:
            await collector.user_feedback.merge_feedback_update(
                feedback_id,
                issues=final_issues or None,
                user_notes=final_user_notes,
            )
        except (CharactersRAGDBError, HTTPException, ValueError) as e:
            logger.exception("Failed to finalize idempotency merge")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="merge_feedback_update_failed",
            ) from e

    return ExplicitFeedbackResponse(ok=True, feedback_id=feedback_id)


# ---------------------------------------------------------------------------
# GET  /feedback  – list feedback for a conversation
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=FeedbackListResponse,
    summary="List feedback for a conversation",
    dependencies=[Depends(check_rate_limit)],
    responses={
        403: {"model": ErrorDetail, "description": "Forbidden – not the conversation owner"},
        404: {"model": ErrorDetail, "description": "Conversation not found"},
    },
)
async def list_feedback(
    conversation_id: str,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FeedbackListResponse:
    try:
        conversation = db.get_conversation_by_id(conversation_id)
        if not conversation or conversation.get("deleted"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        _ensure_conversation_owner(conversation, current_user)

        store = UnifiedFeedbackSystem(chacha_db=db)
        if not store.user_feedback:
            return FeedbackListResponse(ok=True, feedback=[])

        rows = await store.user_feedback.get_conversation_feedback(conversation_id)
        records = [FeedbackRecord(**row) for row in rows]
        return FeedbackListResponse(ok=True, feedback=records)
    except HTTPException:
        raise
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to list feedback") from exc


# ---------------------------------------------------------------------------
# GET  /feedback/{feedback_id}  – get a single feedback entry
# ---------------------------------------------------------------------------

@router.get(
    "/{feedback_id}",
    response_model=FeedbackRecord,
    summary="Get a feedback entry",
    dependencies=[Depends(check_rate_limit)],
    responses={
        403: {"model": ErrorDetail, "description": "Forbidden – not the conversation owner"},
        404: {"model": ErrorDetail, "description": "Feedback record not found"},
    },
)
async def get_feedback(
    feedback_id: str,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FeedbackRecord:
    store = UnifiedFeedbackSystem(chacha_db=db)
    if not store.user_feedback:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")

    record = await store.user_feedback.get_feedback_by_id(feedback_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")

    conv_id = record.get("conversation_id")
    if conv_id:
        conversation = db.get_conversation_by_id(conv_id)
        if not conversation or conversation.get("deleted"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")
        _ensure_conversation_owner(conversation, current_user)

    return FeedbackRecord(**record)


# ---------------------------------------------------------------------------
# DELETE  /feedback/{feedback_id}  – retract a feedback entry
# ---------------------------------------------------------------------------

@router.delete(
    "/{feedback_id}",
    response_model=FeedbackDeleteResponse,
    summary="Delete a feedback entry",
    dependencies=[Depends(check_rate_limit)],
    responses={
        403: {"model": ErrorDetail, "description": "Forbidden – not the conversation owner"},
        404: {"model": ErrorDetail, "description": "Feedback record not found"},
    },
)
async def delete_feedback(
    feedback_id: str,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> FeedbackDeleteResponse:
    try:
        store = UnifiedFeedbackSystem(chacha_db=db)
        if not store.user_feedback:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")

        record = await store.user_feedback.get_feedback_by_id(feedback_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")

        # Validate ownership via conversation
        conv_id = record.get("conversation_id")
        if conv_id:
            conversation = db.get_conversation_by_id(conv_id)
            if conversation:
                _ensure_conversation_owner(conversation, current_user)

        deleted = await store.user_feedback.delete_feedback(feedback_id)
        return FeedbackDeleteResponse(ok=True, deleted=deleted)
    except HTTPException:
        raise
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete feedback") from exc


# ---------------------------------------------------------------------------
# PATCH  /feedback/{feedback_id}  – update issues / user_notes
# ---------------------------------------------------------------------------

@router.patch(
    "/{feedback_id}",
    response_model=ExplicitFeedbackResponse,
    summary="Update a feedback entry (issues / user_notes)",
    dependencies=[Depends(check_rate_limit)],
    responses={
        403: {"model": ErrorDetail, "description": "Forbidden – not the conversation owner"},
        404: {"model": ErrorDetail, "description": "Feedback record not found"},
    },
)
async def update_feedback(
    feedback_id: str,
    payload: FeedbackUpdateRequest,
    current_user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ExplicitFeedbackResponse:
    try:
        store = UnifiedFeedbackSystem(chacha_db=db)
        if not store.user_feedback:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")

        record = await store.user_feedback.get_feedback_by_id(feedback_id)
        if not record:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback record not found")

        # Validate ownership via conversation
        conv_id = record.get("conversation_id")
        if conv_id:
            conversation = db.get_conversation_by_id(conv_id)
            if conversation:
                _ensure_conversation_owner(conversation, current_user)

        await store.user_feedback.merge_feedback_update(
            feedback_id,
            issues=payload.issues,
            user_notes=payload.user_notes,
        )
        return ExplicitFeedbackResponse(ok=True, feedback_id=feedback_id)
    except HTTPException:
        raise
    except CharactersRAGDBError as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update feedback") from exc
