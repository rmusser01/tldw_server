"""Chat loop endpoints for run start, replay, and approval/cancel actions."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from tldw_Server_API.app.api.v1.schemas.chat_loop_schemas import (
    ChatLoopActionResponse,
    ChatLoopApprovalDecisionRequest,
    ChatLoopEventsResponse,
    ChatLoopStartRequest,
    ChatLoopStartResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Chat.chat_loop_store import InMemoryChatLoopStore

router = APIRouter()
_store = InMemoryChatLoopStore()
_run_owners: dict[str, str] = {}


def _assert_run_owner(run_id: str, user_id: str) -> None:
    owner = _run_owners.get(run_id)
    if owner is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    if owner != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")


@router.post("/chat/loop/start", response_model=ChatLoopStartResponse)
async def start_chat_loop_run(
    payload: ChatLoopStartRequest,
    current_user: User = Depends(get_request_user),
) -> ChatLoopStartResponse:
    user_id = str(current_user.id)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    _run_owners[run_id] = user_id
    _store.append(
        run_id,
        "run_started",
        {
            "user_id": user_id,
            "messages_count": len(payload.messages),
        },
    )
    return ChatLoopStartResponse(run_id=run_id)


@router.get("/chat/loop/{run_id}/events", response_model=ChatLoopEventsResponse)
async def list_chat_loop_events(
    run_id: str,
    after_seq: int = Query(0, ge=0),
    _current_user: User = Depends(get_request_user),
) -> ChatLoopEventsResponse:
    _assert_run_owner(run_id, str(_current_user.id))
    events = _store.list_after(run_id, after_seq)
    return ChatLoopEventsResponse(run_id=run_id, events=events)


@router.post("/chat/loop/{run_id}/approve", response_model=ChatLoopActionResponse)
async def approve_chat_loop_call(
    run_id: str,
    payload: ChatLoopApprovalDecisionRequest,
    _current_user: User = Depends(get_request_user),
) -> ChatLoopActionResponse:
    _assert_run_owner(run_id, str(_current_user.id))
    _store.append(
        run_id,
        "approval_resolved",
        {"approval_id": payload.approval_id, "decision": "approve"},
    )
    return ChatLoopActionResponse(ok=True)


@router.post("/chat/loop/{run_id}/reject", response_model=ChatLoopActionResponse)
async def reject_chat_loop_call(
    run_id: str,
    payload: ChatLoopApprovalDecisionRequest,
    _current_user: User = Depends(get_request_user),
) -> ChatLoopActionResponse:
    _assert_run_owner(run_id, str(_current_user.id))
    _store.append(
        run_id,
        "approval_resolved",
        {"approval_id": payload.approval_id, "decision": "reject"},
    )
    return ChatLoopActionResponse(ok=True)


@router.post("/chat/loop/{run_id}/cancel", response_model=ChatLoopActionResponse)
async def cancel_chat_loop_run(
    run_id: str,
    _current_user: User = Depends(get_request_user),
) -> ChatLoopActionResponse:
    _assert_run_owner(run_id, str(_current_user.id))
    _store.append(run_id, "run_cancelled", {})
    return ChatLoopActionResponse(ok=True)
