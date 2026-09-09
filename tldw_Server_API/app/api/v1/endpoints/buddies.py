"""Authenticated independent Buddy profile and client attachment endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, check_rate_limit, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import buddy_turns
from tldw_Server_API.app.api.v1.schemas.buddies import (
    BuddyAcknowledgement,
    BuddyActivityList,
    BuddyAttachmentResponse,
    BuddyAttachmentUpdate,
    BuddyConversationList,
    BuddyConversationSummary,
    BuddyCreate,
    BuddyList,
    BuddyProfile,
    BuddyReplySettings,
    BuddyUpdate,
)
from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.exceptions import BuddyConflictError, BuddyNotFoundError
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualServiceError
from tldw_Server_API.app.core.Persona.visual_starter_catalog import PersonaVisualStarterCatalogError

router = APIRouter(dependencies=[Depends(check_rate_limit)])
router.include_router(buddy_turns.router)
ClientSlot = Annotated[str, Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]


def get_buddy_service(
    user: User = Depends(get_request_user),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> BuddyService:
    """Resolve principal and content database independently of client slot IDs."""
    user_id = str(user.id if user.id is not None else "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return BuddyService(db, user_id)


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except BuddyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BuddyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PersonaVisualStarterCatalogError as exc:
        status_code = 404 if exc.code == "starter_pack_not_found" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except (PersonaVisualServiceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=BuddyList)
def list_buddies(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: BuddyService = Depends(get_buddy_service),
) -> dict:
    """List a bounded page of the principal's Buddy profiles."""
    with _errors():
        return service.list_profiles(limit=limit, offset=offset)


@router.post("", response_model=BuddyProfile, status_code=201)
def create_buddy(body: BuddyCreate, service: BuddyService = Depends(get_buddy_service)) -> dict:
    """Create an independent immutable artwork snapshot."""
    with _errors():
        return service.create(body)


@router.get("/conversation-targets/{conversation_id}", response_model=BuddyConversationSummary)
def conversation_target(conversation_id: str, service: BuddyService = Depends(get_buddy_service)) -> dict:
    """Resolve an owned conversation's current scope before attachment."""
    with _errors():
        return service.conversation_summary(conversation_id)


@router.get("/conversation-targets/{conversation_id}/reply-settings", response_model=BuddyReplySettings)
def conversation_reply_settings(
    conversation_id: str,
    response: Response,
    client_slot: ClientSlot = "default",
    service: BuddyService = Depends(get_buddy_service),
) -> dict:
    """Read the attached target's configured reply model without accepting work."""
    with _errors():
        result = service.conversation_reply_settings(client_slot, conversation_id)
    response.headers["Cache-Control"] = "private, no-store"
    return result


@router.get("/attachment", response_model=BuddyAttachmentResponse)
def get_attachment(client_slot: ClientSlot = "default", service: BuddyService = Depends(get_buddy_service)) -> dict:
    """Read one preference slot after checking its target's current access."""
    with _errors():
        return service.attachment(client_slot)


@router.put("/attachment", response_model=BuddyAttachmentResponse)
def put_attachment(
    body: BuddyAttachmentUpdate, client_slot: ClientSlot = "default", service: BuddyService = Depends(get_buddy_service)
) -> dict:
    """Replace one slot without changing or creating its conversation."""
    with _errors():
        return service.set_attachment(
            client_slot,
            expected_version=body.expected_version,
            attachment=body.model_dump(exclude={"expected_version"}),
        )


@router.delete("/attachment", response_model=BuddyAttachmentResponse)
def delete_attachment(
    expected_version: int = Query(ge=0),
    client_slot: ClientSlot = "default",
    service: BuddyService = Depends(get_buddy_service),
) -> dict:
    """Detach without stopping conversation work or resetting slot revision."""
    with _errors():
        return service.set_attachment(client_slot, expected_version=expected_version, attachment=None)


@router.get("/attachment/conversations", response_model=BuddyConversationList)
def attachment_conversations(
    client_slot: ClientSlot = "default",
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: BuddyService = Depends(get_buddy_service),
) -> dict:
    """Project only the exact attached conversation or owned workspace chats."""
    with _errors():
        return service.conversations(client_slot, limit=limit, offset=offset)


@router.get("/attachment/activity", response_model=BuddyActivityList)
def attachment_activity(
    client_slot: ClientSlot = "default",
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: BuddyService = Depends(get_buddy_service),
) -> dict:
    """Batch latest persisted results without issuing per-chat transcript calls."""
    with _errors():
        return service.activity(client_slot, limit=limit, offset=offset)


@router.post("/attachment/acknowledgements")
def acknowledge_result(
    body: BuddyAcknowledgement, client_slot: ClientSlot = "default", service: BuddyService = Depends(get_buddy_service)
) -> dict[str, bool]:
    """Acknowledge an exact existing result within the current attachment."""
    with _errors():
        return service.acknowledge(client_slot, conversation_id=body.conversation_id, message_id=body.result_message_id)


@router.get("/{buddy_id}", response_model=BuddyProfile)
def get_buddy(buddy_id: str, service: BuddyService = Depends(get_buddy_service)) -> dict:
    """Read a principal-owned Buddy without depending on its source Persona."""
    with _errors():
        return service.get(buddy_id)


@router.patch("/{buddy_id}", response_model=BuddyProfile)
def update_buddy(buddy_id: str, body: BuddyUpdate, service: BuddyService = Depends(get_buddy_service)) -> dict:
    """Update profile preferences with optimistic concurrency."""
    with _errors():
        return service.update(
            buddy_id,
            expected_version=body.expected_version,
            changes=body.model_dump(exclude_unset=True, exclude={"expected_version"}),
        )


@router.delete("/{buddy_id}", status_code=204)
def delete_buddy(
    buddy_id: str, expected_version: int = Query(ge=1), service: BuddyService = Depends(get_buddy_service)
) -> Response:
    """Remove a profile; existing attachment reads then report unavailable."""
    with _errors():
        service.delete(buddy_id, expected_version=expected_version)
    return Response(status_code=204)


@router.get("/{buddy_id}/assets/{asset_id}/content")
def get_buddy_asset(buddy_id: str, asset_id: str, service: BuddyService = Depends(get_buddy_service)) -> Response:
    """Read bounded immutable bytes after ownership and checksum validation."""
    with _errors():
        content, mime_type = service.read_asset(buddy_id, asset_id)
    return Response(
        content,
        media_type=mime_type,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )
