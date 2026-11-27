"""List and attach native conversations for a live Document Workspace item."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user, require_expected_user_if_present
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.document_chat_scope import (
    document_chat_unavailable,
    resolve_document_chat_target,
)
from tldw_Server_API.app.api.v1.schemas.document_chat_schemas import (
    DocumentChatLinkRequest,
    DocumentChatPage,
    DocumentChatSummary,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

router = APIRouter(tags=["Document Workspace"], dependencies=[Depends(require_expected_user_if_present)])


@router.get("/{media_id:int}/chat-conversations")
async def list_document_chats(
    request: Request,
    media_id: Annotated[int, Path(gt=0)],
    media_uuid: UUID,
    current_user: Annotated[User, Depends(get_request_user)],
    db: Annotated[CharactersRAGDB, Depends(get_chacha_db_for_user)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=2**31 - 1)] = 0,
) -> DocumentChatPage:
    """List only owned conversations with saved messages."""
    await resolve_document_chat_target(request, current_user, media_id=media_id, media_uuid=media_uuid)
    rows = await run_in_threadpool(
        db.conversation_store.list_document_chats,
        owner_id=str(current_user.id), media_uuid=str(media_uuid), limit=limit + 1, offset=offset,
    )
    return DocumentChatPage(
        items=[DocumentChatSummary.model_validate(row) for row in rows[:limit]], has_more=len(rows) > limit,
    )


@router.put("/{media_id:int}/chat-conversations/{chat_id}", status_code=204)
async def attach_document_chat(
    request: Request,
    media_id: Annotated[int, Path(gt=0)],
    chat_id: str,
    link: DocumentChatLinkRequest,
    current_user: Annotated[User, Depends(get_request_user)],
    db: Annotated[CharactersRAGDB, Depends(get_chacha_db_for_user)],
) -> Response:
    """Attach an owned, unchanged Chat record or acknowledge its existing link."""
    await resolve_document_chat_target(request, current_user, media_id=media_id, media_uuid=link.media_uuid)
    try:
        await run_in_threadpool(
            db.conversation_store.attach_document_chat, chat_id, owner_id=str(current_user.id),
            media_uuid=str(link.media_uuid), expected_record=link.document_chat_record,
        )
    except NotFoundError:
        raise document_chat_unavailable() from None
    except ConflictError:
        raise HTTPException(status_code=409, detail="This chat is already linked to a document") from None
    return Response(status_code=204)
