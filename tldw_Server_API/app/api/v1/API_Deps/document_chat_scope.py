"""Resolve an optional Document Workspace target through the current Media session."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user


@dataclass(frozen=True)
class DocumentChatTarget:
    """An authorized numeric Media ID and its durable UUID."""

    media_id: int
    media_uuid: UUID


def document_chat_unavailable() -> HTTPException:
    """Use the same response for missing, inaccessible and replaced targets."""
    return HTTPException(status_code=404, detail="Document chat unavailable")


async def resolve_document_chat_target(
    request: Request, current_user: User, *, media_id: int, media_uuid: UUID,
) -> DocumentChatTarget:
    """Require current access to the exact live Media record."""
    async with asynccontextmanager(get_media_db_for_user)(request, current_user) as db:
        record = await run_in_threadpool(db.get_media_by_id, media_id, include_deleted=False, include_trash=False)
    if not record or str(record.get("uuid")) != str(media_uuid):
        raise document_chat_unavailable()
    return DocumentChatTarget(media_id, media_uuid)


async def _document_chat_query(
    request: Request,
    current_user: Annotated[User, Depends(get_request_user)],
    document_media_id: Annotated[int | None, Query(gt=0)] = None,
    document_media_uuid: Annotated[UUID | None, Query()] = None,
) -> DocumentChatTarget | None:
    """Ordinary Chat requests require no additional database session."""
    if document_media_id is None and document_media_uuid is None:
        return None
    if document_media_id is None or document_media_uuid is None:
        raise HTTPException(status_code=422, detail="Both document media ID and UUID are required")
    return await resolve_document_chat_target(
        request, current_user, media_id=document_media_id, media_uuid=document_media_uuid,
    )


DocumentChatQuery = Annotated[DocumentChatTarget | None, Depends(_document_chat_query)]
