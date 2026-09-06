"""Authenticated request-scoped guidance for synchronous Study Assistant replies."""

import asyncio

from fastapi import Depends, HTTPException, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps
from tldw_Server_API.app.api.v1.schemas.flashcards import StudyAssistantRespondRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError
from tldw_Server_API.app.core.Flashcards.study_assistant import resolve_study_assistant_guidance


async def get_study_assistant_guidance(
    request: Request,
    payload: StudyAssistantRespondRequest,
    current_user: User = Depends(get_request_user),
) -> str | None:
    """Capture selected-action guidance from authenticated-owner storage.

    Args:
        request: HTTP request used to acquire the owner's prompt database.
        payload: Validated response request selecting the Study Assistant action.
        current_user: Authenticated owner, shared with the content DB dependency.

    Returns:
        Immutable effective guidance, or None for fact-checking without reading
        prompt storage. Reads and connection cleanup run on the same worker.

    Raises:
        HTTPException: Preserves authentication/storage HTTP failures; maps
            resolution and cleanup failures to a prompt-safe HTTP 500 response.
    """
    if payload.action == "fact_check":
        return None
    try:
        database = await Prompts_DB_Deps.get_prompts_db_for_user(request, current_user)
        return await asyncio.to_thread(resolve_study_assistant_guidance, database, payload.action)
    except HTTPException:
        raise
    except (DatabaseError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        logger.error("Failed to load study assistant guidance (user_id={}, action={})", current_user.id, payload.action)
        raise HTTPException(status_code=500, detail="Failed to load study assistant guidance") from exc
