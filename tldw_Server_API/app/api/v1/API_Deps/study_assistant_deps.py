"""Authenticated request-scoped guidance for synchronous Study Assistant replies."""

import asyncio
from typing import Any

from fastapi import Depends, HTTPException, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps
from tldw_Server_API.app.api.v1.API_Deps.usage_quota_deps import enforce_llm_quota_on_own_key
from tldw_Server_API.app.api.v1.schemas.flashcards import StudyAssistantRespondRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_target_resolution import resolve_chat_target
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError
from tldw_Server_API.app.core.Flashcards.study_assistant import resolve_study_assistant_guidance
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError


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


async def study_assistant_own_key(
    payload: StudyAssistantRespondRequest,
    current_user: User = Depends(get_request_user),
) -> Any:
    """The LLM quota for a study assistant reply, naming its provider; then the user's own key for it, or None.

    An over-limit user whose provider runs on their own key is let through unmarked (TASK-15426). A request
    with no usable provider names none, so an over-limit user gets the 402 before the reply's 400, as before.
    """
    try:
        target = resolve_chat_target(requested_provider=payload.provider, requested_model=payload.model)
    except (ChatConfigurationError, ByokResolutionError):
        return await enforce_llm_quota_on_own_key(current_user.id, None)
    return await enforce_llm_quota_on_own_key(current_user.id, target.provider, model=target.model)
