"""FastAPI dependencies for Chat Macros endpoints."""

from __future__ import annotations

import asyncio

from fastapi import Depends, HTTPException, status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import CurrentPrincipal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.Chat_Macros.service import ChatMacrosService
from tldw_Server_API.app.core.Chat_Macros.storage import ChatMacroStorage
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


async def get_chat_macros_service(
    principal: CurrentPrincipal,
    chacha_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
) -> ChatMacrosService:
    """Return the Chat Macros service for the authenticated user."""
    if principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User identification failed for Chat Macros service.",
        )

    user_id = str(principal.user_id)
    user_base_dir = await asyncio.to_thread(
        DatabasePaths.get_user_base_directory,
        principal.user_id,
    )
    repository = ChatMacroRepository(chacha_db)
    await asyncio.to_thread(repository.ensure_ready)
    return ChatMacrosService(
        user_id=user_id,
        storage=ChatMacroStorage(user_base_dir),
        repository=repository,
    )
