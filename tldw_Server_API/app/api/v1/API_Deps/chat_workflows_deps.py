import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from cachetools import LRUCache
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.ChatWorkflows_DB import ChatWorkflowsDatabase
from tldw_Server_API.app.core.DB_Management.DB_Manager import (
    create_chat_workflows_database,
    get_content_backend_instance,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


MAX_CACHED_INSTANCES = settings.get("MAX_CACHED_CHAT_WORKFLOWS_DB_INSTANCES", 20)
_APP_STATE_KEY = "_chat_workflows_deps_state"

_CHAT_WORKFLOWS_DB_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    RuntimeError,
    AttributeError,
)


@dataclass
class _ChatWorkflowsDepsState:
    """App-scoped chat workflows dependency state."""

    cache: LRUCache
    lock: Any


def _get_chat_workflows_db_path_for_user(user_id: str) -> Path:
    return DatabasePaths.get_chat_workflows_db_path(user_id)


def _get_state(app: FastAPI) -> _ChatWorkflowsDepsState:
    """Return the app-scoped chat-workflows dependency state, creating it lazily."""
    state = getattr(app.state, _APP_STATE_KEY, None)
    if state is None:
        state = _ChatWorkflowsDepsState(
            cache=LRUCache(maxsize=MAX_CACHED_INSTANCES),
            lock=threading.Lock(),
        )
        setattr(app.state, _APP_STATE_KEY, state)
    return state


def _get_or_create_chat_workflows_db(
    app: FastAPI,
    user_id: str,
    client_id: str,
) -> ChatWorkflowsDatabase:
    db_path = _get_chat_workflows_db_path_for_user(user_id)
    backend = get_content_backend_instance()
    state = _get_state(app)

    backend_signature = "sqlite"
    if backend is not None:
        backend_cfg = getattr(backend, "config", None)
        if backend_cfg is not None:
            backend_signature = (
                backend.backend_type.value
                + ":"
                + (
                    backend_cfg.connection_string
                    or backend_cfg.sqlite_path
                    or backend_cfg.pg_database
                    or "default"
                )
            )
        else:
            backend_signature = f"{backend.backend_type.value}:{id(backend)}"

    cache_key = (str(db_path), backend_signature)

    with state.lock:
        if cache_key in state.cache:
            logger.debug("Using cached ChatWorkflowsDatabase for user {}", user_id)
            return state.cache[cache_key]

        try:
            db_instance = create_chat_workflows_database(
                client_id=client_id,
                db_path=db_path,
                backend=backend,
            )
            state.cache[cache_key] = db_instance
            logger.info("Created new ChatWorkflowsDatabase instance for user {}", user_id)
            return db_instance
        except _CHAT_WORKFLOWS_DB_EXCEPTIONS as exc:
            logger.error("Failed to create ChatWorkflowsDatabase (error_type={})", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize chat workflows database",
            ) from exc


async def get_chat_workflows_user(
    request: Request,
    x_client_id: Optional[str] = Header(None),
) -> dict[str, Any]:
    authz = request.headers.get("Authorization")
    api_key_hdr = request.headers.get("X-API-KEY")

    if not authz and not api_key_hdr:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    legacy_token_header = request.headers.get("Token")
    bearer_token: str | None = None
    if authz and isinstance(authz, str):
        scheme, _, credential = authz.partition(" ")
        if scheme.lower() == "bearer":
            bearer_token = credential.strip() or None

    current_user: User = await get_request_user(
        request,
        api_key=api_key_hdr,
        token=bearer_token,
        legacy_token_header=legacy_token_header,
    )

    roles_raw = getattr(current_user, "roles", []) or []
    normalized_roles = {role.lower() for role in roles_raw if isinstance(role, str)}
    permissions = list(getattr(current_user, "permissions", []) or [])
    normalized_permissions = {
        permission.lower()
        for permission in permissions
        if isinstance(permission, str) and permission.strip()
    }
    is_admin = bool(
        getattr(current_user, "is_admin", False)
        or ("admin" in normalized_roles)
        or ("*" in normalized_permissions)
        or ("system.configure" in normalized_permissions)
    )

    user_context: dict[str, Any] = {
        "user_id": str(getattr(current_user, "id", "anonymous")),
        "username": getattr(current_user, "username", None),
        "client_id": x_client_id or "web",
        "is_authenticated": True,
        "is_admin": is_admin,
        "permissions": permissions,
        "rg_policy_id": getattr(request.state, "rg_policy_id", None),
    }
    request.state.user_context = user_context
    return user_context


async def get_chat_workflows_db(
    request: Request,
    user_context: dict[str, Any] = Depends(get_chat_workflows_user),
) -> ChatWorkflowsDatabase:
    """Resolve or create the caller's chat workflows DB adapter off the event loop."""
    user_id = user_context["user_id"]
    client_id = user_context["client_id"]
    return await asyncio.to_thread(
        _get_or_create_chat_workflows_db,
        request.app,
        user_id,
        client_id,
    )


def shutdown_chat_workflows_deps(app: FastAPI | None = None) -> None:
    """Close app-scoped chat workflows DB instances and clear their cache."""
    if app is None:
        return

    state = getattr(app.state, _APP_STATE_KEY, None)
    if state is None:
        return

    with state.lock:
        for db_instance in list(state.cache.values()):
            try:
                db_instance.close()
            except _CHAT_WORKFLOWS_DB_EXCEPTIONS as exc:
                logger.error(
                    "Failed to close ChatWorkflowsDatabase during shutdown (error_type={})",
                    type(exc).__name__,
                )
        state.cache.clear()

    delattr(app.state, _APP_STATE_KEY)
