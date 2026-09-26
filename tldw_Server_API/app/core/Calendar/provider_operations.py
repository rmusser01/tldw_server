"""Scoped credentials, bounded blocking calls, and safe Calendar diagnostics."""

from __future__ import annotations

import inspect
import json
import traceback
from functools import partial
from typing import Any, Callable

from anyio import to_thread
from loguru import logger

from tldw_Server_API.app.core.Calendar.errors import CalendarPermissionDenied, CalendarValidationError
from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase


def resolve_caldav_credentials(
    db: CalendarDatabase,
    *,
    account_id: int,
    actor_user_id: int,
    tenant_id: str,
    overrides: dict[str, Any] | None = None,
    fallback_server_url: str | None = None,
) -> dict[str, str]:
    """Resolve active owner-scoped credentials, preferring explicit request values."""
    account = db.get_external_account(account_id)
    if account.user_id != actor_user_id or account.tenant_id != tenant_id:
        raise CalendarPermissionDenied("External calendar account is outside the current user scope")
    if account.provider.lower() != "caldav":
        raise CalendarValidationError("External calendar account is not a CalDAV account")
    if account.status != "active" or account.revoked_at or account.deleted_at:
        raise CalendarValidationError("External calendar account is not active")
    try:
        metadata = json.loads(account.account_metadata_json or "{}")
    except ValueError as exc:
        raise CalendarValidationError("External calendar account metadata is invalid") from exc
    if not isinstance(metadata, dict):
        raise CalendarValidationError("External calendar account metadata must be an object")
    stored = {}
    if account.secret_ref:
        stored = CalendarSecretStore(db=db, tenant_id=tenant_id).resolve_secret(
            owner_user_id=actor_user_id, secret_ref=account.secret_ref
        )
    request = overrides or {}
    values = {
        "server_url": request.get("server_url") or stored.get("server_url") or metadata.get("server_url") or fallback_server_url,
        "username": request.get("username") or stored.get("username") or metadata.get("username"),
        "password": request.get("password") or request.get("token") or stored.get("password") or stored.get("token"),
    }
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise CalendarValidationError("CalDAV account requires server_url, username, and password/token")
    return values


async def call_provider(operation: Callable[..., Any], **kwargs: Any) -> Any:
    """Run synchronous network work in AnyIO's bounded pool, also accepting async adapters."""
    if inspect.iscoroutinefunction(operation):
        return await operation(**kwargs)
    result = await to_thread.run_sync(partial(operation, **kwargs))
    return await result if inspect.isawaitable(result) else result


def log_calendar_failure(action: str, exc: Exception, **context: Any) -> None:
    """Log traceback locations and operation identifiers without exception text or locals."""
    frames = [(frame.filename, frame.lineno, frame.name) for frame in traceback.extract_tb(exc.__traceback__)]
    logger.bind(action=action, error_type=type(exc).__name__, frames=frames, **context).error(
        "Calendar operation failed: {} ({})", action, type(exc).__name__
    )
