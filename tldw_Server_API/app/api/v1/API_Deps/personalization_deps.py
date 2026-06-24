"""
Personalization dependencies: per-user DB access and event logger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Request
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
    UsageEvent,
)
from tldw_Server_API.app.core.Personalization.companion_user_ids import (
    resolve_existing_companion_storage_user_id,
)


def get_personalization_db_for_user(user: User = Depends(get_request_user)) -> PersonalizationDB:
    """Return a PersonalizationDB instance bound to the current user's DB path."""
    uid = resolve_existing_companion_storage_user_id(user.id)
    return PersonalizationDB.for_user(uid)


@dataclass
class UsageEventLogger:
    user_id: str
    db: PersonalizationDB

    def log_event(
        self,
        event_type: str,
        resource_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        try:
            evt = UsageEvent(user_id=self.user_id, type=event_type, resource_id=resource_id, tags=tags, metadata=metadata)
            return self.db.insert_usage_event(evt)
        except Exception:
            logger.debug("UsageEventLogger failed (non-fatal)")
            return None


def get_usage_event_logger(
    request: Request,
    user: User = Depends(get_request_user),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
) -> UsageEventLogger:
    return UsageEventLogger(user_id=str(user.id), db=db)
