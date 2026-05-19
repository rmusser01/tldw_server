"""
Personalization dependencies: per-user DB access and event logger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Request
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
    UsageEvent,
)


def get_personalization_db_for_user(user: User = Depends(get_request_user)) -> PersonalizationDB:
    """Return a PersonalizationDB instance bound to the current user's DB path."""
    # Accept both numeric and string IDs in tests/single-user flows
    try:
        uid = int(user.id)
    except Exception:
        # Derive a stable numeric from string id (e.g., "test_user")
        try:
            import hashlib
            # Deterministic non-crypto ID derivation for non-integer test/single-user IDs.
            # `usedforsecurity=False` keeps behavior while making intent explicit.
            try:
                digest = hashlib.sha1(str(user.id).encode("utf-8"), usedforsecurity=False).digest()
            except TypeError:  # pragma: no cover - compatibility fallback
                digest = hashlib.sha1(str(user.id).encode("utf-8")).digest()  # nosec B324
            uid = int.from_bytes(digest[:4], byteorder="big", signed=False)
        except Exception:
            uid = 0
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
