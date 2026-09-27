"""Helpers shared by the Prompt Studio repositories."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.Prompts_DB import ConflictError

# What a session can raise from SQL on either backend.
DB_ERRORS = (BackendDatabaseError, sqlite3.Error)


def json_or_none(value: Any) -> Optional[str]:
    return json.dumps(value) if value is not None else None


def is_unique_violation(exc: BaseException) -> bool:
    """SQLite raises IntegrityError("UNIQUE ..."); the PostgreSQL session raises ConflictError."""
    if isinstance(exc, ConflictError):
        return True
    return isinstance(exc, sqlite3.IntegrityError) and "UNIQUE" in str(exc)
