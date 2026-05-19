"""
Guardian & Self-Monitoring dependencies: per-user DB access.
"""
from __future__ import annotations

from fastapi import Depends

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.DB_Management.guardian_db_resolver import (
    resolve_guardian_db_for_user_id,
)


def get_guardian_db_for_user_id(user_id: object) -> GuardianDB:
    """Return a GuardianDB instance bound to the provided user ID."""
    return resolve_guardian_db_for_user_id(user_id)


def get_guardian_db_for_user(user: User = Depends(get_request_user)) -> GuardianDB:
    """Return a GuardianDB instance bound to the current user's DB path."""
    return get_guardian_db_for_user_id(user.id)
