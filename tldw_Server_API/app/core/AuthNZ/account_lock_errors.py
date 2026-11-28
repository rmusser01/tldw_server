"""Account-lock errors and constants (TASK-15679).

A leaf module (fastapi and ``core.AuthNZ.exceptions`` only): ``sessions_repo`` and ``session_manager``
raise these, and ``account_lock`` calls the session manager.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError

ACCOUNT_LOCKED_CODE = "account_locked"
ACCOUNT_LOCK_REVOKE_REASON = "account_locked"
ACCOUNT_RESTORED_REVOKE_REASON = "account_restored"


class AccountSecurityLockedError(HTTPException):
    """The account's owner locked it; only the link mailed to its address unlocks it (423)."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_423_LOCKED,
            detail={
                "code": ACCOUNT_LOCKED_CODE,
                "message": "This account is locked. Use the link sent to its email address to unlock it.",
            },
        )


class StaleCredentialError(InvalidTokenError):
    """A token issued before ``credentials_valid_after``."""

    def __init__(self) -> None:
        super().__init__("issued before the account's credentials were last reset")


class TakeBackUnavailableError(ValueError):
    """The previous address now belongs to another account. Nothing was changed; the account is still locked."""
