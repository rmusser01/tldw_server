"""Self-service email change gate: the current password is required (TASK-15640)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ import password_service, settings


class EmailChangeRefused(Exception):
    """A self-service email change is refused; ``code`` is the API error code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


def _refuse(user_id: int, code: str, detail: str) -> EmailChangeRefused:
    logger.info("Self-service email change refused for user {}: {}", user_id, code)
    return EmailChangeRefused(code, detail)


async def authorize_own_email_change(
    *,
    db_conn: Any,
    is_postgres: bool,
    user_id: int,
    new_emails: Sequence[str],
    current_password: str | None,
) -> None:
    """Return when the user may change their own email address; raise ``EmailChangeRefused`` otherwise.

    Reads only on ``db_conn``: a profile update must fit in its existing PostgreSQL transaction slot.
    ``new_emails`` holds every ``identity.email`` value in the request; one that differs from the
    stored address (ignoring case and surrounding whitespace) requires the password.
    """
    if settings.is_single_user_mode():
        # The API key is the only credential in single-user mode, so there is no password to ask for.
        return
    email, password_hash = await password_service.fetch_email_and_password_hash(
        db_conn, user_id, is_postgres=is_postgres
    )
    stored = (email or "").strip().lower()
    if all(str(value).strip().lower() == stored for value in new_emails):
        return
    if not password_hash:
        raise _refuse(
            user_id,
            "email_change_requires_admin",
            "This account has no password. Ask an administrator to change its email address.",
        )
    if not current_password:
        raise _refuse(user_id, "current_password_required", "Enter your current password to change your email address.")
    if not password_service.verify_password(current_password, password_hash)[0]:
        raise _refuse(
            user_id,
            "current_password_invalid",
            "Current password is incorrect. If you sign in with SSO or an email link, use Forgot password to set one first.",
        )
