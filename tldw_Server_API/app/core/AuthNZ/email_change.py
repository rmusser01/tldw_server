"""Self-service email change gate: the current password is required (TASK-15640)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jinja2
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import account_lock, password_service, settings
from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service
from tldw_Server_API.app.core.AuthNZ.transaction_hooks import defer_until_after_commit
from tldw_Server_API.app.core.DB_Management.user_profile_writes import email_recently_released_by_another_user

_NOTICE_NONCRITICAL_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError, KeyError, jinja2.TemplateError)


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

    A locked account raises ``AccountSecurityLockedError`` (423); the user's row stays locked for the
    rest of ``db_conn``'s transaction (TASK-15671).

    Reads only on ``db_conn``: a profile update must fit in its existing PostgreSQL transaction slot.
    ``new_emails`` holds every ``identity.email`` value in the request; one that differs from the
    stored address (ignoring case and surrounding whitespace) requires the password.
    """
    if settings.is_single_user_mode():
        # The API key is the only credential in single-user mode, so there is no password to ask for.
        return
    # A lock or take-back that starts now waits for this transaction; one that committed is seen (423).
    await account_lock.assert_identity_write_allowed(db_conn, user_id, is_postgres=is_postgres)
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
    # Another account's owner can take its previous address back for 7 days (TASK-15671, Decision 18).
    changed = [value for value in new_emails if str(value).strip().lower() != stored]
    if await email_recently_released_by_another_user(db_conn, changed, user_id=user_id, is_postgres=is_postgres):
        raise _refuse(
            user_id,
            "email_recently_released",
            "This address left another account in the last 7 days and cannot be used yet. Try again later.",
        )


async def notify_previous_address(
    *, user_id: int, old_email: str, new_email: str, username: str, ip_address: str
) -> None:
    """Send the email-changed notice to ``old_email`` once the request transaction commits.

    A failed send is logged (user id only) and never fails the change.
    """

    async def _send() -> None:
        reason = "send failed"
        try:
            sent = await get_email_service().send_email_changed_email(
                to_email=old_email, username=username, new_email=new_email, ip_address=ip_address
            )
        except _NOTICE_NONCRITICAL_EXCEPTIONS as exc:
            sent, reason = False, type(exc).__name__
        if not sent:
            logger.warning("Email-change notice to the previous address was not sent (user {}): {}", user_id, reason)

    if not defer_until_after_commit(_send):
        await _send()
