"""Lock and unlock helpers for the account-lock tests (TASK-15671; not a test module)."""

from __future__ import annotations

import re
import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ import email_service
from tldw_Server_API.app.core.AuthNZ.account_lock import ACCOUNT_LOCK_REVOKE_REASON, clear_lock, set_lock
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager


async def repair_reset_tokens_table() -> None:
    """Rebuild ``password_reset_tokens`` as migration 011 defines it.

    Pre-existing bug, not the lock's: the packaged SQLite schema creates the table with ``token``, and
    migration 011's ``ADD COLUMN token_hash TEXT UNIQUE`` fails silently (SQLite cannot add a UNIQUE
    column), so forgot-password and reset-password cannot store or find a token on a fresh SQLite database.
    """
    with sqlite3.connect((await get_db_pool()).db_path) as conn:
        conn.execute("DROP TABLE password_reset_tokens")
        conn.execute(
            "CREATE TABLE password_reset_tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, "
            "token_hash TEXT UNIQUE, expires_at TIMESTAMP NOT NULL, used_at TIMESTAMP, ip_address TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)"
        )


async def lock_user(user_id: int) -> None:
    """Lock the account and sign every session out, as the lock endpoint does."""
    assert await set_lock(user_id) is not None, "the account was already locked"
    await (await get_session_manager()).revoke_all_user_sessions(user_id, reason=ACCOUNT_LOCK_REVOKE_REASON)


async def unlock_user(user_id: int) -> None:
    """Unlock the account; returns once every token minted before it is older than credentials_valid_after."""
    assert await clear_lock(user_id) is not None, "the account was not locked"


def assert_locked(response) -> None:
    """The 423 a locked account gets, with its machine-readable code."""
    assert response.status_code == 423, response.text
    assert response.json()["detail"]["code"] == "account_locked"


@pytest.fixture
def lockable(monkeypatch):
    """SMTP mail with a public URL, delivery captured: the list of ``(to_address, plain_text_body)`` sent."""
    monkeypatch.setenv("EMAIL_PROVIDER", "smtp")
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.com")
    monkeypatch.setattr(email_service, "_email_service", None)
    outbox: list[tuple[str, str]] = []

    def _capture(self, message) -> None:
        plain = message.get_payload()[0].get_payload(decode=True).decode()
        outbox.append((message["To"], plain))

    monkeypatch.setattr(email_service.EmailService, "_deliver_smtp_message", _capture)
    yield outbox
    email_service._email_service = None


def unlock_token_from(outbox: list[tuple[str, str]], address: str) -> str:
    """The ``token=`` value of the unlock link mailed to ``address``."""
    body = next(text for to, text in outbox if to == address)
    return re.search(r"token=([\w.\-]+)", body).group(1)
