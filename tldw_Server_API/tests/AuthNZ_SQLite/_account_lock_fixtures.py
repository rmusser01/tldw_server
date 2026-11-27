"""Lock and unlock helpers for the account-lock tests (TASK-15671; not a test module)."""

from __future__ import annotations

from tldw_Server_API.app.core.AuthNZ.account_lock import ACCOUNT_LOCK_REVOKE_REASON, clear_lock, set_lock
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager


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
