"""TASK-13387: AuthNZ startup failures must name their cause in the log.

On 2026-09-27 sqlglot 30.20.0 broke every SQLite AuthNZ startup in CI. The real error
was a ProfileUserWriteRejected from the users-bootstrap guard, but two ``raise ... from
None`` sites (Users_DB._create_tables and the pool's _transaction_context) replaced it
with "Failed to create users table", and the exception type was only a loguru extra
that the CI log format does not print. 97 failures showed nothing but the generic text.

The fix logs the chain of exception *types*, never messages: a PostgreSQL error message
can carry row values (a unique-violation detail includes the email), so types are the
safe amount of detail.
"""

from __future__ import annotations

import pytest
from loguru import logger

from tldw_Server_API.app.core.exceptions import exception_type_chain

pytestmark = pytest.mark.unit


class _Inner(Exception):
    pass


class _Outer(Exception):
    pass


def _raise_detached_wrapper() -> None:
    try:
        raise _Inner("user@example.com already exists")
    except _Inner:
        raise _Outer("generic") from None


def test_the_chain_sees_through_from_none() -> None:
    """`from None` only hides the context from display; the chain must still reach it."""
    with pytest.raises(_Outer) as excinfo:
        _raise_detached_wrapper()

    assert exception_type_chain(excinfo.value) == "_Outer <- _Inner"


def test_the_chain_never_includes_exception_messages() -> None:
    """Types only: a message can carry a row value such as an email address."""
    with pytest.raises(_Outer) as excinfo:
        _raise_detached_wrapper()

    chain = exception_type_chain(excinfo.value)
    assert "example.com" not in chain
    assert "generic" not in chain


def test_the_chain_is_bounded_and_survives_a_cycle() -> None:
    """A self-referential context must not loop forever."""
    err = _Outer("x")
    err.__context__ = err

    assert exception_type_chain(err) == "_Outer"


@pytest.mark.asyncio
async def test_a_rejected_users_bootstrap_logs_its_real_cause(tmp_path, monkeypatch) -> None:
    """The exact outage shape: a guard rejection inside _create_tables must be named."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import ProfileUserWriteRejected
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management import Users_DB

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    reset_settings()
    await reset_db_pool()

    async def _rejected(*_args, **_kwargs):
        raise ProfileUserWriteRejected()

    monkeypatch.setattr(Users_DB, "_execute_profile_users_bootstrap", _rejected)

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        users_db = Users_DB.UsersDB(await get_db_pool())
        with pytest.raises(DatabaseError, match="Failed to create users table"):
            await users_db.initialize()
    finally:
        logger.remove(sink_id)
        await reset_db_pool()
        reset_settings()

    assert any("ProfileUserWriteRejected" in m for m in messages), messages
