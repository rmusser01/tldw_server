import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.token_blacklist import TokenBlacklist
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


@pytest.mark.asyncio
async def test_revoke_all_user_tokens_counts_tokens(tmp_path):
    reset_settings()
    db_path = tmp_path / "token_blacklist_counts.sqlite"
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="token-blacklist-secret-1234567890abcd",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(settings)
    await pool.initialize()

    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="charlie",
        email="charlie@example.com",
        password_hash="hashed-password",
    )
    user_id = int(user["id"])

    session_manager = SessionManager(db_pool=pool, settings=settings)
    await session_manager.initialize()

    jwt_service = JWTService(settings=settings)
    access_token = jwt_service.create_access_token(
        user_id=user_id,
        username="charlie",
        role="user",
    )
    refresh_token = jwt_service.create_refresh_token(
        user_id=user_id,
        username="charlie",
    )

    await session_manager.create_session(
        user_id=user_id,
        access_token=access_token,
        refresh_token=refresh_token,
        ip_address="127.0.0.1",
        user_agent="pytest-suite",
    )

    blacklist = TokenBlacklist(db_pool=pool, settings=settings)
    count = await blacklist.revoke_all_user_tokens(
        user_id=user_id,
        reason="test-revoke",
        revoked_by=99,
    )

    assert count == 2

    rows = await pool.fetchall(
        "SELECT token_type FROM token_blacklist WHERE user_id = ? ORDER BY token_type",
        user_id,
    )
    token_types = []
    for row in rows:
        if isinstance(row, dict):
            token_types.append(row["token_type"])
        else:
            try:
                token_types.append(row["token_type"])
            except (TypeError, KeyError):
                token_types.append(row[0])
    assert token_types == ["access", "refresh"]

    await session_manager.shutdown()
    await pool.close()
    reset_settings()


@pytest.mark.asyncio
async def test_revoke_all_user_tokens_preserves_excluded_session(tmp_path):
    reset_settings()
    db_path = tmp_path / "token_blacklist_excluded_session.sqlite"
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="token-blacklist-secret-1234567890abcd",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )
    pool = DatabasePool(settings)
    await pool.initialize()

    try:
        users_db = UsersDB(pool)
        await users_db.initialize()
        user = await users_db.create_user(
            username="charlie",
            email="charlie@example.com",
            password_hash="hashed-password",
        )
        user_id = int(user["id"])

        session_manager = SessionManager(db_pool=pool, settings=settings)
        await session_manager.initialize()
        jwt_service = JWTService(settings=settings)

        first = await session_manager.create_session(
            user_id=user_id,
            access_token=jwt_service.create_access_token(
                user_id=user_id,
                username="charlie",
                role="user",
            ),
            refresh_token=jwt_service.create_refresh_token(
                user_id=user_id,
                username="charlie",
            ),
        )
        excluded = await session_manager.create_session(
            user_id=user_id,
            access_token=jwt_service.create_access_token(
                user_id=user_id,
                username="charlie",
                role="user",
            ),
            refresh_token=jwt_service.create_refresh_token(
                user_id=user_id,
                username="charlie",
            ),
        )
        rows_before = await pool.fetchall(
            """
            SELECT id, access_jti, refresh_jti
            FROM sessions
            WHERE user_id = ?
            ORDER BY id
            """,
            user_id,
        )
        token_metadata = {
            int(row["id"]): (row["access_jti"], row["refresh_jti"])
            for row in rows_before
        }

        blacklist = TokenBlacklist(db_pool=pool, settings=settings)
        count = await blacklist.revoke_all_user_tokens(
            user_id=user_id,
            reason="rotate other sessions",
            revoked_by=99,
            except_session_id=excluded["session_id"],
        )

        rows_after = await pool.fetchall(
            """
            SELECT id, is_active, is_revoked
            FROM sessions
            WHERE user_id = ?
            ORDER BY id
            """,
            user_id,
        )
        session_states = {
            int(row["id"]): (int(row["is_active"]), int(row["is_revoked"]))
            for row in rows_after
        }
        first_jtis = token_metadata[first["session_id"]]
        excluded_jtis = token_metadata[excluded["session_id"]]

        assert count == 2
        assert session_states[first["session_id"]] == (0, 1)
        assert session_states[excluded["session_id"]] == (1, 0)
        assert all([await blacklist.is_blacklisted(jti) for jti in first_jtis])
        assert not any([await blacklist.is_blacklisted(jti) for jti in excluded_jtis])
    finally:
        await pool.close()
        reset_settings()
