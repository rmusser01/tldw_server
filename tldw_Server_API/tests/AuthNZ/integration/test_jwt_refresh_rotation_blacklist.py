import asyncio
import pytest
from datetime import datetime, timedelta

from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist

pytestmark = pytest.mark.integration


def _jwt() -> JWTService:


    return JWTService(
        settings=Settings(
            AUTH_MODE="multi_user",
            JWT_SECRET_KEY="C" * 40,
            JWT_ALGORITHM="HS256",
            ACCESS_TOKEN_EXPIRE_MINUTES=5,
            REFRESH_TOKEN_EXPIRE_DAYS=1,
            ROTATE_REFRESH_TOKENS=True,
        )
    )


@pytest.mark.asyncio
async def test_refresh_rotates_and_blacklists_old_refresh(monkeypatch):
    # Ensure blacklist uses a local SQLite DB to avoid external Postgres dependency
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./Databases/users.db")
    svc = _jwt()
    refresh = svc.create_refresh_token(user_id=7, username="u7")
    # Extract old JTI for assertion
    old_jti = svc.extract_jti(refresh)

    new_access, new_refresh = await svc.refresh_access_token_async(refresh)
    assert isinstance(new_access, str)
    assert isinstance(new_refresh, str)
    assert new_refresh != refresh

    # Old refresh jti should be blacklisted. Allow a short window for
    # background scheduling in async contexts.
    bl = get_token_blacklist()
    for _ in range(10):  # up to ~250ms
        if await bl.is_blacklisted(old_jti):
            break
        await asyncio.sleep(0.025)
    assert await bl.is_blacklisted(old_jti) is True


@pytest.mark.asyncio
async def test_refresh_rotation_raises_when_blacklist_persistence_fails(monkeypatch):
    class _FailingBlacklist:
        def hint_blacklisted(self, *_args, **_kwargs):
            return None

        async def is_blacklisted(self, *_args, **_kwargs):
            return False

        async def revoke_token(self, **_kwargs):
            raise RuntimeError("simulated blacklist persistence failure")

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./Databases/users.db")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.token_blacklist.get_token_blacklist",
        lambda: _FailingBlacklist(),
    )

    svc = _jwt()
    refresh = svc.create_refresh_token(user_id=7, username="u7")

    with pytest.raises(InvalidTokenError, match="rotation"):
        await svc.refresh_access_token_async(refresh)
