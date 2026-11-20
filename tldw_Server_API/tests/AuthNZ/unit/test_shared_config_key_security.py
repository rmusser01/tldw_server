"""Security pins for config keys that move into ``config_sections`` (TASK-13950.21, Task 7).

Run by security-required's TEST_MODE-unset step (C4), which runs this directory.
"""

from __future__ import annotations

import base64
import configparser

import pydantic
import pytest

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.config_sections.auth import load_auth_config

pytestmark = pytest.mark.unit

# The webhook ring takes base64-encoded 32-byte keys.
_KEY_X = base64.b64encode(b"X" * 32).decode()
_KEY_Y = base64.b64encode(b"Y" * 32).decode()


@pytest.mark.parametrize("source", ["env", "config"])
@pytest.mark.parametrize("value", ["multi-user", "MULTI_USER", ""])
def test_invalid_auth_mode_fails_closed_in_every_source(monkeypatch, source, value):
    """A bad AUTH_MODE, in the environment or in config.txt, must stop AuthNZ rather than fall back."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    if source == "env":
        monkeypatch.setenv("AUTH_MODE", value)
    else:
        monkeypatch.delenv("AUTH_MODE", raising=False)
        parser.set("AuthNZ", "auth_mode", value)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
    authnz_settings.reset_settings()
    try:
        with pytest.raises(pydantic.ValidationError):
            authnz_settings.get_settings()
    finally:
        authnz_settings.reset_settings()


def test_config_file_cannot_set_the_single_user_identity(monkeypatch):
    """Row 2: AuthNZ's single-user id comes from the environment alone; config.txt never moves it."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    parser.set("AuthNZ", "single_user_fixed_id", "9")
    monkeypatch.delenv("SINGLE_USER_FIXED_ID", raising=False)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
    authnz_settings.reset_settings()
    try:
        assert authnz_settings.get_settings().SINGLE_USER_FIXED_ID == 1
    finally:
        authnz_settings.reset_settings()


@pytest.mark.parametrize("key", ["DATABASE_URL", "JWT_SECRET_KEY", "SINGLE_USER_API_KEY"])
def test_auth_config_repr_hides_secrets(key):
    """Rows 4, 6 and 7: a URL that can carry a password, the JWT secret and the single-user key never show in the typed auth view's repr."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    assert "s3cret" not in repr(load_auth_config(parser, env={key: "s3cret"}))


def test_webhook_secret_ring_still_decrypts(monkeypatch):
    """Rows 6 and 7: a secret stored under the single-user key id keeps decrypting."""
    from tldw_Server_API.app.core.AuthNZ import admin_webhook_secrets as ring

    for key in (
        "SINGLE_USER_API_KEY", "JWT_SECRET_KEY", "BYOK_ENCRYPTION_KEY",
        "BYOK_SECONDARY_ENCRYPTION_KEY", "SESSION_ENCRYPTION_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("API_KEY", _KEY_X)
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    parser.set("AuthNZ", "single_user_api_key", _KEY_Y)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)

    stored = ring.encrypt_admin_webhook_secret("s")

    assert stored.key_id == "single_user_api_key"
    assert ring.decrypt_admin_webhook_secret(stored.encrypted_blob) == "s"


@pytest.mark.parametrize(("key", "expected_status"), [("env_alias", 401), ("config", 200)])
def test_single_user_api_key_accepted_keys(monkeypatch, key, expected_status):
    """Row 7, D1(e): require_token_scope accepts AuthNZ's resolved key; a legacy env key that differs is refused.

    Y is ``[AuthNZ] single_user_api_key``, which AuthNZ resolves; X is the legacy env ``API_KEY``.
    """
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    for name in ("SINGLE_USER_API_KEY", "SINGLE_USER_TEST_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("API_KEY", "x" * 24)
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    parser.set("AuthNZ", "single_user_api_key", "y" * 24)
    monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
    authnz_settings.reset_settings()

    class _NoKeys:
        async def validate_api_key(self, **_kwargs):
            return None

    async def _manager():
        return _NoKeys()

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _manager)
    dep = auth_deps.require_token_scope("any", require_if_present=True)
    req = SimpleNamespace(
        method="GET", headers={}, scope={"path": "/protected"}, client=SimpleNamespace(host="127.0.0.1"),
    )
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=("x" if key == "env_alias" else "y") * 24)
    try:
        # Accepted: the dependency returns without consulting the API key manager.
        status = 200 if asyncio.run(dep(request=req, credentials=creds, jwt_service=object(), db_pool=object())) is None else 0
    except HTTPException as exc:
        status = exc.status_code
    finally:
        authnz_settings.reset_settings()
    assert status == expected_status
