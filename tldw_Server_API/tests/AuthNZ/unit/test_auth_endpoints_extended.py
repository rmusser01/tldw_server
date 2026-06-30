from types import SimpleNamespace
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi import Response
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.mark.asyncio
async def test_is_mfa_backend_supported_prefers_mfa_service_capability(monkeypatch):
    reset_settings()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _Svc:
        def __init__(self):
            self.init_calls = 0

        async def initialize(self):
            self.init_calls += 1

        def supports_backend(self):
            return True

    svc = _Svc()

    async def _should_not_be_called():
        raise AssertionError("is_postgres_backend fallback should not be used when service supports_backend exists")

    monkeypatch.setattr(auth, "_get_mfa_service", lambda: svc)
    monkeypatch.setattr(auth, "is_postgres_backend", _should_not_be_called)

    assert await auth._is_mfa_backend_supported() is True
    assert svc.init_calls == 1


@pytest.mark.asyncio
async def test_reset_password_weak_and_success(monkeypatch):
    reset_settings()
    # Stub DB adapter expected by endpoint (fetchval/execute/commit)
    class _StubDB:
        async def fetchval(self, *args, **kwargs):
            return None

        async def execute(self, *args, **kwargs):
            class _C:
                lastrowid = 1

                async def fetchone(self):
                    # Simulate existing reset token record
                    return (1, None)

            return _C()

        async def commit(self):
            return True

    class _StubJWT:
        def verify_token(self, token: str, token_type: str | None = None):
            return {"sub": 42}

        def hash_token_candidates(self, token: str) -> list[str]:
            return ["htok"]

        def hash_token(self, token: str) -> str:
            return "htok"

    class _StubPwd:
        def __init__(self, weak: bool = False):
            self.weak = weak
        def validate_password_strength(self, new_password: str, username: str | None = None):
            if self.weak:
                from tldw_Server_API.app.core.AuthNZ.exceptions import WeakPasswordError
                raise WeakPasswordError("Password too weak")
        def hash_password(self, pwd: str) -> str:
            return "HASHED"

    import tldw_Server_API.app.api.v1.endpoints.auth as _auth

    async def _fake_is_pg() -> bool:
        return False

    class _StubBlacklist:
        async def revoke_all_user_tokens(self, *_args, **_kwargs):
            return 0

    monkeypatch.setattr(_auth, "is_postgres_backend", _fake_is_pg)
    monkeypatch.setattr(_auth, "get_token_blacklist", lambda: _StubBlacklist())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/reset-password",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    # Weak password
    with pytest.raises(Exception):
        await _auth.reset_password(
            data=_auth.ResetPasswordRequest(token="tok", new_password="weak"),
            request=request,
            db=_StubDB(),
            jwt_service=_StubJWT(),
            password_service=_StubPwd(weak=True),
        )

    # Success path
    out = await _auth.reset_password(
        data=_auth.ResetPasswordRequest(token="tok", new_password="Strong@12345"),
        request=request,
        db=_StubDB(),
        jwt_service=_StubJWT(),
        password_service=_StubPwd(weak=False),
    )
    assert "success" in out.get("message", "").lower()


@pytest.mark.asyncio
async def test_forgot_password_rate_limited_returns_generic_message(monkeypatch):
    reset_settings()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _deny_rg(*args, **kwargs):
        return False, 1

    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _deny_rg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/forgot-password",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    out = await auth.forgot_password(
        request=request,
        data=auth.ForgotPasswordRequest(email="rate@example.com"),
        db=object(),
        jwt_service=object(),
    )
    assert out["message"] == "If the email exists, a reset link has been sent"


@pytest.mark.asyncio
async def test_reserve_auth_rg_requests_uses_diagnostics_only_shim_when_governor_missing(monkeypatch):
    reset_settings()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _no_governor(_request):
        return None

    monkeypatch.setattr(auth, "_get_auth_endpoint_rg_governor", _no_governor)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/magic-link/request",
        "headers": [],
        "client": ("203.0.113.21", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
        "state": {},
        "app": SimpleNamespace(state=SimpleNamespace()),
    }
    request = Request(scope)

    allowed, retry_after = await auth._reserve_auth_rg_requests(
        request,
        policy_id="authnz.magic_link.request",
        entity="ip:203.0.113.21",
    )
    assert allowed is True
    assert retry_after is None


@pytest.mark.asyncio
async def test_reserve_auth_rg_requests_uses_diagnostics_only_shim_when_policy_missing(monkeypatch):
    reset_settings()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _StubGovernor:
        async def reserve(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("reserve should not run when RG policy is missing")

    async def _governor(_request):
        return _StubGovernor()

    monkeypatch.setattr(auth, "_get_auth_endpoint_rg_governor", _governor)
    monkeypatch.setattr(auth, "_auth_rg_policy_defined", lambda *_args, **_kwargs: False)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/magic-link/request",
        "headers": [],
        "client": ("203.0.113.22", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
        "state": {},
        "app": SimpleNamespace(state=SimpleNamespace()),
    }
    request = Request(scope)

    allowed, retry_after = await auth._reserve_auth_rg_requests(
        request,
        policy_id="authnz.magic_link.request",
        entity="ip:203.0.113.22",
    )

    assert allowed is True
    assert retry_after is None


@pytest.mark.asyncio
async def test_reserve_auth_rg_requests_uses_diagnostics_only_shim_when_rg_reserve_fails(monkeypatch):
    reset_settings()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _BrokenGovernor:
        async def reserve(self, *_args, **_kwargs):
            raise RuntimeError("reserve failed")

    async def _governor(_request):
        return _BrokenGovernor()

    monkeypatch.setattr(auth, "_get_auth_endpoint_rg_governor", _governor)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/magic-link/request",
        "headers": [],
        "client": ("203.0.113.23", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
        "state": {},
        "app": SimpleNamespace(state=SimpleNamespace()),
    }
    request = Request(scope)

    allowed, retry_after = await auth._reserve_auth_rg_requests(
        request,
        policy_id="authnz.magic_link.request",
        entity="ip:203.0.113.23",
    )

    assert allowed is True
    assert retry_after is None


@pytest.mark.asyncio
async def test_request_admin_reauth_uses_dedicated_token_and_email_helpers(monkeypatch):
    reset_settings()
    monkeypatch.setenv("AUTH_MODE", "multi_user")

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    created: dict[str, object] = {}
    sent: dict[str, object] = {}

    class _StubJWT:
        def create_admin_reauth_token(self, **kwargs):  # noqa: ANN003
            created.update(kwargs)
            return "admin-reauth-token"

    class _StubEmailService:
        async def send_admin_reauth_email(self, **kwargs):  # noqa: ANN003
            sent.update(kwargs)
            return True

    monkeypatch.setattr(auth, "_get_email_service", lambda: _StubEmailService())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/admin/reauth/request",
        "headers": [],
        "client": ("203.0.113.50", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    out = await auth.request_admin_reauth(
        request=request,
        current_user=auth.AuthPrincipal(
            kind="user",
            user_id=7,
            subject="user:7",
            username="alice",
            email="alice@example.com",
            roles=["admin"],
            is_admin=True,
        ),
        db=object(),
        jwt_service=_StubJWT(),
    )

    assert out.message == "Admin reauthentication link sent"
    assert created["email"] == "alice@example.com"
    assert created["user_id"] == 7
    assert created["expires_in_minutes"] == 10
    assert sent["to_email"] == "alice@example.com"
    assert sent["username"] == "alice"


@pytest.mark.asyncio
async def test_request_admin_reauth_requires_admin_claims(monkeypatch):
    reset_settings()
    monkeypatch.setenv("AUTH_MODE", "multi_user")

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/admin/reauth/request",
        "headers": [],
        "client": ("203.0.113.51", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    with pytest.raises(HTTPException) as excinfo:
        await auth.request_admin_reauth(
            request=request,
            current_user=auth.AuthPrincipal(
                kind="user",
                user_id=7,
                subject="user:7",
                username="alice",
                email="alice@example.com",
                roles=["user"],
                is_admin=False,
            ),
            db=object(),
            jwt_service=object(),
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "Admin reauthentication is only available to administrators"


@pytest.mark.asyncio
async def test_verify_magic_link_rejects_admin_reauth_purpose(monkeypatch):
    reset_settings()
    monkeypatch.setenv("AUTH_MODE", "multi_user")

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _StubJWT:
        async def verify_token_async(self, token: str, token_type: str | None = None):
            assert token == "admin-reauth-token"
            assert token_type == "magic_link"
            if token_type != "magic_link":
                raise ValueError("unexpected token type")
            return {
                "email": "alice@example.com",
                "user_id": 7,
                "purpose": "admin_reauth",
                "type": "admin_reauth",
            }

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/magic-link/verify",
        "headers": [],
        "client": ("203.0.113.52", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    with pytest.raises(HTTPException) as excinfo:
        await auth.verify_magic_link(
            data=auth.MagicLinkVerifyRequest(token="admin-reauth-token"),
            request=request,
            db=object(),
            jwt_service=_StubJWT(),
            session_manager=object(),
            registration_service=object(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "Admin reauthentication tokens cannot be used for sign-in"


@pytest.mark.asyncio
async def test_login_lockout_uses_trusted_forwarded_client_ip(monkeypatch):
    monkeypatch.setenv("AUTH_TRUST_X_FORWARDED_FOR", "true")
    monkeypatch.setenv("AUTH_TRUSTED_PROXY_IPS", "10.0.0.0/8")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    captured: dict[str, str] = {}

    class _StubGov:
        async def check_lockout(self, identifier: str, *, attempt_type: str = "login", rate_limiter=None):
            _ = attempt_type
            _ = rate_limiter
            captured["identifier"] = identifier
            return True, datetime.now(timezone.utc) + timedelta(minutes=15)

        async def record_auth_failure(self, *args, **kwargs):
            _ = args
            _ = kwargs
            return {"is_locked": False, "remaining_attempts": 5}

    class _StubLimiter:
        enabled = True

    async def _fake_get_auth_governor():
        return _StubGov()

    monkeypatch.setattr(auth, "get_auth_governor", _fake_get_auth_governor)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/login",
        "headers": [(b"x-forwarded-for", b"198.51.100.77, 10.1.2.3")],
        "client": ("10.1.2.3", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    response = Response()
    form_data = SimpleNamespace(username="user1", password="wrong-password")

    with pytest.raises(auth.HTTPException) as exc:
        await auth.login(
            request=request,
            response=response,
            form_data=form_data,
            db=object(),
            jwt_service=object(),
            password_service=object(),
            session_manager=object(),
            rate_limiter=_StubLimiter(),
            settings=auth.get_settings(),
        )

    assert exc.value.status_code == 429
    assert captured["identifier"] == "198.51.100.77"


@pytest.mark.asyncio
async def test_reserve_auth_rg_requests_ignores_untrusted_forwarded_ip(monkeypatch):
    monkeypatch.setenv("AUTH_TRUST_X_FORWARDED_FOR", "true")
    monkeypatch.setenv("AUTH_TRUSTED_PROXY_IPS", "10.0.0.0/8")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _StubGovernor:
        def __init__(self):
            self.entities = []

        async def reserve(self, request_obj, op_id: str):
            _ = op_id
            self.entities.append(request_obj.entity)
            return SimpleNamespace(allowed=True, retry_after=None), None

    governor = _StubGovernor()

    async def _get_governor(_request):
        return governor

    monkeypatch.setattr(auth, "_get_auth_endpoint_rg_governor", _get_governor)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/forgot-password",
        "headers": [(b"x-forwarded-for", b"198.51.100.88, 203.0.113.9")],
        "client": ("203.0.113.9", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
        "state": {},
        "app": SimpleNamespace(state=SimpleNamespace()),
    }
    request = Request(scope)

    allowed, retry_after = await auth._reserve_auth_rg_requests(
        request,
        policy_id="authnz.forgot_password",
    )

    assert allowed is True
    assert retry_after is None
    assert governor.entities == ["ip:203.0.113.9"]


@pytest.mark.asyncio
async def test_reset_password_rate_limited_returns_429(monkeypatch):
    reset_settings()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _deny_rg(*args, **kwargs):
        return False, 7

    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _deny_rg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/reset-password",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    with pytest.raises(auth.HTTPException) as exc:
        await auth.reset_password(
            data=auth.ResetPasswordRequest(token="tok", new_password="Strong@12345"),
            request=request,
            db=object(),
            jwt_service=object(),
            password_service=object(),
        )

    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "7"


@pytest.mark.asyncio
async def test_logout_uses_utc_expiry(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("tzset not available on this platform")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    exp_ts = 1_700_000_000
    captured = {"expires_at": None, "revoked_sessions": []}

    class StubJWT:
        def __init__(self):
            pass

        def extract_jti(self, token: str) -> str:
            return "jti-123"

        def verify_token(self, token: str):
            return {"exp": exp_ts, "session_id": 777}

    class StubBlacklist:
        async def revoke_all_user_tokens(self, **kwargs):
            return 0

        async def revoke_token(
            self,
            *,
            jti,
            expires_at,
            user_id,
            token_type,
            reason,
            revoked_by=None,
            ip_address=None,
        ):
            captured["expires_at"] = expires_at
            captured["token_type"] = token_type
            captured["ip_address"] = ip_address
            return True

    class StubSessionManager:
        async def revoke_all_user_sessions(self, **kwargs):
            captured["revoked_sessions"].append(("all", kwargs))

        async def revoke_session(self, *, session_id, revoked_by=None, reason=None):
            captured["revoked_sessions"].append(
                ("single", {"session_id": session_id, "revoked_by": revoked_by, "reason": reason})
            )

    monkeypatch.setattr(auth, "JWTService", StubJWT)
    stub_blacklist = StubBlacklist()
    monkeypatch.setattr(auth, "get_token_blacklist", lambda: stub_blacklist)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [(b"authorization", b"Bearer my-token")],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    data = auth.LogoutRequest(all_devices=False)
    current_user = SimpleNamespace(id=99)
    session_manager = StubSessionManager()
    jwt_service = StubJWT()

    original_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "US/Pacific"
        time.tzset()
        result = await auth.logout(
            data=data,
            request=request,
            current_user=current_user,
            session_manager=session_manager,
            jwt_service=jwt_service,
        )
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()

    assert result.message == "Successfully logged out"
    assert captured["expires_at"] == datetime.utcfromtimestamp(exp_ts)
    assert captured["token_type"] == "access"
    assert captured["revoked_sessions"][0][0] == "single"


@pytest.mark.asyncio
async def test_login_returns_mfa_challenge_when_enabled(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("MFA_LOGIN_TTL_SECONDS", "300")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_mfa_supported() -> bool:
        return True

    class _StubMFA:
        async def get_user_mfa_status(self, user_id: int):
            return {"enabled": True}

    async def _fake_fetch_user(db, identifier: str):
        return {
            "id": 5,
            "username": "mfa_user",
            "email": "mfa@example.com",
            "password_hash": "HASHED",
            "role": "user",
            "is_active": True,
        }

    class _StubPwd:
        def verify_password(self, password: str, password_hash: str):
            return True, False

    class _StubSessionManager:
        def __init__(self):
            self.cached = {}
            self.created_kwargs = {}
            self.created_at = None
            self.redis_client = object()

        async def create_session(self, **kwargs):
            from datetime import datetime, timezone as _tz
            self.created_kwargs = dict(kwargs)
            self.created_at = datetime.now(_tz.utc)
            return {"session_id": 777}

        async def store_ephemeral_value(self, key: str, value: str, ttl_seconds: int):
            self.cached[key] = (value, ttl_seconds)

        async def update_session_tokens(self, **kwargs):
            raise AssertionError("update_session_tokens should not run for MFA-required login")

    class _StubLimiter:
        enabled = False

    class _StubGov:
        async def check_lockout(self, *args, **kwargs):
            return False, None

        async def record_auth_failure(self, *args, **kwargs):
            return {"is_locked": False, "remaining_attempts": 5}

    async def _fake_get_auth_governor():
        return _StubGov()

    monkeypatch.setattr(auth, "_is_mfa_backend_supported", _fake_mfa_supported)
    monkeypatch.setattr(auth, "_get_mfa_service", lambda: _StubMFA())
    monkeypatch.setattr(auth, "fetch_user_by_login_identifier", _fake_fetch_user)
    monkeypatch.setattr(auth, "get_auth_governor", _fake_get_auth_governor)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/login",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    response = Response()
    form_data = SimpleNamespace(username="mfa_user", password="password")
    session_manager = _StubSessionManager()

    result = await auth.login(
        request=request,
        response=response,
        form_data=form_data,
        db=None,
        jwt_service=object(),
        password_service=_StubPwd(),
        session_manager=session_manager,
        rate_limiter=_StubLimiter(),
        settings=auth.get_settings(),
    )

    assert response.status_code == 202
    assert result.mfa_required is True
    assert result.expires_in == 300
    assert result.session_token
    assert session_manager.cached
    expires_at = session_manager.created_kwargs.get("expires_at_override")
    refresh_expires_at = session_manager.created_kwargs.get("refresh_expires_at_override")
    assert expires_at is not None
    assert refresh_expires_at == expires_at
    # Should be a short-lived expiry (roughly the MFA TTL)
    assert expires_at.tzinfo is not None
    delta = (expires_at - session_manager.created_at).total_seconds()
    assert 290 <= delta <= 305


@pytest.mark.asyncio
async def test_mfa_login_completes_tokens(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_mfa_supported() -> bool:
        return True

    async def _fake_fetch_active_user(db, user_id: int):
        return {
            "id": user_id,
            "username": "mfa_user",
            "email": "mfa@example.com",
            "role": "user",
            "is_active": True,
        }

    class _StubMFA:
        async def get_user_mfa_status(self, user_id: int):
            return {"enabled": True}

        async def get_user_totp_secret(self, user_id: int):
            return "SECRET"

        def verify_totp(self, secret: str, token: str) -> bool:
            return token == "123456"

        async def verify_backup_code(self, user_id: int, code: str) -> bool:
            return False

    class _StubSessionManager:
        def __init__(self):
            self.ephemeral = {}
            self.updated = {}
            self.deleted = []
            self.redis_client = object()

        async def get_ephemeral_value(self, key: str):
            return self.ephemeral.get(key)

        async def delete_ephemeral_value(self, key: str):
            self.deleted.append(key)

        async def update_session_tokens(self, **kwargs):
            self.updated.update(kwargs)

    class _StubLimiter:
        enabled = False

        async def check_user_rate_limit(self, user_id: int, endpoint: str):
            return True, {}

        async def reset_failed_attempts(self, *args, **kwargs):
            return None

    class _StubJWT:
        def create_access_token(self, **kwargs):
            return "ACCESS"

        def create_refresh_token(self, **kwargs):
            return "REFRESH"

    monkeypatch.setattr(auth, "_is_mfa_backend_supported", _fake_mfa_supported)
    monkeypatch.setattr(auth, "fetch_active_user_by_id", _fake_fetch_active_user)
    monkeypatch.setattr(auth, "_get_mfa_service", lambda: _StubMFA())
    async def _noop_async(*args, **kwargs):
        return None

    async def _noop_claims(*args, **kwargs):
        return {}

    monkeypatch.setattr(auth, "update_user_last_login", _noop_async)
    monkeypatch.setattr(auth, "_build_scope_claims", _noop_claims)

    class _StubAuditService:
        async def log_login(self, *args, **kwargs):
            return None

        async def flush(self):
            return None

    async def _fake_audit_service(*args, **kwargs):
        return _StubAuditService()

    monkeypatch.setattr(auth, "get_or_create_audit_service_for_user_id", _fake_audit_service)

    session_manager = _StubSessionManager()
    token = "mfa-session-token"
    cache_key = auth._mfa_login_cache_key(token)
    session_manager.ephemeral[cache_key] = '{"user_id": 5, "session_id": 55}'

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/mfa/login",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    response = Response()

    result = await auth.mfa_login(
        data=auth.MFALoginRequest(session_token=token, mfa_token="123456"),
        request=request,
        response=response,
        db=None,
        jwt_service=_StubJWT(),
        session_manager=session_manager,
        rate_limiter=_StubLimiter(),
        settings=auth.get_settings(),
    )

    assert result.access_token == "ACCESS"
    assert result.refresh_token == "REFRESH"
    assert session_manager.updated["session_id"] == 55
    assert cache_key in session_manager.deleted


@pytest.mark.asyncio
async def test_setup_mfa_accepts_dict_current_user(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_mfa_supported() -> bool:
        return True

    class _StubMFA:
        async def get_user_mfa_status(self, user_id: int):
            return {"enabled": False}

        def generate_secret(self) -> str:
            return "DICTSECRET"

        def generate_totp_uri(self, secret: str, username: str) -> str:
            return f"otpauth://totp/{username}?secret={secret}"

        def generate_qr_code(self, _uri: str) -> bytes:
            return b"png"

        def generate_backup_codes(self) -> list[str]:
            return ["code-a", "code-b"]

    class _StubSessionManager:
        def __init__(self):
            self.redis_client = object()
            self.cached = {}

        async def initialize(self):
            return None

        async def store_ephemeral_value(self, key: str, value: str, ttl_seconds: int):
            self.cached[key] = (value, ttl_seconds)

    monkeypatch.setattr(auth, "_is_mfa_backend_supported", _fake_mfa_supported)
    monkeypatch.setattr(auth, "_get_mfa_service", lambda: _StubMFA())
    sm = _StubSessionManager()

    out = await auth.setup_mfa(
        current_user={
            "id": 7,
            "username": "dictuser",
            "email": "dict@example.com",
            "is_active": True,
            "is_verified": True,
        },
        db=None,
        session_manager=sm,
    )

    assert out.secret == "DICTSECRET"
    assert out.qr_code.startswith("data:image/png;base64,")
    assert len(out.backup_codes) == 2
    assert auth._mfa_setup_cache_key(7) in sm.cached


@pytest.mark.asyncio
async def test_verify_mfa_setup_succeeds_when_email_send_fails(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_mfa_supported() -> bool:
        return True

    class _StubMFA:
        def verify_totp(self, _secret: str, _token: str) -> bool:
            return True

        async def enable_mfa(self, user_id: int, secret: str, backup_codes: list[str]) -> bool:
            return bool(user_id and secret and backup_codes)

    class _StubSessionManager:
        def __init__(self):
            self.redis_client = object()
            self.deleted = []
            self.values = {
                auth._mfa_setup_cache_key(9): json.dumps(
                    {"secret": "SECRET", "backup_codes": ["b1", "b2"]}
                )
            }

        async def initialize(self):
            return None

        async def get_ephemeral_value(self, key: str):
            return self.values.get(key)

        async def delete_ephemeral_value(self, key: str):
            self.deleted.append(key)

    class _FailingEmail:
        async def send_mfa_enabled_email(self, **kwargs):
            raise RuntimeError("mail provider unavailable")

    monkeypatch.setattr(auth, "_is_mfa_backend_supported", _fake_mfa_supported)
    monkeypatch.setattr(auth, "_get_mfa_service", lambda: _StubMFA())
    monkeypatch.setattr(auth, "_get_email_service", lambda: _FailingEmail())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/mfa/verify",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    sm = _StubSessionManager()
    out = await auth.verify_mfa_setup(
        data=auth.MFAVerifyRequest(token="123456"),
        request=request,
        current_user={
            "id": 9,
            "username": "dictmfa",
            "email": "dictmfa@example.com",
            "is_active": True,
            "is_verified": True,
        },
        session_manager=sm,
    )

    assert "enabled" in out.get("message", "").lower()
    assert out.get("backup_codes") == ["b1", "b2"]
    assert auth._mfa_setup_cache_key(9) in sm.deleted


@pytest.mark.asyncio
async def test_verify_mfa_setup_rate_limited_returns_429(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_mfa_supported() -> bool:
        return True

    class _StubSessionManager:
        def __init__(self):
            self.redis_client = object()
            self.values = {auth._mfa_setup_cache_key(9): json.dumps({"secret": "SECRET"})}

        async def initialize(self):
            return None

        async def get_ephemeral_value(self, key: str):
            return self.values.get(key)

    class _StubMFA:
        def verify_totp(self, _secret: str, _token: str) -> bool:
            return True

    async def _deny_rg(*args, **kwargs):
        return False, 3

    monkeypatch.setattr(auth, "_is_mfa_backend_supported", _fake_mfa_supported)
    monkeypatch.setattr(auth, "_get_mfa_service", lambda: _StubMFA())
    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _deny_rg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/mfa/verify",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    with pytest.raises(auth.HTTPException) as exc:
        await auth.verify_mfa_setup(
            data=auth.MFAVerifyRequest(token="123456"),
            request=request,
            current_user={"id": 9, "username": "u9", "email": "u9@example.com"},
            session_manager=_StubSessionManager(),
        )

    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "3"


@pytest.mark.asyncio
async def test_mfa_login_rate_limited_returns_429(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_mfa_supported() -> bool:
        return True

    class _StubSessionManager:
        def __init__(self):
            self.redis_client = object()
            self.ephemeral = {auth._mfa_login_cache_key("session-token"): '{"user_id": 5, "session_id": 55}'}

        async def initialize(self):
            return None

        async def get_ephemeral_value(self, key: str):
            return self.ephemeral.get(key)

    class _StubLimiter:
        enabled = False

        async def reset_failed_attempts(self, *args, **kwargs):
            return None

    async def _deny_rg(*args, **kwargs):
        return False, 9

    monkeypatch.setattr(auth, "_is_mfa_backend_supported", _fake_mfa_supported)
    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _deny_rg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/mfa/login",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    response = Response()

    with pytest.raises(auth.HTTPException) as exc:
        await auth.mfa_login(
            data=auth.MFALoginRequest(session_token="session-token", mfa_token="123456"),
            request=request,
            response=response,
            db=object(),
            jwt_service=object(),
            session_manager=_StubSessionManager(),
            rate_limiter=_StubLimiter(),
            settings=auth.get_settings(),
        )

    assert exc.value.status_code == 429
    assert exc.value.headers.get("Retry-After") == "9"


@pytest.mark.asyncio
async def test_logout_accepts_lowercase_bearer(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    captured = {"revoked": False}

    class StubJWT:
        def extract_jti(self, token: str) -> str:
            return "jti-abc"

        def verify_token(self, token: str):
            return {"exp": 1_700_000_000, "session_id": 123}

    class StubBlacklist:
        async def revoke_all_user_tokens(self, **kwargs):
            return 0

        async def revoke_token(self, **kwargs):
            captured["revoked"] = True
            return True

    class StubSessionManager:
        async def revoke_all_user_sessions(self, **kwargs):
            return None

        async def revoke_session(self, **kwargs):
            return None

    monkeypatch.setattr(auth, "get_token_blacklist", lambda: StubBlacklist())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [(b"authorization", b"bearer my-token")],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    data = auth.LogoutRequest(all_devices=False)
    current_user = SimpleNamespace(id=99)
    session_manager = StubSessionManager()
    jwt_service = StubJWT()

    result = await auth.logout(
        data=data,
        request=request,
        current_user=current_user,
        session_manager=session_manager,
        jwt_service=jwt_service,
    )

    assert result.message == "Successfully logged out"
    assert captured["revoked"] is True


@pytest.mark.asyncio
async def test_logout_all_devices_uses_session_manager_only(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    captured = {"calls": 0, "user_id": None, "reason": None}

    class _StubSessionManager:
        async def revoke_all_user_sessions(
            self,
            *,
            user_id: int,
            except_session_id: int | None = None,
            reason: str = "",
        ) -> int:
            captured["calls"] += 1
            captured["user_id"] = user_id
            captured["reason"] = reason
            return 4

    class _BlacklistShouldNotBeCalled:
        async def revoke_all_user_tokens(self, **kwargs):
            raise AssertionError("logout(all_devices=true) should not call revoke_all_user_tokens directly")

        async def revoke_token(self, **kwargs):
            raise AssertionError("single-token revoke path should not run for logout(all_devices=true)")

    monkeypatch.setattr(auth, "get_token_blacklist", lambda: _BlacklistShouldNotBeCalled())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [(b"authorization", b"Bearer unused-for-all-devices")],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    result = await auth.logout(
        data=auth.LogoutRequest(all_devices=True),
        request=request,
        current_user=SimpleNamespace(id=12),
        session_manager=_StubSessionManager(),
        jwt_service=object(),
    )

    assert result.message == "Logged out from 4 device(s)"
    assert captured["calls"] == 1
    assert captured["user_id"] == 12
    assert captured["reason"] == "User requested logout from all devices"


@pytest.mark.asyncio
async def test_logout_rejects_api_key_authenticated_current_session_logout(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _StubBlacklist:
        async def revoke_all_user_tokens(self, **kwargs):
            raise AssertionError("current-session API-key logout should fail before token revocation")

        async def revoke_token(self, **kwargs):
            raise AssertionError("current-session API-key logout should fail before access token revocation")

    class _StubSessionManager:
        def __init__(self):
            self.calls = []

        async def revoke_all_user_sessions(self, **kwargs):
            self.calls.append(("all", kwargs))
            return 0

        async def revoke_session(self, **kwargs):
            self.calls.append(("single", kwargs))
            return None

    monkeypatch.setattr(auth, "get_token_blacklist", lambda: _StubBlacklist())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [(b"x-api-key", b"test-api-key-123")],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    session_manager = _StubSessionManager()

    with pytest.raises(auth.HTTPException) as exc:
        await auth.logout(
            data=auth.LogoutRequest(all_devices=False),
            request=request,
            current_user=SimpleNamespace(user_id=12, kind="api_key"),
            session_manager=session_manager,
            jwt_service=object(),
        )

    assert exc.value.status_code == 400
    assert "bearer token" in exc.value.detail.lower()
    assert session_manager.calls == []


@pytest.mark.asyncio
async def test_logout_all_devices_ignores_blacklist_factory_failure(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _StubSessionManager:
        async def revoke_all_user_sessions(
            self,
            *,
            user_id: int,
            except_session_id: int | None = None,
            reason: str = "",
        ) -> int:
            return 1

    def _failing_blacklist_factory():
        raise RuntimeError("blacklist unavailable")

    monkeypatch.setattr(auth, "get_token_blacklist", _failing_blacklist_factory)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    result = await auth.logout(
        data=auth.LogoutRequest(all_devices=True),
        request=request,
        current_user=SimpleNamespace(id=12),
        session_manager=_StubSessionManager(),
        jwt_service=object(),
    )

    assert result.message == "Logged out from 1 device(s)"


@pytest.mark.asyncio
async def test_logout_all_devices_raises_when_session_revoke_fails(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _FailingSessionManager:
        async def revoke_all_user_sessions(
            self,
            *,
            user_id: int,
            except_session_id: int | None = None,
            reason: str = "",
        ) -> int:
            raise RuntimeError("session revoke failed")

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    with pytest.raises(auth.HTTPException) as exc:
        await auth.logout(
            data=auth.LogoutRequest(all_devices=True),
            request=request,
            current_user=SimpleNamespace(id=12),
            session_manager=_FailingSessionManager(),
            jwt_service=object(),
        )

    assert exc.value.status_code == 500
    assert "revoke sessions" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_revoke_all_sessions_returns_revoked_count():
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    captured = {}

    class _StubSessionManager:
        async def revoke_all_user_sessions(
            self,
            *,
            user_id: int,
            except_session_id: int | None = None,
            reason: str = "",
        ) -> int:
            captured["user_id"] = user_id
            captured["except_session_id"] = except_session_id
            captured["reason"] = reason
            return 3

    result = await auth.revoke_all_sessions(
        current_user={"id": 77, "username": "carol"},
        session_manager=_StubSessionManager(),
    )

    assert result.message == "Successfully revoked 3 sessions"
    assert result.details == {"sessions_revoked": 3}
    assert captured["user_id"] == 77
    assert captured["except_session_id"] is None
    assert captured["reason"] == "User requested logout from all devices"


@pytest.mark.asyncio
async def test_list_user_sessions_sanitizes_backend_error_in_test_mode(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _FailingSessionManager:
        async def get_user_sessions(self, user_id: int):
            raise RuntimeError(f"session backend exploded for user {user_id}")

    monkeypatch.setattr(auth, "_is_test_mode", lambda: True)

    with pytest.raises(HTTPException) as exc:
        await auth.list_user_sessions(
            current_user=SimpleNamespace(id=77),
            session_manager=_FailingSessionManager(),
        )

    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to retrieve sessions"


@pytest.mark.asyncio
async def test_logout_raises_when_access_token_revoke_fails(monkeypatch):
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    class _StubJWT:
        def extract_jti(self, token: str) -> str:
            return "jti-fail"

        def verify_token(self, token: str):
            return {"exp": 1_700_000_000, "session_id": 456}

    class _FailingBlacklist:
        async def revoke_all_user_tokens(self, **kwargs):
            return 0

        async def revoke_token(self, **kwargs):
            raise RuntimeError("blacklist write failed")

    class _StubSessionManager:
        async def revoke_all_user_sessions(self, **kwargs):
            return None

        async def revoke_session(self, **kwargs):
            return None

    monkeypatch.setattr(auth, "get_token_blacklist", lambda: _FailingBlacklist())

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/logout",
        "headers": [(b"authorization", b"Bearer token-to-revoke")],
        "client": ("203.0.113.5", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    with pytest.raises(auth.HTTPException) as exc:
        await auth.logout(
            data=auth.LogoutRequest(all_devices=False),
            request=request,
            current_user=SimpleNamespace(id=42),
            session_manager=_StubSessionManager(),
            jwt_service=_StubJWT(),
        )

    assert exc.value.status_code == 500
    assert "revoke access token" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_refresh_single_user_respects_ip_allowlist(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "single-key")
    monkeypatch.setenv("SINGLE_USER_ALLOWED_IPS", "127.0.0.1")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_fetch_active_user_by_id(db, user_id: int):
        return {
            "id": user_id,
            "username": "single_user",
            "email": "single@example.com",
            "role": "admin",
            "is_active": True,
        }

    monkeypatch.setattr(auth, "fetch_active_user_by_id", _fake_fetch_active_user_by_id)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/refresh",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    response = Response()

    with pytest.raises(auth.HTTPException) as exc:
        await auth.refresh_token(
            payload=auth.RefreshTokenRequest(refresh_token="single-key"),
            response=response,
            http_request=request,
            jwt_service=object(),
            session_manager=object(),
            db=None,
            settings=auth.get_settings(),
    )
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_login_rejects_single_user_mode_for_enterprise_admin_ui(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "single-key")
    monkeypatch.setenv("ADMIN_UI_ENTERPRISE_MODE", "true")
    reset_settings()

    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    async def _fake_get_auth_governor():
        return SimpleNamespace()

    async def _fake_fetch_user_by_login_identifier(_db, _identifier: str):
        return {
            "id": 42,
            "username": "admin",
            "email": "admin@example.com",
            "role": "admin",
            "is_active": True,
            "password_hash": "hashed-password",
        }

    monkeypatch.setattr(auth, "get_auth_governor", _fake_get_auth_governor)
    monkeypatch.setattr(auth, "fetch_user_by_login_identifier", _fake_fetch_user_by_login_identifier)

    request = Request({
        "type": "http",
        "method": "POST",
        "path": "/api/v1/auth/login",
        "headers": [],
        "client": ("203.0.113.10", 1234),
        "scheme": "http",
        "query_string": b"",
        "server": ("testserver", 80),
    })
    response = Response()

    password_service = SimpleNamespace(
        verify_password=lambda _password, _hash: (True, False),
    )
    rate_limiter = SimpleNamespace(enabled=False)

    with pytest.raises(auth.HTTPException) as exc:
        await auth.login(
            request=request,
            response=response,
            form_data=SimpleNamespace(username="admin", password="password123"),
            db=None,
            jwt_service=object(),
            password_service=password_service,
            session_manager=SimpleNamespace(),
            rate_limiter=rate_limiter,
            settings=auth.get_settings(),
        )

    assert exc.value.status_code == 403
    assert "enterprise admin ui" in exc.value.detail.lower()
