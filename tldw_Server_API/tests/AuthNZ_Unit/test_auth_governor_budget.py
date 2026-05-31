from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.types import Scope

from tldw_Server_API.app.core.AuthNZ.auth_governor import AuthGovernor
from tldw_Server_API.app.core.AuthNZ.llm_budget_guard import enforce_llm_budget
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.debugs: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        self.errors.append(message)

    def debug(self, message: str, *args, **kwargs) -> None:
        self.debugs.append(message)


def _make_request(headers: dict[str, str] | None = None, *, path: str = "/api/v1/chat/completions") -> Request:
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_auth_governor_decorates_over_budget_result_with_principal(monkeypatch):
    async def _fake_is_key_over_budget(_key_id: int):
        return {
            "over": True,
            "reasons": ["day_tokens_exceeded:100/10"],
            "day": {"tokens": 100, "usd": 0.5},
            "month": {"tokens": 2000, "usd": 10.0},
            "limits": {"is_virtual": True},
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.is_key_over_budget",
        _fake_is_key_over_budget,
    )

    principal = AuthPrincipal(
        kind="api_key",
        user_id=1,
        api_key_id=123,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    gov = AuthGovernor()
    result = await gov.check_llm_budget_for_api_key(principal, 123)

    assert result["over"] is True
    assert result["limits"]["is_virtual"] is True
    meta = result.get("principal") or {}
    assert meta.get("principal_id")
    assert meta.get("api_key_id") == 123
    assert meta.get("user_id") == 1


@pytest.mark.asyncio
async def test_auth_governor_under_budget_includes_principal(monkeypatch):
    async def _fake_is_key_over_budget(_key_id: int):
        return {
            "over": False,
            "reasons": [],
            "day": {"tokens": 10, "usd": 0.01},
            "month": {"tokens": 100, "usd": 0.1},
            "limits": {"is_virtual": True},
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.is_key_over_budget",
        _fake_is_key_over_budget,
    )

    principal = AuthPrincipal(
        kind="api_key",
        user_id=2,
        api_key_id=456,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    gov = AuthGovernor()
    result = await gov.check_llm_budget_for_api_key(principal, 456)

    assert result["over"] is False
    assert result["limits"]["is_virtual"] is True
    meta = result.get("principal") or {}
    assert meta.get("principal_id")
    assert meta.get("api_key_id") == 456
    assert meta.get("user_id") == 2


@pytest.mark.asyncio
async def test_auth_governor_budget_check_failure_fails_open(monkeypatch):
    marker = "budget backend down secret"

    async def _fake_is_key_over_budget(_key_id: int):
        raise RuntimeError(marker)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.is_key_over_budget",
        _fake_is_key_over_budget,
    )
    monkeypatch.setenv("AUTH_BUDGET_FAIL_OPEN", "1")

    principal = AuthPrincipal(
        kind="api_key",
        user_id=3,
        api_key_id=789,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    gov = AuthGovernor()
    result = await gov.check_llm_budget_for_api_key(principal, 789)

    assert result["over"] is False
    assert result["reasons"] == []
    meta = result.get("principal") or {}
    assert meta.get("api_key_id") == 789
    assert meta.get("user_id") == 3


@pytest.mark.asyncio
async def test_auth_governor_budget_check_failure_fails_closed(monkeypatch):
    marker = "budget backend down secret"

    async def _fake_is_key_over_budget(_key_id: int):
        raise RuntimeError(marker)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.is_key_over_budget",
        _fake_is_key_over_budget,
    )
    monkeypatch.setenv("AUTH_BUDGET_FAIL_OPEN", "0")

    principal = AuthPrincipal(
        kind="api_key",
        user_id=4,
        api_key_id=321,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    gov = AuthGovernor()
    result = await gov.check_llm_budget_for_api_key(principal, 321)

    assert result["over"] is True
    assert "budget_check_failed" in result.get("reasons", [])
    assert result.get("error") == "budget_check_failed"
    assert marker not in str(result)
    meta = result.get("principal") or {}
    assert meta.get("api_key_id") == 321
    assert meta.get("user_id") == 4


@pytest.mark.asyncio
async def test_auth_governor_budget_check_failure_defaults_to_fail_closed(monkeypatch):
    marker = "budget backend down secret"

    async def _fake_is_key_over_budget(_key_id: int):
        raise RuntimeError(marker)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.is_key_over_budget",
        _fake_is_key_over_budget,
    )
    monkeypatch.delenv("AUTH_BUDGET_FAIL_OPEN", raising=False)

    principal = AuthPrincipal(
        kind="api_key",
        user_id=5,
        api_key_id=654,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    gov = AuthGovernor()
    result = await gov.check_llm_budget_for_api_key(principal, 654)

    assert result["over"] is True
    assert "budget_check_failed" in result.get("reasons", [])
    assert result.get("error") == "budget_check_failed"
    assert marker not in str(result)
    meta = result.get("principal") or {}
    assert meta.get("api_key_id") == 654
    assert meta.get("user_id") == 5


@pytest.mark.asyncio
async def test_enforce_llm_budget_uses_auth_governor_and_raises_402(monkeypatch):
    # Settings with budget enforcement enabled and virtual keys enabled
    fake_settings = SimpleNamespace(
        VIRTUAL_KEYS_ENABLED=True,
        LLM_BUDGET_ENFORCE=True,
    )

    def _fake_get_settings():

        return fake_settings

    async def _fake_resolve_api_key_by_hash(api_key: str, settings=None):
        # settings is accepted for signature compatibility but ignored.
        _ = settings
        return {"id": 123, "user_id": 7}

    async def _fake_get_auth_governor():
        class _FakeGov:
            async def check_llm_budget_for_api_key(self, principal, _api_key_id: int):
                return {
                    "over": True,
                    "reasons": ["day_tokens_exceeded:100/10"],
                    "day": {"tokens": 100, "usd": 0.5},
                    "month": {"tokens": 2000, "usd": 10.0},
                    "limits": {"is_virtual": True},
                    "principal": {
                        "principal_id": principal.principal_id,
                        "kind": principal.kind,
                        "user_id": principal.user_id,
                        "api_key_id": principal.api_key_id,
                    },
                }

        return _FakeGov()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.get_settings",
        _fake_get_settings,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.resolve_api_key_by_hash",
        _fake_resolve_api_key_by_hash,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.get_auth_governor",
        _fake_get_auth_governor,
    )

    req = _make_request(headers={"X-API-KEY": "vk-key"})

    # Pre-populate a minimal AuthContext so the guard can reuse it if desired
    principal = AuthPrincipal(
        kind="api_key",
        user_id=7,
        api_key_id=123,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    req.state.auth = AuthContext(
        principal=principal,
        ip="127.0.0.1",
        user_agent="pytest-agent",
        request_id="req-llm-budget",
    )

    with pytest.raises(HTTPException) as exc_info:
        await enforce_llm_budget(req)

    assert exc_info.value.status_code == 402
    detail = exc_info.value.detail
    assert detail.get("error") == "budget_exceeded"
    assert detail.get("details", {}).get("over") is True


@pytest.mark.asyncio
async def test_auth_governor_lockout_checks_rate_limiter(monkeypatch):
    calls = {}

    class _Limiter:
        enabled = True

        async def check_lockout(self, identifier):
            calls["identifier"] = identifier
            return True, "soon"

    async def _fake_get_rate_limiter():
        return _Limiter()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.get_rate_limiter",
        _fake_get_rate_limiter,
    )

    gov = AuthGovernor()
    locked, expires = await gov.check_lockout("1.2.3.4")
    assert locked is True
    assert expires == "soon"
    assert calls.get("identifier") == "1.2.3.4"


@pytest.mark.asyncio
async def test_auth_governor_record_auth_failure_respects_limiter(monkeypatch):
    calls = {}

    class _Limiter:
        enabled = True

        async def record_failed_attempt(self, *, identifier, attempt_type):
            calls["identifier"] = identifier
            calls["attempt_type"] = attempt_type
            return {"is_locked": False, "remaining_attempts": 2}

    async def _fake_get_rate_limiter():
        return _Limiter()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.get_rate_limiter",
        _fake_get_rate_limiter,
    )

    gov = AuthGovernor()
    result = await gov.record_auth_failure("bad-user", attempt_type="login")
    assert result["is_locked"] is False
    assert result["remaining_attempts"] == 2
    assert calls.get("identifier") == "bad-user"
    assert calls.get("attempt_type") == "login"


@pytest.mark.asyncio
async def test_auth_governor_lockout_fallback_when_limiter_missing(monkeypatch):
    async def _fake_get_rate_limiter():
        raise RuntimeError("no limiter")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.get_rate_limiter",
        _fake_get_rate_limiter,
    )

    gov = AuthGovernor()
    locked, expires = await gov.check_lockout("noop")
    assert locked is False
    assert expires is None

    result = await gov.record_auth_failure("noop")
    assert result["is_locked"] is False
    assert result["remaining_attempts"] == 5


@pytest.mark.asyncio
async def test_auth_governor_rate_limit_delegates_to_limiter(monkeypatch):
    calls: dict[str, Any] = {}

    class _Limiter:
        enabled = True

        async def check_rate_limit(self, identifier, endpoint, **kwargs):
            calls["identifier"] = identifier
            calls["endpoint"] = endpoint
            calls["kwargs"] = kwargs
            return False, {"retry_after": 42}

    async def _fake_get_rate_limiter():
        return _Limiter()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.get_rate_limiter",
        _fake_get_rate_limiter,
    )

    gov = AuthGovernor()
    allowed, meta = await gov.check_rate_limit(
        identifier="1.2.3.4",
        endpoint="auth",
        limit=5,
        window_minutes=1,
    )
    assert allowed is False
    assert meta.get("retry_after") == 42
    assert calls["identifier"] == "1.2.3.4"
    assert calls["endpoint"] == "auth"
    assert calls["kwargs"]["limit"] == 5
    assert calls["kwargs"]["window_minutes"] == 1


@pytest.mark.asyncio
async def test_auth_governor_rate_limit_fails_open_when_limiter_missing(monkeypatch):
    async def _fake_get_rate_limiter():
        raise RuntimeError("no limiter")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.auth_governor.get_rate_limiter",
        _fake_get_rate_limiter,
    )

    gov = AuthGovernor()
    allowed, meta = await gov.check_rate_limit(
        identifier="1.2.3.4",
        endpoint="auth",
        limit=5,
        window_minutes=1,
    )
    assert allowed is True
    assert meta == {}


@pytest.mark.asyncio
async def test_enforce_llm_budget_allows_under_budget_chat(monkeypatch):
    fake_settings = SimpleNamespace(
        VIRTUAL_KEYS_ENABLED=True,
        LLM_BUDGET_ENFORCE=True,
    )

    def _fake_get_settings():

        return fake_settings

    async def _fake_resolve_api_key_by_hash(api_key: str, settings=None):
        _ = settings
        return {"id": 555, "user_id": 9}

    async def _fake_get_auth_governor():
        class _Gov:
            async def check_llm_budget_for_api_key(self, principal, api_key_id: int):
                return {
                    "over": False,
                    "limits": {"is_virtual": True},
                    "principal": {
                        "principal_id": principal.principal_id,
                        "api_key_id": api_key_id,
                        "user_id": principal.user_id,
                    },
                }

        return _Gov()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.get_settings",
        _fake_get_settings,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.resolve_api_key_by_hash",
        _fake_resolve_api_key_by_hash,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.get_auth_governor",
        _fake_get_auth_governor,
    )

    req = _make_request(headers={"X-API-KEY": "vk-allowed"}, path="/api/v1/chat/completions")

    await enforce_llm_budget(req)

    assert getattr(req.state, "api_key_id", None) == 555
    assert getattr(req.state, "user_id", None) == 9


@pytest.mark.asyncio
async def test_enforce_llm_budget_allows_under_budget_embeddings(monkeypatch):
    fake_settings = SimpleNamespace(
        VIRTUAL_KEYS_ENABLED=True,
        LLM_BUDGET_ENFORCE=True,
    )

    def _fake_get_settings():

        return fake_settings

    async def _fake_resolve_api_key_by_hash(api_key: str, settings=None):
        _ = settings
        return {"id": 777, "user_id": 11}

    async def _fake_get_auth_governor():
        class _Gov:
            async def check_llm_budget_for_api_key(self, principal, api_key_id: int):
                return {
                    "over": False,
                    "limits": {"is_virtual": True},
                    "principal": {
                        "principal_id": principal.principal_id,
                        "api_key_id": api_key_id,
                        "user_id": principal.user_id,
                    },
                }

        return _Gov()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.get_settings",
        _fake_get_settings,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.resolve_api_key_by_hash",
        _fake_resolve_api_key_by_hash,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.llm_budget_guard.get_auth_governor",
        _fake_get_auth_governor,
    )

    req = _make_request(headers={"X-API-KEY": "vk-embed"}, path="/api/v1/embeddings")

    await enforce_llm_budget(req)

    assert getattr(req.state, "api_key_id", None) == 777
    assert getattr(req.state, "user_id", None) == 11


@pytest.mark.asyncio
async def test_auth_governor_budget_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import auth_governor as auth_governor_module

    async def _fake_is_key_over_budget(_key_id: int):
        raise RuntimeError("budget backend failed at /private/budget.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(auth_governor_module, "is_key_over_budget", _fake_is_key_over_budget)
    monkeypatch.setattr(auth_governor_module, "logger", logger_stub)
    monkeypatch.setenv("AUTH_BUDGET_FAIL_OPEN", "1")

    principal = AuthPrincipal(
        kind="api_key",
        user_id=12,
        api_key_id=888,
        subject=None,
        token_type="api_key",  # nosec B106
        jti=None,
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    result = await AuthGovernor().check_llm_budget_for_api_key(principal, 888)

    assert result["over"] is False
    assert logger_stub.errors == ["AuthGovernor budget check failed"]
    assert "budget backend failed" not in str(logger_stub.errors)
    assert "/private/budget.db" not in str(logger_stub.errors)
    assert "888" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_auth_governor_limiter_failure_logs_are_sanitized(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import auth_governor as auth_governor_module

    class _FailingLimiter:
        enabled = True

        async def check_lockout(self, identifier, attempt_type="login"):
            raise RuntimeError("lockout backend failed at /private/lockout.db")

        async def record_failed_attempt(self, *, identifier, attempt_type):
            raise RuntimeError("record backend failed at /private/record.db")

        async def check_rate_limit(self, identifier, endpoint, **kwargs):
            raise RuntimeError("rate backend failed at /private/rate.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(auth_governor_module, "logger", logger_stub)

    gov = AuthGovernor()
    locked, expires = await gov.check_lockout("user@example.test", rate_limiter=_FailingLimiter())
    failure = await gov.record_auth_failure("user@example.test", rate_limiter=_FailingLimiter())
    allowed, meta = await gov.check_rate_limit(
        "user@example.test",
        "auth/login",
        rate_limiter=_FailingLimiter(),
    )

    assert (locked, expires) == (False, None)
    assert failure == {"is_locked": False, "remaining_attempts": 5}
    assert (allowed, meta) == (True, {})
    assert logger_stub.debugs == [
        "AuthGovernor lockout check failed",
        "AuthGovernor record failure failed",
        "AuthGovernor rate limit failed",
    ]
    joined = "\n".join(logger_stub.debugs)
    assert "user@example.test" not in joined
    assert "backend failed" not in joined
    assert "/private/" not in joined
