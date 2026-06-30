from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import audit as audit_endpoint
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


class _LoggerStub:
    def __init__(self):
        self.debugs = []

    def debug(self, message, *args, **kwargs):
        self.debugs.append((message, args, kwargs))


def _make_principal(
    *,
    user_id: int = 1,
    kind: str = "user",
    is_admin: bool = False,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=user_id,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def _override_principal(app, principal: AuthPrincipal | None, *, fail_with_401: bool = False) -> None:
    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None, "principal must be provided when fail_with_401 is False"
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal


@asynccontextmanager
async def _get_client(monkeypatch):
    """Yield an httpx.AsyncClient backed by the FastAPI app plus the app itself."""
    monkeypatch.setenv("TEST_MODE", "true")
    from tldw_Server_API.app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=httpx.Timeout(10.0),
    ) as client:
        try:
            yield client, app
        finally:
            app.dependency_overrides.clear()


def test_audit_event_type_value_mapping_debug_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(audit_endpoint, "logger", logger_stub)

    mapped = audit_endpoint._map_event_types("auth.login.success")

    assert mapped == [audit_endpoint.AuditEventType.AUTH_LOGIN_SUCCESS]
    assert logger_stub.debugs == [("Audit event type enum-key mapping failed", (), {})]
    assert "auth.login.success" not in repr(logger_stub.debugs)
    assert "exc_info" not in repr(logger_stub.debugs)


def test_audit_category_value_mapping_debug_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(audit_endpoint, "logger", logger_stub)

    mapped = audit_endpoint._map_categories("data_access")

    assert mapped == [audit_endpoint.AuditEventCategory.DATA_ACCESS]
    assert logger_stub.debugs == [("Audit category enum-key mapping failed", (), {})]
    assert "data_access" not in repr(logger_stub.debugs)
    assert "exc_info" not in repr(logger_stub.debugs)


@pytest.mark.asyncio
async def test_audit_export_requires_admin(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="tester", is_active=True)

        principal = _make_principal(is_admin=False, roles=["user"], permissions=[])
        _override_principal(app, principal)

        r = await client.get("/api/v1/audit/export", headers={"X-API-KEY": "test-api-key-12345"})
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_audit_export_allows_admin_and_returns_payload(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

        class _StubAudit:
            async def export_events(self, **kwargs):
                fmt = (kwargs.get("format") or "json").lower()
                if fmt == "csv":
                    return "event_id,timestamp\n1,2025-01-01T00:00:00Z\n"
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get("/api/v1/audit/export?format=json", headers={"X-API-KEY": "test-api-key-12345"})
        try:
            print("JSON export response:", r.status_code, r.json())
        except Exception:
            print("JSON export response (raw):", r.status_code, r.text)
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/json")
        assert r.text.strip() == "[]"
        assert "attachment" in r.headers.get("content-disposition", "").lower()

        r = await client.get("/api/v1/audit/export?format=csv", headers={"X-API-KEY": "test-api-key-12345"})
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("text/csv")
        assert r.text.splitlines()[0].startswith("event_id,")
        assert "attachment" in r.headers.get("content-disposition", "").lower()


@pytest.mark.asyncio
async def test_audit_export_filename_content_disposition(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

        class _StubAudit:
            async def export_events(self, **kwargs):
                fmt = (kwargs.get("format") or "json").lower()
                if fmt == "csv":
                    return "event_id,timestamp\n1,2025-01-01T00:00:00Z\n"
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=json&filename=custom_audit.json",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/json")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd.lower()
        assert "filename=custom_audit.json" in cd

        r = await client.get(
            "/api/v1/audit/export?format=csv&filename=export.csv",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("text/csv")
        cd = r.headers.get("content-disposition", "")
        assert "attachment" in cd.lower()
        assert "filename=export.csv" in cd


@pytest.mark.asyncio
async def test_audit_export_parses_Z_timestamps(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured["start_time"] = kwargs.get("start_time")
                captured["end_time"] = kwargs.get("end_time")
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        start_z = "2025-01-01T00:00:00Z"
        end_z = "2025-01-02T12:34:56Z"
        r = await client.get(
            f"/api/v1/audit/export?format=json&start_time={start_z}&end_time={end_z}",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/json")
        assert r.text.strip() == "[]"

        from datetime import datetime, timezone

        assert captured.get("start_time") == datetime.fromisoformat("2025-01-01T00:00:00+00:00")
        assert captured.get("end_time") == datetime.fromisoformat("2025-01-02T12:34:56+00:00")
        assert captured["start_time"].tzinfo is not None and captured["end_time"].tzinfo is not None


@pytest.mark.asyncio
async def test_audit_export_streaming_json(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {"stream": None}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured["stream"] = kwargs.get("stream")

                async def _gen():
                    import json as _json

                    yield "["
                    yield _json.dumps({"event_id": "1"})
                    yield ","
                    yield _json.dumps({"event_id": "2"})
                    yield "]"

                if kwargs.get("stream"):
                    return _gen()
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=json&stream=true",
            headers={"X-API-KEY": "test-api-key-12345"},
        )

        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/json")
        assert "attachment" in r.headers.get("content-disposition", "").lower()
        assert r.text == '[{"event_id": "1"},{"event_id": "2"}]'
        assert captured["stream"] is True


@pytest.mark.asyncio
async def test_audit_export_streaming_csv_supported(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {"stream": None}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured["stream"] = kwargs.get("stream")

                async def _gen():
                    yield "event_id,timestamp\n"
                    yield "1,2025-01-01T00:00:00Z\n"

                if kwargs.get("stream"):
                    return _gen()
                return "event_id,timestamp\n1,2025-01-01T00:00:00Z\n"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=csv&stream=true",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("text/csv")
        assert "attachment" in r.headers.get("content-disposition", "").lower()
        assert "event_id" in r.text
        assert captured["stream"] is True


@pytest.mark.asyncio
async def test_audit_export_jsonl_streaming(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

        class _StubAudit:
            async def export_events(self, **kwargs):
                fmt = (kwargs.get("format") or "json").lower()
                if fmt == "jsonl" and kwargs.get("stream") and kwargs.get("file_path") is None:

                    async def _gen():
                        yield '{"event_id": "1"}\n'
                        yield '{"event_id": "2"}\n'

                    return _gen()
                if fmt == "jsonl":
                    return '{"event_id": "1"}\n{"event_id": "2"}\n'
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=jsonl&stream=true",
            headers={"X-API-KEY": "test-api-key-12345"},
        )

        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/x-ndjson")
        assert "attachment" in r.headers.get("content-disposition", "").lower()
        assert r.text == '{"event_id": "1"}\n{"event_id": "2"}\n'


@pytest.mark.asyncio
async def test_audit_export_auto_streams_for_large_max_rows(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {"stream": None}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured["stream"] = kwargs.get("stream")

                async def _gen():
                    yield "[]"

                if kwargs.get("stream"):
                    return _gen()
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=json&max_rows=100000",
            headers={"X-API-KEY": "test-api-key-12345"},
        )

        assert r.status_code == 200
        assert captured["stream"] is True
        assert r.text.strip() == "[]"


@pytest.mark.asyncio
async def test_audit_export_filters_and_max_rows(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured.update(kwargs)
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        qs = (
            "format=json"
            "&user_id=u1&request_id=req123&correlation_id=corr7"
            "&ip_address=10.0.0.1&session_id=sess9&endpoint=/api/x&method=GET"
            "&max_rows=42"
        )
        r = await client.get(f"/api/v1/audit/export?{qs}", headers={"X-API-KEY": "test-api-key-12345"})
        assert r.status_code == 200
        assert r.headers.get("content-type", "").startswith("application/json")
        assert captured.get("user_id") == "u1"
        assert captured.get("request_id") == "req123"
        assert captured.get("correlation_id") == "corr7"
        assert captured.get("ip_address") == "10.0.0.1"
        assert captured.get("session_id") == "sess9"
        assert captured.get("endpoint") == "/api/x"
        assert captured.get("method") == "GET"
        assert captured.get("max_rows") == 42
        assert captured.get("stream") in (False, None)


@pytest.mark.asyncio
async def test_audit_export_rejects_non_positive_max_rows(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        class _StubAudit:
            async def export_events(self, **kwargs):
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r0 = await client.get(
            "/api/v1/audit/export?format=json&max_rows=0",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r0.status_code == 422

        r1 = await client.get(
            "/api/v1/audit/export?format=json&max_rows=-1",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r1.status_code == 422


@pytest.mark.asyncio
async def test_audit_count_endpoint_filters(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def count_events(self, **kwargs):
                captured.update(kwargs)
                return 123

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        qs = (
            "user_id=u7&request_id=r9&correlation_id=c1&ip_address=1.2.3.4"
            "&session_id=sxy&endpoint=/api/z&method=POST&min_risk_score=50"
            "&event_type=DATA_READ,AUTH_LOGIN_SUCCESS&category=SECURITY,api_call"
        )
        r = await client.get(f"/api/v1/audit/count?{qs}", headers={"X-API-KEY": "test-api-key-12345"})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 123
        assert captured.get("user_id") == "u7"
        assert captured.get("request_id") == "r9"
        assert captured.get("correlation_id") == "c1"
        assert captured.get("ip_address") == "1.2.3.4"
        assert captured.get("session_id") == "sxy"
        assert captured.get("endpoint") == "/api/z"
        assert captured.get("method") == "POST"
        assert captured.get("min_risk_score") == 50


@pytest.mark.asyncio
async def test_audit_count_integration_live(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        user_id_int = 777
        app.dependency_overrides[get_request_user] = lambda: User(id=user_id_int, username="admin", is_active=True)
        principal = _make_principal(user_id=user_id_int, is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        r0 = await client.get(f"/api/v1/audit/count?user_id={user_id_int}", headers={"X-API-KEY": "test-api-key-12345"})
        assert r0.status_code == 200
        assert r0.json()["count"] in (0, int(r0.json()["count"]))

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventType

        svc = await audit_deps.get_or_create_audit_service_for_user_id(user_id_int)

        await svc.log_event(
            event_type=AuditEventType.DATA_READ,
            context=AuditContext(user_id=str(user_id_int)),
            resource_type="doc",
            resource_id="int1",
        )
        await svc.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=AuditContext(user_id=str(user_id_int)),
            resource_type="doc",
            resource_id="int2",
        )
        await svc.flush()

        r1 = await client.get(f"/api/v1/audit/count?user_id={user_id_int}", headers={"X-API-KEY": "test-api-key-12345"})
        assert r1.status_code == 200
        assert r1.json()["count"] >= 2


@pytest.mark.asyncio
async def test_audit_export_shared_mode_forces_tenant_for_non_admin(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
        from tldw_Server_API.app.core.config import settings

        monkeypatch.setitem(settings, "AUDIT_STORAGE_MODE", "shared")
        monkeypatch.setitem(settings, "AUDIT_STORAGE_ROLLBACK", False)

        app.dependency_overrides[get_request_user] = lambda: User(
            id=5, username="tester", is_active=True, is_admin=False
        )
        principal = _make_principal(is_admin=False, roles=["user"], permissions=["system.logs"], user_id=5)
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured.update(kwargs)
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=json&user_id=99",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert captured.get("user_id") == "5"
        assert captured.get("allow_cross_tenant") is False


@pytest.mark.asyncio
async def test_audit_count_shared_mode_allows_admin_cross_tenant(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
        from tldw_Server_API.app.core.config import settings

        monkeypatch.setitem(settings, "AUDIT_STORAGE_MODE", "shared")
        monkeypatch.setitem(settings, "AUDIT_STORAGE_ROLLBACK", False)

        app.dependency_overrides[get_request_user] = lambda: User(
            id=1, username="admin", is_active=True, is_admin=True
        )
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"], user_id=1)
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def count_events(self, **kwargs):
                captured.update(kwargs)
                return 1

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/count?user_id=99",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1
        assert captured.get("user_id") == "99"
        assert captured.get("allow_cross_tenant") is True


@pytest.mark.asyncio
async def test_audit_shared_mode_uses_principal_not_user_admin_for_cross_tenant(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
        from tldw_Server_API.app.core.config import settings

        monkeypatch.setitem(settings, "AUDIT_STORAGE_MODE", "shared")
        monkeypatch.setitem(settings, "AUDIT_STORAGE_ROLLBACK", False)

        # Compatibility user object claims admin, but principal does not.
        app.dependency_overrides[get_request_user] = lambda: User(
            id=5, username="legacy-admin", is_active=True, is_admin=True
        )
        principal = _make_principal(is_admin=False, roles=["user"], permissions=["system.logs"], user_id=5)
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def export_events(self, **kwargs):
                captured.update(kwargs)
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=json&user_id=99",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert captured.get("allow_cross_tenant") is False
        assert captured.get("user_id") == "5"


@pytest.mark.asyncio
async def test_audit_shared_mode_allows_principal_admin_even_if_user_not_admin(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
        from tldw_Server_API.app.core.config import settings

        monkeypatch.setitem(settings, "AUDIT_STORAGE_MODE", "shared")
        monkeypatch.setitem(settings, "AUDIT_STORAGE_ROLLBACK", False)

        app.dependency_overrides[get_request_user] = lambda: User(
            id=8, username="non-admin-user", is_active=True, is_admin=False
        )
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"], user_id=8)
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def count_events(self, **kwargs):
                captured.update(kwargs)
                return 3

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/count?user_id=99",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 3
        assert captured.get("allow_cross_tenant") is True
        assert captured.get("user_id") == "99"


@pytest.mark.asyncio
async def test_audit_shared_mode_allows_system_configure_permission_cross_tenant(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
        from tldw_Server_API.app.core.config import settings

        monkeypatch.setitem(settings, "AUDIT_STORAGE_MODE", "shared")
        monkeypatch.setitem(settings, "AUDIT_STORAGE_ROLLBACK", False)

        app.dependency_overrides[get_request_user] = lambda: User(
            id=11, username="ops-user", is_active=True, is_admin=False
        )
        principal = _make_principal(
            is_admin=False,
            roles=["user"],
            permissions=["system.logs", "system.configure"],
            user_id=11,
        )
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps
        captured = {}

        class _StubAudit:
            async def count_events(self, **kwargs):
                captured.update(kwargs)
                return 7

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/count?user_id=99",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 7
        assert captured.get("allow_cross_tenant") is True
        assert captured.get("user_id") == "99"


@pytest.mark.asyncio
async def test_audit_export_filename_extension_normalization(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

        class _StubAudit:
            async def export_events(self, **kwargs):
                fmt = (kwargs.get("format") or "json").lower()
                if fmt == "csv":
                    return "event_id,timestamp\n1,2025-01-01T00:00:00Z\n"
                return "[]"

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=csv&filename=my_export.json",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        cd = r.headers.get("content-disposition", "")
        assert "filename=my_export.csv" in cd

        r = await client.get(
            "/api/v1/audit/export?format=json&filename=report.txt",
            headers={"X-API-KEY": "test-api-key-12345"},
        )
        cd = r.headers.get("content-disposition", "")
        assert "filename=report.json" in cd


@pytest.mark.asyncio
async def test_audit_export_returns_500_on_read_failure(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
    principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
    _override_principal(app, principal)

    class _StubAudit:
        async def export_events(self, **kwargs):
            raise RuntimeError("audit read failed")

    async def _get_stub_service():
        return _StubAudit()

    app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=httpx.Timeout(10.0),
    ) as client:
        try:
            r = await client.get(
                "/api/v1/audit/export?format=json",
                headers={"X-API-KEY": "test-api-key-12345"},
            )
            assert r.status_code == 500
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_audit_count_returns_500_on_read_failure(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

    app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
    principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
    _override_principal(app, principal)

    class _StubAudit:
        async def count_events(self, **kwargs):
            raise RuntimeError("audit count failed")

    async def _get_stub_service():
        return _StubAudit()

    app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=httpx.Timeout(10.0),
    ) as client:
        try:
            r = await client.get(
                "/api/v1/audit/count",
                headers={"X-API-KEY": "test-api-key-12345"},
            )
            assert r.status_code == 500
        finally:
            app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_audit_export_sets_truncation_headers_when_default_cap_applies(monkeypatch):
    async with _get_client(monkeypatch) as (client, app):
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

        app.dependency_overrides[get_request_user] = lambda: User(id=1, username="admin", is_active=True)
        principal = _make_principal(is_admin=True, roles=["admin"], permissions=["system.logs"])
        _override_principal(app, principal)

        from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

        class _StubAudit:
            non_stream_max_rows = 2

            async def export_events(self, **kwargs):
                return "[]"

            async def count_events(self, **kwargs):
                return 5

        async def _get_stub_service():
            return _StubAudit()

        app.dependency_overrides[audit_deps.get_audit_service_for_user] = _get_stub_service

        r = await client.get(
            "/api/v1/audit/export?format=json",
            headers={"X-API-KEY": "test-api-key-12345"},
        )

        assert r.status_code == 200
        assert r.headers.get("x-audit-export-truncated") == "true"
        assert r.headers.get("x-audit-export-row-limit") == "2"
