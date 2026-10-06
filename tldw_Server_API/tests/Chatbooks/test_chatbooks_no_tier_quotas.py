"""Chatbooks quotas come only from limits.chatbooks_* (spec 2 §7): no tier caps remain."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chatbooks import quota_manager as qm

pytestmark = pytest.mark.unit


def test_quota_manager_has_no_tier_tables() -> None:
    """The free/premium/enterprise tables and their checks are gone."""
    for name in ("DEFAULT_QUOTAS", "PREMIUM_QUOTAS", "UserTier", "get_quota_manager"):
        assert not hasattr(qm, name), name
    for method in ("check_export_quota", "check_import_quota", "check_concurrent_jobs",
                   "check_storage_quota", "get_usage_summary"):
        assert not hasattr(qm.QuotaManager, method), method


async def test_file_size_cap_is_the_constant_for_every_tier() -> None:
    """check_file_size uses MAX_CHATBOOK_FILE_SIZE_MB whatever tier string is passed."""
    assert qm.MAX_CHATBOOK_FILE_SIZE_MB == 100
    limit = qm.MAX_CHATBOOK_FILE_SIZE_MB * 1024 * 1024
    for tier in ("free", "premium", "enterprise", "nonsense"):
        manager = qm.QuotaManager("7", tier)
        assert (await manager.check_file_size(limit))[0] is True
        allowed, message = await manager.check_file_size(limit + 1)
        assert allowed is False and "100MB" in message


class _TenExportsDb:
    """DB fake that reports ten jobs for every count query, recording the queries."""

    def __init__(self) -> None:
        """Start with no recorded queries."""
        self.queries: list[str] = []

    def execute_query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Record the query and answer ten for any count."""
        self.queries.append(sql)
        return [{"c": 10, "COUNT(1)": 10}]


class _CaptureService:
    """Export service fake that records create_chatbook calls."""

    def __init__(self) -> None:
        """Attach the ten-exports DB and an empty call list."""
        self.db = _TenExportsDb()
        self.calls: list[dict] = []

    async def create_chatbook(self, **kwargs):
        """Record the call and report a queued job."""
        self.calls.append(kwargs)
        return True, "queued", "job-1"


class _AuditStub:
    """Audit service stub."""

    async def log_event(self, *args, **kwargs) -> None:
        """Ignore the event."""
        return None


async def _user() -> User:
    """Return a fixed authenticated user."""
    return User(id=1, username="tester", email=None, is_active=True)


def test_export_endpoint_has_no_tier_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """With quotas on and no limits.chatbooks_* set, an 11th export in a day is not refused by a tier cap."""
    for name in ("CHATBOOKS_DISABLE_QUOTAS", "TEST_MODE", "TESTING", "PYTEST_CURRENT_TEST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    assert qm.QuotaManager("1", "free")._quotas_disabled is False
    service = _CaptureService()
    app = FastAPI()
    app.include_router(chatbooks_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[chatbooks_endpoints.get_chatbook_service] = lambda: service
    app.dependency_overrides[chatbooks_endpoints.get_request_user] = _user
    app.dependency_overrides[chatbooks_endpoints.get_audit_service_for_user] = lambda: _AuditStub()

    response = TestClient(app).post(
        "/api/v1/chatbooks/export",
        json={"name": "Eleventh", "description": "tier cap check", "async_mode": True},
    )

    assert response.status_code != 429, response.text
    assert len(service.calls) == 1
