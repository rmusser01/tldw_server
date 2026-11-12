from __future__ import annotations

from typing import Any, Optional

import os
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import jobs_admin as jobs_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=1,
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


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(jobs_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
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

    # Stub audit service dependency to avoid hitting real AuthNZ/user flows.
    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

    class _FakeAuditService:
        async def log_event(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            """No-op audit logger for tests."""
            _ = (args, kwargs)
            return None

    async def _fake_get_audit_service_for_user() -> _FakeAuditService:
        return _FakeAuditService()

    app.dependency_overrides[audit_deps.get_audit_service_for_user] = _fake_get_audit_service_for_user

    # Patch JobManager to avoid touching a real DB in these tests
    class _FakeJobManager:
        def __init__(self, backend: Optional[str] = None, db_url: Optional[str] = None) -> None:
            self.backend = backend
            self.db_url = db_url

        def get_queue_flags(self, domain: str, queue: str) -> dict:
            return {"paused": False, "drain": False}

        def set_queue_control(self, domain: str, queue: str, action: str) -> dict:
            # Return a simple, deterministic flags payload; action semantics
            # are not under test here.
            _ = (domain, queue, action)
            return {"paused": False, "drain": False}

        def retry_now_jobs(
            self,
            domain: Optional[str] = None,
            queue: Optional[str] = None,
            job_type: Optional[str] = None,
            job_id: Optional[int] = None,
            only_failed: bool = True,
            dry_run: bool = False,
        ) -> int:
            _ = (domain, queue, job_type, job_id, only_failed, dry_run)
            return 0

    jobs_mod.JobManager = _FakeJobManager  # type: ignore[assignment]

    # Ensure domain-scoped RBAC does not inject additional 403s in these tests
    os.environ.pop("JOBS_DOMAIN_SCOPED_RBAC", None)
    os.environ.pop("JOBS_RBAC_FORCE", None)
    os.environ.pop("JOBS_DOMAIN_RBAC_PRINCIPAL", None)

    return app


def _make_admin_user_from_principal(principal: AuthPrincipal) -> dict[str, Any]:
    """Thin wrapper to mirror jobs_admin helper for tests.

    Delegates to the production helper so tests exercise the same
    principal→admin_user mapping (including username labels) used by
    jobs_admin endpoints.
    """
    return jobs_mod._make_admin_user_from_principal(principal)


@pytest.mark.asyncio
async def test_jobs_queue_status_401_when_principal_unavailable():
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/queue/status", params={"domain": "ps", "queue": "default"})

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_jobs_queue_status_403_when_missing_admin_role():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/queue/status", params={"domain": "ps", "queue": "default"})

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_jobs_queue_status_200_for_admin_principal():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/queue/status", params={"domain": "ps", "queue": "default"})

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("paused") is False
    assert body.get("drain") is False


@pytest.mark.asyncio
async def test_jobs_queue_status_403_when_domain_required_but_missing(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable domain-scoped RBAC and require a domain filter
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_REQUIRE_DOMAIN_FILTER", "1")

    with TestClient(app) as client:
        # Domain parameter is present but empty -> treated as missing/blank
        resp = client.get("/api/v1/jobs/queue/status", params={"domain": "", "queue": "default"})

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert "Domain filter is required" in detail


@pytest.mark.asyncio
async def test_jobs_queue_status_403_when_domain_not_in_allowlist(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable domain-scoped RBAC with an allowlist that does not include requested domain
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "tenant-a,tenant-b")

    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/queue/status", params={"domain": "tenant-x", "queue": "default"})

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert "Not allowed for domain tenant-x" in detail


@pytest.mark.asyncio
async def test_jobs_queue_status_200_when_domain_in_allowlist(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable domain-scoped RBAC with an allowlist that includes requested domain
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "tenant-a,tenant-b")

    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/queue/status", params={"domain": "tenant-a", "queue": "default"})

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("paused") is False
    assert body.get("drain") is False


@pytest.mark.asyncio
async def test_jobs_queue_status_principal_domain_allowlist_matches_user(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable principal-driven domain RBAC and configure allowlist
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "tenant-a,tenant-b")

    with TestClient(app) as client:
        ok = client.get("/api/v1/jobs/queue/status", params={"domain": "tenant-a", "queue": "default"})
        assert ok.status_code == 200

        forbidden = client.get("/api/v1/jobs/queue/status", params={"domain": "tenant-x", "queue": "default"})

    assert forbidden.status_code == 403
    detail = forbidden.json().get("detail", "")
    assert "Not allowed for domain tenant-x" in detail


@pytest.mark.asyncio
async def test_jobs_prune_and_reschedule_respect_principal_domain_rbac(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable principal-driven domain RBAC and configure allowlist for user id 1
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "domain-a,domain-b")

    # Patch JobManager to avoid any real work and to keep responses deterministic
    class _FakeJobManager:
        def __init__(self, backend: Optional[str] = None, db_url: Optional[str] = None) -> None:
            self.backend = backend
            self.db_url = db_url

        def prune_jobs(self, **_kwargs: Any) -> int:
            return 0

        def reschedule_jobs(self, **_kwargs: Any) -> int:
            return 0

    jobs_mod.JobManager = _FakeJobManager  # type: ignore[assignment]

    with TestClient(app) as client:
        # Allowed domain in allowlist should succeed
        ok_prune = client.post(
            "/api/v1/jobs/prune",
            json={
                "statuses": ["completed"],
                "older_than_days": 30,
                "domain": "domain-a",
                "dry_run": True,
            },
        )
        assert ok_prune.status_code == 200, ok_prune.text

        ok_resched = client.post(
            "/api/v1/jobs/reschedule",
            json={
                "domain": "domain-b",
                "queue": "default",
                "job_type": "export",
                "dry_run": True,
            },
        )
        assert ok_resched.status_code == 200, ok_resched.text

        # Domain outside allowlist should be rejected with 403 for both endpoints
        forbidden_prune = client.post(
            "/api/v1/jobs/prune",
            json={
                "statuses": ["completed"],
                "older_than_days": 30,
                "domain": "domain-x",
                "dry_run": True,
            },
        )
        assert forbidden_prune.status_code == 403

        forbidden_resched = client.post(
            "/api/v1/jobs/reschedule",
            json={
                "domain": "domain-x",
                "queue": "default",
                "job_type": "export",
                "dry_run": True,
            },
        )
        assert forbidden_resched.status_code == 403


@pytest.mark.asyncio
async def test_jobs_queue_control_respects_principal_domain_rbac(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable principal-driven domain RBAC and configure allowlist for user id 1
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "domain-a,domain-b")

    with TestClient(app) as client:
        # Allowed domain should succeed
        ok = client.post(
            "/api/v1/jobs/queue/control",
            json={"domain": "domain-a", "queue": "default", "action": "pause"},
        )
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert "paused" in body and "drain" in body

        # Disallowed domain should be rejected with 403
        forbidden = client.post(
            "/api/v1/jobs/queue/control",
            json={"domain": "domain-x", "queue": "default", "action": "pause"},
        )
        assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_jobs_retry_now_respects_principal_domain_rbac(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable principal-driven domain RBAC and configure allowlist for user id 1
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "domain-a,domain-b")

    with TestClient(app) as client:
        # Allowed domain should succeed
        ok = client.post(
            "/api/v1/jobs/retry-now",
            json={"domain": "domain-a", "queue": "default", "job_type": "export", "dry_run": True},
        )
        assert ok.status_code == 200, ok.text
        body = ok.json()
        assert "affected" in body

        # Disallowed domain should be rejected with 403
        forbidden = client.post(
            "/api/v1/jobs/retry-now",
            json={"domain": "domain-x", "queue": "default", "job_type": "export", "dry_run": True},
        )
        assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_jobs_events_respects_principal_domain_allowlist(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable principal-driven domain RBAC and configure allowlist for user id 1
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "domain-a,domain-b")

    # For the list endpoint, we focus on the forbidden path so that RBAC
    # enforcement is exercised before any DB access.
    with TestClient(app) as client:
        forbidden = client.get(
            "/api/v1/jobs/events",
            params={"domain": "domain-x", "queue": "default"},
        )

    assert forbidden.status_code == 403


@pytest.mark.asyncio
async def test_jobs_events_stream_respects_principal_domain_allowlist(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    # Enable principal-driven domain RBAC and configure allowlist for user id 1
    monkeypatch.setenv("JOBS_DOMAIN_SCOPED_RBAC", "1")
    monkeypatch.setenv("JOBS_DOMAIN_RBAC_PRINCIPAL", "1")
    monkeypatch.setenv("JOBS_DOMAIN_ALLOWLIST_1", "domain-a,domain-b")

    # As with the list endpoint, we assert that RBAC enforcement rejects a
    # disallowed domain before the SSE stream is established.
    with TestClient(app) as client:
        forbidden = client.get(
            "/api/v1/jobs/events/stream",
            params={"domain": "domain-x"},
        )

    assert forbidden.status_code == 403
