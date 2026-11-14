"""Admin billing route contract: /api/v1/admin/billing/* (admin-webui perf plan A, stage 1).

Covers GET /api/v1/admin/billing/overview|subscriptions|events against a real
SQLite AuthNZ database. The OSS migrations retired the public billing schema
bootstrap (migrations 030-034 are no-ops), so the hosted billing tables are
provisioned here with the columns ``billing_repo`` addresses.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.billing_repo import AuthnzBillingRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.main import app

# Hosted billing tables as addressed by AuthnzBillingRepo (OSS migrations
# 030-034 retired the public bootstrap, so tests provision them directly).
_BILLING_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS subscription_plans (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT,
    description TEXT,
    stripe_product_id TEXT,
    stripe_price_id TEXT,
    stripe_price_id_yearly TEXT,
    price_usd_monthly REAL,
    price_usd_yearly REAL,
    limits_json TEXT,
    is_active INTEGER DEFAULT 1,
    is_public INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS org_subscriptions (
    id INTEGER PRIMARY KEY,
    org_id INTEGER UNIQUE NOT NULL,
    plan_id INTEGER NOT NULL,
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    stripe_subscription_status TEXT,
    billing_cycle TEXT DEFAULT 'monthly',
    current_period_start TEXT,
    current_period_end TEXT,
    status TEXT DEFAULT 'active',
    trial_end TEXT,
    cancel_at_period_end INTEGER DEFAULT 0,
    custom_limits_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS billing_audit_log (
    id INTEGER PRIMARY KEY,
    org_id INTEGER,
    user_id INTEGER,
    action TEXT NOT NULL,
    details TEXT,
    ip_address TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def _make_billing_db(tmp_path: Path) -> Path:
    """Provision an AuthNZ SQLite DB plus the hosted billing tables."""
    import sqlite3

    db_path = tmp_path / "authnz_admin_billing.sqlite"
    ensure_authnz_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_BILLING_TABLES_SQL)
        conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES (1, 'Acme Corp', 'acme')"
        )
        conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES (2, 'Globex', 'globex')"
        )
        conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES (3, 'Initech', 'initech')"
        )
        conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES (4, 'Umbrella', 'umbrella')"
        )
        conn.execute(
            "INSERT INTO organizations (id, name, slug) VALUES (5, 'Stark Industries', 'stark')"
        )
        conn.execute(
            "INSERT INTO subscription_plans (id, name, display_name, price_usd_monthly, limits_json)"
            " VALUES (10, 'pro', 'Pro', 20.0, '{\"llm_tokens_month\": 100000}')"
        )
    return db_path


def _seed_subscription(
    conn: Any,
    *,
    org_id: int,
    status: str,
    created_at: str,
    plan_id: int = 10,
) -> None:
    conn.execute(
        "INSERT INTO org_subscriptions (org_id, plan_id, status, created_at)"
        " VALUES (?, ?, ?, ?)",
        (org_id, plan_id, status, created_at),
    )


async def _make_pool(db_path: Path) -> DatabasePool:
    pool = DatabasePool(
        Settings(AUTH_MODE="single_user", DATABASE_URL=f"sqlite:///{db_path}")
    )
    await pool.initialize()
    return pool


async def _admin_principal_override(request: Request) -> AuthPrincipal:
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="admin-billing-test",
        token_type="access",  # nosec B106
        jti=None,
        roles=["admin"],
        permissions=["system.configure"],
        is_admin=True,
        org_ids=[],
        team_ids=[],
        email="admin-billing@example.com",
    )
    request.state.auth = AuthContext(
        principal=principal, ip=None, user_agent=None, request_id=None
    )
    return principal


@pytest.fixture
async def billing_api(tmp_path, monkeypatch):
    """Async HTTP client against the real app with the billing DB pool swapped in."""
    import sqlite3

    from tldw_Server_API.app.api.v1.endpoints import billing as billing_endpoints
    from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

    db_path = _make_billing_db(tmp_path)
    pool = await _make_pool(db_path)

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key-admin-billing")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users_admin_billing.db'}")

    async def _fake_get_db_pool() -> DatabasePool:
        return pool

    monkeypatch.setattr(billing_endpoints, "get_db_pool", _fake_get_db_pool)
    app.dependency_overrides[get_auth_principal] = _admin_principal_override
    reset_lifecycle_state(app)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-API-KEY": "unit-test-api-key-admin-billing"},
        ) as client:
            yield client, sqlite3.connect(db_path)
    finally:
        app.dependency_overrides.clear()
        await pool.close()


async def test_overview_counts_by_status(billing_api) -> None:
    """GET /overview returns the four-field shape BillingDashboardPage expects."""
    client, conn = billing_api
    _seed_subscription(conn, org_id=1, status="active", created_at="2026-01-01T00:00:00Z")
    _seed_subscription(conn, org_id=2, status="active", created_at="2026-01-02T00:00:00Z")
    _seed_subscription(conn, org_id=3, status="canceled", created_at="2026-01-03T00:00:00Z")
    conn.commit()

    resp = await client.get("/api/v1/admin/billing/overview")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) == {
        "mrr",
        "active_subscriptions",
        "canceled_subscriptions",
        "past_due_subscriptions",
    }
    assert data["active_subscriptions"] == 2
    assert data["canceled_subscriptions"] == 1
    assert data["past_due_subscriptions"] == 0
    # MRR sums the active subscriptions' plan monthly price (2 x $20).
    assert data["mrr"] == pytest.approx(40.0)


async def test_subscriptions_status_filter_in_sql(tmp_path) -> None:
    """status filtering happens in SQL, not in a Python-side loop."""
    import sqlite3

    db_path = _make_billing_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        _seed_subscription(conn, org_id=1, status="active", created_at="2026-01-01T00:00:00Z")
        _seed_subscription(conn, org_id=2, status="active", created_at="2026-01-02T00:00:00Z")
        _seed_subscription(conn, org_id=3, status="canceled", created_at="2026-01-03T00:00:00Z")

    pool = await _make_pool(db_path)
    statements: list[tuple[str, tuple[Any, ...]]] = []

    class _CountingConn:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def execute(self, query: Any, *args: Any) -> Any:
            statements.append((str(query), tuple(args)))
            return await self._inner.execute(query, *args)

    class _CountingPool:
        def __init__(self, real_pool: DatabasePool) -> None:
            self._real_pool = real_pool

        @asynccontextmanager
        async def acquire(self, *, timeout: float | None = None):
            async with self._real_pool.acquire() as inner:
                yield _CountingConn(inner)

    try:
        repo = AuthnzBillingRepo(db_pool=_CountingPool(pool))  # type: ignore[arg-type]
        items, total = await repo.list_subscriptions(status="active", limit=100, offset=0)
    finally:
        await pool.close()

    assert total == 2
    assert [item["status"] for item in items] == ["active", "active"]

    rows_queries = [
        sql
        for sql, _params in statements
        if "FROM org_subscriptions" in sql and "COUNT(*)" not in sql
    ]
    # Exactly one rows query, and it carries the status filter in its WHERE clause.
    assert len(rows_queries) == 1
    assert "WHERE" in rows_queries[0]
    assert "status" in rows_queries[0]
    # Rows + truthful COUNT: no unfiltered fetch, no org name-map side query.
    assert len(statements) == 2


async def test_subscriptions_paginates_with_truthful_total(billing_api) -> None:
    """limit/offset page in SQL while total stays the truthful COUNT(*)."""
    client, conn = billing_api
    for org_id in range(1, 6):
        _seed_subscription(
            conn,
            org_id=org_id,
            status="active",
            created_at=f"2026-01-0{org_id}T00:00:00Z",
        )
    conn.commit()

    resp = await client.get(
        "/api/v1/admin/billing/subscriptions", params={"limit": 2, "offset": 0}
    )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) == {"items", "total"}
    assert len(data["items"]) == 2
    assert data["total"] == 5
    # Newest first (created_at DESC), org names resolved via the SQL JOIN.
    assert data["items"][0]["org_id"] == 5
    assert data["items"][0]["org_name"] == "Stark Industries"
    assert data["items"][1]["org_name"] == "Umbrella"


async def test_events_pages_billing_audit_log(billing_api) -> None:
    """GET /events pages the real billing audit log with a truthful total."""
    client, conn = billing_api
    conn.executemany(
        "INSERT INTO billing_audit_log (org_id, user_id, action, details, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            (1, 11, "subscription.created", '{"plan": "pro"}', "2026-01-01T00:00:00Z"),
            (2, 12, "subscription.plan_changed", '{"plan": "pro"}', "2026-01-02T00:00:00Z"),
            (3, 13, "subscription.canceled", '{"reason": "churn"}', "2026-01-03T00:00:00Z"),
        ],
    )
    conn.commit()

    resp = await client.get("/api/v1/admin/billing/events", params={"limit": 2})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) == {"items", "total"}
    assert data["total"] == 3
    assert len(data["items"]) == 2
    first = data["items"][0]
    assert first["event_type"] == "subscription.canceled"
    assert first["user_id"] == 13
    assert first["created_at"] == "2026-01-03T00:00:00Z"


async def test_events_empty_shape_is_honest(billing_api) -> None:
    """With no billing events recorded the endpoint returns an empty page, not an error."""
    client, _conn = billing_api

    resp = await client.get("/api/v1/admin/billing/events")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"items": [], "total": 0}


def test_legacy_public_billing_path_removed_and_admin_mount_present() -> None:
    """The old public /api/v1/billing mount stays gone; the admin mount serves the dashboard routes."""
    import importlib

    from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
    from tldw_Server_API.app import main as app_main

    app_main = importlib.reload(app_main)

    route_paths = {
        getattr(route, "path", "") for route in iter_served_routes(app_main.app.routes)
    }
    # The legacy public billing API was removed (test_billing_public_api_removed);
    # no deprecation alias may resurrect it.
    assert not any(path.startswith("/api/v1/billing") for path in route_paths)
    assert "/api/v1/admin/billing/overview" in route_paths
    assert "/api/v1/admin/billing/subscriptions" in route_paths
    assert "/api/v1/admin/billing/events" in route_paths
