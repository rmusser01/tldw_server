from pathlib import Path
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        await reset_db_pool()
        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None


async def _create_admin_user_and_key(*, username: str, email: str) -> str:
    from uuid import uuid4

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username=username,
        email=email,
        password_hash="x",
        role="admin",
        is_active=True,
        is_superuser=True,
        storage_quota_mb=5120,
        uuid_value=uuid4(),
    )
    user_id = int(created_user["id"])
    await AuthnzUsersRepo(db_pool=pool).assign_role_if_missing(
        user_id=user_id,
        role_name="admin",
    )
    mgr = APIKeyManager(pool)
    await mgr.initialize()
    key_rec = await mgr.create_api_key(user_id=user_id, name=f"{username}-key")
    return str(key_rec["key"])


def _reset_rg_state(app) -> None:


    for attr in ("rg_governor", "rg_policy_loader", "rg_policy_store", "rg_policy_version", "rg_policy_count"):
        try:
            if hasattr(app.state, attr):
                setattr(app.state, attr, None)
        except Exception:
            continue
    # Daily-cap helpers cache a process-global ledger instance. Clear it so this
    # module's per-test DATABASE_URL fixtures always apply deterministically.
    try:
        from tldw_Server_API.app.core.Resource_Governance import daily_caps as _daily_caps

        _daily_caps._daily_ledger = None
    except Exception:
        _ = None


@pytest.mark.asyncio
async def test_policy_snapshot_endpoint(monkeypatch, tmp_path):
    base = Path(__file__).resolve().parents[2]
    stub = base / "Config_Files" / "resource_governor_policies.yaml"

    db_path = tmp_path / "authnz_rg_endpoint.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_user_and_key(username="rg-endpoint-admin", email="rg-endpoint-admin@example.com")

    monkeypatch.setenv("RG_POLICY_STORE", "db")
    monkeypatch.setenv("RG_POLICY_PATH", str(stub))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.main import app
    _reset_rg_state(app)
    headers = {"X-API-KEY": api_key}
    # Seed a few known policies via admin API to ensure snapshot exists in DB store
    with TestClient(app) as client:
        seeds = {
            "chat.default": {"requests": {"rpm": 12}},
            "embeddings.default": {"tokens": {"per_min": 1000}},
            "audio.default": {"streams": {"max_concurrent": 2}},
        }
        for pid, payload in seeds.items():
            client.put(
                f"/api/v1/resource-governor/policy/{pid}",
                headers=headers,
                json={"payload": payload, "version": 1},
            )
    with TestClient(app) as client:
        r = client.get("/api/v1/resource-governor/policy?include=ids", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"
        assert data.get("store") in ("db", "file")
        assert data.get("version") >= 1
        ids = data.get("policy_ids") or []
        assert isinstance(ids, list) and len(ids) >= 3
        assert any(i == "chat.default" for i in ids)

        # Basic get (admin-gated)
        # Ensure endpoint is reachable; result may be 404 in file store
        g = client.get(f"/api/v1/resource-governor/policy/{ids[0]}", headers=headers)
        assert g.status_code in (200, 404)


@pytest.mark.asyncio
async def test_policy_admin_upsert_and_delete_sqlite(monkeypatch, tmp_path):
    # Use DB policy store so admin writes update the snapshot on refresh
    db_path = tmp_path / "authnz_rg_admin.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_user_and_key(username="rg-admin", email="rg-admin@example.com")

    monkeypatch.setenv("RG_POLICY_STORE", "db")
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.main import app
    _reset_rg_state(app)
    headers = {"X-API-KEY": api_key}
    with TestClient(app) as client:
        # Upsert a policy
        new_policy_id = "test.policy"
        up = client.put(
            f"/api/v1/resource-governor/policy/{new_policy_id}",
            headers=headers,
            json={
                "payload": {"requests": {"rpm": 42, "burst": 1.0}},
                "version": 1,
            },
        )
        assert up.status_code == 200
        # Verify it's included in snapshot ids
        r = client.get("/api/v1/resource-governor/policy?include=ids", headers=headers)
        assert r.status_code == 200
        ids = r.json().get("policy_ids") or []
        assert new_policy_id in ids
        # Delete it
        de = client.delete(f"/api/v1/resource-governor/policy/{new_policy_id}", headers=headers)
        assert de.status_code == 200
        r2 = client.get("/api/v1/resource-governor/policy?include=ids", headers=headers)
        assert new_policy_id not in (r2.json().get("policy_ids") or [])

        # Optimistic delete: conflict when version mismatches.
        versioned_policy_id = "test.policy.versioned"
        up1 = client.put(
            f"/api/v1/resource-governor/policy/{versioned_policy_id}",
            headers=headers,
            json={"payload": {"requests": {"rpm": 1}}, "version": 1},
        )
        assert up1.status_code == 200
        up2 = client.put(
            f"/api/v1/resource-governor/policy/{versioned_policy_id}",
            headers=headers,
            json={"payload": {"requests": {"rpm": 2}}, "version": 1},
        )
        assert up2.status_code == 200
        de_conflict = client.delete(
            f"/api/v1/resource-governor/policy/{versioned_policy_id}",
            headers=headers,
            params={"version": 1},
        )
        assert de_conflict.status_code == 409
        de_ok = client.delete(
            f"/api/v1/resource-governor/policy/{versioned_policy_id}",
            headers=headers,
            params={"version": 2},
        )
        assert de_ok.status_code == 200
        r3 = client.get("/api/v1/resource-governor/policy?include=ids", headers=headers)
        assert versioned_policy_id not in (r3.json().get("policy_ids") or [])

        # List policies (admin)
        lst = client.get("/api/v1/resource-governor/policies", headers=headers)
        assert lst.status_code == 200
        items = lst.json().get("items")
        assert isinstance(items, list)


@pytest.mark.asyncio
async def test_rg_diag_media_budget_exposes_limits_and_usage(monkeypatch, tmp_path):
    db_path = tmp_path / "authnz_rg_media_budget.db"
    await _init_authnz_sqlite(db_path, monkeypatch)
    api_key = await _create_admin_user_and_key(
        username="rg-media-budget-admin",
        email="rg-media-budget-admin@example.com",
    )

    monkeypatch.setenv("RG_POLICY_STORE", "db")
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
        LedgerEntry,
        ResourceDailyLedger,
    )
    from tldw_Server_API.app.main import app

    ledger = ResourceDailyLedger()
    await ledger.initialize()
    await ledger.add(
        LedgerEntry(
            entity_scope="user",
            entity_value="777",
            category="ingestion_bytes",
            units=256,
            op_id="seed-media-budget-usage",
            occurred_at=datetime.now(timezone.utc),
        )
    )

    _reset_rg_state(app)
    headers = {"X-API-KEY": api_key}
    with TestClient(app) as client:
        up = client.put(
            "/api/v1/resource-governor/policy/media.default",
            headers=headers,
            json={
                "payload": {
                    "requests": {"rpm": 600, "burst": 1.2},
                    "jobs": {"max_concurrent": 3, "ttl_sec": 1800},
                    "ingestion_bytes": {"daily_cap": 1024},
                    "scopes": ["user", "api_key"],
                },
                "version": 1,
            },
        )
        assert up.status_code == 200

        r = client.get(
            "/api/v1/resource-governor/diag/media-budget",
            headers=headers,
            params={"user_id": 777, "policy_id": "media.default"},
        )
        assert r.status_code == 200
        body = r.json()

    assert body["status"] == "ok"
    assert body["entity"] == "user:777"
    assert body["policy_id"] == "media.default"
    assert body["limits"]["jobs_max_concurrent"] == 3
    assert body["limits"]["ingestion_bytes_daily_cap"] == 1024
    assert body["usage"]["jobs_active"] == 0
    assert body["usage"]["jobs_remaining"] == 3
    assert body["usage"]["ingestion_bytes_daily_used"] == 256
    assert body["usage"]["ingestion_bytes_daily_remaining"] == 768
