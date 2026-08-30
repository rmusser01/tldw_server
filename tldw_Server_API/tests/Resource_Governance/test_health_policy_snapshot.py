import os

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_health_reports_policy_snapshot(monkeypatch):
    # Ensure file-based policy loader with known stub path
    from pathlib import Path
    base = Path(__file__).resolve().parents[3]
    stub = base / "Config_Files" / "resource_governor_policies.yaml"

    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_PATH", str(stub))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    # Use SQLite single_user AuthNZ to avoid Postgres dependencies
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{base / 'Databases' / 'users_test_health.db'}")

    # Import app after env setup so lifespan picks it up
    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data == {"status": "ok"}
