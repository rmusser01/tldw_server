import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_api_v1_health_reports_rg_policy_snapshot(monkeypatch, auth_headers):
    # Point to the repo policy file so fallback always works
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]
    stub = base / "Config_Files" / "resource_governor_policies.yaml"

    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.setenv("RG_POLICY_PATH", str(stub))
    monkeypatch.setenv("RG_POLICY_RELOAD_ENABLED", "false")
    # Use SQLite single_user AuthNZ to avoid Postgres
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{base / 'Databases' / 'users_test_health_api_v1.db'}")

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        r = client.get("/api/v1/health", headers=auth_headers)
        assert r.status_code in (200, 206)  # degraded allowed
        data = r.json()
        assert "rg_policy_version" in data
        assert data["rg_policy_version"] >= 1
        assert data.get("rg_policy_store") in {"file", "db", None}
        assert isinstance(data.get("rg_policy_count"), int)

        readiness = client.get("/api/v1/health/ready", headers=auth_headers)
        assert readiness.status_code in (200, 503)
        readiness_data = readiness.json()
        assert readiness_data["rg_policy"]["version"] >= 1
        assert readiness_data["rg_policy"]["store"] in {"file", "db", None}
        assert isinstance(readiness_data["rg_policy"]["policies"], int)
