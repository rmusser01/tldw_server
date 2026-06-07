from __future__ import annotations

import sys

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.config import clear_config_cache
from tldw_Server_API.app.main import app


def _client(monkeypatch) -> TestClient:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "memory")
    clear_config_cache()
    return TestClient(app)


def test_lima_discovery_includes_enforcement_readiness(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_DENY_ALL_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", "0")

    with _client(monkeypatch) as client:
        resp = client.get("/api/v1/sandbox/runtimes")
        assert resp.status_code == 200
        data = resp.json()
        lima = next(rt for rt in data["runtimes"] if rt["name"] == "lima")
        expected_deny_all = not sys.platform.startswith("win")
        assert lima["strict_deny_all_supported"] is expected_deny_all
        assert lima["strict_allowlist_supported"] is False
        assert lima["enforcement_ready"] == {"deny_all": expected_deny_all, "allowlist": False}
        assert isinstance(lima["host"], dict)
        assert "os" in lima["host"]
