import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.api.v1.endpoints import health as health_mod


@contextmanager
def _capture_health_logs() -> Iterator[list[str]]:
    messages: list[str] = []
    sink_id = health_mod.logger.add(
        lambda message: messages.append(str(message))
        if message.record["name"] == health_mod.__name__
        else None,
        level="DEBUG",
    )
    try:
        yield messages
    finally:
        health_mod.logger.remove(sink_id)


@pytest.mark.asyncio
async def test_readyz_sanitizes_workflows_db_check_failure(monkeypatch):
    def _raise_backend_error():
        raise RuntimeError("workflow db exploded at /private/workflows.db")

    monkeypatch.setattr(health_mod, "get_content_backend_instance", _raise_backend_error)

    with _capture_health_logs() as messages:
        response = await health_mod.readyz()

    body = json.loads(response.body.decode("utf-8"))
    joined = "\n".join(messages)

    assert response.status_code == 503
    assert body["ready"] is False
    assert body["db"]["ok"] is False
    assert body["db"]["error"] == "Workflow database health check failed"
    assert "/readyz DB check failed" in joined
    assert "workflow db exploded" not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_api_health_sanitizes_database_probe_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.AuthNZ import database as auth_database
    from tldw_Server_API.app.core.Metrics import metrics_manager

    async def _failing_pool():
        raise RuntimeError("auth database exploded at /private/users.db")

    monkeypatch.setattr(auth_database, "get_db_pool", _failing_pool)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: object())
    monkeypatch.setattr(chacha_deps, "get_chacha_health_snapshot", lambda: {"status": "healthy"})

    response = await health_mod.api_health()
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert body["checks"]["database"]["status"] == "unhealthy"
    assert body["checks"]["database"]["error"] == "Database health check failed"


@pytest.mark.asyncio
async def test_api_health_sanitizes_metrics_probe_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.AuthNZ import database as auth_database
    from tldw_Server_API.app.core.Metrics import metrics_manager

    class _HealthyPool:
        async def health_check(self):
            return {"status": "healthy"}

    async def _healthy_pool():
        return _HealthyPool()

    def _failing_metrics_registry():
        raise RuntimeError("metrics registry exploded at /private/metrics.db")

    monkeypatch.setattr(auth_database, "get_db_pool", _healthy_pool)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", _failing_metrics_registry)
    monkeypatch.setattr(chacha_deps, "get_chacha_health_snapshot", lambda: {"status": "healthy"})

    response = await health_mod.api_health()
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 503
    assert body["checks"]["metrics"]["status"] == "unhealthy"
    assert body["checks"]["metrics"]["error"] == "Metrics health check failed"


@pytest.mark.asyncio
async def test_api_health_sanitizes_chacha_notes_probe_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.AuthNZ import database as auth_database
    from tldw_Server_API.app.core.Metrics import metrics_manager

    class _HealthyPool:
        async def health_check(self):
            return {"status": "healthy"}

    async def _healthy_pool():
        return _HealthyPool()

    def _failing_chacha_snapshot():
        raise RuntimeError("chacha snapshot exploded at /private/ChaChaNotes.db")

    monkeypatch.setattr(auth_database, "get_db_pool", _healthy_pool)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: object())
    monkeypatch.setattr(chacha_deps, "get_chacha_health_snapshot", _failing_chacha_snapshot)

    with _capture_health_logs() as messages:
        response = await health_mod.api_health()

    body = json.loads(response.body.decode("utf-8"))
    joined = "\n".join(messages)

    assert response.status_code == 206
    assert body["checks"]["chacha_notes"]["status"] == "unhealthy"
    assert body["checks"]["chacha_notes"]["error"] == "ChaChaNotes health check failed"
    assert "ChaChaNotes health snapshot failed" in joined
    assert "chacha snapshot exploded" not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_api_health_exposes_chacha_recovery_details_without_path_leak(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.AuthNZ import database as auth_database
    from tldw_Server_API.app.core.Metrics import metrics_manager

    class _HealthyPool:
        async def health_check(self):
            return {"status": "healthy"}

    async def _healthy_pool():
        return _HealthyPool()

    monkeypatch.setattr(auth_database, "get_db_pool", _healthy_pool)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: object())

    with chacha_deps._CHACHA_HEALTH_LOCK:
        chacha_deps._CHACHA_HEALTH.update(
            {
                "init_attempts": 1,
                "init_failures": 1,
                "last_init_ms": 2.5,
                "last_error": "sqlite_corruption",
                "last_init_success": False,
                "cached_instances": 0,
                "consecutive_failures": 1,
                "default_char_ensures": 0,
                "default_char_failures": 0,
                "warm_startups": 2,
                "last_failure": {
                    "reason_code": "sqlite_corruption",
                    "affected_db": "user:42/ChaChaNotes.db",
                    "recovery": {
                        "automatic_repair": False,
                        "documentation": "Docs/Operations/ChaChaNotes_DB_Recovery.md",
                    },
                },
            }
        )

    response = await health_mod.api_health()
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 206
    assert body["status"] == "degraded"
    assert body["checks"]["chacha_notes"]["status"] == "degraded"
    assert body["checks"]["chacha_notes"]["last_init_success"] is False
    assert body["checks"]["chacha_notes"]["consecutive_failures"] == 1
    assert body["checks"]["chacha_notes"]["warm_startups"] == 2
    assert body["checks"]["chacha_notes"]["last_failure"]["affected_db"] == "ChaChaNotes.db"
    assert body["checks"]["chacha_notes"]["last_failure"]["recovery"]["automatic_repair"] is False
    assert body["checks"]["chacha_notes"]["last_failure"]["recovery"]["documentation"] == (
        "Docs/Operations/ChaChaNotes_DB_Recovery.md"
    )
    assert "user:42" not in str(body)
    assert str(tmp_path) not in str(body)
    assert "/private/" not in str(body)


@pytest.mark.asyncio
async def test_api_health_sanitizes_rg_policy_snapshot_failure(monkeypatch, tmp_path):
    from tldw_Server_API.app import main as app_main
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.AuthNZ import database as auth_database
    from tldw_Server_API.app.core.Metrics import metrics_manager
    import yaml

    class _HealthyPool:
        async def health_check(self):
            return {"status": "healthy"}

    async def _healthy_pool():
        return _HealthyPool()

    def _raise_policy_parse_failure(*args, **kwargs):
        raise RuntimeError("rg policy leaked at /private/rg-policy.yaml")

    policy_path = tmp_path / "rg-policy.yaml"
    policy_path.write_text("version: 1\npolicies: {}\n", encoding="utf-8")

    monkeypatch.setattr(auth_database, "get_db_pool", _healthy_pool)
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: object())
    monkeypatch.setattr(chacha_deps, "get_chacha_health_snapshot", lambda: {"status": "healthy"})
    monkeypatch.setattr(app_main.app.state, "rg_policy_version", None, raising=False)
    monkeypatch.setenv("RG_POLICY_PATH", str(policy_path))
    monkeypatch.setattr(yaml, "safe_load", _raise_policy_parse_failure)

    with _capture_health_logs() as messages:
        response = await health_mod.api_health()

    joined = "\n".join(messages)

    assert response.status_code == 200
    assert "Failed to read RG policy file for /health" in joined
    assert "rg policy leaked" not in joined
    assert "/private/" not in joined


@pytest.mark.asyncio
async def test_api_health_metrics_sanitizes_metrics_collection_failure(monkeypatch):
    import psutil

    def _raise_cpu_percent(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("psutil metrics exploded at /private/proc/stat")

    monkeypatch.setattr(psutil, "cpu_percent", _raise_cpu_percent)

    with _capture_health_logs() as messages:
        body = await health_mod.api_health_metrics()

    joined = "\n".join(messages)

    assert body["cpu"] == {"percent": 0.0}
    assert body["memory"]["total"] == 0
    assert body["disk"]["total"] == 0
    assert "health/metrics unavailable" in joined
    assert "psutil metrics exploded" not in joined
    assert "/private/" not in joined


def test_int_env_sanitizes_invalid_value_log(monkeypatch):
    monkeypatch.setenv("AUDIT_SEC_CRITICAL_HIGH_RISK_MIN", "threshold leaked at /private/health.env")

    with _capture_health_logs() as messages:
        value = health_mod._int_env("AUDIT_SEC_CRITICAL_HIGH_RISK_MIN", 7)

    joined = "\n".join(messages)

    assert value == 7
    assert "Invalid integer environment override" in joined
    assert "AUDIT_SEC_CRITICAL_HIGH_RISK_MIN" not in joined
    assert "threshold leaked" not in joined
    assert "/private/" not in joined
