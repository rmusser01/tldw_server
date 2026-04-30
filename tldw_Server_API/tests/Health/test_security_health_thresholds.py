import importlib
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient


@contextmanager
def _capture_health_logs() -> Iterator[list[str]]:
    from tldw_Server_API.app.api.v1.endpoints import health as health_mod

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


def _monkeypatch_audit_summary(monkeypatch, high_risk: int, failures: int):
    from tldw_Server_API.app.api.v1.endpoints import health as health_mod

    class _DummyAudit:
        async def initialize(self, *args, **kwargs):
            return None

        async def get_security_summary(self, hours=24, **_kwargs):
            return {
                "high_risk_events": high_risk,
                "failure_events": failures,
                "unique_security_users": 1,
                "top_failing_ips": ["1.2.3.4"],
                "total_events": high_risk + failures,
            }

    monkeypatch.setattr(health_mod, "UnifiedAuditService", lambda: _DummyAudit())


def _get_client(monkeypatch, env: dict):
    # Ensure test-friendly startup
    for k, v in {"TEST_MODE": "true"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, str(v))

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    return TestClient(app_main.app)


def test_security_critical_when_high_risk_meets_threshold(monkeypatch):
    # Configure thresholds
    client = _get_client(monkeypatch, {
        "AUDIT_SEC_CRITICAL_HIGH_RISK_MIN": 3,
        "AUDIT_SEC_ELEVATED_FAILURE_MIN": 10,
    })
    _monkeypatch_audit_summary(monkeypatch, high_risk=3, failures=0)

    r = client.get("/api/v1/health/security")
    assert r.status_code == 200 or r.status_code == 503 or r.status_code == 206
    data = r.json()
    assert data["risk_level"] == "critical"
    assert data["status"] == "at_risk"


def test_security_elevated_when_failures_meet_threshold(monkeypatch):
    client = _get_client(monkeypatch, {
        "AUDIT_SEC_CRITICAL_HIGH_RISK_MIN": 5,
        "AUDIT_SEC_ELEVATED_FAILURE_MIN": 7,
    })
    _monkeypatch_audit_summary(monkeypatch, high_risk=0, failures=7)

    r = client.get("/api/v1/health/security")
    data = r.json()
    assert data["risk_level"] == "high"
    assert data["status"] == "elevated"


def test_security_low_when_some_failures_below_threshold(monkeypatch):
    client = _get_client(monkeypatch, {
        "AUDIT_SEC_CRITICAL_HIGH_RISK_MIN": 2,
        "AUDIT_SEC_ELEVATED_FAILURE_MIN": 10,
    })
    _monkeypatch_audit_summary(monkeypatch, high_risk=0, failures=1)

    r = client.get("/api/v1/health/security")
    data = r.json()
    assert data["risk_level"] == "low"
    assert data["status"] == "secure"


def test_security_health_shared_mode_scoped(monkeypatch):
    class _ScopedAudit:
        _shared_mode = True

        async def initialize(self, *args, **kwargs):
            return None

        async def get_security_summary(self, hours=24, **kwargs):
            assert kwargs.get("allow_cross_tenant") is False
            return {
                "high_risk_events": 0,
                "failure_events": 0,
                "unique_security_users": 0,
                "top_failing_ips": [],
                "total_events": 0,
            }

    from tldw_Server_API.app.api.v1.endpoints import health as health_mod

    monkeypatch.setattr(health_mod, "UnifiedAuditService", lambda: _ScopedAudit())
    client = _get_client(monkeypatch, {})
    r = client.get("/api/v1/health/security")
    assert r.status_code == 200 or r.status_code == 503 or r.status_code == 206


def test_security_health_sanitizes_audit_service_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import health as health_mod

    class _FailingAudit:
        async def initialize(self, *args, **kwargs):
            return None

        async def get_security_summary(self, hours=24, **_kwargs):
            raise RuntimeError("audit backend exploded at /private/audit.db")

    monkeypatch.setattr(health_mod, "UnifiedAuditService", lambda: _FailingAudit())
    client = _get_client(monkeypatch, {})

    with _capture_health_logs() as messages:
        r = client.get("/api/v1/health/security")

    joined = "\n".join(messages)

    assert r.status_code == 503
    assert r.json()["error"] == "Security health unavailable"
    assert "health/security failed" in joined
    assert "audit backend exploded" not in joined
    assert "/private/" not in joined


def test_security_health_sanitizes_audit_service_stop_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import health as health_mod

    class _AuditWithFailingStop:
        async def initialize(self, *args, **kwargs):
            return None

        async def get_security_summary(self, hours=24, **_kwargs):
            return {
                "high_risk_events": 0,
                "failure_events": 0,
                "unique_security_users": 0,
                "top_failing_ips": [],
                "total_events": 0,
            }

        async def stop(self):
            raise RuntimeError("audit stop leaked at /private/audit.db")

    monkeypatch.setattr(health_mod, "UnifiedAuditService", lambda: _AuditWithFailingStop())
    client = _get_client(monkeypatch, {})

    with _capture_health_logs() as messages:
        r = client.get("/api/v1/health/security")

    joined = "\n".join(messages)

    assert r.status_code == 200
    assert "UnifiedAuditService stop() ignored" in joined
    assert "audit stop leaked" not in joined
    assert "/private/" not in joined
