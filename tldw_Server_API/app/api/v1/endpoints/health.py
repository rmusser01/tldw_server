from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from loguru import logger

from tldw_Server_API.app.core.DB_Management.DB_Manager import create_workflows_database, get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Workflows.engine import WorkflowScheduler
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

_HEALTH_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

try:
    from tldw_Server_API.app.core.Audit.unified_audit_service import UnifiedAuditService as _UnifiedAuditService
except ImportError:  # pragma: no cover - defensive import guard for optional dependencies
    _UnifiedAuditService = None  # type: ignore[assignment]

# Expose symbol for tests to monkeypatch (see test_security_health_thresholds.py)
UnifiedAuditService = _UnifiedAuditService  # type: ignore[assignment]

router = APIRouter()


def _utcnow_iso() -> str:
    import datetime as _dt

    return _dt.datetime.utcnow().isoformat()


def _check_workflows_db() -> dict:
    """Basic connectivity and schema readiness for workflows DB."""
    status = {"ok": False, "backend": None, "schema_version": None, "expected_version": None}
    try:
        backend = get_content_backend_instance()
        db: WorkflowsDatabase = create_workflows_database(backend=backend)
        status["backend"] = backend.backend_type.name if backend else "sqlite"
        # Connectivity probe
        if db._using_backend():
            with db.backend.transaction() as conn:  # type: ignore[union-attr]
                # Lightweight probe
                db._execute_backend("SELECT 1", None, connection=conn)
                # Migration version check (backend only)
                try:
                    status["schema_version"] = int(db._get_backend_schema_version(conn))  # type: ignore[attr-defined]
                    status["expected_version"] = int(db._CURRENT_SCHEMA_VERSION)  # type: ignore[attr-defined]
                except _HEALTH_NONCRITICAL_EXCEPTIONS:
                    pass
        else:
            # SQLite: best-effort probe
            _ = db._conn.cursor().execute("SELECT 1").fetchone()  # type: ignore[attr-defined]
            status["schema_version"] = None
            status["expected_version"] = None
        status["ok"] = True
    except _HEALTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error("/readyz DB check failed")
        status["error"] = "Workflow database health check failed"
    return status


@router.get("/healthz", include_in_schema=False)
async def healthz():
    """Basic liveness check with lightweight engine stats."""
    try:
        qd = WorkflowScheduler.instance().queue_depth()
    except _HEALTH_NONCRITICAL_EXCEPTIONS:
        qd = None
    return {
        "status": "ok",
        "queue_depth": qd,
        "time": _utcnow_iso(),
    }


@router.get("/readyz", include_in_schema=False)
async def readyz():
    """Readiness check: engine stats + DB connectivity and schema version (backend)."""
    try:
        stats = WorkflowScheduler.instance().stats()
    except _HEALTH_NONCRITICAL_EXCEPTIONS:
        stats = {"queue_depth": None, "active_tenants": None, "active_workflows": None}
    db = _check_workflows_db()
    ready = bool(db.get("ok")) and (
        db.get("schema_version") is None or db.get("schema_version") == db.get("expected_version")
    )
    body = {
        "ready": ready,
        "engine": stats,
        "db": db,
        "time": _utcnow_iso(),
    }
    # Fail readiness (HTTP 503) if schema version mismatch or DB not ok
    if not ready:
        return JSONResponse(body, status_code=503)
    return JSONResponse(body, status_code=200)


# Compatibility health endpoints expected by tests (/api/v1/health, /api/v1/health/live, /api/v1/health/ready, /api/v1/health/metrics)


@router.get("/health", tags=["health"], summary="Aggregate health status")
async def api_health():
    """Return aggregate health with a checks map and timestamp."""
    from datetime import datetime as _dt

    checks: dict[str, dict] = {}
    overall = "ok"

    # Database health (AuthNZ pool)
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool as _get_pool

        pool = await _get_pool()
        dbh = await pool.health_check()
        checks["database"] = dbh
        if dbh.get("status") != "healthy":
            overall = "degraded"
    except _HEALTH_NONCRITICAL_EXCEPTIONS as e:
        checks["database"] = {"status": "unhealthy", "error": "Database health check failed"}
        overall = "unhealthy"

    # Metrics registry presence
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry as _get_reg

        reg = _get_reg()
        metrics_ok = bool(reg)
        checks["metrics"] = {"status": "healthy" if metrics_ok else "unhealthy"}
        if not metrics_ok and overall == "ok":
            overall = "degraded"
    except _HEALTH_NONCRITICAL_EXCEPTIONS as e:
        checks["metrics"] = {"status": "unhealthy", "error": "Metrics health check failed"}
        overall = "unhealthy"

    # ChaChaNotes health snapshot
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_health_snapshot

        chacha = get_chacha_health_snapshot()
        checks["chacha_notes"] = chacha
        if chacha.get("status") not in {"healthy", "ok"} and overall == "ok":
            overall = "degraded"
    except _HEALTH_NONCRITICAL_EXCEPTIONS as e:
        logger.warning("ChaChaNotes health snapshot failed")
        checks["chacha_notes"] = {"status": "unhealthy", "error": "ChaChaNotes health check failed"}
        overall = "degraded"

    body = {
        "status": overall,
        "checks": checks,
        "timestamp": _dt.utcnow().isoformat(),
    }
    # Include auth mode for E2E tests and diagnostics
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings  # type: ignore

        _s = _get_settings()
        body["auth_mode"] = getattr(_s, "AUTH_MODE", "single_user")
        # In test environments, optionally expose a test API key only with explicit opt-in
        # This prevents accidental leakage of SINGLE_USER_API_KEY unless HEALTH_EXPOSE_TEST_API_KEY=true
        if (
            is_test_mode()
            and body["auth_mode"] == "single_user"
            and env_flag_enabled("HEALTH_EXPOSE_TEST_API_KEY")
        ):
            _key = getattr(_s, "SINGLE_USER_API_KEY", None)
            if _key:
                body.setdefault("test_api_key", _key)
    except _HEALTH_NONCRITICAL_EXCEPTIONS:
        # Never fail health on settings import issues
        pass
    # Include Resource Governor policy snapshot metadata when available (mirrors top-level /health)
    try:
        from tldw_Server_API.app.main import app as _app

        rgv = getattr(_app.state, "rg_policy_version", None)
        if rgv is not None:
            body["rg_policy_version"] = int(rgv)
            body["rg_policy_store"] = getattr(_app.state, "rg_policy_store", None)
            body["rg_policy_count"] = getattr(_app.state, "rg_policy_count", None)
        else:
            # Fallback: read from configured policy file if available
            import os as _os
            from pathlib import Path as _Path

            import yaml as _yaml

            p = _os.getenv("RG_POLICY_PATH")
            if p and _Path(p).exists():
                try:
                    with _Path(p).open("r", encoding="utf-8") as _f:
                        _data = _yaml.safe_load(_f) or {}
                    body["rg_policy_version"] = int(_data.get("version") or 1)
                    body["rg_policy_store"] = _os.getenv("RG_POLICY_STORE", "file")
                    body["rg_policy_count"] = len((_data.get("policies") or {}).keys())
                except _HEALTH_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Failed to read RG policy file for /health")
    except _HEALTH_NONCRITICAL_EXCEPTIONS:
        pass
    code = status.HTTP_200_OK if overall == "ok" else (206 if overall == "degraded" else 503)
    return JSONResponse(body, status_code=code)


@router.get("/health/live", tags=["health"], summary="Liveness probe")
async def api_liveness():
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"], summary="Readiness probe")
async def api_readiness():
    """Return readiness similar to /readyz with standardized shape."""
    r = await readyz()
    # readyz returns JSONResponse already; normalize body to include 'status'
    try:
        body = r.body  # bytes
        import json as _json

        data = _json.loads(body)
    except _HEALTH_NONCRITICAL_EXCEPTIONS:
        data = {"ready": False}
    status_txt = "ready" if data.get("ready") else "not_ready"
    return JSONResponse({"status": status_txt, **data}, status_code=(200 if data.get("ready") else 503))


@router.get("/health/metrics", tags=["health"], summary="System metrics (CPU/memory/disk)")
async def api_health_metrics():
    """Return basic system metrics for tests/diagnostics."""
    try:
        import psutil

        cpu = {
            "percent": float(psutil.cpu_percent(interval=0.1)),
        }
        vm = psutil.virtual_memory()
        du = psutil.disk_usage("/")
        mem = {
            "total": int(vm.total),
            "available": int(vm.available),
            "percent": float(vm.percent),
            "used": int(vm.used),
            "free": int(vm.free),
        }
        disk = {
            "total": int(du.total),
            "used": int(du.used),
            "free": int(du.free),
            "percent": float(du.percent),
        }
        return {"cpu": cpu, "memory": mem, "disk": disk}
    except _HEALTH_NONCRITICAL_EXCEPTIONS as e:
        logger.warning("health/metrics unavailable")
        return {
            "cpu": {"percent": 0.0},
            "memory": {"total": 0, "available": 0, "percent": 0.0, "used": 0, "free": 0},
            "disk": {"total": 0, "used": 0, "free": 0, "percent": 0.0},
        }


def _int_env(name: str, default: int) -> int:
    """Parse an environment variable into an int with a safe default."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer environment override")
        return default


def _calculate_security_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Derive human-readable security posture from the audit summary."""
    thresholds = {
        "critical_high_risk_min": _int_env("AUDIT_SEC_CRITICAL_HIGH_RISK_MIN", 1),
        "elevated_failure_min": _int_env("AUDIT_SEC_ELEVATED_FAILURE_MIN", 50),
    }
    high_risk = int(summary.get("high_risk_events") or 0)
    failures = int(summary.get("failure_events") or 0)

    risk_level = "low"
    status_text = "secure"

    if thresholds["critical_high_risk_min"] > 0 and high_risk >= thresholds["critical_high_risk_min"]:
        risk_level = "critical"
        status_text = "at_risk"
    elif thresholds["elevated_failure_min"] > 0 and failures >= thresholds["elevated_failure_min"]:
        risk_level = "high"
        status_text = "elevated"
    return {
        "risk_level": risk_level,
        "status": status_text,
        "thresholds": thresholds,
        "high_risk_events": high_risk,
        "failure_events": failures,
    }


@router.get("/health/security", tags=["health"], summary="Security posture overview")
async def api_security_health():
    """Summarize recent security audit activity and map to a risk posture."""
    response: dict[str, Any] = {
        "timestamp": _utcnow_iso(),
        "risk_level": "unknown",
        "status": "unknown",
        "summary": {},
    }

    if UnifiedAuditService is None:
        response.update(
            {
                "error": "UnifiedAuditService unavailable",
            }
        )
        return JSONResponse(response, status_code=503)

    service_instance = None
    try:
        service_instance = UnifiedAuditService()  # type: ignore[operator]
        initialize = getattr(service_instance, "initialize", None)
        if callable(initialize):
            # Prefer a lightweight initialization that doesn't spawn background tasks
            # for this one-off health read path.
            try:
                await initialize(start_background_tasks=False)
            except TypeError:
                # Back-compat for stubs/older signatures (tests monkeypatch this).
                await initialize()
        allow_cross_tenant = not bool(getattr(service_instance, "_shared_mode", False))
        summary = await service_instance.get_security_summary(  # type: ignore[assignment]
            allow_cross_tenant=allow_cross_tenant
        )
        response["summary"] = summary
        status_bits = _calculate_security_status(summary)
        response.update(status_bits)
    except _HEALTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("health/security failed")
        response.update(
            {
                "error": "Security health unavailable",
            }
        )
        return JSONResponse(response, status_code=503)
    finally:
        shutdown = getattr(service_instance, "stop", None)
        if callable(shutdown):
            try:
                await shutdown()
            except _HEALTH_NONCRITICAL_EXCEPTIONS:
                logger.debug("UnifiedAuditService stop() ignored")

    return JSONResponse(response, status_code=200)
