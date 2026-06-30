from __future__ import annotations

import inspect
import os
from typing import Any

from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.responses import JSONResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole
from tldw_Server_API.app.main import app as _app

router = APIRouter()

_RG_ENDPOINT_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
        return int(default)


def _get_app():
    """Return the current app instance, accommodating reloads in tests."""
    try:
        from tldw_Server_API.app import main as _main

        return getattr(_main, "app", _app)
    except ImportError:
        return _app


def _get_or_init_governor() -> Any | None:
    """Return the resource governor from app state or lazily initialize it.

    Tries to create a MemoryResourceGovernor using the configured policy loader
    if no governor is currently present. Returns None if initialization fails
    or no loader is available.
    """
    app = _get_app()
    gov = getattr(app.state, "rg_governor", None)
    if gov is None:
        try:
            from tldw_Server_API.app.core.Resource_Governance import (
                MemoryResourceGovernor,
            )

            loader = getattr(app.state, "rg_policy_loader", None)
            if loader is not None:
                gov = MemoryResourceGovernor(policy_loader=loader)
                app.state.rg_governor = gov
        except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            # Keep behavior consistent with previous code path: best-effort only.
            logger.debug("Resource governor lazy-init skipped")
            gov = None
    return gov


@router.get(
    "/resource-governor/policy",
    dependencies=[Depends(RequireRole("admin"))],
)
async def get_resource_governor_policy(
    include: str | None = Query(None, description="Include extra data: 'ids' or 'full'"),
) -> JSONResponse:
    """
    Return current Resource Governor policy snapshot metadata.

    - include=ids → include policy IDs list
    - include=full → include full policies and tenant payloads (use with caution)
    """
    try:
        app = _get_app()
        loader = getattr(app.state, "rg_policy_loader", None)
        # Prefer process env for store selection when app.state is unset
        try:
            store_env = os.getenv("RG_POLICY_STORE")
        except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            store_env = None
        env_store = (store_env or "").strip().lower() or None
        store = getattr(app.state, "rg_policy_store", None) or (env_store or "file")
        if env_store:
            store = env_store
        # If loader missing or points to a different path than RG_POLICY_PATH, (re)initialize
        try:
            env_path = os.getenv("RG_POLICY_PATH")
        except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            env_path = None
        snap = None
        try:
            snap = loader.get_snapshot() if loader else None
        except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            snap = None
        needs_reload = False
        if loader is None or env_path and snap and str(getattr(snap, "source_path", "")) != str(env_path):
            needs_reload = True
        elif env_store:
            current_store = (getattr(app.state, "rg_policy_store", None) or "file")
            if current_store != env_store:
                needs_reload = True
                store = env_store
        if needs_reload:
            # Initialize a loader based on current store selection
            try:
                from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                    PolicyLoader,
                    PolicyReloadConfig,
                )
                from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                    db_policy_loader as _rg_db_loader,
                )
                from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                    default_policy_loader as _rg_default_loader,
                )
                # Decide store mode: 'db' → AuthNZ-backed; otherwise file-based
                if str(store).lower() == "db":
                    # DB-backed snapshot loader
                    try:
                        from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import (
                            AuthNZPolicyStore as _RGDBStore,
                        )
                        _store = _RGDBStore()
                        interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10")
                        reload_enabled = (os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"})
                        loader = _rg_db_loader(_store, PolicyReloadConfig(enabled=reload_enabled, interval_sec=interval))
                        await loader.load_once()
                        app.state.rg_policy_loader = loader
                        app.state.rg_policy_store = "db"
                    except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                        # Fall back to file loader if DB path can't init
                        logger.warning("RG policy loader DB init failed; falling back to file store")
                        if env_path:
                            reload_enabled = (os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"})
                            interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10")
                            loader = PolicyLoader(env_path, PolicyReloadConfig(enabled=reload_enabled, interval_sec=interval))
                        else:
                            loader = _rg_default_loader()
                        await loader.load_once()
                        app.state.rg_policy_loader = loader
                        app.state.rg_policy_store = "file"
                        store = "file"
                else:
                    # File-based loader
                    if env_path:
                        reload_enabled = (os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"})
                        interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10")
                        loader = PolicyLoader(env_path, PolicyReloadConfig(enabled=reload_enabled, interval_sec=interval))
                    else:
                        loader = _rg_default_loader()
                    await loader.load_once()
                    app.state.rg_policy_loader = loader
                    app.state.rg_policy_store = "file"
                    store = "file"
                # Update snapshot metadata for health/routes that read app.state
                try:
                    snap_meta = loader.get_snapshot()
                    app.state.rg_policy_version = int(getattr(snap_meta, "version", 0) or 0)
                    app.state.rg_policy_count = len(getattr(snap_meta, "policies", {}) or {})
                except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                    # Log with context and stack trace but do not interrupt flow
                    logger.exception("Failed updating app.state RG metadata")
            except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                logger.exception("Resource governor policy loader init failed")
                return JSONResponse({"status": "unavailable", "reason": "policy_loader_not_initialized"}, status_code=503)
        # Ensure response reflects the effective store mode after init/fallback.
        store = getattr(app.state, "rg_policy_store", None) or store
        snap = loader.get_snapshot()
        body: dict[str, Any] = {
            "status": "ok",
            "version": int(getattr(snap, "version", 0) or 0),
            "store": store,
            "policies_count": len(getattr(snap, "policies", {}) or {}),
        }
        if include == "ids":
            body["policy_ids"] = sorted((snap.policies or {}).keys())
        elif include == "full":
            # Caution: large response depending on policy size
            body["policies"] = snap.policies or {}
            body["tenant"] = snap.tenant or {}
        return JSONResponse(body)
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("get_resource_governor_policy failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


# --- Policy admin endpoints (gated by RequireRole('admin')) ---
from pydantic import BaseModel, Field

from tldw_Server_API.app.core.Resource_Governance.policy_admin import (
    AuthNZPolicyAdmin,
    PolicyVersionConflictError,
)


class PolicyUpsertRequest(BaseModel):
    payload: dict[str, Any] = Field(..., description="Policy payload JSON object")
    version: int | None = Field(None, description="Optional explicit version (auto-increments if omitted)")


@router.put(
    "/resource-governor/policy/{policy_id}",
    dependencies=[Depends(RequireRole("admin"))],
)
async def upsert_policy(
    policy_id: str = Path(..., description="Policy identifier, e.g., 'chat.default'"),
    body: PolicyUpsertRequest = Body(...),
):
    try:
        admin = AuthNZPolicyAdmin()
        await admin.upsert_policy(policy_id, body.payload, version=body.version)
        # Best-effort loader refresh when using DB store
        try:
            app = _get_app()
            _env_store = (os.getenv("RG_POLICY_STORE") or "").strip().lower()
            _store_mode = getattr(app.state, "rg_policy_store", None) or (_env_store or "file")
            if _env_store and _store_mode != _env_store:
                _store_mode = _env_store
            if _store_mode == "db":
                loader = getattr(app.state, "rg_policy_loader", None)
                if getattr(app.state, "rg_policy_store", None) != "db":
                    loader = None
                if loader is None:
                    try:
                        from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import (
                            AuthNZPolicyStore as _RGDBStore,
                        )
                        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                            PolicyReloadConfig as _RGReloadCfg,
                        )
                        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                            db_policy_loader as _rg_db_loader,
                        )

                        interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10")
                        reload_enabled = (os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"})
                        loader = _rg_db_loader(_RGDBStore(), _RGReloadCfg(enabled=reload_enabled, interval_sec=interval))
                        await loader.load_once()
                        app.state.rg_policy_loader = loader
                        app.state.rg_policy_store = "db"
                    except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                        logger.debug("Policy upsert DB loader init skipped")
                elif loader is not None:
                    await loader.load_once()
        except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            logger.debug("Policy upsert refresh skipped")
        return JSONResponse({"status": "ok", "policy_id": policy_id})
    except PolicyVersionConflictError:
        logger.debug("upsert_policy version conflict")
        return JSONResponse(
            {
                "status": "conflict",
                "error": "version_conflict",
                "policy_id": policy_id,
                "detail": "The requested policy version is out of date.",
            },
            status_code=409,
        )
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("upsert_policy failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


@router.delete(
    "/resource-governor/policy/{policy_id}",
    dependencies=[Depends(RequireRole("admin"))],
)
async def delete_policy(
    policy_id: str = Path(..., description="Policy identifier"),
    version: int | None = Query(None, ge=1, description="Optional expected version for optimistic delete"),
):
    try:
        admin = AuthNZPolicyAdmin()
        deleted = await admin.delete_policy(policy_id, version=version)
        # Best-effort loader refresh when using DB store
        try:
            app = _get_app()
            _env_store = (os.getenv("RG_POLICY_STORE") or "").strip().lower()
            _store_mode = getattr(app.state, "rg_policy_store", None) or (_env_store or "file")
            if _env_store and _store_mode != _env_store:
                _store_mode = _env_store
            if _store_mode == "db":
                loader = getattr(app.state, "rg_policy_loader", None)
                if getattr(app.state, "rg_policy_store", None) != "db":
                    loader = None
                if loader is None:
                    try:
                        from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import (
                            AuthNZPolicyStore as _RGDBStore,
                        )
                        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                            PolicyReloadConfig as _RGReloadCfg,
                        )
                        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
                            db_policy_loader as _rg_db_loader,
                        )

                        interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10")
                        reload_enabled = (os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"})
                        loader = _rg_db_loader(_RGDBStore(), _RGReloadCfg(enabled=reload_enabled, interval_sec=interval))
                        await loader.load_once()
                        app.state.rg_policy_loader = loader
                        app.state.rg_policy_store = "db"
                    except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                        logger.debug("Policy delete DB loader init skipped")
                elif loader is not None:
                    await loader.load_once()
        except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
            logger.debug("Policy delete refresh skipped")
        return JSONResponse({"status": "ok", "deleted": int(deleted)})
    except PolicyVersionConflictError:
        logger.debug("delete_policy version conflict")
        return JSONResponse(
            {
                "status": "conflict",
                "error": "version_conflict",
                "policy_id": policy_id,
                "detail": "The requested policy version is out of date.",
            },
            status_code=409,
        )
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("delete_policy failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


@router.get(
    "/resource-governor/policies",
    dependencies=[Depends(RequireRole("admin"))],
)
async def list_policies():
    try:
        admin = AuthNZPolicyAdmin()
        rows = await admin.list_policies()
        return JSONResponse({"status": "ok", "items": rows, "count": len(rows)})
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("list_policies failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


@router.get(
    "/resource-governor/policy/{policy_id}",
    dependencies=[Depends(RequireRole("admin"))],
)
async def get_policy(policy_id: str = Path(..., description="Policy identifier")):
    try:
        admin = AuthNZPolicyAdmin()
        rec = await admin.get_policy_record(policy_id)
        if not rec:
            return JSONResponse({"status": "not_found", "policy_id": policy_id}, status_code=404)
        return JSONResponse({"status": "ok", **rec})
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("get_policy failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


# --- Diagnostics (admin) ---
@router.get(
    "/resource-governor/diag/peek",
    dependencies=[Depends(RequireRole("admin"))],
)
async def rg_diag_peek(
    entity: str = Query(..., description="Entity key, e.g., 'user:123'"),
    categories: str = Query(..., description="Comma-separated categories, e.g., 'requests,tokens'"),
    policy_id: str | None = Query(None, description="Optional policy id to use for peek_with_policy when supported"),
):
    try:
        gov = _get_or_init_governor()
        if gov is None:
            return JSONResponse({"status": "unavailable", "reason": "governor_not_initialized"}, status_code=503)
        cats = [c.strip() for c in categories.split(",") if c.strip()]
        # Prefer policy-aware peek when policy_id is provided and supported
        if policy_id and hasattr(gov, "peek_with_policy") and callable(gov.peek_with_policy):
            data = gov.peek_with_policy(entity, cats, policy_id)  # type: ignore[attr-defined]
            if inspect.isawaitable(data):
                data = await data
        else:
            data = gov.peek(entity, cats)
            if inspect.isawaitable(data):
                data = await data
        return JSONResponse({"status": "ok", "entity": entity, "data": data, "policy_id": policy_id})
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("rg_diag_peek failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


@router.get(
    "/resource-governor/diag/query",
    dependencies=[Depends(RequireRole("admin"))],
)
async def rg_diag_query(
    entity: str = Query(..., description="Entity key, e.g., 'user:123'"),
    category: str = Query(..., description="Category name, e.g., 'requests'"),
):
    try:
        gov = _get_or_init_governor()
        if gov is None:
            return JSONResponse({"status": "unavailable", "reason": "governor_not_initialized"}, status_code=503)
        data = gov.query(entity, category)
        if inspect.isawaitable(data):
            data = await data
        return JSONResponse({"status": "ok", "entity": entity, "category": category, "data": data})
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("rg_diag_query failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


@router.get(
    "/resource-governor/diag/media-budget",
    dependencies=[Depends(RequireRole("admin"))],
)
async def rg_diag_media_budget(
    user_id: int = Query(..., ge=1, description="User id to inspect, e.g., 123"),
    policy_id: str = Query("media.default", description="Media policy id"),
):
    """
    Return per-user media ingestion budget limits and current usage.

    This endpoint is intended for admin UI diagnostics and reads from the
    existing Resource Governor + shared daily ledger path.
    """
    try:
        gov = _get_or_init_governor()
        if gov is None:
            return JSONResponse(
                {"status": "unavailable", "reason": "governor_not_initialized"},
                status_code=503,
            )

        app = _get_app()
        loader = getattr(app.state, "rg_policy_loader", None)
        policy: dict[str, Any] = {}
        if loader is not None:
            try:
                policy = dict(loader.get_policy(policy_id) or {})
            except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                policy = {}

        jobs_cfg = dict(policy.get("jobs") or {})
        bytes_cfg = dict(policy.get("ingestion_bytes") or {})
        jobs_limit = _safe_int(jobs_cfg.get("max_concurrent"), 0)
        daily_cap = _safe_int(bytes_cfg.get("daily_cap"), 0)

        categories: dict[str, dict[str, int]] = {}
        if jobs_limit > 0:
            categories["jobs"] = {"units": 1}
        if daily_cap > 0:
            # units=0 requests non-mutating headroom/usage details.
            categories["ingestion_bytes"] = {"units": 0}

        decision: Any | None = None
        if categories:
            try:
                from tldw_Server_API.app.core.Resource_Governance import RGRequest
            except _RG_ENDPOINT_NONCRITICAL_EXCEPTIONS:
                RGRequest = None  # type: ignore
            if RGRequest is not None:
                decision = gov.check(
                    RGRequest(
                        entity=f"user:{int(user_id)}",
                        categories=categories,
                        tags={
                            "policy_id": policy_id,
                            "endpoint": "/api/v1/resource-governor/diag/media-budget",
                        },
                    )
                )
                if inspect.isawaitable(decision):
                    decision = await decision

        details = {}
        if decision is not None:
            details = dict(getattr(decision, "details", {}) or {})
        category_details = dict(details.get("categories") or {})
        jobs_details = dict(category_details.get("jobs") or {})
        bytes_details = dict(category_details.get("ingestion_bytes") or {})

        jobs_limit_eff = _safe_int(jobs_details.get("limit"), jobs_limit)
        jobs_remaining = jobs_details.get("remaining")
        jobs_remaining_int = (
            max(0, _safe_int(jobs_remaining, 0))
            if jobs_remaining is not None
            else None
        )
        jobs_active = (
            max(0, jobs_limit_eff - jobs_remaining_int)
            if jobs_remaining_int is not None and jobs_limit_eff > 0
            else None
        )

        daily_cap_eff = _safe_int(bytes_details.get("daily_cap"), daily_cap)
        daily_used = bytes_details.get("daily_used")
        daily_remaining = bytes_details.get("daily_remaining")

        body = {
            "status": "ok",
            "entity": f"user:{int(user_id)}",
            "policy_id": policy_id,
            "limits": {
                "jobs_max_concurrent": jobs_limit_eff if jobs_limit_eff > 0 else None,
                "ingestion_bytes_daily_cap": daily_cap_eff if daily_cap_eff > 0 else None,
            },
            "usage": {
                "jobs_active": jobs_active,
                "jobs_remaining": jobs_remaining_int,
                "ingestion_bytes_daily_used": (
                    _safe_int(daily_used, 0) if daily_used is not None else None
                ),
                "ingestion_bytes_daily_remaining": (
                    _safe_int(daily_remaining, 0)
                    if daily_remaining is not None
                    else (
                        max(
                            0,
                            daily_cap_eff
                            - _safe_int(daily_used, 0),
                        )
                        if daily_cap_eff > 0 and daily_used is not None
                        else None
                    )
                ),
            },
            "retry_after": (
                _safe_int(getattr(decision, "retry_after", 0), 0)
                if decision is not None and getattr(decision, "retry_after", None) is not None
                else None
            ),
        }
        return JSONResponse(body)
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("rg_diag_media_budget failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


@router.get(
    "/resource-governor/diag/capabilities",
    dependencies=[Depends(RequireRole("admin"))],
)
async def rg_diag_capabilities():
    """Tiny capability probe to report whether Lua or fallback paths are in use."""
    try:
        gov = _get_or_init_governor()
        if gov is None:
            return JSONResponse({"status": "unavailable", "reason": "governor_not_initialized"}, status_code=503)
        caps_fn = getattr(gov, "capabilities", None)
        if callable(caps_fn):
            caps = caps_fn()
            if inspect.isawaitable(caps):
                caps = await caps
        else:
            caps = {"backend": "unknown"}
        return JSONResponse({"status": "ok", "capabilities": caps})
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("rg_diag_capabilities failed")
        return JSONResponse({"status": "error", "error": "internal server error"}, status_code=500)


# ── Coverage audit ──────────────────────────────────────────────────────────

@router.get(
    "/diag/coverage",
    dependencies=[Depends(RequireRole("admin"))],
    summary="Resource Governor endpoint coverage audit",
)
async def rg_coverage_audit():
    """Report which endpoints are governor-protected and which are excluded.

    Returns coverage percentage and lists of protected/unprotected routes.
    """
    try:
        from tldw_Server_API.app.core.Resource_Governance.coverage_audit import (
            audit_governor_coverage,
        )

        app = _get_app()
        result = audit_governor_coverage(app)
        return JSONResponse(result)
    except Exception:  # noqa: BLE001 - generic 500 handler
        logger.exception("rg_coverage_audit failed")
        return JSONResponse(
            {"status": "error", "error": "internal server error"},
            status_code=500,
        )
