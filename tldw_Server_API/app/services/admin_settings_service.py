from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminCleanupSettingsUpdate,
    NotesTitleSettingsUpdate,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.config import settings as app_settings


async def get_cleanup_settings() -> dict[str, Any]:
    """Get cleanup worker settings (enabled, interval in seconds)."""
    try:
        enabled = bool(app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
        interval = int(app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", 1800))
        return {"enabled": enabled, "interval_sec": interval}
    except Exception as exc:
        logger.error("Failed to get cleanup settings")
        raise HTTPException(status_code=500, detail="Failed to get cleanup settings") from exc


async def set_cleanup_settings(payload: AdminCleanupSettingsUpdate) -> dict[str, Any]:
    """Set cleanup worker settings (enabled, interval_sec)."""
    try:
        if payload.enabled is not None:
            app_settings["EPHEMERAL_CLEANUP_ENABLED"] = bool(payload.enabled)
        if payload.interval_sec is not None:
            app_settings["EPHEMERAL_CLEANUP_INTERVAL_SEC"] = int(payload.interval_sec)
        enabled = bool(app_settings.get("EPHEMERAL_CLEANUP_ENABLED", True))
        interval = int(app_settings.get("EPHEMERAL_CLEANUP_INTERVAL_SEC", 1800))
        return {"enabled": enabled, "interval_sec": interval}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to set cleanup settings")
        raise HTTPException(status_code=500, detail="Failed to set cleanup settings") from exc


async def get_notes_title_settings() -> dict[str, Any]:
    """Get Notes auto-title settings (LLM enabled flag and default strategy)."""
    try:
        llm_enabled = bool(app_settings.get("NOTES_TITLE_LLM_ENABLED", False))
        default_strategy = str(app_settings.get("NOTES_TITLE_DEFAULT_STRATEGY", "heuristic")).lower()
        effective_strategy = (
            default_strategy if llm_enabled or default_strategy == "heuristic" else "heuristic"
        )
        return {
            "llm_enabled": llm_enabled,
            "default_strategy": default_strategy,
            "effective_strategy": effective_strategy,
            "strategies": ["heuristic", "llm", "llm_fallback"],
        }
    except Exception as exc:
        logger.error("Failed to get notes title settings")
        raise HTTPException(status_code=500, detail="Failed to get notes title settings") from exc


async def set_notes_title_settings(payload: NotesTitleSettingsUpdate) -> dict[str, Any]:
    """Update Notes auto-title settings."""
    try:
        if payload.llm_enabled is not None:
            app_settings["NOTES_TITLE_LLM_ENABLED"] = bool(payload.llm_enabled)
        if payload.default_strategy is not None:
            app_settings["NOTES_TITLE_DEFAULT_STRATEGY"] = payload.default_strategy
        llm_enabled = bool(app_settings.get("NOTES_TITLE_LLM_ENABLED", False))
        default_strategy = str(app_settings.get("NOTES_TITLE_DEFAULT_STRATEGY", "heuristic")).lower()
        effective_strategy = (
            default_strategy if llm_enabled or default_strategy == "heuristic" else "heuristic"
        )
        return {
            "llm_enabled": llm_enabled,
            "default_strategy": default_strategy,
            "effective_strategy": effective_strategy,
            "strategies": ["heuristic", "llm", "llm_fallback"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to set notes title settings")
        raise HTTPException(status_code=500, detail="Failed to set notes title settings") from exc


# ---------------------------------------------------------------------------
# Security Risk Weights
# ---------------------------------------------------------------------------

_DEFAULT_RISK_WEIGHTS: dict[str, Any] = {
    "mfa_adoption": {"weight": 3, "cap": 40},
    "api_key_age": {"weight": 2, "cap": 25},
    "failed_logins": {"weight": 1, "cap": 20},
    "suspicious_activity": {"weight": 4, "cap": 20},
}

_RISK_WEIGHTS_SETTING_KEY = "security_risk_weights"


def _copy_default_risk_weights() -> dict[str, Any]:
    return {key: dict(value) for key, value in _DEFAULT_RISK_WEIGHTS.items()}


def _validate_risk_weights(weights: dict[str, Any] | None) -> dict[str, Any]:
    normalized = weights if isinstance(weights, dict) else {}
    validated: dict[str, Any] = {}
    for key, defaults in _DEFAULT_RISK_WEIGHTS.items():
        candidate = normalized.get(key)
        if not isinstance(candidate, dict):
            validated[key] = dict(defaults)
            continue

        try:
            weight = int(candidate.get("weight", defaults["weight"]))
        except (TypeError, ValueError):
            weight = defaults["weight"]
        try:
            cap = int(candidate.get("cap", defaults["cap"]))
        except (TypeError, ValueError):
            cap = defaults["cap"]

        validated[key] = {
            "weight": max(0, min(10, weight)),
            "cap": max(0, min(100, cap)),
        }
    return validated


async def _ensure_admin_settings_table() -> Any:
    return await get_db_pool()


async def get_risk_weights() -> dict[str, Any]:
    """Return the current risk weight configuration."""
    try:
        pool = await _ensure_admin_settings_table()
        row = await pool.fetchone(
            "SELECT value_json FROM admin_settings WHERE setting_key = ?",
            _RISK_WEIGHTS_SETTING_KEY,
        )
        raw_payload = row.get("value_json") if row else None
        if not raw_payload:
            return _copy_default_risk_weights()
        try:
            payload = json.loads(str(raw_payload))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode persisted risk weights; falling back to defaults")
            return _copy_default_risk_weights()
        return _validate_risk_weights(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get risk weights")
        raise HTTPException(status_code=500, detail="Failed to get risk weights") from exc


async def set_risk_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Update risk weight configuration."""
    try:
        pool = await _ensure_admin_settings_table()
        existing = await get_risk_weights()
        merged = dict(existing)
        merged.update(weights if isinstance(weights, dict) else {})
        validated = _validate_risk_weights(merged)
        await pool.execute(
            """
            INSERT INTO admin_settings (setting_key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(setting_key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            _RISK_WEIGHTS_SETTING_KEY,
            json.dumps(validated, sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
        )
        return validated
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to set risk weights")
        raise HTTPException(status_code=500, detail="Failed to set risk weights") from exc
