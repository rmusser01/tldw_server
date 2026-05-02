from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.utils.pagination import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    OrgBudgetItem,
    OrgBudgetListResponse,
    OrgBudgetUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Billing.plan_limits import get_plan_limits


async def _emit_budget_audit_event(*args, **kwargs):
    """Dispatch audit events through the admin module to preserve test monkeypatch hooks."""
    try:
        from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
    except Exception:
        admin_mod = None
    else:
        emit_fn = getattr(admin_mod, "emit_budget_audit_event", None)
        if callable(emit_fn):
            # Important: do not swallow emit_fn exceptions. Tests intentionally
            # monkeypatch this symbol and expect failures to block updates.
            return await emit_fn(*args, **kwargs)

    from tldw_Server_API.app.services.budget_audit_service import emit_budget_audit_event

    return await emit_budget_audit_event(*args, **kwargs)


_BUDGET_KEYS = {
    "budget_day_usd",
    "budget_month_usd",
    "budget_day_tokens",
    "budget_month_tokens",
}
_PLATFORM_ADMIN_ROLES = frozenset({"admin", "owner", "super_admin"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _normalized_claim_values(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def _principal_has_platform_admin_claims(principal: AuthPrincipal) -> bool:
    roles = _normalized_claim_values(principal.roles)
    permissions = _normalized_claim_values(principal.permissions)
    if roles & _PLATFORM_ADMIN_ROLES:
        return True
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _parse_json_payload(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug(f"admin budgets: invalid JSON payload: {exc}")
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def _normalize_threshold_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        raise ValueError("Alert thresholds must be a list")
    if not values:
        raise ValueError("Alert thresholds must not be empty")
    cleaned: list[int] = []
    for val in values:
        try:
            num = int(val)
        except (TypeError, ValueError) as exc:
            raise ValueError("Alert thresholds must be integers") from exc
        if num < 1 or num > 100:
            raise ValueError("Alert thresholds must be between 1 and 100")
        cleaned.append(num)
    return sorted(set(cleaned))


def _coerce_alert_thresholds(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return {"global": value}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        if "global" in value:
            out["global"] = value.get("global")
        if "per_metric" in value:
            out["per_metric"] = value.get("per_metric")
        return out or None
    return None


def _coerce_enforcement_mode(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"global": value}
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        if "global" in value:
            out["global"] = value.get("global")
        if "per_metric" in value:
            out["per_metric"] = value.get("per_metric")
        return out or None
    return None


def _normalize_alert_thresholds_update(value: Any) -> dict[str, Any] | None:
    payload = _coerce_alert_thresholds(value)
    if payload is None:
        return None
    out: dict[str, Any] = {}
    if "global" in payload:
        global_value = payload.get("global")
        if global_value is None:
            out["global"] = None
        else:
            out["global"] = _normalize_threshold_list(global_value)
    if "per_metric" in payload:
        per_metric = payload.get("per_metric")
        if per_metric is None:
            out["per_metric"] = None
        elif not isinstance(per_metric, dict):
            raise ValueError("Per-metric thresholds must be a mapping")
        else:
            cleaned: dict[str, Any] = {}
            for key, values in per_metric.items():
                if key not in _BUDGET_KEYS:
                    raise ValueError("Unknown per-metric budget key")
                if values is None:
                    cleaned[key] = None
                else:
                    cleaned[key] = _normalize_threshold_list(values)
            out["per_metric"] = cleaned
    return out or None


def _normalize_enforcement_mode_update(value: Any) -> dict[str, Any] | None:
    payload = _coerce_enforcement_mode(value)
    if payload is None:
        return None
    out: dict[str, Any] = {}
    if "global" in payload:
        global_value = payload.get("global")
        if global_value is None:
            out["global"] = None
        elif global_value in {"none", "soft", "hard"}:
            out["global"] = global_value
        else:
            raise ValueError("Enforcement mode must be none, soft, or hard")
    if "per_metric" in payload:
        per_metric = payload.get("per_metric")
        if per_metric is None:
            out["per_metric"] = None
        elif not isinstance(per_metric, dict):
            raise ValueError("Per-metric enforcement must be a mapping")
        else:
            cleaned: dict[str, Any] = {}
            for key, value in per_metric.items():
                if key not in _BUDGET_KEYS:
                    raise ValueError("Unknown per-metric budget key")
                if value is None:
                    cleaned[key] = None
                elif value in {"none", "soft", "hard"}:
                    cleaned[key] = value
                else:
                    raise ValueError("Enforcement mode must be none, soft, or hard")
            out["per_metric"] = cleaned
    return out or None


def _normalize_alert_thresholds_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    if "global" in value and value.get("global") is not None:
        out["global"] = _normalize_threshold_list(value.get("global"))
    if "per_metric" in value and isinstance(value.get("per_metric"), dict):
        cleaned: dict[str, Any] = {}
        for key, values in value.get("per_metric", {}).items():
            if key not in _BUDGET_KEYS:
                raise ValueError("Unknown per-metric budget key")
            if values is None:
                continue
            cleaned[key] = _normalize_threshold_list(values)
        if cleaned:
            out["per_metric"] = cleaned
    return out or None


def _normalize_enforcement_mode_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    global_value = value.get("global")
    if global_value is not None:
        if global_value not in {"none", "soft", "hard"}:
            raise ValueError("Enforcement mode must be none, soft, or hard")
        out["global"] = global_value
    per_metric = value.get("per_metric")
    if isinstance(per_metric, dict):
        cleaned: dict[str, Any] = {}
        for key, per_value in per_metric.items():
            if key not in _BUDGET_KEYS:
                raise ValueError("Unknown per-metric budget key")
            if per_value is None:
                continue
            if per_value not in {"none", "soft", "hard"}:
                raise ValueError("Enforcement mode must be none, soft, or hard")
            cleaned[key] = per_value
        if cleaned:
            out["per_metric"] = cleaned
    return out or None


def _normalize_budget_payload(raw: Any) -> dict[str, Any]:
    data = _parse_json_payload(raw)
    if not data:
        return {}

    budgets: dict[str, Any] = {}
    if isinstance(data.get("budgets"), dict):
        budgets.update(data.get("budgets") or {})
    provider_budgets = budgets.pop("provider_budgets", None)
    for key in _BUDGET_KEYS:
        if key in data and key not in budgets:
            budgets[key] = data[key]
    if provider_budgets is None and isinstance(data.get("provider_budgets"), dict):
        provider_budgets = data.get("provider_budgets")
    payload = {"budgets": budgets} if budgets else {}
    if isinstance(provider_budgets, dict) and provider_budgets:
        payload["provider_budgets"] = provider_budgets
    thresholds = _coerce_alert_thresholds(data.get("alert_thresholds"))
    if thresholds is not None:
        payload["alert_thresholds"] = thresholds
    enforcement = _coerce_enforcement_mode(data.get("enforcement_mode"))
    if enforcement is not None:
        payload["enforcement_mode"] = enforcement
    return payload


def _flatten_budget_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or not payload:
        return {}
    flat: dict[str, Any] = {}
    if isinstance(payload.get("budgets"), dict):
        flat.update(payload.get("budgets") or {})
    else:
        for key in _BUDGET_KEYS:
            if key in payload:
                flat[key] = payload[key]
    if isinstance(payload.get("provider_budgets"), dict):
        flat["provider_budgets"] = payload.get("provider_budgets")
    if "alert_thresholds" in payload:
        flat["alert_thresholds"] = payload.get("alert_thresholds")
    if "enforcement_mode" in payload:
        flat["enforcement_mode"] = payload.get("enforcement_mode")
    return flat


def _inflate_budget_payload(flat: dict[str, Any]) -> dict[str, Any]:
    if not flat:
        return {}
    payload: dict[str, Any] = {}
    for key in _BUDGET_KEYS:
        if key in flat:
            payload[key] = flat[key]
    if "provider_budgets" in flat:
        payload["provider_budgets"] = flat.get("provider_budgets")
    if "alert_thresholds" in flat:
        payload["alert_thresholds"] = flat.get("alert_thresholds")
    if "enforcement_mode" in flat:
        payload["enforcement_mode"] = flat.get("enforcement_mode")
    return payload


def _apply_budget_response_defaults(budgets: dict[str, Any]) -> dict[str, Any]:
    """Add explicit response defaults for optional budget sub-fields."""
    if not budgets:
        return {}
    normalized = dict(budgets)

    thresholds = normalized.get("alert_thresholds")
    if thresholds is not None:
        thresholds = {} if not isinstance(thresholds, dict) else dict(thresholds)
        if thresholds.get("per_metric") is None:
            thresholds["per_metric"] = {}
        normalized["alert_thresholds"] = thresholds

    enforcement = normalized.get("enforcement_mode")
    enforcement_payload: dict[str, Any] = {}
    if isinstance(enforcement, dict):
        enforcement_payload.update(enforcement)
    if enforcement_payload.get("global") is None:
        enforcement_payload["global"] = "none"
    if enforcement_payload.get("per_metric") is None:
        enforcement_payload["per_metric"] = {}
    normalized["enforcement_mode"] = enforcement_payload
    return normalized


def _build_budget_item(row: dict[str, Any]) -> dict[str, Any]:
    plan_name = row.get("plan_name") or "free"
    plan_display_name = row.get("plan_display_name") or plan_name.title()
    plan_limits = _parse_json_payload(row.get("plan_limits_json"))
    if not plan_limits:
        plan_limits = get_plan_limits(plan_name)

    custom_limits = _parse_json_payload(row.get("custom_limits_json"))
    budgets_payload = _normalize_budget_payload(row.get("budgets_json"))
    if not budgets_payload and isinstance(custom_limits, dict) and "budgets" in custom_limits:
        budgets_payload = _normalize_budget_payload(custom_limits.get("budgets"))
    budgets = _flatten_budget_payload(budgets_payload)
    budgets_with_defaults = _apply_budget_response_defaults(budgets)
    if isinstance(custom_limits, dict) and "budgets" in custom_limits:
        custom_limits = dict(custom_limits)
        custom_limits.pop("budgets", None)

    effective_limits = dict(plan_limits)
    if isinstance(custom_limits, dict) and custom_limits:
        effective_limits.update(custom_limits)
    if budgets_with_defaults:
        effective_limits["budgets"] = _inflate_budget_payload(budgets_with_defaults)

    return {
        "org_id": int(row.get("org_id")),
        "org_name": row.get("org_name") or "",
        "org_slug": row.get("org_slug"),
        "plan_name": plan_name,
        "plan_display_name": plan_display_name,
        "budgets": budgets_with_defaults,
        "custom_limits": custom_limits,
        "effective_limits": effective_limits,
        "updated_at": row.get("budgets_updated_at") or row.get("updated_at"),
    }


async def list_budgets(
    *,
    principal: AuthPrincipal,
    org_id: int | None,
    page: int,
    limit: int,
    db,
) -> OrgBudgetListResponse:
    """List organization budgets and plan context."""
    try:
        del principal  # admin role already enforced by router dependency
        org_ids = [org_id] if org_id is not None else None
        items, total = await list_org_budgets(
            db,
            org_ids=org_ids,
            page=page,
            limit=limit,
        )
        return OrgBudgetListResponse(
            items=[OrgBudgetItem(**item) for item in items],
            total=total,
            page=page,
            limit=limit,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=(page - 1) * limit,
                count=len(items),
            ),
        )
    except Exception as exc:
        logger.error("Failed to list org budgets")
        raise HTTPException(status_code=500, detail="Failed to list org budgets") from exc


async def upsert_budget(
    *,
    payload: OrgBudgetUpdateRequest,
    request,
    principal: AuthPrincipal,
    db,
) -> OrgBudgetItem:
    """Upsert budget settings for an organization with audit logging."""
    budget_updates = None
    if payload.budgets is not None:
        budget_updates = payload.budgets.model_dump(exclude_unset=True, by_alias=True)
    try:
        item, audit_changes = await upsert_org_budget(
            db,
            org_id=payload.org_id,
            budget_updates=budget_updates,
            clear_budgets=payload.clear_budgets,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "org_not_found":
            raise HTTPException(status_code=404, detail="org_not_found") from exc
        if detail == "plan_not_found":
            raise HTTPException(status_code=500, detail="plan_not_found") from exc
        if detail == "subscription_not_found":
            raise HTTPException(status_code=500, detail="subscription_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_budget_update") from exc
    except Exception as exc:
        logger.error("Failed to upsert org budget")
        raise HTTPException(status_code=500, detail="Failed to upsert org budget") from exc

    actor_role = None
    try:
        if _principal_has_platform_admin_claims(principal):
            actor_role = "admin"
        elif principal.roles:
            actor_role = principal.roles[0]
    except Exception:
        actor_role = None

    try:
        await _emit_budget_audit_event(
            request,
            principal,
            org_id=payload.org_id,
            budget_updates=budget_updates,
            audit_changes=audit_changes,
            clear_budgets=payload.clear_budgets,
            actor_role=actor_role,
        )
    except Exception as exc:
        logger.error("Budget audit failed")
        raise HTTPException(status_code=500, detail="audit_failed") from exc

    return OrgBudgetItem(**item)


def merge_budget_settings(
    existing: dict[str, Any],
    updates: dict[str, Any] | None,
    *,
    clear: bool,
) -> dict[str, Any]:
    """Merge budget updates into existing budget settings."""
    if clear:
        return {}
    if not updates:
        return dict(existing)
    merged = dict(existing)
    for key, value in updates.items():
        if key in _BUDGET_KEYS:
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
            continue
        if key == "alert_thresholds":
            merged_thresholds = _merge_alert_thresholds(merged.get(key), value)
            if merged_thresholds is None:
                merged.pop(key, None)
            else:
                merged[key] = merged_thresholds
            continue
        if key == "enforcement_mode":
            merged_enforcement = _merge_enforcement_mode(merged.get(key), value)
            if merged_enforcement is None:
                merged.pop(key, None)
            else:
                merged[key] = merged_enforcement
            continue
        if key == "provider_budgets":
            merged_providers = _merge_provider_budgets(merged.get(key), value)
            if merged_providers is None:
                merged.pop(key, None)
            else:
                merged[key] = merged_providers
            continue
        raise ValueError("invalid_budget_update")
    return merged


def _merge_provider_budgets(
    existing: Any, updates: Any
) -> dict[str, Any] | None:
    """Merge per-provider budget thresholds.

    Structure: ``{"openai": {"month_usd": 50}, "anthropic": {"month_usd": 100}}``
    Setting a provider to ``None`` removes it.
    """
    if updates is None:
        return None
    existing_payload = existing if isinstance(existing, dict) else {}
    merged = dict(existing_payload)
    if not isinstance(updates, dict):
        return merged or None
    for provider, settings in updates.items():
        if settings is None:
            merged.pop(provider, None)
        elif isinstance(settings, dict):
            existing_provider = merged.get(provider)
            if not isinstance(existing_provider, dict):
                existing_provider = {}
            merged[provider] = {**existing_provider, **settings}
        else:
            merged[provider] = settings
    return merged or None


def _merge_alert_thresholds(existing: Any, updates: Any) -> dict[str, Any] | None:
    if updates is None:
        return None
    existing_payload = existing if isinstance(existing, dict) else {}
    merged: dict[str, Any] = {}
    if "global" in existing_payload:
        merged["global"] = existing_payload.get("global")
    if isinstance(existing_payload.get("per_metric"), dict):
        merged["per_metric"] = dict(existing_payload.get("per_metric") or {})

    normalized_updates = _normalize_alert_thresholds_update(updates)
    if not normalized_updates:
        return _normalize_alert_thresholds_payload(merged)

    if "global" in normalized_updates:
        if normalized_updates["global"] is None:
            merged.pop("global", None)
        else:
            merged["global"] = normalized_updates["global"]
    if "per_metric" in normalized_updates:
        if normalized_updates["per_metric"] is None:
            merged.pop("per_metric", None)
        else:
            per_metric = merged.get("per_metric")
            if not isinstance(per_metric, dict):
                per_metric = {}
            for key, values in normalized_updates["per_metric"].items():
                if values is None:
                    per_metric.pop(key, None)
                else:
                    per_metric[key] = values
            if per_metric:
                merged["per_metric"] = per_metric
            else:
                merged.pop("per_metric", None)
    return _normalize_alert_thresholds_payload(merged)


def _merge_enforcement_mode(existing: Any, updates: Any) -> dict[str, Any] | None:
    if updates is None:
        return None
    existing_payload = existing if isinstance(existing, dict) else {}
    merged: dict[str, Any] = {}
    if "global" in existing_payload:
        merged["global"] = existing_payload.get("global")
    if isinstance(existing_payload.get("per_metric"), dict):
        merged["per_metric"] = dict(existing_payload.get("per_metric") or {})

    normalized_updates = _normalize_enforcement_mode_update(updates)
    if not normalized_updates:
        return _normalize_enforcement_mode_payload(merged)

    if "global" in normalized_updates:
        if normalized_updates["global"] is None:
            merged.pop("global", None)
        else:
            merged["global"] = normalized_updates["global"]
    if "per_metric" in normalized_updates:
        if normalized_updates["per_metric"] is None:
            merged.pop("per_metric", None)
        else:
            per_metric = merged.get("per_metric")
            if not isinstance(per_metric, dict):
                per_metric = {}
            for key, value in normalized_updates["per_metric"].items():
                if value is None:
                    per_metric.pop(key, None)
                else:
                    per_metric[key] = value
            if per_metric:
                merged["per_metric"] = per_metric
            else:
                merged.pop("per_metric", None)
    return _normalize_enforcement_mode_payload(merged)


def _infer_change_data_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def build_budget_change_log(
    existing_budgets: dict[str, Any],
    merged_budgets: dict[str, Any],
    budget_updates: dict[str, Any] | None,
    *,
    clear_budgets: bool,
) -> list[dict[str, Any]]:
    """Create audit-friendly change entries for budget updates."""
    if clear_budgets:
        return [
            {
                "field_name": "budgets",
                "old_value": existing_budgets or {},
                "new_value": None,
                "data_type": "object",
                "notes": "clear_budgets=true",
            }
        ]

    if not budget_updates:
        return []

    changes: list[dict[str, Any]] = []
    for key, _ in budget_updates.items():
        if key in _BUDGET_KEYS:
            old_value = existing_budgets.get(key)
            new_value = merged_budgets.get(key)
            if old_value == new_value:
                continue
            data_type = _infer_change_data_type(new_value if new_value is not None else old_value)
            changes.append(
                {
                    "field_name": f"budgets.{key}",
                    "old_value": old_value,
                    "new_value": new_value,
                    "data_type": data_type,
                }
            )
            continue
        if key == "alert_thresholds":
            changes.extend(
                _build_nested_changes(
                    "alert_thresholds",
                    existing_budgets.get("alert_thresholds"),
                    merged_budgets.get("alert_thresholds"),
                    budget_updates.get("alert_thresholds"),
                )
            )
            continue
        if key == "enforcement_mode":
            changes.extend(
                _build_nested_changes(
                    "enforcement_mode",
                    existing_budgets.get("enforcement_mode"),
                    merged_budgets.get("enforcement_mode"),
                    budget_updates.get("enforcement_mode"),
                )
            )
            continue
        if key == "provider_budgets":
            changes.extend(
                _build_generic_map_changes(
                    "provider_budgets",
                    existing_budgets.get("provider_budgets"),
                    merged_budgets.get("provider_budgets"),
                    budget_updates.get("provider_budgets"),
                )
            )
            continue
    return changes


def _build_nested_changes(
    field_name: str,
    existing_value: Any,
    merged_value: Any,
    update_value: Any,
) -> list[dict[str, Any]]:
    if update_value is None:
        if existing_value is None:
            return []
        return [
            {
                "field_name": f"budgets.{field_name}",
                "old_value": existing_value,
                "new_value": None,
                "data_type": _infer_change_data_type(existing_value),
            }
        ]

    if field_name == "alert_thresholds":
        update_payload = _coerce_alert_thresholds(update_value) or {}
    elif field_name == "enforcement_mode":
        update_payload = _coerce_enforcement_mode(update_value) or {}
    else:
        update_payload = {}

    if not update_payload:
        return []

    existing_payload = existing_value if isinstance(existing_value, dict) else {}
    merged_payload = merged_value if isinstance(merged_value, dict) else {}
    changes: list[dict[str, Any]] = []

    if "global" in update_payload:
        old_value = existing_payload.get("global")
        new_value = merged_payload.get("global")
        if old_value != new_value:
            changes.append(
                {
                    "field_name": f"budgets.{field_name}.global",
                    "old_value": old_value,
                    "new_value": new_value,
                    "data_type": _infer_change_data_type(new_value if new_value is not None else old_value),
                }
            )

    if "per_metric" in update_payload:
        per_metric_update = update_payload.get("per_metric")
        if per_metric_update is None:
            old_value = existing_payload.get("per_metric")
            new_value = merged_payload.get("per_metric")
            if old_value != new_value:
                changes.append(
                    {
                        "field_name": f"budgets.{field_name}.per_metric",
                        "old_value": old_value,
                        "new_value": new_value,
                        "data_type": _infer_change_data_type(new_value if new_value is not None else old_value),
                    }
                )
        elif isinstance(per_metric_update, dict):
            old_map = existing_payload.get("per_metric") if isinstance(existing_payload.get("per_metric"), dict) else {}
            new_map = merged_payload.get("per_metric") if isinstance(merged_payload.get("per_metric"), dict) else {}
            for metric_key in per_metric_update:
                old_value = old_map.get(metric_key)
                new_value = new_map.get(metric_key)
                if old_value == new_value:
                    continue
                changes.append(
                    {
                        "field_name": f"budgets.{field_name}.per_metric.{metric_key}",
                        "old_value": old_value,
                        "new_value": new_value,
                        "data_type": _infer_change_data_type(new_value if new_value is not None else old_value),
                    }
                )

    return changes


def _build_generic_map_changes(
    field_name: str,
    existing_value: Any,
    merged_value: Any,
    update_value: Any,
) -> list[dict[str, Any]]:
    if update_value is None:
        if existing_value is None:
            return []
        return [
            {
                "field_name": f"budgets.{field_name}",
                "old_value": existing_value,
                "new_value": None,
                "data_type": _infer_change_data_type(existing_value),
            }
        ]

    if not isinstance(update_value, dict):
        if existing_value == merged_value:
            return []
        changed_value = merged_value if merged_value is not None else existing_value
        return [
            {
                "field_name": f"budgets.{field_name}",
                "old_value": existing_value,
                "new_value": merged_value,
                "data_type": _infer_change_data_type(changed_value),
            }
        ]

    existing_payload = existing_value if isinstance(existing_value, dict) else {}
    merged_payload = merged_value if isinstance(merged_value, dict) else {}
    changes: list[dict[str, Any]] = []
    for key, value in update_value.items():
        changes.extend(
            _build_generic_map_changes(
                f"{field_name}.{key}",
                existing_payload.get(key),
                merged_payload.get(key),
                value,
            )
        )
    return changes


def _is_db_pool_object(db: Any) -> bool:
    return isinstance(db, DatabasePool)


def _is_postgres_connection(db: Any) -> bool:
    """Resolve backend mode from connection/adapter shape without global probing."""
    if _is_db_pool_object(db):
        return getattr(db, "pool", None) is not None

    sqlite_hint = getattr(db, "_is_sqlite", None)
    if isinstance(sqlite_hint, bool):
        return not sqlite_hint

    # SQLite shims in AuthNZ DatabasePool expose underlying aiosqlite connection as `_c`.
    if getattr(db, "_c", None) is not None:
        return False

    module_name = getattr(type(db), "__module__", "")
    if isinstance(module_name, str) and module_name.startswith("asyncpg"):
        return True

    # Last-resort interface hint for test doubles and adapters.
    return callable(getattr(db, "fetchrow", None))


async def _fetchval(db, query: str, params: list[Any], *, pg: bool) -> Any:
    if pg:
        return await db.fetchval(query, *params)
    if _is_db_pool_object(db):
        return await db.fetchval(query, params)
    cur = await db.execute(query, params)
    row = await cur.fetchone()
    return row[0] if row else None


async def _fetchrow(db, query: str, params: list[Any], *, pg: bool) -> dict[str, Any] | None:
    if pg:
        row = await db.fetchrow(query, *params)
        return dict(row) if row and not isinstance(row, dict) else row
    if _is_db_pool_object(db):
        row = await db.fetchrow(query, params)
        return dict(row) if row and not isinstance(row, dict) else row
    cur = await db.execute(query, params)
    row = await cur.fetchone()
    if not row:
        return None
    if hasattr(row, "keys"):
        return {key: row[key] for key in row}
    return dict(row) if not isinstance(row, dict) else row


async def _fetchrows(db, query: str, params: list[Any], *, pg: bool) -> list[Any]:
    if pg:
        return await db.fetch(query, *params)
    if _is_db_pool_object(db):
        return await db.fetchall(query, params)
    cur = await db.execute(query, params)
    return await cur.fetchall()


async def list_org_budgets(
    db,
    *,
    org_ids: list[int] | None,
    page: int,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    offset = (page - 1) * limit
    pg = _is_postgres_connection(db)
    if org_ids is not None and len(org_ids) == 0:
        return [], 0

    conditions: list[str] = []
    params: list[Any] = []
    if org_ids is not None:
        if pg:
            conditions.append(f"o.id = ANY(${len(params) + 1})")
            params.append(org_ids)
        else:
            placeholders = ",".join("?" for _ in org_ids)
            conditions.append(f"o.id IN ({placeholders})")
            params.extend(org_ids)

    where_clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    count_sql_template = "SELECT COUNT(*) FROM organizations o{where_clause}"
    count_sql = count_sql_template.format_map(locals())  # nosec B608
    total = int(await _fetchval(db, count_sql, params, pg=pg) or 0)

    if pg:
        limit_placeholder = f"${len(params) + 1}"
        offset_placeholder = f"${len(params) + 2}"
    else:
        limit_placeholder = "?"
        offset_placeholder = "?"

    list_budgets_sql_template = (
        "SELECT o.id as org_id, o.name as org_name, o.slug as org_slug, "
        "ob.budgets_json, ob.updated_at as budgets_updated_at "
        "FROM organizations o "
        "LEFT JOIN org_budgets ob ON ob.org_id = o.id "
        "{where_clause} "
        "ORDER BY o.name ASC LIMIT {limit_placeholder} OFFSET {offset_placeholder}"
    )
    sql = list_budgets_sql_template.format_map(locals())  # nosec B608

    rows = await _fetchrows(db, sql, params + [limit, offset], pg=pg)
    items = []
    for row in rows:
        row_dict = dict(row) if not isinstance(row, dict) else row
        items.append(_build_budget_item(row_dict))
    return items, total


async def upsert_org_budget(
    db,
    *,
    org_id: int,
    budget_updates: dict[str, Any] | None,
    clear_budgets: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pg = _is_postgres_connection(db)

    org_row = await _fetchrow(
        db,
        "SELECT id, name, slug FROM organizations WHERE id = $1",
        [org_id],
        pg=pg,
    )
    if not org_row:
        raise ValueError("org_not_found")
    org_data = dict(org_row) if not isinstance(org_row, dict) else org_row

    custom_limits: dict[str, Any] = {}
    row_dict: dict[str, Any] = {}

    budget_row = await _fetchrow(
        db,
        "SELECT org_id, budgets_json, updated_at FROM org_budgets WHERE org_id = $1",
        [org_id],
        pg=pg,
    )
    budgets_payload = _normalize_budget_payload(
        budget_row.get("budgets_json") if budget_row else None
    )
    legacy_budgets = False
    if not budgets_payload and isinstance(custom_limits, dict) and "budgets" in custom_limits:
        budgets_payload = _normalize_budget_payload(custom_limits.get("budgets"))
        legacy_budgets = True

    existing_budgets = _flatten_budget_payload(budgets_payload)
    merged_budgets = merge_budget_settings(existing_budgets, budget_updates, clear=clear_budgets)
    audit_changes = build_budget_change_log(
        existing_budgets,
        merged_budgets,
        budget_updates,
        clear_budgets=clear_budgets,
    )
    updated_payload = _inflate_budget_payload(merged_budgets)

    now = datetime.utcnow()
    should_upsert_budget = budget_updates is not None or clear_budgets or legacy_budgets
    if should_upsert_budget:
        payload = json.dumps(updated_payload) if updated_payload else None
        if pg:
            await db.execute(
                """
                INSERT INTO org_budgets (org_id, budgets_json, updated_at)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT (org_id)
                DO UPDATE SET budgets_json = EXCLUDED.budgets_json, updated_at = EXCLUDED.updated_at
                """,
                org_id,
                payload,
                now,
            )
        else:
            await db.execute(
                """
                INSERT INTO org_budgets (org_id, budgets_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(org_id)
                DO UPDATE SET budgets_json = excluded.budgets_json, updated_at = excluded.updated_at
                """,
                (org_id, payload, now),
            )
        row_dict["budgets_json"] = payload
        row_dict["budgets_updated_at"] = now
    else:
        row_dict["budgets_json"] = budgets_payload
        row_dict["budgets_updated_at"] = budget_row.get("updated_at") if budget_row else None

    row_dict.update(
        {
            "org_id": org_data.get("id"),
            "org_name": org_data.get("name"),
            "org_slug": org_data.get("slug"),
        }
    )
    return _build_budget_item(row_dict), audit_changes
