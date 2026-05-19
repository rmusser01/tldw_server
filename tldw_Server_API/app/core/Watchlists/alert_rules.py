"""
Watchlist Alert Rules — Evaluation Engine

Evaluates user-defined alert rules against completed watchlist run statistics
and creates notifications when conditions are met.

Supported condition types:
- no_items: Run produced zero items
- error_rate_above: Error rate exceeds threshold (0.0–1.0)
- items_below: Total items below threshold
- items_above: Total items above threshold (unusual activity)
- run_failed: Run ended with failed status
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.watchlist_alert_rules_db import (
    AlertRule,
    create_watchlist_alert_rule,
    delete_watchlist_alert_rule,
    ensure_watchlist_alert_rules_table,
    get_watchlist_alert_rule,
    list_watchlist_alert_rules,
    update_watchlist_alert_rule,
)


class AlertConditionType(str, Enum):
    NO_ITEMS = "no_items"
    ERROR_RATE_ABOVE = "error_rate_above"
    ITEMS_BELOW = "items_below"
    ITEMS_ABOVE = "items_above"
    RUN_FAILED = "run_failed"


ALERT_CONDITION_TYPE_VALUES = frozenset(condition.value for condition in AlertConditionType)
DEFAULT_ERROR_RATE_THRESHOLD = 0.5
DEFAULT_ITEMS_BELOW_THRESHOLD = 1
DEFAULT_ITEMS_ABOVE_THRESHOLD = 1000
WATCHLIST_HEALTH_NOTIFICATION_TYPE = "watchlist_health_issue"
WATCHLIST_LEGACY_ALERT_KIND = "watchlist_alert"


def ensure_alert_rules_table(db_path: str) -> None:
    """Create the alert rules table if it doesn't exist."""
    ensure_watchlist_alert_rules_table(db_path)


def list_alert_rules(db_path: str, user_id: str, job_id: int | None = None) -> list[AlertRule]:
    return list_watchlist_alert_rules(db_path, user_id, job_id=job_id)


def create_alert_rule(
    db_path: str,
    user_id: str,
    name: str,
    condition_type: str,
    condition_value: dict[str, Any] | None = None,
    job_id: int | None = None,
    severity: str = "warning",
) -> AlertRule:
    _validate_condition_type(condition_type)
    return create_watchlist_alert_rule(
        db_path,
        user_id=user_id,
        name=name,
        condition_type=condition_type,
        condition_value=condition_value,
        job_id=job_id,
        severity=severity,
    )


def get_alert_rule(db_path: str, rule_id: int, user_id: str) -> AlertRule | None:
    return get_watchlist_alert_rule(db_path, rule_id, user_id)


def update_alert_rule(db_path: str, rule_id: int, user_id: str, **fields: Any) -> AlertRule | None:
    updates = {key: value for key, value in fields.items() if value is not None}
    if not updates:
        return None

    if "condition_type" in updates:
        _validate_condition_type(str(updates["condition_type"]))
    return update_watchlist_alert_rule(db_path, rule_id, user_id, **updates)


def delete_alert_rule(db_path: str, rule_id: int, user_id: str) -> bool:
    return delete_watchlist_alert_rule(db_path, rule_id, user_id)


def evaluate_rules_for_run(
    db_path: str,
    user_id: str,
    job_id: int,
    run_id: int,
    stats: dict[str, Any],
    status: str,
) -> list[dict[str, Any]]:
    """Evaluate all matching alert rules against a completed run."""
    rules = list_alert_rules(db_path, user_id, job_id=job_id)
    triggered: list[dict[str, Any]] = []

    items_found = stats.get("items_found", 0)
    items_ingested = stats.get("items_ingested", 0)
    error_rate = 1.0 - (items_ingested / items_found) if items_found > 0 else 0.0

    for rule in rules:
        if not rule.enabled:
            continue

        try:
            cv = json.loads(rule.condition_value)
        except (json.JSONDecodeError, TypeError):
            cv = {}

        match = False
        detail = ""

        if rule.condition_type == AlertConditionType.NO_ITEMS.value:
            match = items_ingested == 0
            detail = f"Run produced 0 items (found {items_found})"

        elif rule.condition_type == AlertConditionType.ERROR_RATE_ABOVE.value:
            threshold = _coerce_threshold(
                cv.get("threshold", DEFAULT_ERROR_RATE_THRESHOLD),
                default=DEFAULT_ERROR_RATE_THRESHOLD,
                caster=float,
                rule=rule,
            )
            if threshold is None:
                continue
            match = error_rate > threshold
            detail = f"Error rate {error_rate:.0%} exceeds {threshold:.0%} threshold"

        elif rule.condition_type == AlertConditionType.ITEMS_BELOW.value:
            threshold = _coerce_threshold(
                cv.get("threshold", DEFAULT_ITEMS_BELOW_THRESHOLD),
                default=DEFAULT_ITEMS_BELOW_THRESHOLD,
                caster=int,
                rule=rule,
            )
            if threshold is None:
                continue
            match = items_ingested < threshold
            detail = f"Only {items_ingested} items ingested (threshold: {threshold})"

        elif rule.condition_type == AlertConditionType.ITEMS_ABOVE.value:
            threshold = _coerce_threshold(
                cv.get("threshold", DEFAULT_ITEMS_ABOVE_THRESHOLD),
                default=DEFAULT_ITEMS_ABOVE_THRESHOLD,
                caster=int,
                rule=rule,
            )
            if threshold is None:
                continue
            match = items_ingested > threshold
            detail = f"{items_ingested} items ingested exceeds {threshold} threshold"

        elif rule.condition_type == AlertConditionType.RUN_FAILED.value:
            match = status == "failed"
            detail = f"Run failed: {stats.get('error_msg', 'unknown error')}"

        if match:
            triggered.append({
                "rule": rule,
                "notification_kwargs": {
                    "kind": WATCHLIST_HEALTH_NOTIFICATION_TYPE,
                    "type": WATCHLIST_HEALTH_NOTIFICATION_TYPE,
                    "legacy_kind": WATCHLIST_LEGACY_ALERT_KIND,
                    "category": "health",
                    "title": f"Alert: {rule.name}",
                    "message": detail,
                    "severity": rule.severity,
                    "source_job_id": str(job_id),
                    "source_domain": "watchlists",
                    "source_job_type": "watchlist_run",
                    "link_type": "watchlist_run",
                    "link_id": str(run_id),
                    "dedupe_key": f"watchlist-alert:{rule.id}:{run_id}",
                },
            })

    return triggered


def _validate_condition_type(condition_type: str) -> None:
    if condition_type not in ALERT_CONDITION_TYPE_VALUES:
        raise ValueError(
            f"Invalid condition_type. Must be one of: {', '.join(sorted(ALERT_CONDITION_TYPE_VALUES))}"
        )


def _coerce_threshold(
    raw_value: Any,
    *,
    default: float | int,
    caster: type[float] | type[int],
    rule: AlertRule,
) -> float | int | None:
    value = default if raw_value is None else raw_value
    try:
        return caster(value)
    except (TypeError, ValueError):
        logger.warning(
            "Skipping alert rule {} because threshold {!r} is invalid for condition_type={}",
            rule.id,
            raw_value,
            rule.condition_type,
        )
        return None
