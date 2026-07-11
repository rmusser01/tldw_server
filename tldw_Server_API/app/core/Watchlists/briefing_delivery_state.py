"""Pure aggregate-state helpers for Watchlists briefing delivery."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def aggregate_delivery_status(
    adapters: Iterable[str],
    stages: Mapping[str, Mapping[str, Any]],
) -> str:
    """Derive top-level delivery status from configured adapter outcomes."""
    outcomes: list[str | None] = []
    for adapter in adapters:
        stage = stages.get(f"deliver:{adapter}")
        outcome = stage.get("outcome") if isinstance(stage, Mapping) else None
        outcomes.append(str(outcome) if outcome else None)
    if not outcomes:
        return "not_configured"
    if any(outcome == "sending" for outcome in outcomes):
        return "delivering"
    if any(outcome == "unknown" for outcome in outcomes):
        return "unknown"
    if all(outcome == "successful" for outcome in outcomes):
        return "delivered"
    if any(outcome in {"successful", "partial"} for outcome in outcomes):
        return "partially_delivered"
    if any(outcome == "failed" for outcome in outcomes):
        return "failed"
    return "waiting_for_artifacts"


def aggregate_delivery_stage(delivery_status: str, *, finished_at: str) -> dict[str, Any]:
    """Build the persisted aggregate stage for a top-level delivery status."""
    return {
        "status": {
            "not_configured": "skipped",
            "waiting_for_artifacts": "not_started",
            "delivering": "running",
            "delivered": "ready",
            "partially_delivered": "failed",
            "failed": "failed",
            "unknown": "failed",
        }[delivery_status],
        "code": None if delivery_status == "delivered" else delivery_status,
        "retryable": delivery_status in {"failed", "partially_delivered"},
        "finished_at": (
            finished_at
            if delivery_status in {"not_configured", "delivered", "partially_delivered", "failed", "unknown"}
            else None
        ),
    }
