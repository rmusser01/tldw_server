from __future__ import annotations

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_RUN_STATUSES = frozenset({"pending", "running"})


def normalize_run_status(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw == "canceled":
        return "cancelled"
    return raw or "pending"


def can_transition_run_status(current: str | None, target: str | None) -> bool:
    current_norm = normalize_run_status(current)
    target_norm = normalize_run_status(target)
    if current_norm in TERMINAL_RUN_STATUSES and target_norm != current_norm:
        return False
    return True
