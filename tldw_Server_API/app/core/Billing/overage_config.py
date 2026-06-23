"""
Overage handling configuration.

Configures behavior when usage exceeds plan limits:
- hard_block: Reject requests immediately
- degraded: Allow with reduced quality/rate
- notify_only: Allow but alert admin
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

_ALLOWED_OVERAGE_MODES = {"hard_block", "degraded", "notify_only"}
_DEFAULT_OVERAGE_MODE = "notify_only"
_DEFAULT_GRACE_PERCENTAGE = 10.0
_DEFAULT_NOTIFICATION_THRESHOLD = 80.0


def _safe_float_from_env(env_name: str, default: float) -> float:
    """Parse a non-negative percentage from the environment."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid {} configured; using default", env_name)
        return default
    if value < 0:
        logger.warning("Negative {} configured; using default", env_name)
        return default
    return value


@dataclass
class OveragePolicy:
    """Policy that governs how the system reacts when usage exceeds plan limits."""

    mode: str  # hard_block, degraded, notify_only
    grace_percentage: float  # Allow X% over limit before enforcement
    notification_threshold: float  # Alert at X% of limit

    @classmethod
    def from_env(cls) -> OveragePolicy:
        """Create an :class:`OveragePolicy` from environment variables.

        Recognised variables:
        - ``BILLING_OVERAGE_MODE`` (default ``notify_only``)
        - ``BILLING_OVERAGE_GRACE_PCT`` (default ``10``)
        - ``BILLING_OVERAGE_NOTIFY_PCT`` (default ``80``)
        """
        mode = os.getenv("BILLING_OVERAGE_MODE", _DEFAULT_OVERAGE_MODE).strip().lower()
        if mode not in _ALLOWED_OVERAGE_MODES:
            logger.warning("Invalid BILLING_OVERAGE_MODE configured; using default")
            mode = _DEFAULT_OVERAGE_MODE

        return cls(
            mode=mode,
            grace_percentage=_safe_float_from_env(
                "BILLING_OVERAGE_GRACE_PCT",
                _DEFAULT_GRACE_PERCENTAGE,
            ),
            notification_threshold=_safe_float_from_env(
                "BILLING_OVERAGE_NOTIFY_PCT",
                _DEFAULT_NOTIFICATION_THRESHOLD,
            ),
        )

    def should_block(self, usage_pct: float) -> bool:
        """Return ``True`` if usage should be hard-blocked."""
        if self.mode == "hard_block":
            return usage_pct > (100 + self.grace_percentage)
        return False

    def should_degrade(self, usage_pct: float) -> bool:
        """Return ``True`` if usage should trigger degraded service."""
        if self.mode == "degraded":
            return usage_pct > (100 + self.grace_percentage)
        return False

    def should_notify(self, usage_pct: float) -> bool:
        """Return ``True`` if a notification should be sent."""
        return usage_pct >= self.notification_threshold

    def evaluate(self, usage_pct: float) -> dict[str, Any]:
        """Evaluate a usage percentage against the policy.

        Returns a dict summarising the mode, actual percentage, and which
        actions (block / degrade / notify) should be taken.
        """
        return {
            "mode": self.mode,
            "usage_pct": usage_pct,
            "blocked": self.should_block(usage_pct),
            "degraded": self.should_degrade(usage_pct),
            "notify": self.should_notify(usage_pct),
        }
