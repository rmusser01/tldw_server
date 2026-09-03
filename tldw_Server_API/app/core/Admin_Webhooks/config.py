"""Validated process configuration for the canonical admin webhook surface."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

WEBHOOK_KEYS_ENV = "TLDW_ADMIN_WEBHOOK_KEYS_JSON"
WEBHOOK_PRIMARY_KEY_ID_ENV = "TLDW_ADMIN_WEBHOOK_PRIMARY_KEY_ID"


class AdminWebhookMode(str, Enum):
    """Canonical control-plane operating mode."""

    OFF = "off"
    MIGRATE = "migrate"
    ON = "on"


_PRODUCTION_NAMES = frozenset({"prod", "production"})
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_DECIMAL_PATTERN = re.compile(r"^[0-9]+$")


def _parse_strict_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{name} must be true or false")


def _parse_bounded_positive_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    maximum: int,
) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip()
    if not _DECIMAL_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be a positive integer")
    value = int(normalized)
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def is_production_environment_mapping(environ: Mapping[str, str]) -> bool:
    """Return whether a pure environment mapping denotes production."""
    if any(environ.get(name, "").strip().lower() in _TRUTHY_VALUES for name in ("tldw_production", "TLDW_PRODUCTION")):
        return True
    return any(
        environ.get(name, "").strip().lower() in _PRODUCTION_NAMES
        for name in ("ENV", "APP_ENV", "TLDW_ENV", "ENVIRONMENT")
    )


@dataclass(frozen=True)
class AdminWebhookSettings:
    """Validated immutable settings used by the canonical webhook package."""

    mode: AdminWebhookMode
    registration_limit: int
    active_limit: int
    allow_http_dev: bool
    idempotency_ttl_seconds: int
    rollback_window_days: int
    allow_e2e_loopback: bool = False
    delivery_claim_ttl_seconds: int = 60
    delivery_loop_interval_seconds: int = 1
    delivery_heartbeat_interval_seconds: int = 10
    delivery_heartbeat_freshness_seconds: int = 30
    activation_max_backlog_age_seconds: int = 300

    @property
    def delivery_retry_delays_seconds(self) -> tuple[int, int, int]:
        """Return the fixed protocol retry schedule."""
        return (60, 300, 1800)

    @property
    def delivery_max_attempts(self) -> int:
        """Return the hard protocol ceiling for network attempts."""
        return 4

    @property
    def jobs_quarantine_threshold(self) -> int:
        """Return the fixed Jobs quarantine threshold for webhook work."""
        return 5

    @property
    def delivery_infrastructure_defer_seconds(self) -> int:
        """Return the fixed no-attempt infrastructure defer interval."""
        return 30

    @property
    def delivery_expiry_seconds(self) -> int:
        """Return the fixed automatic-delivery expiry interval."""
        return 72 * 60 * 60

    @property
    def delivery_retention_days(self) -> int:
        """Return the fixed terminal-delivery retention interval."""
        return 30

    @property
    def delivery_commit_margin_seconds(self) -> int:
        """Return the fixed pre-I/O commit safety margin."""
        return 30

    @property
    def delivery_stale_attempt_margin_seconds(self) -> int:
        """Return the fixed stale-attempt recovery margin."""
        return 90

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str],
    ) -> AdminWebhookSettings:
        """Parse webhook settings without consulting ambient process state."""
        raw_mode = environ.get("TLDW_ADMIN_WEBHOOKS_MODE", "off").strip().lower()
        try:
            mode = AdminWebhookMode(raw_mode)
        except ValueError as exc:
            raise ValueError("TLDW_ADMIN_WEBHOOKS_MODE must be off, migrate, or on") from exc

        legacy = _parse_strict_bool(
            environ.get("TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT", "false"),
            name="TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT",
        )
        if legacy:
            raise ValueError("Legacy webhook compatibility is no longer supported")
        registration_limit = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_REGISTRATION_LIMIT",
            100,
            1_000,
        )
        active_limit = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT",
            25,
            1_000,
        )
        if active_limit > registration_limit:
            raise ValueError("TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT cannot exceed registration limit")

        allow_http_dev = _parse_strict_bool(
            environ.get("TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV", "false"),
            name="TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV",
        )
        if allow_http_dev and is_production_environment_mapping(environ):
            raise ValueError("Webhook HTTP development override is forbidden in production")
        allow_e2e_loopback = _parse_strict_bool(
            environ.get("TLDW_ADMIN_WEBHOOKS_E2E_LOOPBACK", "false"),
            name="TLDW_ADMIN_WEBHOOKS_E2E_LOOPBACK",
        )
        if allow_e2e_loopback and (
            not allow_http_dev
            or environ.get("ENABLE_ADMIN_E2E_TEST_MODE", "").strip().lower() != "true"
            or environ.get("TEST_MODE", "").strip().lower() != "true"
            or environ.get("PYTEST_CURRENT_TEST", "").strip()
            != "admin-ui-real-backend-e2e"
            or is_production_environment_mapping(environ)
        ):
            raise ValueError(
                "TLDW_ADMIN_WEBHOOKS_E2E_LOOPBACK requires the isolated admin real-backend test gates"
            )
        rollback_window_days = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS",
            7,
            30,
        )
        delivery_claim_ttl_seconds = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_DELIVERY_CLAIM_TTL_SECONDS",
            60,
            300,
        )
        if delivery_claim_ttl_seconds < 5:
            raise ValueError("TLDW_ADMIN_WEBHOOK_DELIVERY_CLAIM_TTL_SECONDS must be between 5 and 300")
        delivery_loop_interval_seconds = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_DELIVERY_LOOP_INTERVAL_SECONDS",
            1,
            60,
        )
        delivery_heartbeat_interval_seconds = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_INTERVAL_SECONDS",
            10,
            60,
        )
        delivery_heartbeat_freshness_seconds = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_FRESHNESS_SECONDS",
            30,
            60,
        )
        if delivery_heartbeat_freshness_seconds <= delivery_heartbeat_interval_seconds:
            raise ValueError("TLDW_ADMIN_WEBHOOK_DELIVERY_HEARTBEAT_FRESHNESS_SECONDS must exceed heartbeat interval")
        activation_max_backlog_age_seconds = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_ACTIVATION_MAX_BACKLOG_AGE_SECONDS",
            300,
            86_400,
        )
        return cls(
            mode=mode,
            registration_limit=registration_limit,
            active_limit=active_limit,
            allow_http_dev=allow_http_dev,
            idempotency_ttl_seconds=86_400,
            rollback_window_days=rollback_window_days,
            allow_e2e_loopback=allow_e2e_loopback,
            delivery_claim_ttl_seconds=delivery_claim_ttl_seconds,
            delivery_loop_interval_seconds=delivery_loop_interval_seconds,
            delivery_heartbeat_interval_seconds=delivery_heartbeat_interval_seconds,
            delivery_heartbeat_freshness_seconds=delivery_heartbeat_freshness_seconds,
            activation_max_backlog_age_seconds=(activation_max_backlog_age_seconds),
        )
