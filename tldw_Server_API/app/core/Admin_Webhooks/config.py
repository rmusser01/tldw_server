"""Validated process configuration for the canonical admin webhook surface."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class AdminWebhookMode(str, Enum):
    """Canonical control-plane operating mode."""

    OFF = "off"
    MIGRATE = "migrate"
    ON = "on"


class WebhookRouteSelection(str, Enum):
    """Webhook route family selected for this process."""

    CANONICAL = "canonical"
    LEGACY = "legacy"


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
    if any(
        environ.get(name, "").strip().lower() in _TRUTHY_VALUES
        for name in ("tldw_production", "TLDW_PRODUCTION")
    ):
        return True
    return any(
        environ.get(name, "").strip().lower() in _PRODUCTION_NAMES
        for name in ("ENV", "APP_ENV", "TLDW_ENV", "ENVIRONMENT")
    )


@dataclass(frozen=True)
class AdminWebhookSettings:
    """Validated immutable settings used by the canonical webhook package."""

    mode: AdminWebhookMode
    route_selection: WebhookRouteSelection
    registration_limit: int
    active_limit: int
    allow_http_dev: bool
    idempotency_ttl_seconds: int
    rollback_window_days: int

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
            raise ValueError(
                "TLDW_ADMIN_WEBHOOKS_MODE must be off, migrate, or on"
            ) from exc

        legacy = _parse_strict_bool(
            environ.get("TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT", "false"),
            name="TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT",
        )
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
            raise ValueError(
                "TLDW_ADMIN_WEBHOOK_ACTIVE_LIMIT cannot exceed registration limit"
            )

        allow_http_dev = _parse_strict_bool(
            environ.get("TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV", "false"),
            name="TLDW_ADMIN_WEBHOOKS_ALLOW_HTTP_DEV",
        )
        if allow_http_dev and is_production_environment_mapping(environ):
            raise ValueError("Webhook HTTP development override is forbidden in production")
        if legacy and mode is not AdminWebhookMode.OFF:
            raise ValueError("Legacy webhook compatibility requires canonical mode off")

        rollback_window_days = _parse_bounded_positive_int(
            environ,
            "TLDW_ADMIN_WEBHOOK_ROLLBACK_WINDOW_DAYS",
            7,
            30,
        )
        return cls(
            mode=mode,
            route_selection=(
                WebhookRouteSelection.LEGACY
                if legacy
                else WebhookRouteSelection.CANONICAL
            ),
            registration_limit=registration_limit,
            active_limit=active_limit,
            allow_http_dev=allow_http_dev,
            idempotency_ttl_seconds=86_400,
            rollback_window_days=rollback_window_days,
        )
