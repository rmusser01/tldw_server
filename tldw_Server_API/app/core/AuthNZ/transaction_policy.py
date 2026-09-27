"""Sanitized AuthNZ transaction retry and timeout policy."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

_POLICY_SETTING_NAMES = (
    "AUTHNZ_SQLITE_LOCK_MAX_RETRIES",
    "AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS",
    "AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS",
    "AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS",
    "AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS",
)


@dataclass(frozen=True, slots=True)
class AuthnzTransactionPolicy:
    """Resolved transaction controls shared by storage and adapter boundaries."""

    sqlite_lock_max_retries: int = 2
    sqlite_lock_retry_base_seconds: float = 0.05
    sqlite_lock_retry_max_seconds: float = 0.25
    busy_retry_after_seconds: int = 1
    db_pool_acquire_timeout_seconds: float = 5.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AuthnzTransactionPolicy:
        """Resolve policy values without logging raw configuration input."""
        defaults = cls()
        max_retries = cls._non_negative_int(
            values.get("AUTHNZ_SQLITE_LOCK_MAX_RETRIES"),
            defaults.sqlite_lock_max_retries,
        )
        retry_base = cls._non_negative_float(
            values.get("AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS"),
            defaults.sqlite_lock_retry_base_seconds,
        )
        retry_max = cls._non_negative_float(
            values.get("AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS"),
            defaults.sqlite_lock_retry_max_seconds,
        )
        retry_after = cls._non_negative_int(
            values.get("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS"),
            defaults.busy_retry_after_seconds,
        )
        acquire_timeout = cls._bounded_acquire_timeout(
            values.get("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS"),
            defaults.db_pool_acquire_timeout_seconds,
        )
        return cls(
            sqlite_lock_max_retries=max_retries,
            sqlite_lock_retry_base_seconds=retry_base,
            sqlite_lock_retry_max_seconds=max(retry_max, retry_base),
            busy_retry_after_seconds=retry_after,
            db_pool_acquire_timeout_seconds=acquire_timeout,
        )

    @classmethod
    def from_settings(
        cls,
        settings: object | None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> AuthnzTransactionPolicy:
        """Resolve environment overrides over a settings object's raw values."""
        environment = os.environ if environ is None else environ
        values: dict[str, Any] = {}
        for name in _POLICY_SETTING_NAMES:
            if name in environment:
                values[name] = environment[name]
            elif settings is not None:
                values[name] = getattr(settings, name, None)
        return cls.from_mapping(values)

    @staticmethod
    def _non_negative_int(raw: Any, default: int) -> int:
        if raw is None or isinstance(raw, bool):
            return default
        try:
            parsed = int(str(raw).strip())
        except (TypeError, ValueError):
            return default
        return max(parsed, 0)

    @staticmethod
    def _non_negative_float(raw: Any, default: float) -> float:
        if raw is None or isinstance(raw, bool):
            return default
        try:
            parsed = float(str(raw).strip())
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return max(parsed, 0.0)

    @staticmethod
    def _bounded_acquire_timeout(raw: Any, default: float) -> float:
        if raw is None or isinstance(raw, bool):
            return default
        try:
            parsed = float(str(raw).strip())
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed) or parsed < 0.0:
            return default
        return parsed


@lru_cache(maxsize=4)
def _fallback_values_for_generation(generation: int) -> tuple[tuple[str, Any], ...]:
    """Cache only non-process settings sources for one settings generation."""
    del generation
    from tldw_Server_API.app.core.AuthNZ.settings import (
        get_authnz_transaction_policy_fallback_values,
    )

    return tuple(get_authnz_transaction_policy_fallback_values().items())


def get_authnz_transaction_policy() -> AuthnzTransactionPolicy:
    """Resolve live process overrides over non-stale settings sources."""
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings_generation

    values = dict(_fallback_values_for_generation(get_settings_generation()))
    for name in _POLICY_SETTING_NAMES:
        if name in os.environ:
            values[name] = os.environ[name]
    return AuthnzTransactionPolicy.from_mapping(values)
