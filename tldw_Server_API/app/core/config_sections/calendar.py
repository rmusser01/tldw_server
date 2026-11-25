"""Live, env-only Calendar settings; secrets stay outside aggregate config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.testing import is_truthy

from .precedence import resolve


@dataclass(frozen=True)
class CalendarSyncWorkerConfig:
    """Settings passed to the Calendar Jobs consumer without extra validation."""

    worker_id: str
    lease_seconds: int
    renew_threshold_seconds: int
    renew_jitter_seconds: int


@dataclass(frozen=True)
class CalendarSyncStartupConfig:
    """Opt-in switches, independent of worker parsing and route admission."""

    jobs_worker_enabled: bool
    scheduler_enabled: bool


def load_calendar_sync_worker_config(
    *, default_worker_id: str, env: Mapping[str, str] | None = None,
) -> CalendarSyncWorkerConfig:
    """Keep blank values, integer failures and the unset-only Jobs lease fallback."""
    worker_id = resolve(("CALENDAR_SYNC_WORKER_ID",), env=env, blank_env=True).value
    jobs_lease = resolve(("JOBS_LEASE_SECONDS",), env=env, default="120", blank_env=True).value
    lease = resolve(("CALENDAR_SYNC_LEASE_SECONDS",), env=env, default=jobs_lease, blank_env=True).value
    return CalendarSyncWorkerConfig(
        worker_id=(worker_id or default_worker_id).strip(),
        lease_seconds=int(lease or "120"),
        renew_threshold_seconds=int(resolve(
            ("CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS",), env=env, default="10", blank_env=True,
        ).value or "10"),
        renew_jitter_seconds=int(resolve(
            ("CALENDAR_SYNC_RENEW_JITTER_SECONDS",), env=env, default="0", blank_env=True,
        ).value or "0"),
    )


def _flag_enabled(key: str, env: Mapping[str, str] | None) -> bool:
    """Preserve env_flag_enabled's false default, vocabulary and fail-closed reads."""
    try:
        return is_truthy(resolve((key,), env=env, blank_env=True).value)
    except Exception:  # noqa: BLE001 - match the existing startup flag policy.
        return False


def load_calendar_sync_startup_config(env: Mapping[str, str] | None = None) -> CalendarSyncStartupConfig:
    """Read only the startup flags; callers still apply the lifecycle route gate."""
    return CalendarSyncStartupConfig(
        jobs_worker_enabled=_flag_enabled("CALENDAR_SYNC_JOBS_WORKER_ENABLED", env),
        scheduler_enabled=_flag_enabled("CALENDAR_SYNC_SCHEDULER_ENABLED", env),
    )


def calendar_sync_scheduler_interval_seconds(env: Mapping[str, str] | None = None) -> float:
    """Parse the env interval with the existing five-second minimum; caller handles errors."""
    raw = resolve(("CALENDAR_SYNC_SCHEDULER_INTERVAL_SECONDS",), env=env, default="60", blank_env=True).value
    return max(5.0, float(raw or "60"))


def calendar_secret_encryption_key(env: Mapping[str, str] | None = None) -> str:
    """Read the stripped env-only key on demand, never storing it in a config object."""
    return (resolve(("CALENDAR_SECRET_ENCRYPTION_KEY",), env=env, default="", blank_env=True).value or "").strip()
