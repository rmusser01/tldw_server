from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from tldw_Server_API.app.core.testing import is_truthy


class JobsSettingMode(str, Enum):
    CONSTRUCTION_TIME = "construction_time"
    SNAPSHOT_REFRESHABLE = "snapshot_refreshable"
    OPERATION_TIME = "operation_time"
    UNCLASSIFIED = "unclassified"


def _env_value(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    value = env.get(key)
    if value is None:
        return default
    return str(value)


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _env_value(env, key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = _env_value(env, key)
    if raw is None:
        return default
    return is_truthy(str(raw))


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class JobsSettings:
    db_url: str | None = None
    db_path: str | None = None
    max_json_bytes: int = 1_048_576
    lease_max_seconds: int = 3_600
    events_outbox_enabled: bool = False
    counters_enabled: bool = False
    allowed_queue_extras: tuple[str, ...] = ()
    allowed_queue_extras_by_domain: tuple[tuple[str, tuple[str, ...]], ...] = ()

    CONSTRUCTION_TIME_KEYS = frozenset(
        {"JOBS_BACKEND", "JOBS_DB_URL", "JOBS_DB_PATH", "JOBS_PG_SKIP_SCHEMA_INIT", "JOBS_TEST_NOW_EPOCH"}
    )
    SNAPSHOT_REFRESHABLE_KEYS = frozenset(
        {
            "JOBS_ALLOWED_QUEUES",
            "JOBS_MAX_JSON_BYTES",
            "JOBS_LEASE_MAX_SECONDS",
            "JOBS_EVENTS_OUTBOX",
            "JOBS_COUNTERS_ENABLED",
            "JOBS_LEASE_SECONDS",
            "JOBS_LEASE_RENEW_SECONDS",
            "JOBS_LEASE_RENEW_JITTER_SECONDS",
            "JOBS_RENEW_JITTER_SECONDS",
            "JOBS_RENEW_THRESHOLD_SECONDS",
            "JOBS_LEASE_RENEW_THRESHOLD_SECONDS",
        }
    )
    SNAPSHOT_REFRESHABLE_PREFIXES = ("JOBS_ALLOWED_QUEUES_",)
    OPERATION_TIME_KEYS = frozenset(
        {
            "JOBS_EVENTS_ENABLED",
            "JOBS_EXPOSE_PROGRESS",
            "JOBS_REQUIRE_CONFIRM",
            "JOBS_REQUIRE_COMPLETION_TOKEN",
            "JOBS_ENFORCE_LEASE_ACK",
            "JOBS_DISABLE_LEASE_ENFORCEMENT",
            "JOBS_ENCRYPT",
            "JOBS_OWNER_STRICT",
            "JOBS_EVENTS_POLL_INTERVAL",
            "JOBS_METRICS_GAUGES_ENABLED",
            "JOBS_METRICS_RECONCILE_ENABLE",
            "JOBS_UPDATE_GAUGES_ON_PRUNE",
            "JOBS_WEBHOOKS_ENABLED",
            "JOBS_ADAPTIVE_LEASE_ENABLE",
            "JOBS_ADAPTIVE_LEASE_MIN_SECONDS",
            "JOBS_ADAPTIVE_LEASE_HEADROOM",
            "JOBS_ADAPTIVE_LEASE_WINDOW_HOURS",
            "JOBS_ACQUIRE_TIE_BREAK",
            "JOBS_ACQUIRE_PRIORITY_DESC_DOMAINS",
            "JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE",
            "JOBS_PG_SINGLE_UPDATE_ACQUIRE",
            "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
            "JOBS_ALLOWED_JOB_TYPES",
            "JOBS_ARCHIVE_BEFORE_DELETE",
            "JOBS_ARCHIVE_COMPRESS",
            "JOBS_ARCHIVE_COMPRESS_DROP_JSON",
            "JOBS_DOMAIN_RBAC_PRINCIPAL",
            "JOBS_DOMAIN_SCOPED_RBAC",
            "JOBS_GAUGES_DEBOUNCE_MS",
            "JOBS_JSON_TRUNCATE",
            "JOBS_MAX_PER_ORG",
            "JOBS_MAX_PER_USER",
            "JOBS_PG_RLS_DEBUG",
            "JOBS_PG_RLS_ROLE",
            "JOBS_QUARANTINE_THRESHOLD",
            "JOBS_QUOTA_MAX_INFLIGHT",
            "JOBS_QUOTA_MAX_QUEUED",
            "JOBS_QUOTA_SUBMITS_PER_MIN",
            "JOBS_RBAC_FORCE",
            "JOBS_REQUIRE_DOMAIN_FILTER",
            "JOBS_SECRET_DENY_KEYS",
            "JOBS_SECRET_PATTERNS",
            "JOBS_SECRET_REDACT",
            "JOBS_SECRET_REJECT",
            "JOBS_SSE_TEST_MAX_SECONDS",
        }
    )
    OPERATION_TIME_PREFIXES = (
        "JOBS_ACQUIRE_TIE_BREAK_",
        "JOBS_ALLOWED_JOB_TYPES_",
        "JOBS_DOMAIN_ALLOWLIST_",
        "JOBS_ENCRYPT_",
        "JOBS_PG_ACQUIRE_",
        "JOBS_POSTGRES_ACQUIRE_",
        "JOBS_POSTGRESQL_ACQUIRE_",
        "JOBS_QUOTA_MAX_INFLIGHT_",
        "JOBS_QUOTA_MAX_QUEUED_",
        "JOBS_QUOTA_SUBMITS_PER_MIN_",
        "JOBS_RETENTION_DAYS_",
        "JOBS_SQLITE_ACQUIRE_",
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "JobsSettings":
        source = os.environ if env is None else env
        domain_queues = []
        for key, value in source.items():
            if key.startswith("JOBS_ALLOWED_QUEUES_"):
                domain = key.removeprefix("JOBS_ALLOWED_QUEUES_").strip().lower()
                if domain:
                    domain_queues.append((domain, _split_csv(str(value))))
        domain_queues.sort(key=lambda item: item[0])

        return cls(
            db_url=_env_value(source, "JOBS_DB_URL"),
            db_path=_env_value(source, "JOBS_DB_PATH"),
            max_json_bytes=_env_int(source, "JOBS_MAX_JSON_BYTES", 1_048_576),
            lease_max_seconds=_env_int(source, "JOBS_LEASE_MAX_SECONDS", 3_600),
            events_outbox_enabled=_env_bool(source, "JOBS_EVENTS_OUTBOX", False),
            counters_enabled=_env_bool(source, "JOBS_COUNTERS_ENABLED", False),
            allowed_queue_extras=_split_csv(_env_value(source, "JOBS_ALLOWED_QUEUES", "")),
            allowed_queue_extras_by_domain=tuple(domain_queues),
        )

    def refresh(self, env: Mapping[str, str] | None = None) -> "JobsSettings":
        refreshed = type(self).from_env(env)
        return replace(refreshed, db_url=self.db_url, db_path=self.db_path)

    def allowed_queue_extras_for_domain(self, domain: str | None) -> list[str]:
        values = list(self.allowed_queue_extras)
        normalized_domain = str(domain or "").strip().lower()
        if normalized_domain:
            for configured_domain, queues in self.allowed_queue_extras_by_domain:
                if configured_domain == normalized_domain:
                    values.extend(queues)
                    break

        seen: set[str] = set()
        result: list[str] = []
        for queue in values:
            if queue in seen:
                continue
            seen.add(queue)
            result.append(queue)
        return result

    @classmethod
    def setting_mode(cls, key: str) -> JobsSettingMode:
        normalized = str(key or "").strip().upper()
        if normalized in cls.CONSTRUCTION_TIME_KEYS:
            return JobsSettingMode.CONSTRUCTION_TIME
        if normalized in cls.SNAPSHOT_REFRESHABLE_KEYS:
            return JobsSettingMode.SNAPSHOT_REFRESHABLE
        if any(normalized.startswith(prefix) for prefix in cls.SNAPSHOT_REFRESHABLE_PREFIXES):
            return JobsSettingMode.SNAPSHOT_REFRESHABLE
        if normalized in cls.OPERATION_TIME_KEYS:
            return JobsSettingMode.OPERATION_TIME
        if any(normalized.startswith(prefix) for prefix in cls.OPERATION_TIME_PREFIXES):
            return JobsSettingMode.OPERATION_TIME
        return JobsSettingMode.UNCLASSIFIED


__all__ = ["JobsSettingMode", "JobsSettings"]
