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
    allowed_queues: tuple[str, ...] = ()
    allowed_queues_by_domain: tuple[tuple[str, tuple[str, ...]], ...] = ()

    CONSTRUCTION_TIME_KEYS = frozenset({"JOBS_DB_URL", "JOBS_DB_PATH"})
    SNAPSHOT_REFRESHABLE_KEYS = frozenset(
        {
            "JOBS_MAX_JSON_BYTES",
            "JOBS_LEASE_MAX_SECONDS",
            "JOBS_EVENTS_OUTBOX",
            "JOBS_COUNTERS_ENABLED",
        }
    )
    OPERATION_TIME_KEYS = frozenset({"JOBS_ALLOWED_QUEUES"})
    OPERATION_TIME_PREFIXES = ("JOBS_ALLOWED_QUEUES_",)

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
            allowed_queues=_split_csv(_env_value(source, "JOBS_ALLOWED_QUEUES", "")),
            allowed_queues_by_domain=tuple(domain_queues),
        )

    def refresh(self, env: Mapping[str, str] | None = None) -> "JobsSettings":
        refreshed = type(self).from_env(env)
        return replace(refreshed, db_url=self.db_url, db_path=self.db_path)

    def allowed_queues_for_domain(self, domain: str | None) -> list[str]:
        values = list(self.allowed_queues)
        normalized_domain = str(domain or "").strip().lower()
        if normalized_domain:
            for configured_domain, queues in self.allowed_queues_by_domain:
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
        if normalized in cls.OPERATION_TIME_KEYS:
            return JobsSettingMode.OPERATION_TIME
        if any(normalized.startswith(prefix) for prefix in cls.OPERATION_TIME_PREFIXES):
            return JobsSettingMode.OPERATION_TIME
        return JobsSettingMode.OPERATION_TIME


__all__ = ["JobsSettingMode", "JobsSettings"]
