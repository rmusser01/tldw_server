from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class JobsConfig:
    prune_enforce: bool
    prune_interval_sec: int
    prune_dry_run: bool
    prune_domain: list[str]
    prune_queue: list[str]
    prune_job_type: list[str]
    retention_days_terminal: int | None
    retention_days_nonterminal: int
    retention_days_completed: int
    retention_days_failed: int
    retention_days_cancelled: int
    retention_days_quarantined: int


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    *,
    env_key: str,
    option: str,
    default: str,
) -> str:
    env_value = env_map.get(env_key)
    if env_value is not None and str(env_value).strip() != "":
        return str(env_value)

    raw = config_parser.get("Jobs", option, fallback=default)
    text = str(raw).strip()
    return text or default


def _parse_bool(raw: object, default: bool) -> bool:
    text = str(raw).strip().lower()
    if not text:
        return default
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _parse_int(raw: object, default: int) -> int:
    text = str(raw).strip()
    if not text:
        return default
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _parse_int_optional(raw: object) -> int | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _parse_csv(raw: object) -> list[str]:
    text = str(raw).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def load_jobs_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> JobsConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    prune_enforce = _parse_bool(
        _get_raw(config_parser, env_map, env_key="JOBS_PRUNE_ENFORCE", option="prune_enforce", default="false"),
        False,
    )
    prune_interval_sec = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_PRUNE_INTERVAL_SEC",
            option="prune_interval_sec",
            default="86400",
        ),
        86400,
    )
    prune_dry_run = _parse_bool(
        _get_raw(config_parser, env_map, env_key="JOBS_PRUNE_DRY_RUN", option="prune_dry_run", default="false"),
        False,
    )
    prune_domain = _parse_csv(
        _get_raw(config_parser, env_map, env_key="JOBS_PRUNE_DOMAIN", option="prune_domain", default="")
    )
    prune_queue = _parse_csv(
        _get_raw(config_parser, env_map, env_key="JOBS_PRUNE_QUEUE", option="prune_queue", default="")
    )
    prune_job_type = _parse_csv(
        _get_raw(config_parser, env_map, env_key="JOBS_PRUNE_JOB_TYPE", option="prune_job_type", default="")
    )
    retention_days_terminal = _parse_int_optional(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_RETENTION_DAYS_TERMINAL",
            option="retention_days_terminal",
            default="",
        )
    )
    retention_days_nonterminal = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_RETENTION_DAYS_NONTERMINAL",
            option="retention_days_nonterminal",
            default="0",
        ),
        0,
    )
    retention_days_completed = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_RETENTION_DAYS_COMPLETED",
            option="retention_days_completed",
            default="30",
        ),
        30,
    )
    retention_days_failed = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_RETENTION_DAYS_FAILED",
            option="retention_days_failed",
            default="60",
        ),
        60,
    )
    retention_days_cancelled = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_RETENTION_DAYS_CANCELLED",
            option="retention_days_cancelled",
            default="60",
        ),
        60,
    )
    retention_days_quarantined = _parse_int(
        _get_raw(
            config_parser,
            env_map,
            env_key="JOBS_RETENTION_DAYS_QUARANTINED",
            option="retention_days_quarantined",
            default="90",
        ),
        90,
    )

    return JobsConfig(
        prune_enforce=prune_enforce,
        prune_interval_sec=prune_interval_sec,
        prune_dry_run=prune_dry_run,
        prune_domain=prune_domain,
        prune_queue=prune_queue,
        prune_job_type=prune_job_type,
        retention_days_terminal=retention_days_terminal,
        retention_days_nonterminal=retention_days_nonterminal,
        retention_days_completed=retention_days_completed,
        retention_days_failed=retention_days_failed,
        retention_days_cancelled=retention_days_cancelled,
        retention_days_quarantined=retention_days_quarantined,
    )
