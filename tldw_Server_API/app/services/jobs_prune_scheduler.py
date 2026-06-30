"""
Jobs prune scheduler.

Runs periodic maintenance to prune jobs by status and retention window.

Enable via env:
  - JOBS_PRUNE_ENFORCE=true
  - JOBS_PRUNE_INTERVAL_SEC=86400 (default daily)
  - JOBS_PRUNE_DRY_RUN=false (log counts only)
  - JOBS_PRUNE_DOMAIN=chatbooks,embeddings (optional scope)
  - JOBS_PRUNE_QUEUE=default,high (optional scope)
  - JOBS_PRUNE_JOB_TYPE=export,import (optional scope)

Retention defaults (when enabled):
  - completed: 30 days
  - failed: 60 days
  - cancelled: 60 days
  - quarantined: 90 days

Overrides:
  - JOBS_RETENTION_DAYS_TERMINAL
  - JOBS_RETENTION_DAYS_NONTERMINAL (queued/processing)
  - JOBS_RETENTION_DAYS_<STATUS> (e.g., JOBS_RETENTION_DAYS_COMPLETED)
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.config import get_config_value
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import is_truthy

_DEFAULT_RETENTION_DAYS: dict[str, int] = {
    "completed": 30,
    "failed": 60,
    "cancelled": 60,
    "quarantined": 90,
}
_NONTERMINAL_STATUSES = ("queued", "processing")


def _normalize_value(val: str | None) -> str | None:
    if val is None:
        return None
    val = str(val).strip()
    return val or None


def _raw_setting(env_name: str, config_key: str, default: str | None = None) -> str | None:
    env_val = _normalize_value(os.getenv(env_name))
    if env_val is not None:
        return env_val
    cfg_val = _normalize_value(get_config_value("Jobs", config_key))
    if cfg_val is not None:
        return cfg_val
    return _normalize_value(default)


def _is_truthy(val: str | None) -> bool:
    if val is None:
        return False
    return is_truthy(str(val).strip().lower())


def _int_optional(env_name: str, config_key: str) -> int | None:
    raw = _raw_setting(env_name, config_key)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            f"jobs_prune: invalid {env_name}; using default"
        )
        return None


def _int_setting(env_name: str, config_key: str, default: int) -> int:
    val = _int_optional(env_name, config_key)
    return default if val is None else val


def _split_csv(env_name: str, config_key: str) -> list[str]:
    raw = _raw_setting(env_name, config_key, "") or ""
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _retention_for_status(
    status: str,
    terminal_override: int | None,
    nonterminal_override: int | None,
) -> int:
    per_status = _int_optional(
        f"JOBS_RETENTION_DAYS_{status.upper()}",
        f"retention_days_{status.lower()}",
    )
    if per_status is not None:
        return per_status
    if status in _DEFAULT_RETENTION_DAYS:
        if terminal_override is not None:
            return terminal_override
        return _DEFAULT_RETENTION_DAYS[status]
    if status in _NONTERMINAL_STATUSES:
        return nonterminal_override or 0
    return 0


def _build_retention_groups() -> dict[int, list[str]]:
    terminal_override = _int_optional("JOBS_RETENTION_DAYS_TERMINAL", "retention_days_terminal")
    nonterminal_override = _int_optional("JOBS_RETENTION_DAYS_NONTERMINAL", "retention_days_nonterminal")
    statuses = list(_DEFAULT_RETENTION_DAYS.keys()) + list(_NONTERMINAL_STATUSES)
    grouped: dict[int, list[str]] = {}
    for status in statuses:
        days = _retention_for_status(status, terminal_override, nonterminal_override)
        if days <= 0:
            continue
        grouped.setdefault(days, []).append(status)
    return grouped


def _iter_scopes(values: list[str]) -> list[str | None]:
    if values:
        return values
    return [None]


async def start_jobs_prune_scheduler() -> asyncio.Task | None:
    if not _is_truthy(_raw_setting("JOBS_PRUNE_ENFORCE", "prune_enforce")):
        return None

    interval = _int_setting("JOBS_PRUNE_INTERVAL_SEC", "prune_interval_sec", 86400)
    interval = max(60, interval)
    dry_run = _is_truthy(_raw_setting("JOBS_PRUNE_DRY_RUN", "prune_dry_run"))
    domains = _iter_scopes(_split_csv("JOBS_PRUNE_DOMAIN", "prune_domain"))
    queues = _iter_scopes(_split_csv("JOBS_PRUNE_QUEUE", "prune_queue"))
    job_types = _iter_scopes(_split_csv("JOBS_PRUNE_JOB_TYPE", "prune_job_type"))

    async def _run_once() -> None:
        groups = _build_retention_groups()
        if not groups:
            logger.info("Jobs prune scheduler enabled but no retention windows configured")
            return
        jm = JobManager()
        total = 0
        JobManager.set_rls_context(is_admin=True, domain_allowlist=None, owner_user_id="system")
        try:
            for days, statuses in sorted(groups.items()):
                for domain in domains:
                    for queue in queues:
                        for job_type in job_types:
                            total += jm.prune_jobs(
                                statuses=statuses,
                                older_than_days=days,
                                domain=domain,
                                queue=queue,
                                job_type=job_type,
                                dry_run=dry_run,
                            )
        finally:
            JobManager.clear_rls_context()
        if total:
            group_summary = dict(sorted(groups.items()))
            logger.info(
                f"Jobs prune run complete: total={total} dry_run={dry_run} "
                f"groups={group_summary} "
                f"domain={','.join([d for d in domains if d]) or 'all'} "
                f"queue={','.join([q for q in queues if q]) or 'all'} "
                f"job_type={','.join([jt for jt in job_types if jt]) or 'all'}"
            )

    async def _runner() -> None:
        await asyncio.sleep(min(interval, 60))
        while True:
            try:
                await _run_once()
            except Exception as exc:
                logger.bind(error_type=type(exc).__name__).warning("Jobs prune run failed")
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="jobs_prune_scheduler")
    logger.info(f"Started Jobs prune scheduler: interval={interval}s dry_run={dry_run}")
    return task
