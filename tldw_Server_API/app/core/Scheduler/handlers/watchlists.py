"""
Scheduler task handler for Watchlists runs.

Task names:
- 'watchlist_run'
- 'watchlists_enrich_output'
Inputs expected in payload:
  payload = {
    'inputs': { 'watchlist_job_id': <int> },
    'user_id': '<user id>',
    'tenant_id': 'default' | str
  }

The handler creates a scrape_run row, performs a minimal fetch→ingest stub
and then updates run status and job history (last_run_at/next_run_at). When the
real scraping is implemented, this stub can be replaced with the actual pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.Scheduler.base.registry import task
from tldw_Server_API.app.core.Watchlists import output_enrichment_handler
from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _compute_next_run(cron: str | None, timezone_str: str | None) -> str | None:
    if not cron:
        return None
    try:
        from apscheduler.triggers.cron import CronTrigger
        tz = (timezone_str or "UTC")
        trigger = CronTrigger.from_crontab(cron, timezone=tz)
        now = datetime.now(trigger.timezone)
        nxt = trigger.get_next_fire_time(None, now)
        return nxt.isoformat() if nxt else None
    except Exception:
        return None


@task(name="watchlist_run", max_retries=0, timeout=3600, queue="watchlists")
async def watchlist_run(payload: dict[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs") or {}
    if not isinstance(inputs, dict):
        raise ValueError("watchlist_run: inputs must be a dict")
    job_id = inputs.get("watchlist_job_id")
    if not job_id:
        raise ValueError("watchlist_run: missing watchlist_job_id")
    user_id = payload.get("user_id")
    if user_id is None:
        raise ValueError("watchlist_run: missing user_id")
    try:
        uid_int = int(user_id)
    except Exception:
        raise ValueError("watchlist_run: user_id must be int-like") from None

    tenant_id = payload.get("tenant_id")
    if tenant_id is not None:
        tenant_id = str(tenant_id)

    # Execute the real pipeline (handles run row creation, stats, and job history)
    result = await run_watchlist_job(uid_int, int(job_id), tenant_id=tenant_id)
    status = str(result.get("status") or "succeeded")
    return {"run_id": result.get("run_id"), "status": status, "items_ingested": int(result.get("items_ingested", 0))}


@task(name="watchlists_enrich_output", max_retries=1, timeout=3600, queue="watchlists")
async def watchlists_enrich_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Run deferred enrichment for a generated watchlist output."""
    output_id = payload.get("output_id")
    if output_id is None:
        raise ValueError("watchlists_enrich_output: missing output_id")
    user_id = payload.get("user_id")
    if user_id is None:
        raise ValueError("watchlists_enrich_output: missing user_id")
    try:
        output_id_int = int(output_id)
        user_id_int = int(user_id)
    except Exception:
        raise ValueError("watchlists_enrich_output: output_id and user_id must be int-like") from None

    grouping_config = payload.get("grouping_config")
    if grouping_config is not None and not isinstance(grouping_config, dict):
        raise ValueError("watchlists_enrich_output: grouping_config must be a dict when provided")
    summary_config = payload.get("summary_config")
    if summary_config is not None and not isinstance(summary_config, dict):
        raise ValueError("watchlists_enrich_output: summary_config must be a dict when provided")

    await output_enrichment_handler.enrich_output(
        output_id=output_id_int,
        user_id=user_id_int,
        grouping_config=grouping_config,
        summary_config=summary_config,
    )
    return {"output_id": output_id_int, "status": "completed"}
