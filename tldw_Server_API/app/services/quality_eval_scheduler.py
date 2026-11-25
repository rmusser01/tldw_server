"""
Nightly RAG quality evaluation scheduler.

Runs a lightweight, configurable evaluation over a small eval set and emits
Prometheus/OTel metrics for Grafana dashboards (faithfulness/coverage trend).

Enable via env:
  - RAG_QUALITY_EVAL_ENABLED=true
  - RAG_QUALITY_EVAL_INTERVAL_SEC=86400  (default)
  - RAG_QUALITY_EVAL_DATASET=Docs/Deployment/Monitoring/Evals/nightly_rag_eval.jsonl

Dataset format (JSONL):
  {"query": "...", "expect": "optional expected aspects", "namespace": "optional"}

This scheduler runs inside the FastAPI lifespan if enabled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.Metrics.metrics_manager import (
    set_gauge,
)
from tldw_Server_API.app.core.testing import env_flag_enabled

DEFAULT_EVAL_PATH = Path("Docs/Deployment/Monitoring/Evals/nightly_rag_eval.jsonl")


@dataclass
class EvalRecord:
    query: str
    expect: str | None = None
    namespace: str | None = None


def _load_eval_set(path: Path) -> list[EvalRecord]:
    items: list[EvalRecord] = []
    if not path.exists():
        logger.warning(f"RAG quality eval dataset not found at {path}")
        return items
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                q = str(data.get("query", "")).strip()
                if not q:
                    continue
                items.append(EvalRecord(query=q, expect=data.get("expect"), namespace=data.get("namespace")))
    except Exception as e:
        logger.warning(f"Failed to load eval set: {e}")
    return items


async def _run_eval_once(dataset_path: Path, dataset_label: str) -> None:
    """Run a single evaluation pass and publish metrics."""
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import RagRunConfig, run_rag_pipeline  # lazy import
    try:
        items = _load_eval_set(dataset_path)
        if not items:
            return

        faithfulness_scores: list[float] = []
        coverage_scores: list[float] = []
        # Keep the pass bounded; cap to 50 queries
        for rec in items[:50]:
            try:
                resp = await run_rag_pipeline(RagRunConfig.from_kwargs(
                    query=rec.query,
                    index_namespace=rec.namespace,
                    search_mode="hybrid",
                    enable_reranking=True,
                    enable_citations=True,
                    enable_generation=True,
                    enable_post_verification=True,
                    # Keep it lean for nightly runs
                    adaptive_max_claims=12,
                    adaptive_time_budget_sec=10.0,
                ))
                # Extract faithfulness: 1 - unsupported ratio if present
                md = getattr(resp, "metadata", {}) if hasattr(resp, "metadata") else (resp.get("metadata", {}) if isinstance(resp, dict) else {})
                pv = md.get("post_verification") or {}
                unsupported = float(pv.get("unsupported_ratio") or 0.0)
                faithfulness_scores.append(max(0.0, min(1.0, 1.0 - unsupported)))

                # Extract coverage from hard citations if present
                hc = md.get("hard_citations") or {}
                cov = hc.get("coverage")
                if cov is not None:
                    with contextlib.suppress(Exception):
                        coverage_scores.append(max(0.0, min(1.0, float(cov))))
            except Exception as e:
                logger.debug(f"Eval query failed: {e}")
                continue

        # Publish gauges (averages)
        try:
            if faithfulness_scores:
                avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
                set_gauge("rag_eval_faithfulness_score", avg_faith, labels={"dataset": dataset_label})
            if coverage_scores:
                avg_cov = sum(coverage_scores) / len(coverage_scores)
                set_gauge("rag_eval_coverage_score", avg_cov, labels={"dataset": dataset_label})
            import time as _t
            set_gauge("rag_eval_last_run_timestamp", _t.time(), labels={"dataset": dataset_label})
        except Exception as e:
            logger.debug(f"Failed to publish eval gauges: {e}")
    except Exception as e:
        logger.warning(f"RAG quality evaluation run failed: {e}")


async def start_quality_eval_scheduler() -> asyncio.Task | None:
    """Start the periodic quality evaluation scheduler if enabled.

    Returns an asyncio.Task or None if disabled.
    """
    enabled = env_flag_enabled("RAG_QUALITY_EVAL_ENABLED")
    if not enabled:
        return None

    try:
        interval = int(os.getenv("RAG_QUALITY_EVAL_INTERVAL_SEC", "86400"))
    except Exception:
        interval = 86400
    dataset = os.getenv("RAG_QUALITY_EVAL_DATASET", str(DEFAULT_EVAL_PATH))
    dataset_path = Path(dataset)
    dataset_label = dataset_path.stem

    async def _runner():
        # Initial delay to avoid startup thundering herd
        await asyncio.sleep(min(30, interval))
        while True:
            await _run_eval_once(dataset_path, dataset_label)
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="rag_quality_eval_scheduler")
    logger.info(
        f"Started RAG quality eval scheduler: every {interval}s using dataset={dataset_path}"
    )
    return task
