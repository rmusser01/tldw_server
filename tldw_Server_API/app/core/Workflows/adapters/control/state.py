"""State management adapters: batch, cache_result, retry, checkpoint.

These adapters handle workflow state and caching operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._common import resolve_artifacts_dir
from tldw_Server_API.app.core.Workflows.adapters._registry import get_adapter, registry
from tldw_Server_API.app.core.Workflows.adapters.control._config import (
    BatchConfig,
    CacheResultConfig,
    CheckpointConfig,
    RetryConfig,
)

_STATE_ADAPTER_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_STATE_JSON_EXCEPTIONS = (TypeError, ValueError, json.JSONDecodeError)


def _get_workflow_cache_collection(collection_name: str):
    """Resolve a Chroma collection for workflow cache storage."""
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import get_default_chroma_manager

    manager = get_default_chroma_manager()
    if not manager:
        return None
    return manager.get_or_create_collection(collection_name=collection_name)


@registry.register(
    "batch",
    category="control",
    description="Batch items into chunks for processing",
    parallelizable=False,
    tags=["control", "data"],
    config_model=BatchConfig,
)
async def run_batch_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Batch items into chunks for processing.

    Config:
      - items: list (or from prev)
      - batch_size: int (default 10)
    Output: { "batches": [...], "batch_count": n, "total_items": n }
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    items = config.get("items")
    if items is None:
        prev = context.get("prev") or context.get("last") or {}
        items = prev.get("items") or prev.get("documents") or prev.get("records")
        if items is None and isinstance(prev, list):
            items = prev

    if not isinstance(items, list):
        return {"error": "missing_items", "batches": [], "batch_count": 0}

    batch_size = int(config.get("batch_size", 10))
    if batch_size < 1:
        batch_size = 1

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    return {"batches": batches, "batch_count": len(batches), "total_items": len(items), "batch_size": batch_size}


@registry.register(
    "cache_result",
    category="control",
    description="Cache step result by key for reuse",
    parallelizable=False,
    tags=["control", "cache"],
    config_model=CacheResultConfig,
)
async def run_cache_result_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Cache step result by key for reuse.

    Config:
      - key: str (cache key)
      - ttl_seconds: int (default 3600)
      - action: get|set|get_or_set|invalidate (default get_or_set)
      - data: any (data to cache, defaults to prev)
    Output: { "cached": bool, "data": any, ... }
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    cache_key = config.get("key")
    if not cache_key:
        return {"error": "missing_cache_key", "cached": False}

    ttl_seconds = int(config.get("ttl_seconds", 3600))
    action = config.get("action", "get_or_set")  # get, set, get_or_set, invalidate

    # Get data to cache (for set operations)
    data = config.get("data")
    if data is None:
        prev = context.get("prev") or context.get("last") or {}
        data = prev

    try:
        cache_collection_name = "workflow_cache"
        try:
            collection = _get_workflow_cache_collection(cache_collection_name)
        except _STATE_ADAPTER_EXCEPTIONS:
            return {"cached": False, "data": data, "error": "cache_collection_error"}
        if not collection:
            return {"cached": False, "data": data, "error": "cache_unavailable"}

        if action == "invalidate":
            with contextlib.suppress(_STATE_ADAPTER_EXCEPTIONS):
                collection.delete(ids=[cache_key])
            return {"invalidated": True, "key": cache_key}

        if action in ("get", "get_or_set"):
            try:
                result = collection.get(ids=[cache_key], include=["metadatas", "documents"])
                if result and result.get("ids") and result["ids"]:
                    meta = result.get("metadatas", [{}])[0] or {}
                    cached_at = meta.get("cached_at", 0)
                    if time.time() - cached_at <= ttl_seconds:
                        cached_data = meta.get("data")
                        if isinstance(cached_data, str):
                            with contextlib.suppress(_STATE_JSON_EXCEPTIONS):
                                cached_data = json.loads(cached_data)
                        return {"cached": True, "data": cached_data, "key": cache_key, "age_seconds": int(time.time() - cached_at)}
            except _STATE_ADAPTER_EXCEPTIONS:
                pass

        if action in ("set", "get_or_set"):
            try:
                data_str = json.dumps(data) if not isinstance(data, str) else data
                collection.upsert(
                    ids=[cache_key],
                    documents=[cache_key],
                    metadatas=[{"data": data_str, "cached_at": time.time()}],
                )
                return {"cached": False, "stored": True, "data": data, "key": cache_key}
            except _STATE_ADAPTER_EXCEPTIONS:
                return {"cached": False, "data": data, "error": "cache_store_error"}

        return {"cached": False, "data": data}

    except _STATE_ADAPTER_EXCEPTIONS:
        logger.exception("Cache result error")
        return {"cached": False, "data": data, "error": "cache_result_error"}


@registry.register(
    "retry",
    category="control",
    description="Wrap a step with retry logic",
    parallelizable=False,
    tags=["control", "error-handling"],
    config_model=RetryConfig,
)
async def run_retry_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Wrap a step with retry logic.

    Config:
      - step_type: str
      - step_config: dict
      - max_retries: int (default 3)
      - backoff_base: float (default 2.0)
      - backoff_max: float (default 30.0)
      - retry_on_errors: list[str] (patterns to retry on)
    Output: { "result": any, "attempts": n, "success": bool }
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    step_type = config.get("step_type")
    step_config = config.get("step_config", {})
    if not step_type:
        return {"error": "missing_step_type", "result": None}

    max_retries = int(config.get("max_retries", 3))
    backoff_base = float(config.get("backoff_base", 2.0))
    backoff_max = float(config.get("backoff_max", 30.0))
    retry_on_errors = config.get("retry_on_errors")  # List of error patterns to retry on

    adapter = get_adapter(step_type)
    if adapter is None:
        return {"error": f"unknown_step_type: {step_type}", "result": None}
    last_error = None

    for attempt in range(max_retries + 1):
        if callable(context.get("is_cancelled")) and context["is_cancelled"]():
            return {"__status__": "cancelled"}

        try:
            result = await adapter(step_config, context)

            # Check if result indicates an error
            if isinstance(result, dict) and result.get("error"):
                error_str = str(result["error"])
                should_retry = any(pat in error_str for pat in retry_on_errors) if retry_on_errors else True

                if should_retry and attempt < max_retries:
                    last_error = error_str
                    delay = min(backoff_base ** attempt, backoff_max)
                    await asyncio.sleep(delay)
                    continue

            return {"result": result, "attempts": attempt + 1, "success": True}

        except _STATE_ADAPTER_EXCEPTIONS:
            last_error = "retry_step_error"
            if attempt < max_retries:
                delay = min(backoff_base ** attempt, backoff_max)
                await asyncio.sleep(delay)
            else:
                return {"error": last_error, "attempts": attempt + 1, "success": False}

    return {"error": last_error, "attempts": max_retries + 1, "success": False}


@registry.register(
    "checkpoint",
    category="control",
    description="Save workflow state for recovery",
    parallelizable=False,
    tags=["control", "state"],
    config_model=CheckpointConfig,
)
async def run_checkpoint_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Save workflow state for recovery.

    Config:
      - checkpoint_id: str (auto-generated if not provided)
      - data: any (defaults to prev + inputs)
    Output: { "checkpoint_id": str, "saved": bool, "run_id": str }
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    checkpoint_id = config.get("checkpoint_id") or f"checkpoint_{int(time.time()*1000)}"
    data = config.get("data")
    if data is None:
        prev = context.get("prev") or context.get("last") or {}
        data = {"prev": prev, "inputs": context.get("inputs", {})}

    run_id = context.get("run_id")

    try:
        # Store checkpoint as an event
        if callable(context.get("append_event")):
            context["append_event"]("checkpoint", {
                "checkpoint_id": checkpoint_id,
                "data": data,
            })

        # Also store as artifact for persistence
        if callable(context.get("add_artifact")):
            step_run_id = str(context.get("step_run_id") or checkpoint_id)
            art_dir = resolve_artifacts_dir(step_run_id)
            art_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = art_dir / f"{checkpoint_id}.json"
            checkpoint_file.write_text(json.dumps(data, default=str), encoding="utf-8")

            context["add_artifact"](
                type="checkpoint",
                uri=f"file://{checkpoint_file}",
                size_bytes=checkpoint_file.stat().st_size,
                mime_type="application/json",
                metadata={"checkpoint_id": checkpoint_id},
            )

        return {"checkpoint_id": checkpoint_id, "saved": True, "run_id": run_id}

    except _STATE_ADAPTER_EXCEPTIONS:
        logger.exception("Checkpoint error")
        return {"error": "checkpoint_error", "checkpoint_id": checkpoint_id, "saved": False}
