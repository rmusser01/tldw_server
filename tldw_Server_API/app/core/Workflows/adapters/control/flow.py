"""Control flow adapters: prompt, delay, log, branch, map, parallel.

These adapters handle workflow control flow operations.
"""

from __future__ import annotations

import asyncio
import re
import time
import types
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.Metrics import start_async_span as _start_span
from tldw_Server_API.app.core.testing import is_truthy as _is_truthy
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import get_adapter, registry
from tldw_Server_API.app.core.Workflows.adapters.control._config import (
    BranchConfig,
    DelayConfig,
    LogConfig,
    MapConfig,
    ParallelConfig,
    PromptConfig,
)

_FLOW_NONCRITICAL_EXCEPTIONS = (
    AdapterError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    re.error,
)


@registry.register(
    "prompt",
    category="control",
    description="Render a prompt using the sandboxed Jinja engine",
    parallelizable=True,
    tags=["core", "template"],
    config_model=PromptConfig,
)
async def run_prompt_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Render a prompt using the sandboxed Jinja engine.

    Config:
      - template: str (preferred) or prompt: str
      - variables: dict (optional) merged into context
    Output:
      - {"text": rendered}
    """
    template = config.get("template") or config.get("prompt") or ""
    variables = config.get("variables") or {}
    # Merge base inputs
    data = {**context}
    # Ensure dot-access for inputs in templates (e.g., inputs.name)
    try:
        if isinstance(data.get("inputs"), dict):
            data["inputs"] = types.SimpleNamespace(**data["inputs"])  # type: ignore[arg-type]
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        logger.debug("Prompt adapter: failed to namespace inputs")
    try:
        # Keep a shallow namespace for convenience
        data.update(variables)
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        logger.debug("Prompt adapter: failed to merge variables into context")

    # Pre-pass: replacements for common tokens to be robust in sandbox
    try:
        if isinstance(context.get("inputs"), dict):
            # Handle {{ inputs.key || '' }}
            def repl_fallback(m):
                key = m.group(1)
                return str(context["inputs"].get(key, ""))
            template = re.sub(r"\{\{\s*inputs\.(\w+)\s*\|\|\s*''\s*\}\}", repl_fallback, template)
            # Handle {{ inputs.key }}
            def repl_simple(m):
                key = m.group(1)
                return str(context["inputs"].get(key, ""))
            template = re.sub(r"\{\{\s*inputs\.(\w+)\s*\}\}", repl_simple, template)
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        logger.debug("Prompt adapter: pre-pass templating fallback failed")

    # Optional simulated delay/error for testing retries/timeouts
    try:
        delay_ms = int(config.get("simulate_delay_ms", 0))
        if delay_ms > 0:
            remaining = delay_ms / 1000.0
            # Sleep in small chunks to allow cooperative cancel during tests
            while remaining > 0:
                if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                    return {"__status__": "cancelled"}
                sl = min(0.05, remaining)
                await asyncio.sleep(sl)
                remaining -= sl
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        logger.debug("Prompt adapter: simulate_delay handling failed")
    # Force-error handling (test-friendly)
    fe = config.get("force_error")
    if isinstance(fe, str):
        fe = _is_truthy(fe.strip())
    if fe or str(config.get("template", "")).strip().lower() == "bad":
        raise AdapterError("forced_error")

    rendered = apply_template_to_string(template, data) or ""
    logger.debug(f"Prompt adapter rendered length={len(rendered)}")
    # Optional artifact persistence
    try:
        if bool(config.get("save_artifact")) and callable(context.get("add_artifact")):
            step_run_id = str(context.get("step_run_id") or "")
            art_dir = resolve_artifacts_dir(step_run_id or f"prompt_{int(time.time()*1000)}")
            art_dir.mkdir(parents=True, exist_ok=True)
            fpath = art_dir / "prompt.txt"
            fpath.write_text(rendered or "", encoding="utf-8")
            context["add_artifact"](
                type="prompt_text",
                uri=f"file://{fpath}",
                size_bytes=len((rendered or "").encode("utf-8")),
                mime_type="text/plain",
                metadata={"step": "prompt"},
            )
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        logger.debug("Prompt adapter: failed to persist prompt artifact")
    return {"text": rendered}


@registry.register(
    "delay",
    category="control",
    description="Wait for specified milliseconds",
    parallelizable=True,
    tags=["control", "utility"],
    config_model=DelayConfig,
)
async def run_delay_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Simple delay step; waits for the specified milliseconds.

    Config:
      - milliseconds: int (default 1000)
    Output: { "delayed_ms": n }
    """
    try:
        ms = int(config.get("milliseconds", 1000))
    except (TypeError, ValueError):
        ms = 1000
    remaining = max(0, ms) / 1000.0
    while remaining > 0:
        if callable(context.get("is_cancelled")) and context["is_cancelled"]():
            return {"__status__": "cancelled"}
        sl = min(0.05, remaining)
        await asyncio.sleep(sl)
        remaining -= sl
    return {"delayed_ms": ms}


@registry.register(
    "log",
    category="control",
    description="Log a templated message for debugging pipelines",
    parallelizable=True,
    tags=["control", "debug"],
    config_model=LogConfig,
)
async def run_log_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Log a templated message; useful for debugging pipelines.

    Config:
      - message: str (templated)
      - level: str (debug|info|warning|error) default info
    Output: { "logged": true, "message": ... }
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl
    msg_t = str(config.get("message", ""))
    level = str(config.get("level", "info")).lower()
    # Pre-pass replacements for common templates like {{ inputs.name || '' }} and {{ inputs.name }}
    try:
        if isinstance(context.get("inputs"), dict):
            # Handle {{ inputs.key || '' }}
            def repl_fallback(m):
                key = m.group(1)
                return str(context["inputs"].get(key, ""))
            msg_t = re.sub(r"\{\{\s*inputs\.(\w+)\s*\|\|\s*''\s*\}\}", repl_fallback, msg_t)
            # Handle {{ inputs.key }}
            def repl_simple(m):
                key = m.group(1)
                return str(context["inputs"].get(key, ""))
            msg_t = re.sub(r"\{\{\s*inputs\.(\w+)\s*\}\}", repl_simple, msg_t)
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        message = _tmpl(msg_t, context) or msg_t
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        # Fall back to the pre-pass content if templating fails
        message = msg_t
    # Optional PII redaction in logs
    try:
        import os as _os
        redact = _is_truthy(_os.getenv("WORKFLOWS_REDACT_LOGS", "true"))
        if redact:
            try:
                from tldw_Server_API.app.core.Audit.unified_audit_service import PIIDetector
                message = PIIDetector().redact(message)
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                pass
        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        pass
    return {"logged": True, "message": message, "level": level}


@registry.register(
    "branch",
    category="control",
    description="Evaluate a condition and select the next step",
    parallelizable=False,
    tags=["control", "conditional"],
    config_model=BranchConfig,
)
async def run_branch_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a simple boolean condition and select the next step.

    Config:
      - condition: str (templated). Treated as true by the shared truthy parser.
      - true_next: str (step id)
      - false_next: str (step id)
    Output: { "__next__": step_id, "branch": "true"|"false" }
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl
    cond_t = str(config.get("condition", "")).strip()
    rendered = (_tmpl(cond_t, context) or cond_t).strip().lower()
    is_true = _is_truthy(rendered)
    next_id = str(config.get("true_next") if is_true else config.get("false_next") or "").strip()
    # Do not force if not provided; engine will fall back to natural order
    out: dict[str, Any] = {"branch": "true" if is_true else "false"}
    if next_id:
        out["__next__"] = next_id
    # Trace decision as a child span for better visibility
    try:
        async with _start_span("workflows.branch", attributes={
            "condition_template": cond_t,
            "rendered": rendered,
            "decision": out["branch"],
            "next_id": next_id or ""
        }):
            pass
    except _FLOW_NONCRITICAL_EXCEPTIONS:
        pass
    return out


@registry.register(
    "map",
    category="control",
    description="Fan-out over a list of items and apply a step to each",
    parallelizable=False,
    tags=["control", "parallel"],
    config_model=MapConfig,
)
async def run_map_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Fan-out over a list of items and apply a simple step to each item.

    Config:
      - items: list | str (templated path)
      - step: {type, config}
      - concurrency: int (default 4)
    Output: { "results": [ ... ], "count": n }
    """
    from tldw_Server_API.app.core.Workflows.adapters._registry import get_parallelizable

    items_cfg = config.get("items")
    items: list
    if isinstance(items_cfg, list):
        items = items_cfg
    else:
        from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl
        raw = _tmpl(str(items_cfg or ""), context) or str(items_cfg or "")
        try:
            import json as _json
            parsed = _json.loads(raw)
            items = parsed if isinstance(parsed, list) else [raw]
        except (TypeError, ValueError):
            items = [s.strip() for s in str(raw).split(",") if str(s).strip()]

    sub = config.get("step") or {}
    sub_type = str(sub.get("type") or "").strip()
    sub_cfg = sub.get("config") or {}
    if not sub_type:
        raise AdapterError("missing_substep_type")

    # Use registry to check if substep type is parallelizable
    parallelizable = get_parallelizable()
    if sub_type not in parallelizable:
        raise AdapterError(f"unsupported_substep_type:{sub_type}")

    concurrency = max(1, int(config.get("concurrency", 4)))
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(idx: int, item: Any) -> dict[str, Any]:
        async with sem:
            # Honour cancellation before running each sub-step
            try:
                if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                    return {"__status__": "cancelled"}
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                pass
            sub_ctx = {**context, "item": item}
            # Child span per item
            try:
                preview = str(item)
                if len(preview) > 80:
                    preview = preview[:77] + "…"
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                preview = ""
            try:
                async with _start_span("workflows.map.item", attributes={
                    "index": int(idx),
                    "sub_type": sub_type,
                    "item_preview": preview,
                }):
                    adapter = get_adapter(sub_type)
                    if adapter is not None:
                        return await adapter(sub_cfg, sub_ctx)
                    return {"error": f"unsupported_substep:{sub_type}"}
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                # If tracing fails, still attempt the sub-step
                adapter = get_adapter(sub_type)
                if adapter is not None:
                    return await adapter(sub_cfg, sub_ctx)
                return {"error": f"unsupported_substep:{sub_type}"}

    results = await asyncio.gather(*[_run_one(i, it) for i, it in enumerate(items)], return_exceptions=False)
    return {"results": results, "count": len(results)}


@registry.register(
    "parallel",
    category="control",
    description="Execute multiple steps in parallel",
    parallelizable=False,
    tags=["control", "parallel"],
    config_model=ParallelConfig,
)
async def run_parallel_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Execute multiple steps in parallel.

    Config:
      - steps: list of {type, config}
      - max_concurrency: int (default 5)
      - fail_fast: bool (default False)
    Output: { "results": [...], "count": n, "errors": [...] }
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        return {"error": "missing_steps", "results": []}

    max_concurrency = int(config.get("max_concurrency", 5))
    fail_fast = config.get("fail_fast", False)

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list = [None] * len(steps)
    errors: list = []

    async def run_step(idx: int, step_config: dict[str, Any]) -> None:
        async with semaphore:
            if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                return

            step_type = step_config.get("type")
            step_cfg = step_config.get("config", {})

            try:
                adapter = get_adapter(step_type)
                if adapter is not None:
                    result = await adapter(step_cfg, context)
                    results[idx] = result
                else:
                    results[idx] = {"error": f"unknown_step_type: {step_type}"}
            except _FLOW_NONCRITICAL_EXCEPTIONS:
                results[idx] = {"error": "parallel_step_error"}
                if fail_fast:
                    errors.append("parallel_step_error")

    tasks = [run_step(i, step) for i, step in enumerate(steps)]
    await asyncio.gather(*tasks, return_exceptions=not fail_fast)

    return {"results": results, "count": len(results), "errors": errors if errors else None}
