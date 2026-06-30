"""Utility adapters for workflow steps.

This module provides adapters for utility operations like timing,
diff detection, document operations, sandbox execution, and scheduling.
"""

from __future__ import annotations

import contextlib
import datetime
import difflib
import json
import os
import tempfile
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_context_user_id,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.utility._config import (
    ContextBuildConfig,
    DiffChangeDetectorConfig,
    DocumentDiffConfig,
    DocumentMergeConfig,
    EmbedConfig,
    SandboxExecConfig,
    ScheduleWorkflowConfig,
    ScreenshotCaptureConfig,
    TimingStartConfig,
    TimingStopConfig,
)


@registry.register(
    "timing_start",
    category="utility",
    description="Start a named timer",
    parallelizable=True,
    tags=["utility", "timing"],
    config_model=TimingStartConfig,
)
async def run_timing_start_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Start a named timer.

    Config:
      - timer_name: str - Name for the timer (default: "default")
    Output:
      - timer_name: str
      - started_at: float
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    timer_name = config.get("timer_name", "default")
    started_at = time.time()

    # Store in context for retrieval by timing_stop
    context[f"__timer_{timer_name}__"] = started_at

    return {
        "timer_name": timer_name,
        "started_at": started_at,
        "started_at_iso": datetime.datetime.utcnow().isoformat(),
    }


@registry.register(
    "timing_stop",
    category="utility",
    description="Stop timer and return elapsed time",
    parallelizable=True,
    tags=["utility", "timing"],
    config_model=TimingStopConfig,
)
async def run_timing_stop_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Stop timer and return elapsed time.

    Config:
      - timer_name: str - Name of the timer (default: "default")
    Output:
      - timer_name: str
      - elapsed_ms: float
      - elapsed_seconds: float
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    timer_name = config.get("timer_name", "default")
    stopped_at = time.time()

    # Try to get start time from context
    started_at = context.get(f"__timer_{timer_name}__")

    if started_at is None:
        # Try from inputs
        inputs = context.get("inputs", {})
        started_at = inputs.get(f"timer_{timer_name}_started_at")

    if started_at is None:
        return {
            "timer_name": timer_name,
            "error": "timer_not_found",
            "elapsed_ms": 0,
            "elapsed_seconds": 0,
        }

    elapsed_seconds = stopped_at - float(started_at)
    elapsed_ms = elapsed_seconds * 1000

    return {
        "timer_name": timer_name,
        "elapsed_ms": elapsed_ms,
        "elapsed_seconds": elapsed_seconds,
        "stopped_at": stopped_at,
        "stopped_at_iso": datetime.datetime.utcnow().isoformat(),
    }


@registry.register(
    "diff_change_detector",
    category="utility",
    description="Compare last vs current text to detect changes",
    parallelizable=True,
    tags=["utility", "diff"],
    config_model=DiffChangeDetectorConfig,
)
async def run_diff_change_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Compare last vs current text to detect changes.

    Config:
      - current: str (templated) or take from inputs.text
      - method: 'ratio'|'unified' (default 'ratio')
      - threshold: float (for ratio; default 0.9)

    Output:
      - { changed: bool, ratio?, diff?, text }
    """
    prev = context.get("prev") or {}
    prev_text = str(prev.get("text") or prev.get("content") or "")
    cur_t = str(config.get("current") or "").strip()
    if cur_t:
        current_text = apply_template_to_string(cur_t, context) or cur_t
    else:
        current_text = str((context.get("inputs") or {}).get("text") or "")
    method = str(config.get("method") or "ratio").strip().lower()
    th = float(config.get("threshold", 0.9))
    if method == "unified":
        diff = "\n".join(
            difflib.unified_diff(
                prev_text.splitlines(),
                current_text.splitlines(),
                fromfile="prev",
                tofile="current",
                lineterm="",
            )
        )
        changed = prev_text != current_text
        return {"changed": changed, "diff": diff, "text": current_text}
    else:
        sm = difflib.SequenceMatcher(a=prev_text, b=current_text)
        ratio = sm.ratio()
        changed = ratio < th
        return {"changed": changed, "ratio": ratio, "text": current_text}


@registry.register(
    "document_merge",
    category="utility",
    description="Merge multiple documents into one",
    parallelizable=True,
    tags=["utility", "merge"],
    config_model=DocumentMergeConfig,
)
async def run_document_merge_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple documents into one.

    Config:
      - documents: list[str] - List of document texts to merge
      - separator: str - Separator between documents (default: "\n\n")
      - add_headers: bool - Add section headers (default: False)
    Output:
      - merged: str - The merged document
      - document_count: int
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    documents = config.get("documents") or []
    separator = config.get("separator", "\n\n")
    add_headers = bool(config.get("add_headers", False))

    # Try to get documents from previous step if not provided
    if not documents:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            documents = prev.get("documents") or prev.get("texts") or []
            if not documents and prev.get("text"):
                documents = [prev.get("text")]

    # Template each document
    processed = []
    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            doc = apply_template_to_string(doc, context) or doc
        elif isinstance(doc, dict):
            doc = doc.get("content") or doc.get("text") or str(doc)
        else:
            doc = str(doc)

        if add_headers:
            processed.append(f"## Document {i + 1}\n\n{doc}")
        else:
            processed.append(doc)

    merged = separator.join(processed)
    return {"merged": merged, "text": merged, "document_count": len(processed)}


@registry.register(
    "document_diff",
    category="utility",
    description="Compare two documents and output diff",
    parallelizable=True,
    tags=["utility", "diff"],
    config_model=DocumentDiffConfig,
)
async def run_document_diff_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Compare two documents and output diff.

    Config:
      - document_a: str - First document
      - document_b: str - Second document
      - context_lines: int - Lines of context around changes (default: 3)
      - output_format: str - "unified", "html", or "side_by_side" (default: "unified")
    Output:
      - diff: str - The diff output
      - has_changes: bool
      - additions: int
      - deletions: int
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    doc_a = config.get("document_a") or ""
    doc_b = config.get("document_b") or ""

    if isinstance(doc_a, str):
        doc_a = apply_template_to_string(doc_a, context) or doc_a
    if isinstance(doc_b, str):
        doc_b = apply_template_to_string(doc_b, context) or doc_b

    context_lines = int(config.get("context_lines", 3))
    output_format = str(config.get("output_format", "unified")).lower()

    lines_a = doc_a.splitlines(keepends=True)
    lines_b = doc_b.splitlines(keepends=True)

    if output_format == "html":
        differ = difflib.HtmlDiff()
        diff_output = differ.make_file(lines_a, lines_b, context=True, numlines=context_lines)
    elif output_format == "side_by_side":
        differ = difflib.Differ()
        diff_output = "\n".join(differ.compare(lines_a, lines_b))
    else:
        diff_output = "".join(difflib.unified_diff(lines_a, lines_b, lineterm="", n=context_lines))

    # Count additions and deletions
    additions = sum(1 for line in diff_output.split("\n") if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_output.split("\n") if line.startswith("-") and not line.startswith("---"))

    return {
        "diff": diff_output,
        "has_changes": bool(additions or deletions),
        "additions": additions,
        "deletions": deletions,
    }


@registry.register(
    "context_build",
    category="utility",
    description="Build context from multiple sources",
    parallelizable=True,
    tags=["utility", "context"],
    config_model=ContextBuildConfig,
)
async def run_context_build_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Build context from multiple sources."""
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    sources = config.get("sources") or []
    max_tokens = int(config.get("max_tokens", 4000))
    separator = config.get("separator", "\n\n---\n\n")

    context_parts = []
    total_chars = 0
    char_limit = max_tokens * 4  # Rough estimate

    # Include inputs if specified
    if config.get("include_inputs"):
        inputs = context.get("inputs", {})
        if inputs:
            context_parts.append(f"**Inputs:**\n{json.dumps(inputs, indent=2)}")

    # Include previous step output
    if config.get("include_prev"):
        prev = context.get("prev") or context.get("last") or {}
        prev_text = prev.get("text") or prev.get("content") or ""
        if prev_text:
            context_parts.append(f"**Previous Output:**\n{prev_text}")

    # Process additional sources
    for source in sources:
        if total_chars >= char_limit:
            break

        if isinstance(source, str):
            source = apply_template_to_string(source, context) or source
            context_parts.append(source)
            total_chars += len(source)
        elif isinstance(source, dict):
            source_type = source.get("type")
            if source_type == "text":
                text = source.get("text") or source.get("content") or ""
                if isinstance(text, str):
                    text = apply_template_to_string(text, context) or text
                label = source.get("label", "Content")
                context_parts.append(f"**{label}:**\n{text}")
                total_chars += len(text)
            elif source_type == "documents":
                docs = source.get("documents") or []
                for doc in docs:
                    if total_chars >= char_limit:
                        break
                    doc_text = doc.get("content") or doc.get("text") or str(doc)
                    context_parts.append(doc_text)
                    total_chars += len(doc_text)

    combined_context = separator.join(context_parts)

    # Truncate if needed
    if len(combined_context) > char_limit:
        combined_context = combined_context[:char_limit] + "\n... [truncated]"

    return {"context": combined_context, "source_count": len(context_parts), "total_chars": len(combined_context)}


@registry.register(
    "embed",
    category="utility",
    description="Generate embeddings and store in vector database",
    parallelizable=True,
    tags=["utility", "embeddings"],
    config_model=EmbedConfig,
)
async def run_embed_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Embed texts and upsert into vector store (Chroma) directly.

    Config:
      - texts: list[str] | str (defaults to last.text)
      - collection: str (default: user_{user_id}_workflows)
      - model_id: str (optional override)
      - metadata: dict (optional global metadata per text)

    Output: { upserted: n, collection: name }
    """
    import uuid as _uuid

    from tldw_Server_API.app.core.config import settings as _settings
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import create_embeddings_batch_async

    # Resolve texts
    texts_cfg = config.get("texts")
    texts: list[str]
    if isinstance(texts_cfg, list):
        texts = [str(t) for t in texts_cfg if str(t).strip()]
    elif isinstance(texts_cfg, str) and texts_cfg.strip():
        texts = [texts_cfg]
    else:
        prev = context.get("prev") or {}
        txt = str(prev.get("text") or prev.get("content") or "").strip()
        if not txt:
            return {"error": "no_text"}
        texts = [txt]

    user_id = resolve_context_user_id(context)
    if not user_id:
        return {"error": "missing_user_id"}
    collection = str(config.get("collection") or f"user_{user_id}_workflows")
    model_id = str(config.get("model_id") or "") or None
    md_global = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}

    # Build embedding config
    user_app_config = dict(_settings.get("EMBEDDING_CONFIG", {}))
    user_app_config["USER_DB_BASE_DIR"] = _settings.get("USER_DB_BASE_DIR")
    embeds = await create_embeddings_batch_async(texts=texts, user_app_config=user_app_config, model_id_override=model_id)

    ids = [f"wf_{_uuid.uuid4().hex}" for _ in texts]
    metadatas = []
    for _t in texts:
        m = {"run_id": context.get("run_id"), "step_run_id": context.get("step_run_id")}
        if md_global:
            with contextlib.suppress(Exception):
                m.update(dict(md_global.items()))
        metadatas.append(m)

    # Upsert into per-user collection
    mgr = ChromaDBManager(user_id=user_id, user_embedding_config=user_app_config)
    mgr.store_in_chroma(
        collection_name=collection,
        texts=texts,
        embeddings=embeds,
        ids=ids,
        metadatas=metadatas,
        embedding_model_id_for_dim_check=model_id,
    )
    return {"upserted": len(texts), "collection": collection}


@registry.register(
    "sandbox_exec",
    category="utility",
    description="Execute code in an isolated sandbox environment",
    parallelizable=False,
    tags=["utility", "execution"],
    config_model=SandboxExecConfig,
)
async def run_sandbox_exec_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Execute code in an isolated sandbox environment.

    Config:
      - code: str (templated) - code to execute
      - language: Literal["python", "bash", "javascript"] = "python"
      - timeout_seconds: int = 30
      - memory_limit_mb: int = 256
      - stdin: Optional[str] (templated) - input to provide via stdin
      - base_image: Optional[str] - Docker image to use
    Output:
      - {"stdout": str, "stderr": str, "exit_code": int, "duration_ms": float, "timed_out": bool}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = resolve_context_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except Exception:
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    # Template rendering
    code_t = str(config.get("code") or "").strip()
    if code_t:
        code = apply_template_to_string(code_t, context) or code_t
    else:
        return {"error": "missing_code"}

    if not code.strip():
        return {"error": "missing_code"}

    language = str(config.get("language") or "python").strip().lower()
    if language not in ("python", "bash", "sh", "javascript", "js", "node"):
        return {"error": f"unsupported_language:{language}"}
    # Normalize aliases
    if language == "sh":
        language = "bash"
    if language in ("js", "node"):
        language = "javascript"

    timeout_seconds = int(config.get("timeout_seconds") or config.get("timeout_sec") or 30)
    timeout_seconds = max(1, min(timeout_seconds, 300))  # Cap at 5 minutes

    memory_limit_mb = int(config.get("memory_limit_mb") or config.get("memory_mb") or 256)
    memory_limit_mb = max(64, min(memory_limit_mb, 1024))  # 64MB to 1GB

    stdin_t = config.get("stdin")
    stdin_val = None
    if stdin_t is not None:
        stdin_val = apply_template_to_string(str(stdin_t), context) or str(stdin_t)

    # Test mode simulation
    if is_test_mode():
        simulated_stdout = f"[TEST_MODE] Code executed successfully\nLanguage: {language}\nCode length: {len(code)} chars"
        if stdin_val:
            simulated_stdout += f"\nStdin provided: {len(stdin_val)} chars"
        return {
            "stdout": simulated_stdout,
            "stderr": "",
            "exit_code": 0,
            "duration_ms": 42.0,
            "timed_out": False,
            "simulated": True,
            "language": language,
        }

    try:
        from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType
        from tldw_Server_API.app.core.Sandbox.service import SandboxService

        service = SandboxService()

        # Determine base image based on language
        base_image = config.get("base_image")
        if not base_image:
            if language == "python":
                base_image = "python:3.11-slim"
            elif language == "javascript":
                base_image = "node:20-slim"
            else:
                base_image = "ubuntu:24.04"

        # Build command based on language
        if language == "python":
            command = ["python", "-c", code]
        elif language == "javascript":
            command = ["node", "-e", code]
        else:  # bash
            command = ["bash", "-c", code]

        # Create run spec
        spec = RunSpec(
            session_id=None,
            runtime=RuntimeType.docker,
            base_image=base_image,
            command=command,
            env=dict(config.get("env") or {}),
            timeout_sec=timeout_seconds,
            memory_mb=memory_limit_mb,
            network_policy="deny_all",
        )

        # Execute
        import uuid as _uuid

        idem_key = f"wf-sandbox-{_uuid.uuid4()}"
        status = service.start_run_scaffold(
            user_id=user_id,
            spec=spec,
            spec_version="1.0",
            idem_key=idem_key,
            raw_body={"code": code[:100], "language": language},
        )

        # Extract results
        from tldw_Server_API.app.core.Sandbox.models import RunPhase

        timed_out = status.phase == RunPhase.timed_out
        exit_code = status.exit_code if status.exit_code is not None else (124 if timed_out else 1)

        # Get stdout/stderr from artifacts if available
        stdout = ""
        stderr = ""
        if status.artifacts:
            stdout = (status.artifacts.get("stdout") or b"").decode("utf-8", errors="replace")
            stderr = (status.artifacts.get("stderr") or b"").decode("utf-8", errors="replace")
        elif status.message:
            stderr = status.message

        duration_ms = 0.0
        if status.started_at and status.finished_at:
            duration_ms = (status.finished_at - status.started_at).total_seconds() * 1000

        result: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "timed_out": timed_out,
            "language": language,
            "run_id": status.id,
        }

        if stdout:
            result["text"] = stdout.strip()

        return result

    except Exception:
        logger.exception("Sandbox exec adapter error")
        return {"error": "sandbox_exec_error"}


@registry.register(
    "screenshot_capture",
    category="utility",
    description="Capture screenshot of URL using playwright",
    parallelizable=False,
    tags=["utility", "screenshot"],
    config_model=ScreenshotCaptureConfig,
)
async def run_screenshot_capture_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Capture screenshot of URL using playwright.

    Config:
      - url: str - URL to capture
      - full_page: bool - Capture full page (default: False)
      - width: int - Viewport width (default: 1280)
      - height: int - Viewport height (default: 720)
      - format: str - "png" or "jpeg" (default: "png")
      - timeout: int - Navigation timeout in ms (default: 30000)
    Output:
      - screenshot_path: str
      - screenshot_base64: str (if return_base64 is True)
    """
    from tldw_Server_API.app.core.Security.egress import evaluate_url_policy

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # TEST_MODE: return simulated result without real browser
    if is_test_mode():
        url = config.get("url") or ""
        if isinstance(url, str):
            url = apply_template_to_string(url, context) or url
        if not url:
            return {"error": "missing_url", "simulated": True}
        return {
            "screenshot_path": os.path.join(tempfile.gettempdir(), "simulated_screenshot.png"),
            "url": url,
            "simulated": True,
        }

    url = config.get("url") or ""
    if isinstance(url, str):
        url = apply_template_to_string(url, context) or url

    if not url:
        return {"error": "missing_url"}

    # SSRF protection: validate URL before navigation
    policy_result = evaluate_url_policy(url)
    if not policy_result.allowed:
        return {"error": f"url_blocked: {policy_result.reason}"}

    full_page = bool(config.get("full_page", False))
    width = int(config.get("width", 1280))
    height = int(config.get("height", 720))
    img_format = config.get("format", "png")
    return_base64 = bool(config.get("return_base64", False))
    nav_timeout = int(config.get("timeout", 30000))

    try:
        from playwright.async_api import async_playwright

        step_run_id = str(context.get("step_run_id") or f"screenshot_{int(time.time() * 1000)}")
        art_dir = resolve_artifacts_dir(step_run_id)
        art_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = art_dir / f"screenshot.{img_format}"

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page(viewport={"width": width, "height": height})
            await page.goto(url, wait_until="load", timeout=nav_timeout)
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=5000)
            await page.screenshot(path=str(screenshot_path), full_page=full_page, type=img_format)
            await browser.close()

        result = {"screenshot_path": str(screenshot_path), "url": url}

        if return_base64:
            import base64

            with open(screenshot_path, "rb") as f:
                result["screenshot_base64"] = base64.b64encode(f.read()).decode("utf-8")

        # Add artifact
        if callable(context.get("add_artifact")):
            context["add_artifact"](
                type="screenshot",
                uri=f"file://{screenshot_path}",
                size_bytes=screenshot_path.stat().st_size,
                mime_type=f"image/{img_format}",
                metadata={"url": url},
            )

        return result

    except ImportError:
        return {"error": "playwright_not_installed"}
    except Exception:
        logger.exception("Screenshot capture error")
        return {"error": "screenshot_capture_error"}


@registry.register(
    "schedule_workflow",
    category="utility",
    description="Schedule a workflow for future execution",
    parallelizable=False,
    tags=["utility", "scheduling"],
    config_model=ScheduleWorkflowConfig,
)
async def run_schedule_workflow_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Schedule a workflow for future execution.

    Config:
      - workflow_id: str - Workflow to schedule
      - delay_seconds: int - Delay before execution
      - cron: str - Cron expression (alternative to delay)
      - inputs: dict - Inputs for the workflow
    Output:
      - scheduled: bool
      - schedule_id: str
      - run_at: str
    """
    from datetime import timedelta

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    workflow_id = config.get("workflow_id")
    if isinstance(workflow_id, str):
        workflow_id = apply_template_to_string(workflow_id, context) or workflow_id

    if not workflow_id:
        return {"scheduled": False, "error": "missing_workflow_id"}

    delay_seconds = config.get("delay_seconds")
    cron = config.get("cron")
    inputs = config.get("inputs") or {}

    if not delay_seconds and not cron:
        return {"scheduled": False, "error": "missing_delay_or_cron"}

    try:
        # Calculate run time
        run_at = datetime.datetime.utcnow() + timedelta(seconds=int(delay_seconds)) if delay_seconds else None

        schedule_id = f"sched_{int(time.time() * 1000)}"

        # Store schedule in database
        try:
            from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

            db = WorkflowsDatabase()
            tenant_id = str(context.get("tenant_id", "default"))
            db.create_schedule(
                schedule_id=schedule_id,
                tenant_id=tenant_id,
                workflow_id=workflow_id,
                cron=cron,
                next_run_at=run_at.isoformat() if run_at else None,
                inputs_json=json.dumps(inputs),
            )
        except Exception:
            logger.debug("Schedule storage error")

        return {
            "scheduled": True,
            "schedule_id": schedule_id,
            "workflow_id": workflow_id,
            "run_at": run_at.isoformat() if run_at else None,
            "cron": cron,
        }

    except Exception:
        logger.exception("Schedule workflow error")
        return {"scheduled": False, "error": "schedule_workflow_error"}
