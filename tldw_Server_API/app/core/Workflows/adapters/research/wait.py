"""Workflow adapter for waiting on deep research sessions."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from tldw_Server_API.app.core.Utils.template_render import apply_template_to_string
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_context_user_id,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.research._config import (
    DeepResearchWaitConfig,
)

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_CHECKPOINT_PHASES = {
    "awaiting_plan_review",
    "awaiting_source_review",
    "awaiting_outline_review",
}


def _build_research_service():
    from tldw_Server_API.app.core.Research.service import ResearchService

    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)


def _now() -> float:
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _resolve_run_id(validated: DeepResearchWaitConfig, context: dict[str, Any]) -> str:
    raw_run_id = validated.run_id
    rendered_run_id = (
        apply_template_to_string(raw_run_id, context)
        if isinstance(raw_run_id, str)
        else raw_run_id
    )
    run_id = str(rendered_run_id or "").strip()
    if run_id:
        return run_id
    run_obj = validated.run if isinstance(validated.run, dict) else {}
    run_obj_id = str(run_obj.get("run_id") or "").strip()
    if run_obj_id:
        return run_obj_id
    raise ValueError("run_id must not be empty")


def _write_wait_artifact(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
) -> None:
    add_artifact = context.get("add_artifact")
    if not callable(add_artifact):
        return

    artifact_dir = resolve_artifacts_dir(str(context.get("step_run_id") or "deep_research_wait"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "deep_research_wait.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    add_artifact(
        type="deep_research_wait",
        uri=f"file://{artifact_path}",
        size_bytes=artifact_path.stat().st_size,
        mime_type="application/json",
    )


@registry.register(
    "deep_research_wait",
    category="research",
    description=(
        "waits for a launched deep-research run to finish and can return the final bundle"
    ),
    parallelizable=False,
    tags=["research", "deep-research", "wait"],
    config_model=DeepResearchWaitConfig,
)
async def run_deep_research_wait_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Wait for a deep research session to complete and return terminal metadata."""
    validated = DeepResearchWaitConfig.model_validate(config or {})
    owner_user_id = resolve_context_user_id(context)
    if not owner_user_id:
        raise ValueError("missing_user_id")

    is_cancelled = context.get("is_cancelled")
    if callable(is_cancelled) and is_cancelled():
        return {"__status__": "cancelled"}

    run_id = _resolve_run_id(validated, context)
    timeout_seconds = int(validated.timeout_seconds or 300)
    prev = context.get("prev") if isinstance(context.get("prev"), dict) else {}
    elapsed_poll_seconds = float(prev.get("active_poll_seconds") or 0.0)
    service = _build_research_service()

    while True:
        if callable(is_cancelled) and is_cancelled():
            return {"__status__": "cancelled"}

        session = service.get_session(
            owner_user_id=owner_user_id,
            session_id=run_id,
        )
        if session.status in _TERMINAL_STATUSES:
            bundle = None
            if session.status == "completed" and validated.include_bundle:
                try:
                    bundle = service.get_bundle(
                        owner_user_id=owner_user_id,
                        session_id=run_id,
                    )
                except KeyError:
                    bundle = None

            result = {
                "run_id": session.id,
                "status": session.status,
                "phase": session.phase,
                "control_state": session.control_state,
                "completed_at": session.completed_at,
                "bundle_url": f"/api/v1/research/runs/{session.id}/bundle",
                "bundle": bundle,
            }

            if session.status == "failed" and validated.fail_on_failed:
                raise RuntimeError("research_run_failed")
            if session.status == "cancelled" and validated.fail_on_cancelled:
                raise RuntimeError("research_run_cancelled")

            if validated.save_artifact is not False:
                _write_wait_artifact(result=result, context=context)
            return result

        if session.status == "waiting_human" and session.phase in _CHECKPOINT_PHASES:
            checkpoint_id = str(getattr(session, "latest_checkpoint_id", "") or "").strip() or None
            checkpoint_type = None
            try:
                snapshot = service.get_stream_snapshot(
                    owner_user_id=owner_user_id,
                    session_id=run_id,
                )
                checkpoint = getattr(snapshot, "checkpoint", None)
                if isinstance(checkpoint, dict):
                    checkpoint_type = str(checkpoint.get("checkpoint_type") or "").strip() or None
                elif checkpoint is not None:
                    checkpoint_type = str(getattr(checkpoint, "checkpoint_type", "") or "").strip() or None
                    checkpoint_id = checkpoint_id or str(getattr(checkpoint, "checkpoint_id", "") or "").strip() or None
            except (AttributeError, KeyError, TypeError, ValueError):
                checkpoint_type = None

            return {
                "__status__": "waiting_human",
                "reason": "research_checkpoint",
                "run_id": session.id,
                "research_phase": session.phase,
                "research_control_state": session.control_state,
                "research_checkpoint_id": checkpoint_id,
                "research_checkpoint_type": checkpoint_type,
                "research_console_url": f"/research?run={session.id}",
                "active_poll_seconds": elapsed_poll_seconds,
            }

        if elapsed_poll_seconds >= timeout_seconds:
            raise TimeoutError("deep_research_wait_timed_out")
        before_sleep = _now()
        await _sleep(validated.poll_interval_seconds)
        elapsed_poll_seconds += max(0.0, _now() - before_sleep)
