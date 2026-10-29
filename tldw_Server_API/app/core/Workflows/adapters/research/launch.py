"""Workflow adapter for launching deep research sessions."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.Utils.template_render import apply_template_to_string
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_context_user_id,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.research._config import DeepResearchConfig


def _build_research_service():
    from tldw_Server_API.app.core.Research.service import ResearchService

    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)


def _normalize_query(raw_query: str, context: dict[str, Any]) -> str:
    rendered = apply_template_to_string(raw_query, context) if isinstance(raw_query, str) else raw_query
    query = str(rendered or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    return query


def _write_launch_artifact(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
) -> None:
    add_artifact = context.get("add_artifact")
    if not callable(add_artifact):
        return

    artifact_dir = resolve_artifacts_dir(str(context.get("step_run_id") or "deep_research_launch"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "deep_research_launch.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    add_artifact(
        type="deep_research_launch",
        uri=f"file://{artifact_path}",
        size_bytes=artifact_path.stat().st_size,
        mime_type="application/json",
    )


@registry.register(
    "deep_research",
    category="research",
    description=(
        "Launch a deep research session and return its run reference; "
        "does not wait for completion"
    ),
    parallelizable=False,
    tags=["research", "deep-research", "launch"],
    config_model=DeepResearchConfig,
)
async def run_deep_research_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Launch a deep research session for downstream workflow use."""
    validated = DeepResearchConfig.model_validate(config or {})
    owner_user_id = resolve_context_user_id(context)
    if not owner_user_id:
        raise ValueError("missing_user_id")

    query = _normalize_query(validated.query, context)
    service = _build_research_service()
    session = service.create_session(
        owner_user_id=owner_user_id,
        query=query,
        source_policy=validated.source_policy,
        autonomy_mode=validated.autonomy_mode,
        limits_json=validated.limits_json,
        provider_overrides=validated.provider_overrides,
    )

    result = {
        "run_id": session.id,
        "status": session.status,
        "phase": session.phase,
        "control_state": session.control_state,
        "console_url": f"/research?run={session.id}",
        "bundle_url": f"/api/v1/research/runs/{session.id}/bundle",
        "query": query,
        "source_policy": validated.source_policy,
        "autonomy_mode": validated.autonomy_mode,
    }

    if validated.save_artifact is not False:
        _write_launch_artifact(result=result, context=context)

    return result
