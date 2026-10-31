"""Workflow adapter for loading completed deep research bundle references."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.Utils.template_render import apply_template_to_string
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_context_user_id,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.research._config import (
    DeepResearchLoadBundleConfig,
)


def _build_research_service():
    from tldw_Server_API.app.core.Research.service import ResearchService

    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)


def _resolve_run_id(validated: DeepResearchLoadBundleConfig, context: dict[str, Any]) -> str:
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


def _normalize_outline_titles(bundle: dict[str, Any]) -> list[str]:
    outline = bundle.get("outline")
    if isinstance(outline, dict):
        sections = outline.get("sections")
    elif isinstance(outline, list):
        sections = outline
    else:
        sections = []

    titles: list[str] = []
    if not isinstance(sections, list):
        return titles
    for section in sections:
        if isinstance(section, dict):
            title = str(section.get("title") or "").strip()
        else:
            title = str(section or "").strip()
        if title:
            titles.append(title)
    return titles


def _build_bundle_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    claims = bundle.get("claims")
    source_inventory = bundle.get("source_inventory")
    unresolved_questions = bundle.get("unresolved_questions")
    return {
        "question": str(bundle.get("question") or ""),
        "outline_titles": _normalize_outline_titles(bundle),
        "claim_count": len(claims) if isinstance(claims, list) else 0,
        "source_count": len(source_inventory) if isinstance(source_inventory, list) else 0,
        "unresolved_question_count": (
            len(unresolved_questions) if isinstance(unresolved_questions, list) else 0
        ),
    }


def _normalize_artifacts(snapshot: Any) -> list[dict[str, Any]]:
    artifacts = getattr(snapshot, "artifacts", []) or []
    normalized: list[dict[str, Any]] = []
    for artifact in artifacts:
        if hasattr(artifact, "model_dump"):
            item = artifact.model_dump()
        elif isinstance(artifact, dict):
            item = artifact
        else:
            item = {
                "artifact_name": getattr(artifact, "artifact_name", None),
                "artifact_version": getattr(artifact, "artifact_version", None),
                "content_type": getattr(artifact, "content_type", None),
                "phase": getattr(artifact, "phase", None),
                "job_id": getattr(artifact, "job_id", None),
            }
        normalized.append(
            {
                "artifact_name": item.get("artifact_name"),
                "artifact_version": item.get("artifact_version"),
                "content_type": item.get("content_type"),
                "phase": item.get("phase"),
                "job_id": item.get("job_id"),
            }
        )
    return normalized


def _write_bundle_ref_artifact(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
) -> None:
    add_artifact = context.get("add_artifact")
    if not callable(add_artifact):
        return

    artifact_dir = resolve_artifacts_dir(str(context.get("step_run_id") or "deep_research_load_bundle"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "deep_research_bundle_ref.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    add_artifact(
        type="deep_research_bundle_ref",
        uri=f"file://{artifact_path}",
        size_bytes=artifact_path.stat().st_size,
        mime_type="application/json",
    )


@registry.register(
    "deep_research_load_bundle",
    category="research",
    description=(
        "Loads references from a completed deep research run without returning the full bundle"
    ),
    parallelizable=False,
    tags=["research", "deep-research", "bundle"],
    config_model=DeepResearchLoadBundleConfig,
)
async def run_deep_research_load_bundle_adapter(
    config: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Load bundle references from a completed deep research session."""
    validated = DeepResearchLoadBundleConfig.model_validate(config or {})
    owner_user_id = resolve_context_user_id(context)
    if not owner_user_id:
        raise ValueError("missing_user_id")

    run_id = _resolve_run_id(validated, context)
    service = _build_research_service()
    session = service.get_session(owner_user_id=owner_user_id, session_id=run_id)
    if session.status != "completed":
        raise RuntimeError("deep_research_load_bundle is for completed runs only")

    bundle = service.get_bundle(owner_user_id=owner_user_id, session_id=run_id)
    snapshot = service.get_stream_snapshot(owner_user_id=owner_user_id, session_id=run_id)

    result = {
        "run_id": session.id,
        "status": session.status,
        "phase": session.phase,
        "control_state": session.control_state,
        "completed_at": session.completed_at,
        "bundle_url": f"/api/v1/research/runs/{session.id}/bundle",
        "bundle_summary": _build_bundle_summary(bundle),
        "artifacts": _normalize_artifacts(snapshot),
    }

    if validated.save_artifact is not False:
        _write_bundle_ref_artifact(result=result, context=context)

    return result
