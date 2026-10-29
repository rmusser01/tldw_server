"""Workflow adapter for selecting canonical fields from a completed deep research bundle."""

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
    DeepResearchSelectBundleFieldsConfig,
)

MAX_SELECTED_FIELDS_BYTES = 256 * 1024


def _build_research_service():
    from tldw_Server_API.app.core.Research.service import ResearchService

    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)


def _resolve_run_id(
    validated: DeepResearchSelectBundleFieldsConfig, context: dict[str, Any]
) -> str:
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


def _serialize_selected_fields(selected_fields: dict[str, Any]) -> str:
    serialized = json.dumps(selected_fields, sort_keys=True)
    if len(serialized.encode("utf-8")) > MAX_SELECTED_FIELDS_BYTES:
        raise RuntimeError(
            "deep_research_select_bundle_fields output exceeds inline size limit; "
            "select fewer fields or use deep_research_load_bundle"
        )
    return serialized


def _write_selected_fields_artifact(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
) -> None:
    add_artifact = context.get("add_artifact")
    if not callable(add_artifact):
        return

    artifact_dir = resolve_artifacts_dir(
        str(context.get("step_run_id") or "deep_research_select_bundle_fields")
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "deep_research_selected_fields.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    add_artifact(
        type="deep_research_selected_fields",
        uri=f"file://{artifact_path}",
        size_bytes=artifact_path.stat().st_size,
        mime_type="application/json",
    )


@registry.register(
    "deep_research_select_bundle_fields",
    category="research",
    description=(
        "Loads selected canonical bundle fields from a completed deep research run and "
        "returns null for missing allowed fields"
    ),
    parallelizable=False,
    tags=["research", "deep-research", "bundle", "selection"],
    config_model=DeepResearchSelectBundleFieldsConfig,
)
async def run_deep_research_select_bundle_fields_adapter(
    config: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Select a fixed allowlist of top-level fields from a completed research bundle."""
    validated = DeepResearchSelectBundleFieldsConfig.model_validate(config or {})
    owner_user_id = resolve_context_user_id(context)
    if not owner_user_id:
        raise ValueError("missing_user_id")

    run_id = _resolve_run_id(validated, context)
    service = _build_research_service()
    session = service.get_session(owner_user_id=owner_user_id, session_id=run_id)
    if session.status != "completed":
        raise RuntimeError("deep_research_select_bundle_fields is for completed runs only")

    bundle = service.get_bundle(owner_user_id=owner_user_id, session_id=run_id)
    selected_fields = {field_name: bundle.get(field_name) for field_name in validated.fields}
    _serialize_selected_fields(selected_fields)

    result = {
        "run_id": session.id,
        "status": session.status,
        "selected_fields": selected_fields,
    }

    if validated.save_artifact is not False:
        _write_selected_fields_artifact(result=result, context=context)

    return result
