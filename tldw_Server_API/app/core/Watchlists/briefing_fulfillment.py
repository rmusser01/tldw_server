"""Idempotent artifact fulfillment for completed Watchlists runs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Watchlists.briefing_contract import (
    BRIEFING_PIPELINE_KEY,
    briefing_selection_limit,
    get_briefing_contract,
)
from tldw_Server_API.app.services.outputs_service import (
    _build_output_filename,
    _outputs_dir_for_user,
    _resolve_output_path_for_user,
    build_items_context_from_content_items,
    render_output_template,
)

_STAGE_NAMES = (
    "collect",
    "select",
    "render_text",
    "persist_text",
    "compose_audio_script",
    "persist_audio_script",
    "generate_audio",
    "persist_audio",
    "deliver",
)
_AUDIO_STAGES = (
    "compose_audio_script",
    "persist_audio_script",
    "generate_audio",
    "persist_audio",
)


@dataclass(frozen=True)
class BriefingSelection:
    """One bounded, ordered selection shared by text and audio."""

    items: tuple[Any, ...]
    candidate_count: int
    selected_count: int
    omitted_count: int


@dataclass(frozen=True)
class FulfillmentResult:
    """Durable briefing occurrence projection returned to pipeline callers."""

    occurrence_id: int
    output_id: int | None
    audio_task_id: str | None
    artifact_status: str
    delivery_status: str
    selected_count: int
    omitted_count: int
    stages: dict[str, dict[str, Any]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage(
    status: str = "not_started",
    *,
    started_at: str | None = None,
    finished_at: str | None = None,
    code: str | None = None,
    retryable: bool = False,
    **details: Any,
) -> dict[str, Any]:
    return {
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "code": code,
        "retryable": retryable,
        **details,
    }


def _initial_stages(run: Any, *, audio_enabled: bool, delivery_configured: bool) -> dict[str, dict[str, Any]]:
    now = _utcnow_iso()
    stages = {name: _stage() for name in _STAGE_NAMES}
    stages["collect"] = _stage(
        "ready",
        started_at=getattr(run, "started_at", None),
        finished_at=getattr(run, "finished_at", None) or now,
    )
    if not audio_enabled:
        for name in _AUDIO_STAGES:
            stages[name] = _stage("skipped", finished_at=now, code="audio_not_selected")
    if not delivery_configured:
        stages["deliver"] = _stage("skipped", finished_at=now, code="delivery_not_configured")
    return stages


def _read_stages(
    occurrence: Any,
    *,
    run: Any,
    audio_enabled: bool,
    delivery_configured: bool,
) -> dict[str, dict[str, Any]]:
    defaults = _initial_stages(
        run,
        audio_enabled=audio_enabled,
        delivery_configured=delivery_configured,
    )
    try:
        stored = json.loads(getattr(occurrence, "stages_json", None) or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        stored = {}
    if not isinstance(stored, dict):
        return defaults
    for name, value in stored.items():
        if name in defaults and isinstance(value, dict):
            defaults[name] = {**defaults[name], **value}
    return defaults


def _transition(
    stages: dict[str, dict[str, Any]],
    name: str,
    status: str,
    *,
    code: str | None = None,
    retryable: bool = False,
    **details: Any,
) -> None:
    previous = stages.get(name, _stage())
    now = _utcnow_iso()
    started_at = previous.get("started_at")
    if status in {"running", "queued"} and not started_at:
        started_at = now
    finished_at = now if status in {"ready", "failed", "skipped", "cancelled"} else None
    stages[name] = _stage(
        status,
        started_at=started_at,
        finished_at=finished_at,
        code=code,
        retryable=retryable,
        **details,
    )


def _save_occurrence(watchlists_db: Any, occurrence_id: int, stages: dict[str, dict[str, Any]], **patch: Any) -> Any:
    return watchlists_db.update_briefing_occurrence(
        occurrence_id,
        stages=copy.deepcopy(stages),
        **patch,
    )


def _result(occurrence: Any, stages: dict[str, dict[str, Any]]) -> FulfillmentResult:
    return FulfillmentResult(
        occurrence_id=int(occurrence.id),
        output_id=int(occurrence.output_id) if occurrence.output_id is not None else None,
        audio_task_id=str(occurrence.audio_task_id) if occurrence.audio_task_id else None,
        artifact_status=str(occurrence.artifact_status),
        delivery_status=str(occurrence.delivery_status),
        selected_count=int(occurrence.selected_count or 0),
        omitted_count=int(occurrence.omitted_count or 0),
        stages=copy.deepcopy(stages),
    )


def _job_output_prefs(job: Any) -> dict[str, Any]:
    raw = getattr(job, "output_prefs_json", None)
    if isinstance(raw, Mapping):
        return copy.deepcopy(dict(raw))
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _occurrence_key(user_id: int, job: Any, run: Any, contract: Mapping[str, Any]) -> str:
    version = int(contract.get("version") or 1)
    return f"user:{user_id}:job:{int(job.id)}:run:{int(run.id)}:v{version}"


def audio_request_id_for_occurrence(occurrence_key: str, *, output_version: int = 1) -> str:
    """Return the stable opaque request ID for an occurrence or explicit version."""
    source = occurrence_key if output_version <= 1 else f"{occurrence_key}:output:v{output_version}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return f"wla_{digest[:32]}"


def _stable_item_id(item: Any) -> str:
    raw = item.get("id") if isinstance(item, Mapping) else getattr(item, "id", "")
    try:
        return f"{int(raw):020d}"
    except (TypeError, ValueError):
        return str(raw or "")


def _published_at(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("published_at") or item.get("created_at") or "")
    return str(getattr(item, "published_at", None) or getattr(item, "created_at", None) or "")


def _normalize_items(rows: tuple[Any, ...]) -> list[dict[str, Any]]:
    normalized = build_items_context_from_content_items(rows)
    for row, item in zip(rows, normalized):
        stable_id = row.get("id") if isinstance(row, Mapping) else getattr(row, "id", None)
        source_id = row.get("source_id") if isinstance(row, Mapping) else getattr(row, "source_id", None)
        item["id"] = stable_id
        item["source_id"] = source_id
    return normalized


async def _load_selection(
    watchlists_db: Any,
    *,
    run_id: int,
    limit: int,
) -> BriefingSelection:
    rows, total = await asyncio.to_thread(
        watchlists_db.list_items,
        run_id=run_id,
        status="ingested",
        sort="published_desc",
        limit=limit,
        offset=0,
    )
    ordered = tuple(
        sorted(
            rows,
            key=lambda item: (_published_at(item), _stable_item_id(item)),
            reverse=True,
        )
    )
    candidate_count = max(len(ordered), int(total or 0))
    return BriefingSelection(
        items=ordered,
        candidate_count=candidate_count,
        selected_count=len(ordered),
        omitted_count=max(0, candidate_count - len(ordered)),
    )


def _run_stats(run: Any) -> dict[str, Any]:
    raw = getattr(run, "stats_json", None)
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _source_counts(run: Any) -> dict[str, int]:
    counts = {"succeeded": 0, "failed": 0, "deferred": 0}
    statuses = _run_stats(run).get("source_statuses")
    if not isinstance(statuses, list):
        return counts
    for entry in statuses:
        status = str(entry.get("status") or "") if isinstance(entry, Mapping) else ""
        if status == "deferred" or status.startswith("not_modified_backoff"):
            counts["deferred"] += 1
        elif status.startswith(("error", "partial:")):
            counts["failed"] += 1
        else:
            counts["succeeded"] += 1
    return counts


def no_material_updates_markdown(
    *,
    title: str,
    checked_at: str,
    next_run_at: str | None,
    source_counts: Mapping[str, int],
) -> str:
    """Render deterministic zero-item status Markdown without an LLM."""
    return "\n".join(
        (
            f"# {title}",
            "",
            "No qualifying new material was found.",
            "",
            f"- Sources succeeded: {int(source_counts.get('succeeded', 0))}",
            f"- Sources failed: {int(source_counts.get('failed', 0))}",
            f"- Sources deferred: {int(source_counts.get('deferred', 0))}",
            f"- Checked: {checked_at}",
            f"- Next run: {next_run_at or 'Not scheduled'}",
            "",
        )
    )


def _render_briefing_text(
    *,
    title: str,
    items: list[dict[str, Any]],
    contract: Mapping[str, Any],
    checked_at: str,
    next_run_at: str | None,
    source_counts: Mapping[str, int],
) -> str:
    if not items:
        return no_material_updates_markdown(
            title=title,
            checked_at=checked_at,
            next_run_at=next_run_at,
            source_counts=source_counts,
        )

    text = contract["text"]
    template_name = str(text.get("template_name") or "").strip()
    context = {
        "title": title,
        "generated_at": checked_at,
        "items": items,
        "item_count": len(items),
    }
    if template_name:
        from tldw_Server_API.app.core.Watchlists import template_store

        template = template_store.load_template(template_name)
        return render_output_template(template.content, context)

    lines = [f"# {title}", ""]
    for index, item in enumerate(items, 1):
        item_title = str(item.get("title") or "Untitled")
        item_url = str(item.get("url") or "")
        summary = str(item.get("summary") or "")
        lines.append(f"{index}. [{item_title}]({item_url})" if item_url else f"{index}. {item_title}")
        if summary:
            lines.append(f"   {summary[:200]}")
        lines.append("")
    return "\n".join(lines)


def _provenance(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.get("id"),
            "source_id": item.get("source_id"),
            "url": item.get("url"),
            "published_at": item.get("published_at"),
        }
        for item in items
    ]


def _metadata(
    *,
    occurrence: Any,
    contract: Mapping[str, Any],
    selection: BriefingSelection,
    items: list[dict[str, Any]],
    run: Any,
    job: Any,
    checked_at: str,
    next_run_at: str | None,
    source_counts: Mapping[str, int],
    output_version: int = 1,
) -> dict[str, Any]:
    editorial = copy.deepcopy(dict(contract["editorial"]))
    audio_enabled = bool(contract["audio"]["enabled"])
    return {
        "origin": "watchlists",
        "generation_mode": "auto_output",
        "fulfillment_mode": "briefing_occurrence",
        "auto_output": True,
        "run_id": int(run.id),
        "job_id": int(job.id),
        "occurrence_id": int(occurrence.id),
        "occurrence_key": str(occurrence.occurrence_key),
        "output_version": output_version,
        "candidate_count": selection.candidate_count,
        "selected_count": selection.selected_count,
        "included_count": selection.selected_count,
        "omitted_count": selection.omitted_count,
        "item_count": selection.selected_count,
        "item_ids": [item.get("id") for item in items],
        "briefing_items": copy.deepcopy(items),
        "no_material_updates": selection.selected_count == 0,
        "checked_at": checked_at,
        "next_run_at": next_run_at,
        "source_counts": dict(source_counts),
        "editorial": editorial,
        "program_format": editorial.get("program_format"),
        "outcome_noun": editorial.get("outcome_noun"),
        "show_name": editorial.get("show_name"),
        "show_identity": {
            "name": editorial.get("show_name"),
            "premise": editorial.get("show_premise"),
        },
        "provenance": _provenance(items),
        "ai_generated_speech": audio_enabled,
        "speech_disclosure": "Synthetic AI-generated speech" if audio_enabled else None,
        "audio_output_version": output_version,
    }


async def _persist_text_output(
    *,
    user_id: int,
    collections_db: Any,
    occurrence: Any,
    job: Any,
    run: Any,
    contract: Mapping[str, Any],
    content: str,
    metadata: Mapping[str, Any],
    output_version: int = 1,
) -> int:
    output_format = str(contract["text"].get("format") or "md")
    output_type = str(contract["text"].get("type") or "briefing_markdown")
    show_name = str(contract["editorial"].get("show_name") or "").strip()
    title = show_name or f"{getattr(job, 'name', 'Watchlist')} briefing"
    suffix = f"occurrence-{int(occurrence.id)}-v{output_version}"
    digest = hashlib.sha256(str(occurrence.occurrence_key).encode("utf-8")).hexdigest()[:12]
    filename = _build_output_filename(title, suffix, digest, output_format)
    output_dir = _outputs_dir_for_user(user_id)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    path: Path = _resolve_output_path_for_user(user_id, filename)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")
    artifact = await asyncio.to_thread(
        collections_db.create_output_artifact,
        type_=output_type,
        title=title,
        format_=output_format,
        storage_path=filename,
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        job_id=int(job.id),
        run_id=int(run.id),
    )
    return int(artifact.id)


def _existing_output(collections_db: Any, occurrence: Any) -> tuple[Any, dict[str, Any]] | None:
    if occurrence.output_id is None:
        return None
    try:
        row = collections_db.get_output_artifact(int(occurrence.output_id))
        metadata = json.loads(getattr(row, "metadata_json", None) or "{}")
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    if metadata.get("occurrence_id") != int(occurrence.id):
        return None
    if metadata.get("occurrence_key") != str(occurrence.occurrence_key):
        return None
    return row, metadata


async def trigger_audio_briefing(**kwargs: Any) -> Any:
    """Resolve the existing audio bridge lazily so legacy patches still apply."""
    from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
        trigger_audio_briefing as trigger,
    )

    return await trigger(**kwargs)


async def _update_output_audio_metadata(
    collections_db: Any,
    output_id: int,
    audio_result: Any,
    *,
    output_version: int,
) -> None:
    from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
        apply_audio_briefing_result_metadata,
    )

    row = await asyncio.to_thread(collections_db.get_output_artifact, output_id)
    try:
        metadata = json.loads(getattr(row, "metadata_json", None) or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["audio_output_version"] = output_version
    apply_audio_briefing_result_metadata(metadata, audio_result, requested=True, retry=output_version > 1)
    await asyncio.to_thread(
        collections_db.update_output_artifact_metadata,
        output_id,
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )


def _delivery_configured(contract: Mapping[str, Any]) -> bool:
    delivery = contract.get("delivery")
    if not isinstance(delivery, Mapping):
        return False
    return any(
        bool(config.get("enabled"))
        for name, config in delivery.items()
        if name != "reports" and isinstance(config, Mapping)
    )


async def _submit_audio(
    *,
    user_id: int,
    job: Any,
    run: Any,
    contract: Mapping[str, Any],
    occurrence: Any,
    stages: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
    collections_db: Any,
    watchlists_db: Any,
    scheduler: Any | None,
    output_version: int,
) -> FulfillmentResult:
    _transition(stages, "compose_audio_script", "running")
    occurrence = _save_occurrence(
        watchlists_db,
        int(occurrence.id),
        stages,
        artifact_status="running",
    )
    request_id = audio_request_id_for_occurrence(
        str(occurrence.occurrence_key),
        output_version=output_version,
    )
    no_material = not items
    audio_items = items or [
        {
            "id": f"status:{occurrence.id}",
            "title": "No material updates",
            "summary": "No qualifying new material was found during this source check.",
            "url": "",
            "status_kind": "no_material_updates",
        }
    ]
    try:
        audio_result = await trigger_audio_briefing(
            user_id=user_id,
            job_id=int(job.id),
            run_id=int(run.id),
            output_prefs={BRIEFING_PIPELINE_KEY: copy.deepcopy(dict(contract))},
            db=watchlists_db,
            scheduler=scheduler,
            audio_request_id=request_id,
            items=copy.deepcopy(audio_items),
            occurrence_id=int(occurrence.id),
            output_id=int(occurrence.output_id),
            editorial=copy.deepcopy(dict(contract["editorial"])),
            status_audio=no_material,
        )
    except asyncio.CancelledError:
        _transition(stages, "compose_audio_script", "cancelled", code="audio_submit_cancelled", retryable=True)
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="cancelled",
        )
        raise
    except Exception:  # noqa: BLE001 - orchestration boundary persists provider failures
        _transition(stages, "compose_audio_script", "failed", code="audio_submit_failed", retryable=True)
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="failed",
        )
        return _result(occurrence, stages)

    if not audio_result.submitted:
        code = str(audio_result.reason or audio_result.status)
        _transition(
            stages,
            "compose_audio_script",
            "failed",
            code=code,
            retryable=audio_result.status in {"queue_unavailable", "enqueue_failed"},
            trigger_status=audio_result.status,
            audio_request_id=audio_result.audio_request_id or request_id,
        )
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="failed",
        )
        return _result(occurrence, stages)

    try:
        await _update_output_audio_metadata(
            collections_db,
            int(occurrence.output_id),
            audio_result,
            output_version=output_version,
        )
    except Exception:  # noqa: BLE001 - orchestration boundary persists metadata failures
        _transition(
            stages,
            "compose_audio_script",
            "failed",
            code="audio_metadata_persist_failed",
            retryable=True,
            audio_request_id=request_id,
            task_id=audio_result.task_id,
        )
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="failed",
            audio_task_id=audio_result.task_id,
        )
        return _result(occurrence, stages)

    _transition(
        stages,
        "compose_audio_script",
        "queued",
        audio_request_id=request_id,
        task_id=audio_result.task_id,
    )
    occurrence = _save_occurrence(
        watchlists_db,
        int(occurrence.id),
        stages,
        artifact_status="running",
        audio_task_id=audio_result.task_id,
    )
    return _result(occurrence, stages)


async def fulfill_watchlist_briefing(
    *,
    user_id: int,
    job: Any,
    run: Any,
    watchlists_db: Any,
    collections_db: Any,
    scheduler: Any | None = None,
) -> FulfillmentResult:
    """Fulfill one logical Watchlists occurrence without duplicate artifacts."""
    output_prefs = _job_output_prefs(job)
    contract = get_briefing_contract(
        output_prefs,
        scheduled=bool(getattr(job, "schedule_expr", None)),
    )
    occurrence_key = _occurrence_key(user_id, job, run, contract)
    occurrence = watchlists_db.create_or_get_briefing_occurrence(
        run_id=int(run.id),
        occurrence_key=occurrence_key,
        contract_json=json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    audio_enabled = bool(contract["audio"]["enabled"])
    delivery_configured = _delivery_configured(contract)
    stages = _read_stages(
        occurrence,
        run=run,
        audio_enabled=audio_enabled,
        delivery_configured=delivery_configured,
    )
    expected_delivery_status = "waiting_for_artifacts" if delivery_configured else "not_configured"
    if occurrence.delivery_status != expected_delivery_status:
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            delivery_status=expected_delivery_status,
        )
    existing = _existing_output(collections_db, occurrence)
    if existing is not None:
        if not audio_enabled or occurrence.audio_task_id:
            return _result(occurrence, stages)
        if stages["compose_audio_script"]["status"] == "failed":
            return _result(occurrence, stages)

    try:
        _transition(stages, "select", "running")
        occurrence = _save_occurrence(watchlists_db, int(occurrence.id), stages)
        selection = await _load_selection(
            watchlists_db,
            run_id=int(run.id),
            limit=briefing_selection_limit(contract),
        )
    except asyncio.CancelledError:
        _transition(stages, "select", "cancelled", code="selection_cancelled", retryable=True)
        _save_occurrence(watchlists_db, int(occurrence.id), stages, artifact_status="cancelled")
        raise
    except Exception:  # noqa: BLE001 - orchestration boundary persists selection failures
        _transition(stages, "select", "failed", code="selection_load_failed", retryable=True)
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="failed",
        )
        return _result(occurrence, stages)

    _transition(
        stages,
        "select",
        "ready",
        candidate_count=selection.candidate_count,
        selected_count=selection.selected_count,
        omitted_count=selection.omitted_count,
    )
    occurrence = _save_occurrence(
        watchlists_db,
        int(occurrence.id),
        stages,
        selected_count=selection.selected_count,
        omitted_count=selection.omitted_count,
    )
    items = _normalize_items(selection.items)
    checked_at = str(getattr(run, "finished_at", None) or _utcnow_iso())
    try:
        refreshed_job = watchlists_db.get_job(int(job.id))
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        refreshed_job = job
    next_run_at = getattr(refreshed_job, "next_run_at", None) or getattr(job, "next_run_at", None)
    source_counts = _source_counts(run)
    title = str(contract["editorial"].get("show_name") or getattr(job, "name", "Watchlist briefing"))

    _transition(stages, "render_text", "running")
    occurrence = _save_occurrence(watchlists_db, int(occurrence.id), stages)
    try:
        content = _render_briefing_text(
            title=title,
            items=items,
            contract=contract,
            checked_at=checked_at,
            next_run_at=next_run_at,
            source_counts=source_counts,
        )
    except Exception:  # noqa: BLE001 - orchestration boundary persists renderer failures
        _transition(stages, "render_text", "failed", code="text_render_failed", retryable=True)
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="failed",
        )
        return _result(occurrence, stages)
    _transition(stages, "render_text", "ready")
    occurrence = _save_occurrence(watchlists_db, int(occurrence.id), stages)

    metadata = _metadata(
        occurrence=occurrence,
        contract=contract,
        selection=selection,
        items=items,
        run=run,
        job=job,
        checked_at=checked_at,
        next_run_at=next_run_at,
        source_counts=source_counts,
    )
    _transition(stages, "persist_text", "running")
    occurrence = _save_occurrence(watchlists_db, int(occurrence.id), stages)
    try:
        output_id = await _persist_text_output(
            user_id=user_id,
            collections_db=collections_db,
            occurrence=occurrence,
            job=job,
            run=run,
            contract=contract,
            content=content,
            metadata=metadata,
        )
    except asyncio.CancelledError:
        _transition(stages, "persist_text", "cancelled", code="text_persist_cancelled", retryable=True)
        _save_occurrence(watchlists_db, int(occurrence.id), stages, artifact_status="cancelled")
        raise
    except Exception:  # noqa: BLE001 - orchestration boundary persists storage failures
        _transition(stages, "persist_text", "failed", code="text_persist_failed", retryable=True)
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="failed",
        )
        return _result(occurrence, stages)
    _transition(stages, "persist_text", "ready", artifact_id=output_id)
    occurrence = _save_occurrence(
        watchlists_db,
        int(occurrence.id),
        stages,
        output_id=output_id,
        artifact_status="running" if audio_enabled else "ready",
        delivery_status="waiting_for_artifacts" if delivery_configured else "not_configured",
    )

    if not audio_enabled:
        return _result(occurrence, stages)
    return await _submit_audio(
        user_id=user_id,
        job=job,
        run=run,
        contract=contract,
        occurrence=occurrence,
        stages=stages,
        items=items,
        collections_db=collections_db,
        watchlists_db=watchlists_db,
        scheduler=scheduler,
        output_version=1,
    )


async def retry_briefing_stage(
    *,
    user_id: int,
    occurrence_id: int,
    stage: str,
    watchlists_db: Any,
    collections_db: Any,
    scheduler: Any | None = None,
    regenerate: bool = False,
) -> FulfillmentResult:
    """Retry one failed/missing artifact stage while reusing durable outputs."""
    if stage not in {"select", "render_text", "persist_text", "compose_audio_script"}:
        raise ValueError("unsupported_briefing_retry_stage")
    occurrence = watchlists_db.get_briefing_occurrence(occurrence_id)
    job = watchlists_db.get_job(int(occurrence.job_id))
    run = watchlists_db.get_run(int(occurrence.run_id))
    try:
        contract = json.loads(occurrence.contract_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_briefing_occurrence_contract") from exc
    if not isinstance(contract, dict):
        raise ValueError("invalid_briefing_occurrence_contract")
    audio_enabled = bool(contract.get("audio", {}).get("enabled"))
    stages = _read_stages(
        occurrence,
        run=run,
        audio_enabled=audio_enabled,
        delivery_configured=_delivery_configured(contract),
    )
    current_status = stages[stage]["status"]
    if not regenerate and current_status not in {"failed", "not_started"}:
        return _result(occurrence, stages)

    if stage in {"select", "render_text", "persist_text"}:
        _transition(stages, stage, "running")
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            artifact_status="running",
        )
        try:
            selection = await _load_selection(
                watchlists_db,
                run_id=int(run.id),
                limit=briefing_selection_limit(contract),
            )
            items = _normalize_items(selection.items)
            checked_at = str(getattr(run, "finished_at", None) or _utcnow_iso())
            next_run_at = getattr(job, "next_run_at", None)
            source_counts = _source_counts(run)
            title = str(contract["editorial"].get("show_name") or getattr(job, "name", "Watchlist briefing"))
            content = _render_briefing_text(
                title=title,
                items=items,
                contract=contract,
                checked_at=checked_at,
                next_run_at=next_run_at,
                source_counts=source_counts,
            )
            existing = _existing_output(collections_db, occurrence)
            existing_metadata = existing[1] if existing else {}
            output_version = int(existing_metadata.get("output_version") or 1) + (1 if regenerate else 0)
            metadata = _metadata(
                occurrence=occurrence,
                contract=contract,
                selection=selection,
                items=items,
                run=run,
                job=job,
                checked_at=checked_at,
                next_run_at=next_run_at,
                source_counts=source_counts,
                output_version=output_version,
            )
            output_id = await _persist_text_output(
                user_id=user_id,
                collections_db=collections_db,
                occurrence=occurrence,
                job=job,
                run=run,
                contract=contract,
                content=content,
                metadata=metadata,
                output_version=output_version,
            )
        except asyncio.CancelledError:
            _transition(stages, stage, "cancelled", code=f"{stage}_retry_cancelled", retryable=True)
            _save_occurrence(
                watchlists_db,
                int(occurrence.id),
                stages,
                artifact_status="cancelled",
            )
            raise
        except Exception:  # noqa: BLE001 - retry boundary persists every stage failure
            failure_code = {
                "select": "selection_load_failed",
                "render_text": "text_render_failed",
                "persist_text": "text_persist_failed",
            }[stage]
            _transition(stages, stage, "failed", code=failure_code, retryable=True)
            occurrence = _save_occurrence(
                watchlists_db,
                int(occurrence.id),
                stages,
                artifact_status="failed",
            )
            return _result(occurrence, stages)
        _transition(
            stages,
            "select",
            "ready",
            candidate_count=selection.candidate_count,
            selected_count=selection.selected_count,
            omitted_count=selection.omitted_count,
        )
        _transition(stages, "render_text", "ready")
        _transition(stages, "persist_text", "ready", artifact_id=output_id)
        occurrence = _save_occurrence(
            watchlists_db,
            int(occurrence.id),
            stages,
            output_id=output_id,
            artifact_status="running" if audio_enabled else "ready",
            selected_count=selection.selected_count,
            omitted_count=selection.omitted_count,
            delivery_status=("waiting_for_artifacts" if _delivery_configured(contract) else "not_configured"),
        )
        return _result(occurrence, stages)

    if not audio_enabled:
        raise ValueError("audio_not_selected")
    existing = _existing_output(collections_db, occurrence)
    if existing is None:
        raise ValueError("ready_text_output_required")
    _, metadata = existing
    items = metadata.get("briefing_items")
    if not isinstance(items, list):
        raise ValueError("briefing_selection_missing")
    current_version = int(metadata.get("audio_output_version") or 1)
    output_version = current_version + 1 if regenerate else current_version
    return await _submit_audio(
        user_id=user_id,
        job=job,
        run=run,
        contract=contract,
        occurrence=occurrence,
        stages=stages,
        items=copy.deepcopy(items),
        collections_db=collections_db,
        watchlists_db=watchlists_db,
        scheduler=scheduler,
        output_version=output_version,
    )


__all__ = [
    "BriefingSelection",
    "FulfillmentResult",
    "audio_request_id_for_occurrence",
    "fulfill_watchlist_briefing",
    "no_material_updates_markdown",
    "retry_briefing_stage",
]
