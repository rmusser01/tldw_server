"""Audio briefing workflow bridge.

Triggers the audio briefing workflow pipeline after a watchlist run completes
when `briefing_pipeline.audio.enabled` is true, with legacy preferences normalized
through the canonical briefing contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from loguru import logger
from starlette.concurrency import run_in_threadpool

from tldw_Server_API.app.core.TTS.tts_request_resolution import resolve_tts_request_defaults
from tldw_Server_API.app.core.Watchlists.briefing_contract import (
    briefing_selection_limit,
    get_briefing_contract,
)

AudioBriefingTriggerStatus = Literal[
    "disabled",
    "submitted",
    "skipped_no_items",
    "configuration_required",
    "queue_unavailable",
    "enqueue_failed",
]


@dataclass(frozen=True)
class AudioBriefingTriggerResult:
    """Outcome contract for requesting a Watchlists audio briefing workflow.

    Attributes:
        status: Normalized trigger outcome persisted to output metadata.
        task_id: Scheduler task ID when workflow submission succeeds.
        audio_request_id: Stable request ID for correlating workflow inputs and run state.
        reason: Stable diagnostic code for non-submitted or failed outcomes.
    """

    status: AudioBriefingTriggerStatus
    task_id: str | None = None
    audio_request_id: str | None = None
    reason: str | None = None

    @property
    def submitted(self) -> bool:
        return self.status == "submitted" and bool(self.task_id)


def persisted_audio_briefing_status(result: AudioBriefingTriggerResult) -> str:
    """Return the externally visible status for an audio briefing trigger."""
    return "queued" if result.submitted else result.status


def apply_audio_briefing_result_metadata(
    target: MutableMapping[str, Any],
    result: AudioBriefingTriggerResult,
    *,
    requested: bool | None = None,
    retry: bool = False,
) -> str:
    """Persist trigger result fields while clearing stale task/reason values."""
    if requested is not None:
        target["audio_briefing_requested"] = requested

    status = persisted_audio_briefing_status(result)
    target["audio_briefing_status"] = status
    target.pop("audio_briefing_error", None)

    if result.task_id:
        target["audio_briefing_task_id"] = result.task_id
        if retry:
            target["audio_briefing_retry_task_id"] = result.task_id
    else:
        target.pop("audio_briefing_task_id", None)
        if retry:
            target.pop("audio_briefing_retry_task_id", None)

    if result.reason:
        target["audio_briefing_reason"] = result.reason
    else:
        target.pop("audio_briefing_reason", None)

    if result.audio_request_id:
        target["audio_request_id"] = result.audio_request_id
    else:
        target.pop("audio_request_id", None)

    return status


# ---------------------------------------------------------------------------
# Built-in workflow definition
# ---------------------------------------------------------------------------

AUDIO_BRIEFING_WORKFLOW_DEF: dict[str, Any] = {
    "name": "audio_briefing",
    "version": 1,
    "description": "Spoken-word multi-voice audio briefing from watchlist items",
    "steps": [
        {
            "id": "compose_script",
            "type": "audio_briefing_compose",
            "config": {
                "items": "{{ inputs.items }}",
                "target_audio_minutes": "{{ inputs.target_audio_minutes }}",
                "output_language": "{{ inputs.audio_language }}",
                "provider": "{{ inputs.llm_provider }}",
                "model": "{{ inputs.llm_model }}",
                "multi_voice": True,
                "voice_map": "{{ inputs.voice_map }}",
                "audio_cast": "{{ inputs.audio_cast }}",
                "persona_summarize": "{{ inputs.persona_summarize }}",
                "persona_id": "{{ inputs.persona_id }}",
                "persona_provider": "{{ inputs.persona_provider }}",
                "persona_model": "{{ inputs.persona_model }}",
                "editorial": "{{ inputs.editorial }}",
                "is_no_material_update": "{{ inputs.is_no_material_update }}",
            },
            "timeout_seconds": 120,
        },
        {
            "id": "clean_script",
            "type": "text_clean",
            "config": {
                "operations": [
                    "strip_markdown",
                    "normalize_whitespace",
                    "normalize_unicode",
                    "remove_urls",
                ],
            },
            "timeout_seconds": 10,
        },
        {
            "id": "generate_audio",
            "type": "multi_voice_tts",
            "config": {
                "sections": "{{ compose_script.sections }}",
                "voice_assignments": "{{ compose_script.voice_assignments }}",
                "default_provider": "{{ inputs.tts_provider }}",
                "default_model": "{{ inputs.tts_model }}",
                "default_voice": "{{ inputs.tts_voice }}",
                "response_format": "mp3",
                "speed": "{{ inputs.tts_speed }}",
                "normalize": True,
                "target_lufs": -16.0,
                "background_audio_uri": "{{ inputs.background_audio_uri }}",
                "background_volume": "{{ inputs.background_volume }}",
                "background_delay_ms": "{{ inputs.background_delay_ms }}",
                "background_fade_seconds": "{{ inputs.background_fade_seconds }}",
            },
            "timeout_seconds": 600,
            "retry": 1,
            "on_success": "_end",
            "on_failure": "tts_single_voice_fallback",
        },
        {
            "id": "tts_single_voice_fallback",
            "type": "tts",
            "config": {
                "input": "{{ compose_script.text }}",
                "provider": "{{ inputs.tts_provider }}",
                "model": "{{ inputs.tts_model }}",
                "voice": "{{ inputs.tts_voice }}",
                "response_format": "mp3",
                "speed": "{{ inputs.tts_speed }}",
                "artifact_metadata": {
                    "final_artifact": True,
                    "fallback_artifact": True,
                    "single_voice_fallback": True,
                    "fallback_reason": "multi_voice_tts_failed",
                },
            },
            "timeout_seconds": 600,
            "retry": 1,
        },
    ],
}


def _normalize_audio_cast_voice_map(audio_cast: Any) -> dict[str, str] | None:
    """Build a voice_map-compatible marker map from structured audio_cast speakers."""
    if not isinstance(audio_cast, dict):
        return None
    speakers = audio_cast.get("speakers")
    if not isinstance(speakers, list):
        return None

    voice_map: dict[str, str] = {}
    for speaker in speakers:
        if not isinstance(speaker, dict):
            continue
        voice = speaker.get("voice")
        if not isinstance(voice, str) or not voice.strip():
            continue
        raw_key = speaker.get("id") or speaker.get("label")
        if not isinstance(raw_key, str) or not raw_key.strip():
            continue
        normalized_key = "".join(char.upper() if char.isalnum() else "_" for char in raw_key.strip()).strip("_")
        if normalized_key:
            voice_map[normalized_key] = voice.strip()

    return voice_map or None


def _first_non_empty_pref(output_prefs: Mapping[str, Any], *keys: str) -> Any:
    """Return the first non-empty preference value for current and legacy keys."""
    for key in keys:
        value = output_prefs.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        if value is not None:
            return value
    return None


def _resolve_workflow_tts_defaults(audio_prefs: Mapping[str, Any]) -> tuple[str | None, str, str] | None:
    """Resolve Watchlists audio prefs through the same defaults as /audio/speech."""
    provider = _first_non_empty_pref(audio_prefs, "provider")
    model = _first_non_empty_pref(audio_prefs, "model")
    voice = _first_non_empty_pref(audio_prefs, "voice")
    try:
        resolved = resolve_tts_request_defaults(
            provider=provider,
            model=model,
            voice=voice,
        )
    except Exception as exc:
        logger.warning("Audio briefing: failed to resolve TTS defaults (error_type={})", type(exc).__name__)
        return None

    provider = str(resolved.provider or "").strip() or None
    model = str(resolved.model or "").strip()
    voice = str(resolved.voice or "").strip()
    if not model or not voice:
        return None
    return provider, model, voice


def _get_scheduler_queue_worker_count(scheduler: Any, queue_name: str) -> int | None:
    """Read current queue worker count without requiring Scheduler internals."""
    worker_pool = getattr(scheduler, "worker_pool", None)
    get_status = getattr(worker_pool, "get_status", None)
    if not callable(get_status):
        return None
    status = get_status()
    if not isinstance(status, dict):
        return None
    workers_by_queue = status.get("workers_by_queue")
    if not isinstance(workers_by_queue, dict):
        return None
    raw_count = workers_by_queue.get(queue_name)
    if isinstance(raw_count, int):
        return raw_count
    return None


async def _ensure_workflows_queue_has_worker(scheduler: Any) -> int:
    """Ensure the workflows queue has at least one worker without downscaling it."""
    current_count = _get_scheduler_queue_worker_count(scheduler, "workflows")
    if current_count is not None and current_count >= 1:
        return current_count
    return await scheduler.scale_workers(1, "workflows")


def _new_audio_request_id() -> str:
    """Create an opaque Watchlists audio request ID for retries and artifacts."""
    return f"wla_{uuid4().hex}"


def _build_workflow_inputs(
    items: list[dict[str, Any]],
    output_prefs: dict[str, Any],
    *,
    editorial: Mapping[str, Any] | None = None,
    status_audio: bool = False,
) -> dict[str, Any] | None:
    """Build workflow inputs dict from watchlist output_prefs."""
    contract = get_briefing_contract(output_prefs, scheduled=False)
    audio_prefs = contract["audio"]
    tts_defaults = _resolve_workflow_tts_defaults(audio_prefs)
    if tts_defaults is None:
        return None
    tts_provider, tts_model, tts_voice = tts_defaults

    audio_cast = audio_prefs.get("cast")
    voice_map = audio_prefs.get("voice_map")
    if not isinstance(voice_map, dict):
        voice_map = _normalize_audio_cast_voice_map(audio_cast)

    return {
        "items": items,
        "target_audio_minutes": 1 if status_audio else audio_prefs.get("target_minutes", 10),
        "audio_language": audio_prefs.get("language") or "en",
        "tts_provider": tts_provider,
        "tts_model": tts_model,
        "tts_voice": tts_voice,
        "tts_speed": audio_prefs.get("speed") or 1.0,
        "llm_provider": audio_prefs.get("llm_provider"),
        "llm_model": audio_prefs.get("llm_model"),
        "voice_map": voice_map,
        "audio_cast": audio_cast if isinstance(audio_cast, dict) else None,
        "persona_summarize": bool(audio_prefs.get("persona_summarize", False)),
        "persona_id": audio_prefs.get("persona_id"),
        "persona_provider": audio_prefs.get("persona_provider"),
        "persona_model": audio_prefs.get("persona_model"),
        "background_audio_uri": audio_prefs.get("background_audio_uri"),
        "background_volume": audio_prefs.get("background_volume", 0.15),
        "background_delay_ms": audio_prefs.get("background_delay_ms", 0),
        "background_fade_seconds": audio_prefs.get("background_fade_seconds", 2.0),
        "editorial": dict(editorial or contract["editorial"]),
        "is_no_material_update": status_audio,
    }


async def trigger_audio_briefing(
    *,
    user_id: int,
    job_id: int,
    run_id: int,
    output_prefs: dict[str, Any],
    db: Any,
    scheduler: Any | None = None,
    audio_request_id: str | None = None,
    items: list[dict[str, Any]] | None = None,
    occurrence_id: int | None = None,
    output_id: int | None = None,
    editorial: Mapping[str, Any] | None = None,
    status_audio: bool = False,
) -> AudioBriefingTriggerResult:
    """Trigger the audio briefing workflow for a completed watchlist run.

    Args:
        user_id: The user who owns the watchlist.
        job_id: The watchlist job ID.
        run_id: The watchlist run ID that just completed.
        output_prefs: The job's output_prefs dict.
        db: The WatchlistsDB instance.
        scheduler: Optional scheduler instance. Defaults to the global Scheduler.
        audio_request_id: Optional caller-supplied request ID for deterministic retries/tests.
        items: Optional caller-selected normalized items. Avoids a second database load.
        occurrence_id: Optional durable briefing occurrence correlation ID.
        output_id: Optional persisted text output correlation ID.
        editorial: Optional canonical editorial configuration.
        status_audio: Whether this is a short deterministic no-update status intent.

    Returns:
        Structured status for the trigger attempt.
    """
    contract = get_briefing_contract(output_prefs, scheduled=False)
    if not contract["audio"]["enabled"]:
        return AudioBriefingTriggerResult(status="disabled")

    # Older callers may still ask this bridge to select items. Fulfillment callers
    # pass the canonical bounded selection so text and audio cannot drift.
    if items is None:
        try:
            scraped_items, _ = await run_in_threadpool(
                db.list_items,
                run_id=run_id,
                status="ingested",
                limit=briefing_selection_limit(contract),
                offset=0,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Audio briefing: could not load scraped items for run {} (error_type={})",
                run_id,
                type(exc).__name__,
            )
            return AudioBriefingTriggerResult(status="enqueue_failed", reason="item_load_failed")
    else:
        scraped_items = items

    if not scraped_items:
        logger.info(f"Audio briefing: no ingested items for run {run_id}, skipping")
        return AudioBriefingTriggerResult(status="skipped_no_items", reason="no_ingested_items")

    # Build items context (title, summary, url)
    normalized_items: list[dict[str, Any]] = []
    for item in scraped_items:
        if items is not None:
            normalized_items.append(dict(item))
            continue
        if isinstance(item, dict):
            row = item
        elif hasattr(item, "_asdict"):
            row = item._asdict()
        else:
            row = {
                "title": getattr(item, "title", ""),
                "summary": getattr(item, "summary", ""),
                "url": getattr(item, "url", ""),
                "snippet": getattr(item, "snippet", ""),
                "source_url": getattr(item, "source_url", ""),
            }
        normalized_items.append(
            {
                "title": row.get("title", ""),
                "summary": row.get("summary", row.get("snippet", "")),
                "url": row.get("url", row.get("source_url", "")),
            }
        )

    workflow_inputs = _build_workflow_inputs(
        normalized_items,
        output_prefs,
        editorial=editorial,
        status_audio=status_audio,
    )
    if workflow_inputs is None:
        return AudioBriefingTriggerResult(
            status="configuration_required",
            reason="tts_defaults_unavailable",
        )
    if audio_request_id is not None:
        audio_request_id = str(audio_request_id).strip()
        if not audio_request_id.startswith("wla_") or len(audio_request_id) <= len("wla_"):
            raise ValueError("audio_request_id must start with 'wla_'")
    active_audio_request_id = audio_request_id or _new_audio_request_id()
    workflow_inputs = {
        **workflow_inputs,
        "audio_request_id": active_audio_request_id,
        "occurrence_id": occurrence_id,
        "output_id": output_id,
    }

    # Submit as a scheduler workflow task.
    try:
        if scheduler is None:
            from tldw_Server_API.app.core.Scheduler import get_global_scheduler

            scheduler = await get_global_scheduler()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Audio briefing: failed to resolve scheduler for run {} (error_type={})",
            run_id,
            type(exc).__name__,
        )
        return AudioBriefingTriggerResult(status="queue_unavailable", reason="scheduler_unavailable")

    try:
        worker_count = await _ensure_workflows_queue_has_worker(scheduler)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Audio briefing: workflows queue unavailable for run {} (error_type={})",
            run_id,
            type(exc).__name__,
        )
        return AudioBriefingTriggerResult(status="queue_unavailable", reason="workflows_queue_scale_failed")

    if worker_count < 1:
        return AudioBriefingTriggerResult(status="queue_unavailable", reason="workflows_queue_has_no_workers")

    try:
        metadata = {
            "source": "watchlist_audio_briefing",
            "watchlist_job_id": job_id,
            "watchlist_run_id": run_id,
            "audio_request_id": active_audio_request_id,
        }
        if occurrence_id is not None:
            metadata["briefing_occurrence_id"] = occurrence_id
        if output_id is not None:
            metadata["watchlist_output_id"] = output_id
        scheduler_metadata = {**metadata, "user_id": str(user_id)}
        task_id = await scheduler.submit(
            "workflow_run",
            payload={
                "user_id": user_id,
                "definition_snapshot": AUDIO_BRIEFING_WORKFLOW_DEF,
                "inputs": workflow_inputs,
                "mode": "async",
                "metadata": metadata,
            },
            queue_name="workflows",
            idempotency_key=(f"watchlist-audio-briefing:{user_id}:{job_id}:{run_id}:{active_audio_request_id}"),
            metadata=scheduler_metadata,
            max_retries=1,
        )
        logger.info(
            f"Audio briefing workflow submitted for watchlist run {run_id}, "
            f"task_id={task_id}, items={len(normalized_items)}"
        )
        return AudioBriefingTriggerResult(
            status="submitted",
            task_id=task_id,
            audio_request_id=active_audio_request_id,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Audio briefing: failed to submit workflow for run {} (error_type={})",
            run_id,
            type(exc).__name__,
        )
        return AudioBriefingTriggerResult(status="enqueue_failed", reason="scheduler_submit_failed")
