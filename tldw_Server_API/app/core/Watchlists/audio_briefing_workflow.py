"""Audio briefing workflow bridge.

Triggers the audio briefing workflow pipeline after a watchlist run completes,
when the job's output_prefs has generate_audio=True.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from starlette.concurrency import run_in_threadpool

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
                "model": "{{ inputs.tts_model }}",
                "voice": "{{ inputs.tts_voice }}",
                "response_format": "mp3",
                "speed": "{{ inputs.tts_speed }}",
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


def _build_workflow_inputs(
    items: list[dict[str, Any]],
    output_prefs: dict[str, Any],
) -> dict[str, Any]:
    """Build workflow inputs dict from watchlist output_prefs."""
    audio_cast = output_prefs.get("audio_cast")
    voice_map = output_prefs.get("voice_map")
    if not isinstance(voice_map, dict):
        voice_map = _normalize_audio_cast_voice_map(audio_cast)

    return {
        "items": items,
        "target_audio_minutes": output_prefs.get("target_audio_minutes", 10),
        "audio_language": output_prefs.get("audio_language") or "en",
        "tts_model": output_prefs.get("audio_model") or "kokoro",
        "tts_voice": output_prefs.get("audio_voice") or "af_heart",
        "tts_speed": output_prefs.get("audio_speed") or 1.0,
        "llm_provider": output_prefs.get("llm_provider"),
        "llm_model": output_prefs.get("llm_model"),
        "voice_map": voice_map,
        "audio_cast": audio_cast if isinstance(audio_cast, dict) else None,
        "persona_summarize": bool(output_prefs.get("persona_summarize", False)),
        "persona_id": output_prefs.get("persona_id"),
        "persona_provider": output_prefs.get("persona_provider"),
        "persona_model": output_prefs.get("persona_model"),
        "background_audio_uri": output_prefs.get("background_audio_uri"),
        "background_volume": output_prefs.get("background_volume", 0.15),
        "background_delay_ms": output_prefs.get("background_delay_ms", 0),
        "background_fade_seconds": output_prefs.get("background_fade_seconds", 2.0),
    }


async def trigger_audio_briefing(
    *,
    user_id: int,
    job_id: int,
    run_id: int,
    output_prefs: dict[str, Any],
    db: Any,
) -> str | None:
    """Trigger the audio briefing workflow for a completed watchlist run.

    Args:
        user_id: The user who owns the watchlist.
        job_id: The watchlist job ID.
        run_id: The watchlist run ID that just completed.
        output_prefs: The job's output_prefs dict.
        db: The WatchlistsDB instance.

    Returns:
        The workflow run_id if successfully enqueued, None otherwise.
    """
    if not output_prefs.get("generate_audio"):
        return None

    # Gather scraped items for this run
    try:
        scraped_items, _ = await run_in_threadpool(
            db.list_items,
            run_id=run_id,
            status="ingested",
            limit=100,
            offset=0,
        )
    except Exception as exc:
        logger.warning(f"Audio briefing: could not load scraped items for run {run_id}: {exc}")
        return None

    if not scraped_items:
        logger.info(f"Audio briefing: no ingested items for run {run_id}, skipping")
        return None

    # Build items context (title, summary, url)
    items: list[dict[str, Any]] = []
    for item in scraped_items:
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
        items.append(
            {
                "title": row.get("title", ""),
                "summary": row.get("summary", row.get("snippet", "")),
                "url": row.get("url", row.get("source_url", "")),
            }
        )

    workflow_inputs = _build_workflow_inputs(items, output_prefs)

    # Enqueue as a scheduler task
    try:
        from tldw_Server_API.app.core.Scheduler import get_global_scheduler
        from tldw_Server_API.app.core.Scheduler.base.task import Task

        scheduler = await get_global_scheduler()
        task = Task(
            handler="workflow_run",
            payload={
                "user_id": user_id,
                "definition_snapshot": AUDIO_BRIEFING_WORKFLOW_DEF,
                "inputs": workflow_inputs,
                "mode": "async",
                "metadata": {
                    "source": "watchlist_audio_briefing",
                    "watchlist_job_id": job_id,
                    "watchlist_run_id": run_id,
                },
            },
            timeout=3600,
            max_retries=1,
        )
        task_id = await scheduler.enqueue(task)
        logger.info(
            f"Audio briefing workflow enqueued for watchlist run {run_id}, " f"task_id={task_id}, items={len(items)}"
        )
        return task_id
    except Exception as exc:
        logger.warning(f"Audio briefing: failed to enqueue workflow for run {run_id}: {exc}")
        return None
