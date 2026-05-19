from __future__ import annotations

import asyncio
import contextlib
import os
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Usage.audio_quota import (
    can_start_job,
    finish_job,
    increment_jobs_started,
)

DOMAIN = "audio"


async def _handle_gpu_audio_transcribe_stage(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Handle the GPU `audio_transcribe` stage for a single job payload.

    This mirrors the CPU worker's behavior but delegates STT to the shared
    registry/adapter helper (`run_stt_job_via_registry`) so that provider
    selection and normalized artifacts stay consistent across entrypoints.
    """
    wav_path = payload.get("wav_path")
    if not wav_path:
        raise ValueError("missing wav_path in payload")

    raw_model = payload.get("model")
    model = (raw_model.strip() if isinstance(raw_model, str) else raw_model) or None
    hotwords = payload.get("hotwords")
    temp_dir = payload.get("temp_dir")
    # Constrain adapter file access to the job's temp dir for path safety.
    base_dir = Path(temp_dir) if temp_dir else None

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
        run_stt_job_via_registry,
    )

    artifact = await asyncio.to_thread(
        partial(
            run_stt_job_via_registry,
            wav_path,
            model,
            None,
            hotwords=hotwords,
            base_dir=base_dir,
        )
    )

    segments_list = artifact.get("segments") or []
    if not isinstance(segments_list, list):
        raise ValueError("unexpected transcription result format; expected list of segments")

    text_merged = artifact.get("text")
    if not isinstance(text_merged, str) or not text_merged.strip():
        text_merged = " ".join(
            (seg.get("Text", "").strip() if isinstance(seg, dict) else "") for seg in segments_list
        ).strip()

    updated_payload = dict(payload)
    updated_payload["segments"] = segments_list
    updated_payload["text"] = text_merged
    updated_payload["normalized_stt"] = artifact
    return updated_payload


async def run_audio_transcribe_gpu_worker(stop_event: asyncio.Event | None = None) -> None:
    """GPU-focused worker that only processes the `audio_transcribe` stage.

    This is a stub container-friendly worker meant to be deployed on GPU nodes.
    It acquires jobs from the `audio` domain and processes only `audio_transcribe` jobs,
    handing off other stages back to the queue with a small retryable backoff.

    Env:
      - JOBS_POLL_INTERVAL_SECONDS (float): poll interval
      - JOBS_LEASE_SECONDS (int): lease duration per job
    """
    jm = JobManager()
    worker_id = "audio-gpu-worker"
    poll_sleep = float(os.getenv("JOBS_POLL_INTERVAL_SECONDS", "1.0") or "1.0")
    lease_seconds = int(os.getenv("JOBS_LEASE_SECONDS", "180") or "180")

    logger.info("Starting Audio GPU worker (transcription-only stub)")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping Audio GPU worker on shutdown signal")
            return
        job = None
        owner = None
        acquired_slot = False
        try:
            job = jm.acquire_next_job(domain=DOMAIN, queue="transcribe", lease_seconds=lease_seconds, worker_id=worker_id)
            if not job:
                await asyncio.sleep(poll_sleep)
                continue
            owner = job.get("owner_user_id") or "0"

            # Only process audio_transcribe jobs here
            jtype = str(job.get("job_type") or "").lower()
            if jtype != "audio_transcribe":
                jm.fail_job(int(job["id"]), error="not_transcribe_stage", retryable=True, backoff_seconds=2, worker_id=worker_id, lease_id=str(job.get("lease_id")))
                continue

            # Per-user concurrency guard for fairness
            ok_job, msg = await can_start_job(int(owner))
            if not ok_job:
                jm.fail_job(int(job["id"]), error=msg or "concurrency limit", retryable=True, backoff_seconds=10, worker_id=worker_id, lease_id=str(job.get("lease_id")))
                continue
            acquired_slot = True
            with contextlib.suppress(Exception):
                await increment_jobs_started(int(owner))

            payload: dict[str, Any] = job.get("payload") or {}
            updated_payload = await _handle_gpu_audio_transcribe_stage(payload)

            # Complete and enqueue next stage
            jm.complete_job(int(job["id"]), worker_id=worker_id, lease_id=str(job.get("lease_id")))
            next_type: str | None = "audio_chunk" if payload.get("perform_chunking") else "audio_store"
            jm.create_job(
                domain=DOMAIN,
                queue="default",
                job_type=next_type,
                payload=updated_payload,
                owner_user_id=str(owner),
                priority=5,
            )

        except Exception as e:
            try:
                if job:
                    jm.fail_job(int(job["id"]), error=str(e), retryable=True, backoff_seconds=15, worker_id=worker_id, lease_id=str(job.get("lease_id")))
            except Exception:
                logger.debug("Audio GPU worker failed to mark job as failed")
            logger.error("Audio GPU worker error")
        finally:
            try:
                if acquired_slot and owner is not None:
                    await finish_job(int(owner))
            except Exception:
                logger.debug("Audio GPU worker failed to release owner slot")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_audio_transcribe_gpu_worker())
