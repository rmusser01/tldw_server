from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Usage.audio_quota import (
    can_start_job,
    finish_job,
    get_limits_for_user,
    increment_jobs_started,
)

DOMAIN = "audio"


async def _handle_audio_transcribe_stage(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """
    Handle the `audio_transcribe` stage for a single job payload.

    This helper delegates STT to the shared registry/adapter helper
    (`run_stt_job_via_registry`) and normalizes the payload into the
    expected shape for downstream stages:
      - `segments`: list of segment dicts
      - `text`: merged transcript string
      - `normalized_stt`: full normalized STT artifact
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
    # Attach full normalized artifact for future stages without breaking
    # existing consumers that rely only on 'text' and 'segments'.
    updated_payload["normalized_stt"] = artifact

    next_type: str | None = "audio_chunk" if payload.get("perform_chunking") else "audio_store"
    return updated_payload, next_type


async def run_audio_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """MVP in-process worker handling audio pipeline stages.

    Stages (job_type):
      - audio_download: download URL to local temp
      - audio_convert: convert to 16k mono WAV
      - audio_transcribe: transcribe to text/segments
      - audio_chunk: optional chunking
      - audio_analyze: optional analysis
      - audio_store: finalize/store results (placeholder)
    """
    jm = JobManager()
    worker_id = "audio-worker"
    poll_sleep = float(os.getenv("JOBS_POLL_INTERVAL_SECONDS", "1.0") or "1.0")

    logger.info("Starting Audio Jobs worker (MVP)")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping Audio Jobs worker on shutdown signal")
            return
        # Per-iteration state used in the outer finally block. These must be
        # initialized before any `continue` to avoid leaking previous values.
        owner: str | None = None
        acquired_slot = False
        try:
            lease_seconds = int(os.getenv("JOBS_LEASE_SECONDS", "120") or "120")
            # Optional strict owner-aware acquisition: try to pick an owner under cap
            job = None
            try:
                if os.getenv("AUDIO_JOBS_OWNER_STRICT", "false").lower() in {"true", "1", "yes", "y", "on"}:
                    conn = jm._connect()
                    owner_candidate = None
                    try:
                        if jm.backend == "postgres":
                            with jm._pg_cursor(conn) as cur:  # type: ignore[attr-defined]
                                cur.execute(
                                    "SELECT owner_user_id FROM jobs WHERE domain=%s AND queue=%s AND status='queued' AND owner_user_id IS NOT NULL "
                                    "ORDER BY priority ASC, COALESCE(available_at, created_at) ASC, created_at ASC LIMIT 50",
                                    (DOMAIN, "default"),
                                )
                                rows = cur.fetchall() or []
                                owners = [r["owner_user_id"] for r in rows if r and r.get("owner_user_id")]
                        else:
                            rows = conn.execute(
                                "SELECT owner_user_id FROM jobs WHERE domain=? AND queue=? AND status='queued' AND owner_user_id IS NOT NULL "
                                "ORDER BY priority ASC, COALESCE(available_at, created_at) ASC, created_at ASC LIMIT 50",
                                (DOMAIN, "default"),
                            ).fetchall() or []
                            owners = [r[0] for r in rows if r and r[0] is not None]
                    finally:
                        conn.close()
                    # Choose first owner with headroom
                    for cand in owners:
                        try:
                            limits = await get_limits_for_user(int(cand))
                        except (OSError, RuntimeError, TypeError, ValueError) as e:
                            logger.warning(
                                f"Failed to get limits for owner candidate {cand}; assuming unlimited: {e}"
                            )
                            limits = {}
                        try:
                            max_jobs = int(limits.get("concurrent_jobs") or 0)
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not parse concurrent_jobs for owner candidate {cand}; assuming 0 (unlimited): {e}"
                            )
                            max_jobs = 0
                        if max_jobs == 0:
                            owner_candidate = cand
                            break
                        # Count current processing for this owner
                        try:
                            conn2 = jm._connect()
                            if jm.backend == "postgres":
                                with jm._pg_cursor(conn2) as cur2:  # type: ignore[attr-defined]
                                    cur2.execute(
                                        "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND status='processing' AND owner_user_id=%s",
                                        (DOMAIN, str(cand)),
                                    )
                                    rowp = cur2.fetchone()
                                    cur_count = int(rowp["c"]) if rowp else 0
                            else:
                                rowp = conn2.execute(
                                    "SELECT COUNT(*) FROM jobs WHERE domain=? AND status='processing' AND owner_user_id=?",
                                    (DOMAIN, str(cand)),
                                ).fetchone()
                                cur_count = int(rowp[0]) if rowp else 0
                        except (OSError, RuntimeError, TypeError, ValueError) as e:
                            logger.warning(
                                f"Failed to count processing jobs for owner candidate {cand}; assuming 0: {e}"
                            )
                            cur_count = 0
                        finally:
                            with contextlib.suppress(OSError, RuntimeError):
                                conn2.close()
                        if cur_count < max_jobs:
                            owner_candidate = cand
                            break
                    if owner_candidate is not None:
                        job = jm.acquire_next_job(domain=DOMAIN, queue="default", lease_seconds=lease_seconds, worker_id=worker_id, owner_user_id=str(owner_candidate))
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Owner-aware acquisition failed; falling back to default acquisition: {e}")
                job = None
            if not job:
                job = jm.acquire_next_job(domain=DOMAIN, queue="default", lease_seconds=lease_seconds, worker_id=worker_id)
            if not job:
                await asyncio.sleep(poll_sleep)
                continue

            owner = job.get("owner_user_id")
            if not owner:
                jm.fail_job(int(job["id"]), error="missing owner_user_id", retryable=False, worker_id=worker_id, lease_id=str(job.get("lease_id")))
                continue
            # Cross-process fairness: enforce concurrent processing cap across all workers
            try:
                limits_owner = await get_limits_for_user(int(owner))
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to get limits for owner {owner}; assuming unlimited: {e}"
                )
                limits_owner = {}
            try:
                max_jobs = int(limits_owner.get("concurrent_jobs") or 0)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse concurrent_jobs for owner {owner}; assuming 0 (unlimited): {e}"
                )
                max_jobs = 0
            if max_jobs:
                # Count current processing jobs for this owner in 'audio' domain
                try:
                    conn = jm._connect()  # use manager connection for simplicity
                    count = 0
                    if jm.backend == "postgres":
                        with conn:
                            with jm._pg_cursor(conn) as cur:  # type: ignore[attr-defined]
                                cur.execute(
                                    "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND status='processing' AND owner_user_id=%s",
                                    (DOMAIN, str(owner)),
                                )
                                row = cur.fetchone()
                                count = int(row["c"]) if row else 0
                    else:
                        with conn:
                            row = conn.execute(
                                "SELECT COUNT(*) FROM jobs WHERE domain=? AND status='processing' AND owner_user_id=?",
                                (DOMAIN, str(owner)),
                            ).fetchone()
                            count = int(row[0]) if row else 0
                    conn.close()
                except (OSError, RuntimeError, TypeError, ValueError) as e:
                    logger.warning(
                        f"Failed to count processing jobs for owner {owner}; assuming 0: {e}"
                    )
                    count = 0
                if count > max_jobs:
                    # Put job back with backoff to allow other owners to proceed
                    lease_id = str(job.get("lease_id"))
                    jm.fail_job(
                        int(job["id"]),
                        error="owner concurrency cap",
                        retryable=True,
                        backoff_seconds=10,
                        worker_id=worker_id,
                        lease_id=lease_id,
                        completion_token=lease_id,
                    )
                    continue

            # Enforce per-user concurrent job cap
            ok_job, msg = await can_start_job(int(owner))
            if not ok_job:
                # Reschedule with backoff using fail_job(retryable=True)
                lease_id = str(job.get("lease_id"))
                jm.fail_job(
                    int(job["id"]),
                    error=msg or "concurrency limit",
                    retryable=True,
                    backoff_seconds=15,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
                continue
            try:
                await increment_jobs_started(int(owner))
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(
                    f"Failed to increment jobs started for owner {owner}: {e}"
                )
            else:
                acquired_slot = True

            payload: dict[str, Any] = job.get("payload") or {}
            jtype = str(job.get("job_type") or "").lower()
            next_type: str | None = None
            updated_payload = dict(payload)
            ok = True
            msg_err = ""
            try:
                if jtype == "audio_download":
                    # Download to temp path
                    url = payload.get("url")
                    temp_dir = payload.get("temp_dir") or os.getenv("AUDIO_JOBS_TEMP", tempfile.gettempdir())
                    if not url:
                        raise ValueError("missing url in payload")
                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import (
                        download_audio_file,
                    )
                    local_path = await asyncio.to_thread(download_audio_file, url, temp_dir, False, None)
                    updated_payload["local_path"] = local_path
                    next_type = "audio_convert"
                elif jtype == "audio_convert":
                    path = payload.get("local_path")
                    if not path:
                        raise ValueError("missing local_path in payload")
                    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
                        convert_to_wav,
                    )
                    out_path = await asyncio.to_thread(convert_to_wav, path, 0, False)
                    updated_payload["wav_path"] = out_path
                    next_type = "audio_transcribe"
                elif jtype == "audio_transcribe":
                    updated_payload, next_type = await _handle_audio_transcribe_stage(updated_payload)
                elif jtype == "audio_chunk":
                    from tldw_Server_API.app.core.Chunking import improved_chunking_process
                    # Prefer merged text; fallback to joining segments on demand
                    text_for_chunking = updated_payload.get("text")
                    if not text_for_chunking:
                        segs = updated_payload.get("segments") or []
                        if not isinstance(segs, list):
                            raise ValueError("missing transcript text/segments for chunking")
                        text_for_chunking = " ".join(
                            (s.get("Text", "").strip() if isinstance(s, dict) else "") for s in segs
                        ).strip()
                    chunk_options = {"method": "sentences", "max_size": 500, "overlap": 200}
                    chunks = await asyncio.to_thread(improved_chunking_process, text_for_chunking, chunk_options)
                    updated_payload["chunks"] = chunks
                    next_type = payload.get("perform_analysis") and "audio_analyze" or "audio_store"
                elif jtype == "audio_analyze":
                    from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
                    text = updated_payload.get("text")
                    if not text:
                        # Fallback to segments
                        segs = updated_payload.get("segments") or []
                        text = " ".join((s.get("Text", "").strip() if isinstance(s, dict) else "") for s in segs).strip()
                    analysis = await asyncio.to_thread(analyze, text, payload.get("api_name") or "openai")
                    updated_payload["analysis"] = analysis
                    next_type = "audio_store"
                elif jtype == "audio_store":
                    # Placeholder: integration to store into DB or export artifacts
                    next_type = None
                else:
                    ok = False
                    msg_err = f"Unknown job_type: {jtype}"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                ok = False
                msg_err = str(e)

            if ok:
                if next_type:
                    # Create next stage
                    jm.complete_job(int(job["id"]), worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
                    jm.create_job(
                        domain=DOMAIN,
                        queue=("transcribe" if next_type == "audio_transcribe" else "default"),
                        job_type=next_type,
                        payload=updated_payload,
                        owner_user_id=str(owner),
                        priority=5,
                    )
                else:
                    jm.complete_job(int(job["id"]), result={"payload": updated_payload}, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))
            else:
                jm.fail_job(int(job["id"]), error=msg_err, retryable=True, worker_id=worker_id, lease_id=str(job.get("lease_id")), completion_token=str(job.get("lease_id")))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Audio worker loop error: {e}")
        finally:
            try:
                if acquired_slot and owner is not None:
                    await finish_job(int(owner))  # type: ignore[arg-type]
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to release job slot: {e}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_audio_jobs_worker())
