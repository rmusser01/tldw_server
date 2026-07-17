"""
TTS Jobs worker for long-form speech generation.

- domain = "audio"
- queue = os.getenv("TTS_JOBS_QUEUE", "default")
- job_type = "tts_longform"
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.Jobs.event_stream import emit_job_event
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.TTS.gateway_preflight import gateway_route_provenance
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError, is_retryable_error
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
from tldw_Server_API.app.core.TTS.utils import (
    build_tts_segments_payload,
    compute_tts_history_text_hash,
    parse_bool,
    tts_history_text_length,
)

TTS_DOMAIN = "audio"
TTS_JOB_TYPE = "tts_longform"
_TTS_JOBS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    OverflowError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


class TTSJobError(Exception):
    def __init__(self, message: str, *, retryable: bool = True, backoff_seconds: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = int(backoff_seconds)


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> str:
    candidate = payload.get("user_id") or job.get("owner_user_id")
    if candidate is None or str(candidate).strip() == "":
        raise TTSJobError("missing user_id", retryable=False)
    return str(candidate)


def _tts_history_config() -> dict[str, Any]:
    return {
        "enabled": parse_bool(getattr(settings, "TTS_HISTORY_ENABLED", False), default=False),
        "store_text": parse_bool(getattr(settings, "TTS_HISTORY_STORE_TEXT", True), default=True),
        "store_failed": parse_bool(getattr(settings, "TTS_HISTORY_STORE_FAILED", True), default=True),
        "hash_key": getattr(settings, "TTS_HISTORY_HASH_KEY", None),
    }


def _open_media_db_for_history(user_id: str) -> Any | None:
    try:
        db_path = DatabasePaths.get_media_db_path(user_id)
        return create_media_database(
            client_id="tts_jobs_worker",
            db_path=str(db_path),
        )
    except _TTS_JOBS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("TTS jobs worker: failed to open media db for history: {}", exc)
        return None


def _sanitize_params_json(
    request: OpenAISpeechRequest,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params_json: dict[str, Any] = {"speed": request.speed}
    if request.extra_params:
        try:
            extra_params = dict(request.extra_params)
        except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
            extra_params = None
        if extra_params:
            extra_params.pop("voice_reference", None)
            params_json["extra_params"] = extra_params
    if request.lang_code:
        params_json["lang_code"] = request.lang_code
    route = gateway_route_provenance(
        requested_backend=request.backend,
        requested_model=request.model,
        metadata=metadata,
    )
    if route:
        if route["requested_backend"] != route["actual_backend"]:
            params_json["requested_backend"] = route["requested_backend"]
        params_json["fallback_used"] = route["fallback_used"]
        params_json["conversion_used"] = route["conversion_used"]
    return params_json


def _build_voice_info(request: OpenAISpeechRequest, metadata: dict[str, Any]) -> dict[str, Any] | None:
    voice_info: dict[str, Any] = {}
    meta_voice_info = metadata.get("voice_info")
    if isinstance(meta_voice_info, dict):
        voice_info.update(meta_voice_info)
    voice_info.pop("voice_reference", None)
    if request.voice_reference:
        voice_info["has_voice_reference"] = True
    if request.reference_duration_min is not None:
        voice_info["reference_duration_min"] = request.reference_duration_min
    return voice_info or None


async def _handle_tts_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    job_type = str(job.get("job_type") or payload.get("job_type") or "").strip().lower()
    if job_type and job_type != TTS_JOB_TYPE:
        raise TTSJobError(f"unsupported job_type: {job_type}", retryable=False)

    speech_payload = payload.get("speech_request")
    if not isinstance(speech_payload, dict):
        raise TTSJobError("missing speech_request payload", retryable=False)

    # Force non-streaming in jobs worker
    speech_payload["stream"] = False
    try:
        request = OpenAISpeechRequest(**speech_payload)
    except Exception as exc:
        raise TTSJobError(f"invalid speech_request: {exc}", retryable=False) from exc

    explicit_backend = request.backend is not None
    user_id = _resolve_user_id(job, {} if explicit_backend else payload)
    provider_hint = None if explicit_backend else payload.get("provider_hint")
    provider_overrides = None if explicit_backend else payload.get("provider_overrides")

    jm = JobManager()
    tts_service = await get_tts_service_v2()
    job_id = int(job.get("id") or 0)
    request_id = str(job.get("request_id") or payload.get("request_id") or "")
    start_ts = asyncio.get_event_loop().time()

    history_cfg = _tts_history_config()
    history_enabled = history_cfg.get("enabled", False)
    history_db: Any | None = _open_media_db_for_history(user_id) if history_enabled else None
    history_written = False

    def _record_history(
        status: str,
        *,
        error_message: str | None = None,
        output_id: int | None = None,
        artifact_ids: list[Any] | None = None,
    ) -> None:
        nonlocal history_written
        if history_written or history_db is None:
            return
        if status == "failed" and not history_cfg.get("store_failed", True):
            return
        try:
            text_hash = compute_tts_history_text_hash(request.input, history_cfg.get("hash_key"))
        except _TTS_JOBS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "TTS jobs worker: failed to compute text hash: {} (job_id={}, request_id={})",
                exc,
                job_id,
                request_id or "unknown",
            )
            return
        metadata = getattr(request, "_tts_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        provider = (
            metadata.get("actual_backend")
            or metadata.get("actual_provider")
            or metadata.get("provider")
            or provider_hint
        )
        model = metadata.get("model") or request.model
        voice_name = metadata.get("voice") or request.voice
        voice_id = metadata.get("voice_id")
        fmt = metadata.get("format") or request.response_format

        duration_ms = None
        duration_val = metadata.get("duration_ms")
        if isinstance(duration_val, (int, float)):
            duration_ms = int(duration_val)
        else:
            duration_seconds = metadata.get("duration_seconds") or metadata.get("duration")
            if isinstance(duration_seconds, (int, float)):
                duration_ms = int(float(duration_seconds) * 1000)

        segments_json = build_tts_segments_payload(metadata.get("segments"))
        generation_time_ms = int(max(0.0, (asyncio.get_event_loop().time() - start_ts) * 1000))
        text_length = tts_history_text_length(request.input)
        text_value = request.input if history_cfg.get("store_text", True) else None
        params_json = _sanitize_params_json(request, metadata)
        voice_info = _build_voice_info(request, metadata)

        final_status = status
        if final_status == "success" and parse_bool(metadata.get("partial"), default=False):
            final_status = "partial"

        try:
            insert_start = asyncio.get_event_loop().time()
            history_db.create_tts_history_entry(
                user_id=str(user_id),
                text_hash=text_hash,
                text=text_value,
                text_length=text_length,
                provider=str(provider) if provider is not None else None,
                model=str(model) if model is not None else None,
                voice_id=str(voice_id) if voice_id is not None else None,
                voice_name=str(voice_name) if voice_name is not None else None,
                voice_info=voice_info,
                format=str(fmt) if fmt is not None else None,
                duration_ms=duration_ms,
                generation_time_ms=generation_time_ms,
                params_json=params_json if params_json else None,
                status=final_status,
                segments_json=segments_json,
                job_id=job_id if job_id > 0 else None,
                output_id=output_id,
                artifact_ids=artifact_ids,
                error_message=error_message,
            )
            try:
                log_counter(
                    "tts_history_writes_total",
                    labels={
                        "status": str(final_status or "unknown"),
                        "provider": str(provider or "unknown"),
                    },
                )
                log_histogram(
                    "tts_history_write_latency_ms",
                    value=max(0.0, (asyncio.get_event_loop().time() - insert_start) * 1000),
                    labels={"status": str(final_status or "unknown")},
                )
            except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
                pass
            history_written = True
        except _TTS_JOBS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "TTS jobs worker: failed to write history record: {} (job_id={}, request_id={})",
                exc,
                job_id,
                request_id or "unknown",
            )

    def _emit_progress(percent: float, message: str, eta_seconds: float | None = None) -> None:
        if job_id <= 0:
            return
        with contextlib.suppress(_TTS_JOBS_NONCRITICAL_EXCEPTIONS):
            jm.update_job_progress(job_id, progress_percent=percent, progress_message=message)
        try:
            attrs = {"progress_percent": percent, "progress_message": message}
            if eta_seconds is not None:
                attrs["eta_seconds"] = max(0.0, float(eta_seconds))
            emit_job_event("job.progress", job={"id": job_id}, attrs=attrs)
        except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
            pass

    try:
        _emit_progress(5.0, "tts_started")
        try:
            speech_iter = tts_service.generate_speech(
                request,
                provider=provider_hint,
                fallback=not explicit_backend,
                provider_overrides=provider_overrides,
                user_id=int(user_id),
            )
            audio_bytes = bytearray()
            last_update = start_ts
            expected_chars_per_sec = 15.0
            try:
                expected_chars_per_sec = float(
                    (request.extra_params or {}).get("audio_expected_chars_per_sec", expected_chars_per_sec)
                )
            except _TTS_JOBS_NONCRITICAL_EXCEPTIONS:
                expected_chars_per_sec = 15.0
            expected_sec = max(1.0, len(request.input or "") / max(1.0, expected_chars_per_sec))
            async for chunk in speech_iter:
                if chunk:
                    audio_bytes.extend(chunk)
                now = asyncio.get_event_loop().time()
                if now - last_update >= 1.0:
                    elapsed = now - start_ts
                    percent = min(80.0, (elapsed / expected_sec) * 80.0)
                    eta = max(0.0, expected_sec - elapsed)
                    _emit_progress(percent, "tts_synthesizing", eta_seconds=eta)
                    last_update = now
        except TTSError as exc:
            if history_enabled:
                _record_history("failed", error_message=str(exc))
            retryable = is_retryable_error(exc)
            raise TTSJobError(str(exc), retryable=retryable) from exc
        except Exception as exc:
            if history_enabled:
                _record_history("failed", error_message=str(exc))
            raise TTSJobError(str(exc), retryable=True) from exc

        if not audio_bytes:
            if history_enabled:
                _record_history("failed", error_message="empty_audio")
            raise TTSJobError("empty_audio", retryable=False)

        _emit_progress(85.0, "tts_synthesis_complete")
        outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
        filename = f"tts_job_{job.get('id')}.{request.response_format}"

        _emit_progress(92.0, "tts_writing_output")
        with CollectionsDatabase.for_user(user_id=user_id) as cdb:
            safe_name = cdb.resolve_output_storage_path(filename)
            output_path = outputs_dir / safe_name
            try:
                output_path.write_bytes(bytes(audio_bytes))
            except Exception as exc:
                logger.error("tts jobs worker: failed to write output {}: {}", output_path, exc)
                if history_enabled:
                    _record_history("failed", error_message="write_failed")
                raise TTSJobError("write_failed", retryable=True) from exc

            request_metadata = getattr(request, "_tts_metadata", None)
            if not isinstance(request_metadata, dict):
                request_metadata = {}
            route = gateway_route_provenance(
                requested_backend=request.backend,
                requested_model=request.model,
                metadata=request_metadata,
            )
            metadata = {
                "artifact_type": "tts_audio",
                "provider": route.get("actual_backend", provider_hint),
                "model": route.get("actual_model", request.model),
                "voice": request_metadata.get("voice") or request.voice,
                "format": request.response_format,
                **route,
            }
            row = cdb.create_output_artifact(
                type_="tts_audio",
                title=f"TTS Job {job.get('id')}",
                format_=request.response_format,
                storage_path=safe_name,
                metadata_json=json.dumps(metadata),
                job_id=int(job.get("id")),
            )

        if history_enabled:
            artifact_ids: list[Any] | None = None
            if getattr(row, "id", None) is not None:
                artifact_ids = [f"output:{int(row.id)}"]
            _record_history("success", output_id=row.id, artifact_ids=artifact_ids)

        _emit_progress(100.0, "tts_completed", eta_seconds=0.0)
        return {
            "output_id": row.id,
            "storage_path": row.storage_path,
            "format": row.format,
            "bytes": len(audio_bytes),
            **route,
        }
    finally:
        if history_db is not None:
            with contextlib.suppress(_TTS_JOBS_NONCRITICAL_EXCEPTIONS):
                history_db.close_connection()


async def run_tts_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("TTS_JOBS_WORKER_ID") or f"tts-jobs-{os.getpid()}").strip()
    queue = (os.getenv("TTS_JOBS_QUEUE") or "default").strip() or "default"
    lease_seconds = _coerce_int(os.getenv("TTS_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
    renew_jitter = _coerce_int(os.getenv("TTS_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"), 5)
    renew_threshold = _coerce_int(os.getenv("TTS_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"), 10)
    cfg = WorkerConfig(
        domain=TTS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    _stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        _stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info("TTS Jobs worker starting (queue={}, worker_id={})", queue, worker_id)
    try:
        await sdk.run(handler=_handle_tts_job)
    finally:
        if _stop_watcher_task is not None and not _stop_watcher_task.done():
            _stop_watcher_task.cancel()
            with contextlib.suppress(_TTS_JOBS_NONCRITICAL_EXCEPTIONS):
                await _stop_watcher_task


if __name__ == "__main__":
    asyncio.run(run_tts_jobs_worker())
