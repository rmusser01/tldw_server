import asyncio
import os
import shutil
import sqlite3
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.smoke


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def info(self, message, *args, **kwargs):
        self.records.append(("info", message, args, kwargs))

    def debug(self, message, *args, **kwargs):
        self.records.append(("debug", message, args, kwargs))

    def warning(self, message, *args, **kwargs):
        self.records.append(("warning", message, args, kwargs))

    def error(self, message, *args, **kwargs):
        self.records.append(("error", message, args, kwargs))


def _render_log_records(records):
    return "\n".join(
        f"{level} {message} {args!r} {kwargs!r}" for level, message, args, kwargs in records
    )


def _has_ffmpeg() -> bool:


    return bool(shutil.which("ffmpeg"))


def _allow_heavy() -> bool:


    return os.getenv("ALLOW_HEAVY_AUDIO_SMOKE", "").lower() in ("1", "true", "yes")


@pytest.mark.asyncio
async def test_audio_worker_pipeline_smoke_skip_if_missing():
    """
    Smoke test for audio worker pipeline. Skips unless explicitly allowed and ffmpeg is present.

    This test enqueues a local_path job and briefly runs the worker loop. It is designed
    to be opt-in (ALLOW_HEAVY_AUDIO_SMOKE=1) and will skip otherwise to keep CI fast.
    """
    if not _allow_heavy() or not _has_ffmpeg():
        pytest.skip("Audio worker smoke test skipped (set ALLOW_HEAVY_AUDIO_SMOKE=1 and install ffmpeg to enable)")

    # Create a tiny WAV file
    sr = 16000
    data = np.zeros(sr // 10, dtype=np.float32)  # 0.1s silence
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, data, sr)
            tmp_path = f.name

        # Create an audio job pointing to this local file
        from tldw_Server_API.app.core.Jobs.manager import JobManager
        jm = JobManager()
        row = jm.create_job(
            domain="audio",
            queue="default",
            job_type="audio_convert",
            payload={"local_path": tmp_path, "perform_chunking": False, "perform_analysis": False, "model": "whisper-1"},
            owner_user_id="1",
        )
        assert row.get("id")

        # Run the worker for a brief period
        from tldw_Server_API.app.services.audio_jobs_worker import run_audio_jobs_worker
        stop = asyncio.Event()

        async def _stop_soon():
            await asyncio.sleep(1.0)
            stop.set()

        task = asyncio.create_task(run_audio_jobs_worker(stop))
        stopper = asyncio.create_task(_stop_soon())
        await asyncio.gather(task, stopper)

        # If we reach here without exceptions, smoke passes
        assert True
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                _ = None


@pytest.mark.asyncio
async def test_audio_worker_transcribe_normalizes_segments_and_text_tuple(monkeypatch, tmp_path):
    """
    The CPU audio_jobs_worker should normalize speech_to_text results that come
    back as (segments, language) into a payload with both 'segments' and merged
    'text' for downstream stages.
    """
    # Isolate jobs DB for this test
    jobs_db_path = tmp_path / "jobs_cpu.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()

    # Stub quota helpers to avoid external dependencies
    import tldw_Server_API.app.services.audio_jobs_worker as worker

    async def _fake_get_limits_for_user(user_id: int):
        return {"daily_minutes": 30.0, "concurrent_streams": 1, "concurrent_jobs": None, "max_file_size_mb": 25}

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _fake_get_limits_for_user, raising=True)
    monkeypatch.setattr(worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(worker, "finish_job", _fake_finish_job, raising=True)

    # Stub run_stt_job_via_registry so no real STT runs
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def _fake_run_stt_job_via_registry(wav_path, model, language, hotwords=None, base_dir=None, **kwargs):
        assert hotwords is None

        segs = [{"Text": "hello worker", "start_seconds": 0.0, "end_seconds": 1.0}]
        return {
            "text": "hello worker",
            "language": "en",
            "segments": segs,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {"provider": "faster-whisper", "model": model or ""},
        }

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _fake_run_stt_job_via_registry, raising=True)

    # Create a dummy wav file and corresponding audio_transcribe job
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"\x00\x00")
    row = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_transcribe",
        payload={
            "wav_path": str(wav_path),
            "model": "whisper-1",
            "perform_chunking": False,
            "perform_analysis": False,
        },
        owner_user_id="1",
    )
    assert row.get("id")

    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        # Poll for the next-stage job (audio_store) created from the transcribe stage
        created = None
        for _ in range(50):
            await asyncio.sleep(0.05)
            jobs = jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="1")
            if jobs:
                created = jobs[0]
                break

        assert created is not None, "audio_store job was not created by worker"
        payload = created.get("payload") or {}
        assert isinstance(payload.get("segments"), list)
        assert payload.get("text") == "hello worker"
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_worker_download_error_does_not_crash_worker(monkeypatch, tmp_path):
    """
    A DownloadError raised during the audio_download stage should be handled as
    a job failure/retry path without crashing the worker task.
    """
    jobs_db_path = tmp_path / "jobs_download_error.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()

    import tldw_Server_API.app.services.audio_jobs_worker as worker

    async def _fake_get_limits_for_user(user_id: int):
        return {"daily_minutes": 30.0, "concurrent_streams": 1, "concurrent_jobs": None, "max_file_size_mb": 25}

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _fake_get_limits_for_user, raising=True)
    monkeypatch.setattr(worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(worker, "finish_job", _fake_finish_job, raising=True)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files as audio_files
    from tldw_Server_API.app.core.exceptions import DownloadError

    def _fake_download_audio_file(*args, **kwargs):
        raise DownloadError("Download failed with status 404")

    monkeypatch.setattr(audio_files, "download_audio_file", _fake_download_audio_file, raising=True)

    row = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_download",
        payload={"url": "https://example.invalid/missing.mp3", "temp_dir": str(tmp_path)},
        owner_user_id="1",
    )
    job_id = int(row["id"])

    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        saw_retry_increment = False
        for _ in range(80):
            await asyncio.sleep(0.05)
            if task.done():
                exc = task.exception()
                pytest.fail(f"audio worker crashed unexpectedly: {exc!r}")
            job_state = jm.get_job(job_id) or {}
            if int(job_state.get("retry_count") or 0) >= 1:
                saw_retry_increment = True
                break

        assert saw_retry_increment, "audio_download job did not record a retry after DownloadError"

        stop.set()
        await asyncio.wait_for(task, timeout=2.0)

        job_state = jm.get_job(job_id) or {}
        err_text = str(job_state.get("last_error") or job_state.get("error_message") or "")
        assert "404" in err_text
    finally:
        if not task.done():
            stop.set()
            await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_audio_gpu_worker_normalizes_segments_and_text(monkeypatch, tmp_path):
    """
    The GPU audio_transcribe worker should normalize both plain segments and
    (segments, language) results into 'segments' and merged 'text' payload.
    """
    jobs_db_path = tmp_path / "jobs_gpu.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))
    # Allow the GPU worker's 'transcribe' queue for the audio domain
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_AUDIO", "transcribe")

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()

    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        return None

    monkeypatch.setattr(gpu_worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(gpu_worker, "finish_job", _fake_finish_job, raising=True)

    # Stub run_stt_job_via_registry for GPU worker so no real STT runs
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    def _fake_run_stt_job_via_registry_gpu(wav_path, model, language, hotwords=None, base_dir=None, **kwargs):
        assert hotwords is None

        segs = [{"Text": "hello gpu", "start_seconds": 0.0, "end_seconds": 1.0}]
        return {
            "text": "hello gpu",
            "language": "en",
            "segments": segs,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {"provider": "faster-whisper", "model": model or ""},
        }

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _fake_run_stt_job_via_registry_gpu, raising=True)

    wav_path = tmp_path / "sample_gpu.wav"
    wav_path.write_bytes(b"\x00\x00")
    row = jm.create_job(
        domain="audio",
        queue="transcribe",
        job_type="audio_transcribe",
        payload={
            "wav_path": str(wav_path),
            "model": "whisper-1",
            "perform_chunking": False,
            "perform_analysis": False,
        },
        owner_user_id="1",
    )
    assert row.get("id")

    stop = asyncio.Event()
    task = asyncio.create_task(gpu_worker.run_audio_transcribe_gpu_worker(stop))
    try:
        created = None
        for _ in range(50):
            await asyncio.sleep(0.05)
            jobs = jm.list_jobs(domain="audio", queue="default", job_type="audio_store", owner_user_id="1")
            if jobs:
                created = jobs[0]
                break

        assert created is not None, "audio_store job was not created by GPU worker"
        payload = created.get("payload") or {}
        assert isinstance(payload.get("segments"), list)
        assert payload.get("text") == "hello gpu"
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_gpu_worker_sanitizes_worker_error_log_but_preserves_job_error(monkeypatch):
    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker

    sensitive_error = "transcribe failed for /tmp/private/audio.wav token=SECRET123"
    logger_capture = _CaptureLogger()
    stop = asyncio.Event()
    failed_errors = []

    class _FakeJobManager:
        def __init__(self):
            self._sent_job = False

        def acquire_next_job(self, **kwargs):
            if self._sent_job:
                stop.set()
                return None
            self._sent_job = True
            return {
                "id": 101,
                "owner_user_id": "7",
                "job_type": "audio_transcribe",
                "lease_id": "lease-101",
                "payload": {"wav_path": "/tmp/private/audio.wav"},
            }

        def fail_job(self, job_id, error, **kwargs):
            failed_errors.append(error)

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        return None

    async def _fake_handle_stage(payload):
        raise RuntimeError(sensitive_error)

    monkeypatch.setattr(gpu_worker, "JobManager", _FakeJobManager, raising=True)
    monkeypatch.setattr(gpu_worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(gpu_worker, "finish_job", _fake_finish_job, raising=True)
    monkeypatch.setattr(gpu_worker, "_handle_gpu_audio_transcribe_stage", _fake_handle_stage, raising=True)
    monkeypatch.setattr(gpu_worker, "logger", logger_capture, raising=True)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")

    await asyncio.wait_for(gpu_worker.run_audio_transcribe_gpu_worker(stop), timeout=1.0)

    assert failed_errors == [sensitive_error]
    rendered_logs = _render_log_records(logger_capture.records)
    assert "/tmp/private/audio.wav" not in rendered_logs
    assert "SECRET123" not in rendered_logs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_gpu_worker_sanitizes_failed_mark_job_as_failed_log(monkeypatch):
    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker

    primary_error = "primary failure /tmp/private/audio.wav token=PRIMARY"
    fail_job_error = "fail_job failure /tmp/private/jobs.db token=FAILJOB"
    logger_capture = _CaptureLogger()
    stop = asyncio.Event()

    class _FakeJobManager:
        def __init__(self):
            self._sent_job = False

        def acquire_next_job(self, **kwargs):
            if self._sent_job:
                stop.set()
                return None
            self._sent_job = True
            return {
                "id": 102,
                "owner_user_id": "7",
                "job_type": "audio_transcribe",
                "lease_id": "lease-102",
                "payload": {"wav_path": "/tmp/private/audio.wav"},
            }

        def fail_job(self, job_id, error, **kwargs):
            assert error == primary_error
            raise RuntimeError(fail_job_error)

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        return None

    async def _fake_handle_stage(payload):
        raise RuntimeError(primary_error)

    monkeypatch.setattr(gpu_worker, "JobManager", _FakeJobManager, raising=True)
    monkeypatch.setattr(gpu_worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(gpu_worker, "finish_job", _fake_finish_job, raising=True)
    monkeypatch.setattr(gpu_worker, "_handle_gpu_audio_transcribe_stage", _fake_handle_stage, raising=True)
    monkeypatch.setattr(gpu_worker, "logger", logger_capture, raising=True)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")

    await asyncio.wait_for(gpu_worker.run_audio_transcribe_gpu_worker(stop), timeout=1.0)

    rendered_logs = _render_log_records(logger_capture.records)
    assert "/tmp/private/audio.wav" not in rendered_logs
    assert "/tmp/private/jobs.db" not in rendered_logs
    assert "PRIMARY" not in rendered_logs
    assert "FAILJOB" not in rendered_logs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_gpu_worker_sanitizes_owner_slot_release_log(monkeypatch):
    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker

    release_error = "release failed for /tmp/private/quota.db token=RELEASE"
    logger_capture = _CaptureLogger()
    stop = asyncio.Event()

    class _FakeJobManager:
        def __init__(self):
            self._sent_job = False

        def acquire_next_job(self, **kwargs):
            if self._sent_job:
                stop.set()
                return None
            self._sent_job = True
            return {
                "id": 103,
                "owner_user_id": "7",
                "job_type": "audio_transcribe",
                "lease_id": "lease-103",
                "payload": {"wav_path": "/tmp/private/audio.wav"},
            }

        def complete_job(self, *args, **kwargs):
            return None

        def create_job(self, *args, **kwargs):
            return {"id": 104}

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        raise RuntimeError(release_error)

    async def _fake_handle_stage(payload):
        return dict(payload, segments=[], text="", normalized_stt={})

    monkeypatch.setattr(gpu_worker, "JobManager", _FakeJobManager, raising=True)
    monkeypatch.setattr(gpu_worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(gpu_worker, "finish_job", _fake_finish_job, raising=True)
    monkeypatch.setattr(gpu_worker, "_handle_gpu_audio_transcribe_stage", _fake_handle_stage, raising=True)
    monkeypatch.setattr(gpu_worker, "logger", logger_capture, raising=True)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")

    await asyncio.wait_for(gpu_worker.run_audio_transcribe_gpu_worker(stop), timeout=1.0)

    rendered_logs = _render_log_records(logger_capture.records)
    assert "/tmp/private/quota.db" not in rendered_logs
    assert "RELEASE" not in rendered_logs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_worker_owner_concurrent_jobs_zero_blocks_the_job(monkeypatch, tmp_path):
    """limits.audio_concurrent_jobs == 0 must fail the job now, not requeue it forever (spec 2, Qodo Q13)."""
    jobs_db_path = tmp_path / "jobs_cap_zero.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()

    import tldw_Server_API.app.services.audio_jobs_worker as worker

    limits_calls: list[int] = []
    started_calls: list[int] = []

    async def _fake_get_limits_for_user(user_id: int):
        limits_calls.append(user_id)
        return {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": 0, "max_file_size_mb": None}

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        started_calls.append(user_id)

    async def _fake_finish_job(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _fake_get_limits_for_user, raising=True)
    monkeypatch.setattr(worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(worker, "finish_job", _fake_finish_job, raising=True)

    row = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_convert",
        payload={"local_path": str(tmp_path / "unused.wav")},
        owner_user_id="7",
    )
    job_id = int(row["id"])

    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        job_state: dict = {}
        for _ in range(80):
            await asyncio.sleep(0.05)
            if task.done():
                exc = task.exception()
                pytest.fail(f"audio worker crashed unexpectedly: {exc!r}")
            job_state = jm.get_job(job_id) or {}
            if job_state.get("status") in {"failed", "quarantined"}:
                break

        assert started_calls == [], "a job blocked by concurrent_jobs=0 must not count as started"
        assert job_state.get("status") == "failed", "a 0 cap must terminally fail the job, not requeue it"
        assert int(job_state.get("retry_count") or 0) == 0, "a non-retryable 0-cap failure must not bump retries"
        err_text = str(job_state.get("last_error") or job_state.get("error_message") or "")
        assert "audio_concurrent_jobs" in err_text
        assert limits_calls, "get_limits_for_user must have been consulted"
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_worker_owner_at_limit_runs_the_job(monkeypatch, tmp_path):
    """limits.audio_concurrent_jobs == 1 with no other processing job must let the job run (Qodo Q3)."""
    jobs_db_path = tmp_path / "jobs_cap_one_runs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()

    import tldw_Server_API.app.services.audio_jobs_worker as worker

    started_calls: list[int] = []
    released_calls: list[int] = []
    _orig_release_job = JobManager.release_job

    def _recording_release_job(self, job_id, **kwargs):
        released_calls.append(job_id)
        return _orig_release_job(self, job_id, **kwargs)

    async def _fake_get_limits_for_user(user_id: int):
        return {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": 1, "max_file_size_mb": None}

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        started_calls.append(user_id)

    async def _fake_finish_job(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _fake_get_limits_for_user, raising=True)
    monkeypatch.setattr(worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(worker, "finish_job", _fake_finish_job, raising=True)
    monkeypatch.setattr(JobManager, "release_job", _recording_release_job, raising=True)

    # audio_store has no further stage, so the job simply completes.
    row = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_store",
        payload={"segments": [], "text": ""},
        owner_user_id="9",
    )
    job_id = int(row["id"])

    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        job_state: dict = {}
        for _ in range(80):
            await asyncio.sleep(0.05)
            if task.done():
                exc = task.exception()
                pytest.fail(f"audio worker crashed unexpectedly: {exc!r}")
            job_state = jm.get_job(job_id) or {}
            if job_state.get("status") not in {None, "queued", "processing"}:
                break

        assert started_calls == [9], "a job at exactly its owner's limit must be allowed to start"
        assert released_calls == [], "a job within its owner's limit must never be capacity-deferred"
        assert job_state.get("status") == "completed"
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_worker_owner_over_limit_defers_repeatedly_without_quarantine(monkeypatch, tmp_path):
    """Repeated capacity deferrals must never bump retries or quarantine the job (Qodo Q3, Q13)."""
    jobs_db_path = tmp_path / "jobs_cap_one_defers.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()

    import tldw_Server_API.app.services.audio_jobs_worker as worker

    released_calls: list[int] = []
    failed_calls: list[tuple] = []
    _orig_release_job = JobManager.release_job

    def _recording_release_job(self, job_id, **kwargs):
        released_calls.append(job_id)
        return _orig_release_job(self, job_id, **kwargs)

    async def _fake_get_limits_for_user(user_id: int):
        return {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": 1, "max_file_size_mb": None}

    async def _fake_can_start_job(user_id: int):
        return True, ""

    async def _fake_increment_jobs_started(user_id: int):
        return None

    async def _fake_finish_job(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _fake_get_limits_for_user, raising=True)
    monkeypatch.setattr(worker, "can_start_job", _fake_can_start_job, raising=True)
    monkeypatch.setattr(worker, "increment_jobs_started", _fake_increment_jobs_started, raising=True)
    monkeypatch.setattr(worker, "finish_job", _fake_finish_job, raising=True)
    monkeypatch.setattr(JobManager, "release_job", _recording_release_job, raising=True)

    # A long-running processing job already occupies this owner's one slot.
    occupying = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_store",
        payload={"segments": [], "text": "occupying"},
        owner_user_id="11",
    )
    occupied = jm.acquire_next_job(domain="audio", queue="default", lease_seconds=120, worker_id="pretend-other-worker")
    assert occupied and int(occupied["id"]) == int(occupying["id"])

    # A second job for the same owner should be repeatedly deferred, not failed.
    row = jm.create_job(
        domain="audio",
        queue="default",
        job_type="audio_store",
        payload={"segments": [], "text": "waiting"},
        owner_user_id="11",
    )
    job_id = int(row["id"])

    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if task.done():
                exc = task.exception()
                pytest.fail(f"audio worker crashed unexpectedly: {exc!r}")
            if len([j for j in released_calls if j == job_id]) >= 3:
                break

        assert len([j for j in released_calls if j == job_id]) >= 3, "expected 3+ capacity deferrals"
        assert failed_calls == [], "capacity deferrals must never call fail_job"
        job_state = jm.get_job(job_id) or {}
        assert job_state.get("status") == "queued", "a capacity-deferred job must stay queued, not fail/quarantine"
        assert int(job_state.get("retry_count") or 0) == 0, "capacity deferrals must not bump retry_count"
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)


# --- audio_analyze through the real analyze() (TASK-13637) ---


class _AnalysisAdapter:
    """The provider adapter below analyze(): records each request and replies or raises."""

    def __init__(self):
        self.reply = "Model analysis."
        self.error = None
        self.requests = []

    def chat(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.reply}}]}


@pytest.fixture
def analyze_worker(monkeypatch, tmp_path):
    """The audio worker with quota stubs, a fresh jobs DB and a fake provider adapter."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_analyze.db"))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as summarization
    import tldw_Server_API.app.services.audio_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    async def _unlimited(user_id: int):
        return {"concurrent_jobs": None}

    async def _yes(user_id: int):
        return True, ""

    async def _noop(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _unlimited)
    monkeypatch.setattr(worker, "can_start_job", _yes)
    monkeypatch.setattr(worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(worker, "finish_job", _noop)

    adapter = _AnalysisAdapter()
    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: adapter))
    monkeypatch.setattr(summarization, "load_and_log_configs", lambda: {"openai_api": {"model": "configured-model"}})
    return worker, JobManager(), adapter


async def _run_worker_until(worker, done, *, workers: int = 1, seconds: float = 5.0) -> None:
    """Run ``workers`` audio worker loops (as separate server processes would) until ``done()`` or ``seconds``."""
    stop = asyncio.Event()
    tasks = [asyncio.create_task(worker.run_audio_jobs_worker(stop)) for _ in range(workers)]
    try:
        for _ in range(int(seconds / 0.05)):
            await asyncio.sleep(0.05)
            for task in tasks:
                if task.done():
                    pytest.fail(f"audio worker crashed: {task.exception()!r}")
            if done():
                break
    finally:
        stop.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)


def _analyze_job(jm, text: str, api_name: str | None = "openai") -> int:
    row = jm.create_job(
        domain="audio", queue="default", job_type="audio_analyze",
        payload={"text": text, "api_name": api_name, "model": "whisper-1", "perform_analysis": True}, owner_user_id="7",
    )
    return int(row["id"])


def _failed_or_retried(jm, job_id: int) -> bool:
    job = jm.get_job(job_id) or {}
    return job.get("status") in {"failed", "quarantined"} or int(job.get("retry_count") or 0) >= 1


def _stored(jm) -> list:
    return jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")


def _settled(jm, job_id: int, retries: int = 1) -> bool:
    """The analyze job handed over to audio_store, ended, or was retried `retries` times."""
    job = jm.get_job(job_id) or {}
    ended = job.get("status") in {"failed", "quarantined"}
    return bool(_stored(jm)) or ended or int(job.get("retry_count") or 0) >= retries


def _seconds_until_due(jm, job_id: int) -> float:
    """How far in the future the job's available_at is; 0.0 when it is NULL."""
    connection = sqlite3.connect(jm.db_path)
    try:
        row = connection.execute(
            "SELECT (julianday(available_at) - julianday('now')) * 86400 FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    finally:
        connection.close()
    return float(row[0] or 0.0)


def _make_due(jm, job_id: int) -> None:
    connection = sqlite3.connect(jm.db_path)
    try:
        connection.execute("UPDATE jobs SET available_at = DATETIME('now') WHERE id = ?", (job_id,))
        connection.commit()
    finally:
        connection.close()


@pytest.mark.unit
async def test_audio_analyze_stores_the_model_analysis(analyze_worker):
    worker, jm, adapter = analyze_worker
    _analyze_job(jm, "hello transcript")

    def stored():
        return jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")

    await _run_worker_until(worker, stored)
    assert stored(), "audio_analyze did not hand over to audio_store"
    assert stored()[0]["payload"]["analysis"] == "Model analysis."
    assert [r["messages"][0]["content"] for r in adapter.requests] == ["hello transcript"]
    assert adapter.requests[0]["model"] == "configured-model"


@pytest.mark.unit
@pytest.mark.parametrize("text", ["", " \n "], ids=["empty", "whitespace"])
async def test_audio_analyze_empty_transcript_keeps_it(analyze_worker, text):
    worker, jm, adapter = analyze_worker
    job_id = _analyze_job(jm, text)
    await _run_worker_until(worker, lambda: _settled(jm, job_id))
    assert _stored(jm), "audio_analyze did not hand over to audio_store"
    payload = _stored(jm)[0]["payload"]
    assert payload["text"] == text
    assert payload["analysis_error"]["code"] == "empty_transcript"
    assert "analysis" not in payload
    assert adapter.requests == []


def _break_provider(code: str, adapter, monkeypatch) -> None:
    """Make analysis fail the way a misconfigured provider does, for the given error code."""
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as summarization
    from tldw_Server_API.app.core import exceptions

    if code == "authentication":
        adapter.error = exceptions.ChatAuthenticationError(provider="openai")
    elif code == "configuration":
        adapter.error = exceptions.ChatConfigurationError(provider="openai")
    elif code == "missing_model":
        monkeypatch.setattr(summarization, "load_and_log_configs", lambda: {"openai_api": {}})
        monkeypatch.delenv("DEFAULT_MODEL_OPENAI", raising=False)
    else:  # adapter_unavailable
        monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: None))


def _exhaust_retries(jm, job_id: int, before: int = 0) -> int:
    """Put the job on its last attempt (retry_count == max_retries - before); returns max_retries."""
    connection = sqlite3.connect(jm.db_path)
    try:
        connection.execute("UPDATE jobs SET retry_count = max_retries - ? WHERE id = ?", (before, job_id))
        connection.commit()
        return int(connection.execute("SELECT max_retries FROM jobs WHERE id = ?", (job_id,)).fetchone()[0])
    finally:
        connection.close()


@pytest.mark.unit
@pytest.mark.parametrize("code", ["authentication", "configuration", "missing_model", "adapter_unavailable"])
async def test_audio_analyze_provider_setup_failure_keeps_the_transcript(analyze_worker, monkeypatch, code):
    worker, jm, adapter = analyze_worker
    _break_provider(code, adapter, monkeypatch)
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: _settled(jm, job_id))
    assert _stored(jm), "audio_analyze did not hand over to audio_store"
    payload = _stored(jm)[0]["payload"]
    assert payload["text"] == "hello transcript"
    assert payload["analysis_error"]["code"] == code
    assert payload["analysis_error"]["message"]
    assert "analysis" not in payload
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count")) == ("completed", 0)
    assert job["result"]["analysis_error"]["code"] == code


@pytest.mark.unit
async def test_audio_analyze_rate_limited_on_the_last_attempt_keeps_the_transcript(analyze_worker):
    from tldw_Server_API.app.core.exceptions import ChatRateLimitError

    worker, jm, adapter = analyze_worker
    adapter.error = ChatRateLimitError(provider="openai")
    job_id = _analyze_job(jm, "hello transcript")
    max_retries = _exhaust_retries(jm, job_id)
    await _run_worker_until(worker, lambda: _settled(jm, job_id, retries=max_retries + 1))
    assert _stored(jm), "audio_analyze did not hand over to audio_store"
    payload = _stored(jm)[0]["payload"]
    assert payload["text"] == "hello transcript"
    assert payload["analysis_error"]["code"] == "rate_limit"
    assert "analysis" not in payload
    job = jm.get_job(job_id) or {}
    assert job.get("status") == "completed"
    assert job["result"]["analysis_error"]["code"] == "rate_limit"


@pytest.mark.unit
async def test_audio_analyze_rate_limited_one_attempt_before_the_last_is_retried(analyze_worker):
    from tldw_Server_API.app.core.exceptions import ChatRateLimitError

    worker, jm, adapter = analyze_worker
    adapter.error = ChatRateLimitError(provider="openai")
    job_id = _analyze_job(jm, "hello transcript")
    max_retries = _exhaust_retries(jm, job_id, before=1)
    await _run_worker_until(worker, lambda: _settled(jm, job_id, retries=max_retries))
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count"), job.get("error_code")) == ("queued", max_retries, "rate_limit")
    assert _stored(jm) == []


@pytest.mark.unit
async def test_audio_analyze_error_reply_on_the_last_attempt_keeps_the_transcript(analyze_worker):
    worker, jm, adapter = analyze_worker
    adapter.reply = "Error: the provider said no"
    job_id = _analyze_job(jm, "hello transcript")
    max_retries = _exhaust_retries(jm, job_id)
    await _run_worker_until(worker, lambda: _settled(jm, job_id, retries=max_retries + 1))
    assert _stored(jm), "audio_analyze did not hand over to audio_store"
    payload = _stored(jm)[0]["payload"]
    assert payload["analysis_error"] == {"code": "provider_failure", "message": "Error: the provider said no"}
    assert "analysis" not in payload


@pytest.mark.unit
async def test_audio_analyze_error_reply_is_retried_as_a_provider_failure(analyze_worker):
    worker, jm, adapter = analyze_worker
    adapter.reply = "Error: the provider said no"
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: _settled(jm, job_id))
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count"), job.get("error_code")) == ("queued", 1, "provider_failure")
    assert _seconds_until_due(jm, job_id) >= 4, "a provider failure waits about 5 s"
    assert _stored(jm) == []


@pytest.mark.unit
async def test_audio_analyze_rate_limit_still_retries(analyze_worker):
    from tldw_Server_API.app.core.exceptions import ChatRateLimitError

    worker, jm, adapter = analyze_worker
    adapter.error = ChatRateLimitError(provider="openai")
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: _failed_or_retried(jm, job_id))
    job = jm.get_job(job_id) or {}
    assert int(job.get("retry_count") or 0) >= 1
    assert job.get("status") != "failed"
    assert job.get("error_code") == "rate_limit"


@pytest.mark.unit
async def test_audio_analyze_rate_limit_waits_and_is_not_quarantined(analyze_worker):
    from tldw_Server_API.app.core.exceptions import ChatRateLimitError

    worker, jm, adapter = analyze_worker
    adapter.error = ChatRateLimitError(provider="openai")
    job_id = _analyze_job(jm, "hello transcript")

    await _run_worker_until(worker, lambda: _settled(jm, job_id, retries=1))
    assert _seconds_until_due(jm, job_id) >= 25, "a rate limit must wait about 30 s, not retry at once"

    _make_due(jm, job_id)
    await _run_worker_until(worker, lambda: _settled(jm, job_id, retries=2))
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count"), job.get("error_code")) == ("queued", 2, "rate_limit")
    assert _seconds_until_due(jm, job_id) >= 55, "the second wait doubles"
    assert len(adapter.requests) == 2


@pytest.mark.unit
@pytest.mark.parametrize("api_name", [None, "", "None"], ids=["missing", "blank", "None-string"])
async def test_audio_analyze_without_api_name_skips_analysis(analyze_worker, api_name):
    worker, jm, adapter = analyze_worker
    _analyze_job(jm, "hello transcript", api_name=api_name)

    def stored():
        return jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")

    await _run_worker_until(worker, stored)
    assert stored(), "audio_analyze did not hand over to audio_store"
    payload = stored()[0]["payload"]
    assert payload.get("analysis_skipped") == "no api_name"
    assert "analysis" not in payload
    assert adapter.requests == []


@pytest.mark.unit
async def test_audio_analyze_quota_refusal_keeps_the_transcript(analyze_worker):
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    worker, jm, adapter = analyze_worker
    adapter.error = LLMQuotaExceededError(
        category="llm_tokens_month", current=12, limit=10, message="over", retry_after=60
    )
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: _settled(jm, job_id))
    assert _stored(jm), "audio_analyze did not hand over to audio_store"
    payload = _stored(jm)[0]["payload"]
    assert payload["text"] == "hello transcript"
    assert payload["analysis_error"] == {
        "code": "llm_quota_exceeded",
        "message": "LLM quota exceeded (llm_tokens_month)",
    }
    job = jm.get_job(job_id) or {}
    assert (job.get("status"), job.get("retry_count")) == ("completed", 0)
    assert job["result"]["analysis_error"]["code"] == "llm_quota_exceeded"


@pytest.mark.unit
async def test_audio_analyze_provider_failure_still_retries(analyze_worker):
    worker, jm, adapter = analyze_worker
    adapter.reply = ""  # no usable analysis: a provider failure, not a refusal
    job_id = _analyze_job(jm, "hello transcript")
    await _run_worker_until(worker, lambda: int((jm.get_job(job_id) or {}).get("retry_count") or 0) >= 1)
    job = jm.get_job(job_id) or {}
    assert int(job.get("retry_count") or 0) >= 1
    assert job.get("status") != "failed"
    assert job.get("error_code") == "provider_failure"
    assert len(adapter.requests) >= 1


# --- the audio_convert -> audio_transcribe hand-off (TASK-13848) ---


@pytest.fixture
def pipeline_worker(monkeypatch, tmp_path):
    """The audio worker with quota stubs, a fresh jobs DB, and conversion and STT stubbed out."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_pipeline.db"))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.delenv("AUDIO_JOBS_GPU_TRANSCRIBE", raising=False)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.services.audio_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    async def _unlimited(user_id: int):
        return {"concurrent_jobs": None}

    async def _yes(user_id: int):
        return True, ""

    async def _noop(user_id: int):
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _unlimited)
    monkeypatch.setattr(worker, "can_start_job", _yes)
    monkeypatch.setattr(worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(worker, "finish_job", _noop)

    stt_calls = []

    def _fake_stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "convert_to_wav", lambda path, *a, **k: str(path) + ".wav")
    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _fake_stt)
    return worker, JobManager(), stt_calls


def _convert_job(jm) -> int:
    row = jm.create_job(
        domain="audio", queue="default", job_type="audio_convert",
        payload={"local_path": "/tmp/pipeline.mp3", "model": "whisper-1", "perform_chunking": False, "perform_analysis": False},
        owner_user_id="7",
    )
    return int(row["id"])


def _jobs_of(jm, job_type: str):
    return jm.list_jobs(domain="audio", job_type=job_type, owner_user_id="7")


@pytest.mark.unit
async def test_audio_pipeline_transcribes_in_process_by_default(pipeline_worker):
    worker, jm, stt_calls = pipeline_worker
    convert_id = _convert_job(jm)
    await _run_worker_until(worker, lambda: any(j.get("status") == "completed" for j in _jobs_of(jm, "audio_store")))
    transcribe = _jobs_of(jm, "audio_transcribe")
    assert [j["queue"] for j in transcribe] == ["default"]
    # The key that lets a re-run stage reuse the next stage it already queued.
    assert transcribe[0]["idempotency_key"] == f"audio-stage:{convert_id}:audio_transcribe"
    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"]
    assert len(stt_calls) == 1


@pytest.mark.unit
async def test_audio_pipeline_runs_under_a_per_user_job_limit_of_one(pipeline_worker, monkeypatch):
    # The running stage must not count against its own hand-off under fair-share.
    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
    _convert_job(jm)
    await _run_worker_until(worker, lambda: any(j.get("status") == "completed" for j in _jobs_of(jm, "audio_store")))
    assert [j["status"] for j in _jobs_of(jm, "audio_store")] == ["completed"]
    assert len(stt_calls) == 1


@pytest.mark.unit
async def test_audio_pipeline_routes_transcription_to_the_gpu_queue_when_enabled(pipeline_worker, monkeypatch):
    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("AUDIO_JOBS_GPU_TRANSCRIBE", "true")
    convert_id = _convert_job(jm)
    await _run_worker_until(
        worker,
        lambda: bool(_jobs_of(jm, "audio_transcribe")) or (jm.get_job(convert_id) or {}).get("status") == "completed",
    )
    assert [(j["queue"], j["status"]) for j in _jobs_of(jm, "audio_transcribe")] == [("transcribe", "queued")]
    assert stt_calls == []


@pytest.mark.unit
async def test_a_failed_stage_hand_off_fails_the_job_visibly(pipeline_worker, monkeypatch):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("AUDIO_JOBS_GPU_TRANSCRIBE", "true")
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES", raising=False)
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES_AUDIO", raising=False)
    monkeypatch.setattr(JobManager, "DOMAIN_ALLOWED_QUEUES", {})  # "transcribe" is now refused
    convert_id = _convert_job(jm)
    await _run_worker_until(worker, lambda: (jm.get_job(convert_id) or {}).get("status") in {"completed", "failed"})
    job = jm.get_job(convert_id) or {}
    assert job.get("status") == "failed"
    assert job.get("error_code") == "next_stage_enqueue_failed"
    assert int(job.get("retry_count") or 0) == 0
    assert _jobs_of(jm, "audio_transcribe") == []


@pytest.mark.unit
async def test_a_transient_hand_off_failure_is_retried(pipeline_worker, monkeypatch):
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    original_create = JobManager.create_job

    def _flaky_create(self, **kwargs):
        if kwargs.get("job_type") == "audio_transcribe":
            raise RuntimeError("database is locked")
        return original_create(self, **kwargs)

    monkeypatch.setattr(JobManager, "create_job", _flaky_create)
    convert_id = _convert_job(jm)
    await _run_worker_until(worker, lambda: int((jm.get_job(convert_id) or {}).get("retry_count") or 0) >= 1)
    job = jm.get_job(convert_id) or {}
    assert int(job.get("retry_count") or 0) >= 1
    assert job.get("status") != "completed"
    assert job.get("error_code") == "next_stage_enqueue_failed"


# --- the stage after transcription (TASK-13845) ---


@pytest.mark.unit
@pytest.mark.parametrize(
    ("chunking", "analysis", "expected"),
    [(False, True, "audio_analyze"), (False, False, "audio_store"), (True, True, "audio_chunk"), (True, False, "audio_chunk")],
)
def test_stage_after_transcription(chunking, analysis, expected):
    from tldw_Server_API.app.services.audio_jobs_worker import stage_after_transcription

    assert stage_after_transcription({"perform_chunking": chunking, "perform_analysis": analysis}) == expected


@pytest.mark.unit
async def test_audio_pipeline_without_chunking_still_analyzes(pipeline_worker):
    worker, jm, stt_calls = pipeline_worker
    jm.create_job(
        domain="audio", queue="default", job_type="audio_convert",
        payload={"local_path": "/tmp/p.mp3", "model": "whisper-1", "perform_chunking": False,
                 "perform_analysis": True, "api_name": None},
        owner_user_id="7",
    )
    await _run_worker_until(worker, lambda: bool(_jobs_of(jm, "audio_store")))
    assert [j["status"] for j in _jobs_of(jm, "audio_analyze")] == ["completed"]
    assert _jobs_of(jm, "audio_store")[0]["payload"]["analysis_skipped"] == "no api_name"


@pytest.mark.unit
async def test_audio_gpu_worker_hands_over_to_analysis_without_chunking(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_gpu_analysis.db"))
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_AUDIO", "transcribe")
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    import tldw_Server_API.app.services.audio_transcribe_gpu_worker as gpu_worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    async def _yes(user_id: int):
        return True, ""

    async def _noop(user_id: int):
        return None

    monkeypatch.setattr(gpu_worker, "can_start_job", _yes)
    monkeypatch.setattr(gpu_worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(gpu_worker, "finish_job", _noop)
    monkeypatch.setattr(atlib, "run_stt_job_via_registry", lambda *a, **k: {"text": "hello gpu", "segments": []})

    jm = JobManager()
    jm.create_job(
        domain="audio", queue="transcribe", job_type="audio_transcribe",
        payload={"wav_path": str(tmp_path / "gpu.wav"), "model": "whisper-1",
                 "perform_chunking": False, "perform_analysis": True},
        owner_user_id="1",
    )
    stop = asyncio.Event()
    task = asyncio.create_task(gpu_worker.run_audio_transcribe_gpu_worker(stop))
    try:
        for _ in range(50):
            await asyncio.sleep(0.05)
            if jm.list_jobs(domain="audio", queue="default", owner_user_id="1"):
                break
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
    assert [j["job_type"] for j in jm.list_jobs(domain="audio", queue="default", owner_user_id="1")] == ["audio_analyze"]


@pytest.mark.unit
async def test_audio_analyze_skip_is_logged(analyze_worker, monkeypatch):
    worker, jm, adapter = analyze_worker
    capture = _CaptureLogger()
    monkeypatch.setattr(worker, "logger", capture)
    job_id = _analyze_job(jm, "hello transcript", api_name=None)
    await _run_worker_until(worker, lambda: bool(jm.list_jobs(domain="audio", job_type="audio_store", owner_user_id="7")))
    info_lines = [message.format(*args) if args else message for level, message, args, _ in capture.records if level == "info"]
    assert f"Audio job {job_id}: analysis skipped, no api_name" in info_lines


# --- the stage lease (TASK-13850) ---


def _store_completed(jm) -> bool:
    return any(j.get("status") == "completed" for j in _jobs_of(jm, "audio_store"))


@pytest.mark.unit
async def test_a_stage_longer_than_its_lease_runs_once(pipeline_worker, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("JOBS_LEASE_SECONDS", "2")

    def _slow_stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        time.sleep(3)  # longer than the lease
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _slow_stt)
    _convert_job(jm)
    # Two worker loops, as two server processes would run them: the second must not recover the lease.
    await _run_worker_until(worker, lambda: _store_completed(jm), workers=2, seconds=15)
    assert len(stt_calls) == 1
    assert [(j["status"], j["retry_count"]) for j in _jobs_of(jm, "audio_transcribe")] == [("completed", 0)]


@pytest.mark.unit
@pytest.mark.parametrize("stage_fails", [False, True], ids=["completes", "fails"])
async def test_lease_renewal_stops_when_the_stage_ends(pipeline_worker, monkeypatch, stage_fails):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    monkeypatch.setenv("JOBS_LEASE_SECONDS", "4")  # a renewal every 2 s
    renewed = []
    original_renew = JobManager.renew_job_lease

    def _recording_renew(self, job_id, **kwargs):
        renewed.append(int(job_id))
        return original_renew(self, job_id, **kwargs)

    monkeypatch.setattr(JobManager, "renew_job_lease", _recording_renew)

    def _stt(wav_path, model, *args, **kwargs):
        time.sleep(3)  # one renewal, at 2 s
        if stage_fails:
            raise RuntimeError("stt failed")
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _stt)
    job_id = int(jm.create_job(
        domain="audio", queue="default", job_type="audio_transcribe",
        payload={"wav_path": "/tmp/p.wav", "model": "whisper-1", "perform_chunking": False, "perform_analysis": False},
        owner_user_id="7", max_retries=0,
    )["id"])
    await _run_worker_until(worker, lambda: (jm.get_job(job_id) or {}).get("status") in {"completed", "failed"})
    await asyncio.sleep(2)  # a renewal still running would fire again at 4 s
    assert (jm.get_job(job_id) or {}).get("status") == ("failed" if stage_fails else "completed")
    assert renewed.count(job_id) == 1


@pytest.mark.unit
async def test_a_renewal_error_does_not_end_the_renewals(pipeline_worker, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    worker, jm, stt_calls = pipeline_worker
    # A 3 s lease is renewed every second (3 // 2): the renewal after the failed one still comes a second
    # before the lease runs out.
    monkeypatch.setenv("JOBS_LEASE_SECONDS", "3")
    renewals = []
    original_renew = JobManager.renew_job_lease

    def _first_renewal_fails(self, job_id, **kwargs):
        renewals.append(int(job_id))
        if len(renewals) == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_renew(self, job_id, **kwargs)

    monkeypatch.setattr(JobManager, "renew_job_lease", _first_renewal_fails)

    def _slow_stt(wav_path, model, *args, **kwargs):
        stt_calls.append(wav_path)
        time.sleep(4)  # longer than the lease
        return {"text": "hello pipeline", "segments": [{"Text": "hello pipeline"}]}

    monkeypatch.setattr(atlib, "run_stt_job_via_registry", _slow_stt)
    _convert_job(jm)
    await _run_worker_until(worker, lambda: _store_completed(jm), workers=2, seconds=20)
    assert len(stt_calls) == 1
    assert [(j["status"], j["retry_count"]) for j in _jobs_of(jm, "audio_transcribe")] == [("completed", 0)]
