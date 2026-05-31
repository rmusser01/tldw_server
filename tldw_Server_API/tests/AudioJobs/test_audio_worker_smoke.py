import os
import shutil
import asyncio
import tempfile
import numpy as np
import soundfile as sf
import pytest


pytestmark = pytest.mark.smoke


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def info(self, message, *args, **kwargs):
        self.records.append(("info", message, args, kwargs))

    def debug(self, message, *args, **kwargs):
        self.records.append(("debug", message, args, kwargs))

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
        return {"daily_minutes": 30.0, "concurrent_streams": 1, "concurrent_jobs": 0, "max_file_size_mb": 25}

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
        return {"daily_minutes": 30.0, "concurrent_streams": 1, "concurrent_jobs": 0, "max_file_size_mb": 25}

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

    from tldw_Server_API.app.core.exceptions import DownloadError
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files as audio_files

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
