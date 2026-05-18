from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Local_LLM.llamacpp_acquisition_jobs import (
    LLAMACPP_DOWNLOAD_JOB_TYPE,
)

pytestmark = pytest.mark.unit


class _FakeDownloadStream:
    def __init__(self, chunks: list[bytes], *, total_bytes: int | None = None) -> None:
        self._chunks = chunks
        self.total_bytes = total_bytes

    async def __aenter__(self) -> _FakeDownloadStream:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _RecordingJobManager:
    def __init__(self, *, status: str = "processing") -> None:
        self.status = status
        self.progress_updates: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []

    def update_job_progress(
        self,
        job_id: int,
        *,
        progress_percent: float | None = None,
        progress_message: str | None = None,
    ) -> bool:
        self.progress_updates.append(
            {
                "job_id": job_id,
                "progress_percent": progress_percent,
                "progress_message": progress_message,
            }
        )
        return True

    def get_job(self, job_id: int) -> dict[str, Any]:
        return {"id": job_id, "status": self.status}

    def finalize_cancelled(self, job_id: int, *, reason: str | None = None) -> bool:
        self.cancelled.append({"job_id": job_id, "reason": reason})
        self.status = "cancelled"
        return True


def _job_for(final_path: Path, *, payload_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": "download",
        "source_url": "https://example.com/releases/model.gguf",
        "source_label": "Example model",
        "destination_path": str(final_path),
        "expected_sha256": None,
        "expected_size_bytes": None,
        "overwrite": False,
        "register_asset": True,
        "warnings": ["dns warning"],
    }
    if payload_overrides:
        payload.update(payload_overrides)
    return {
        "id": 41,
        "job_type": LLAMACPP_DOWNLOAD_JOB_TYPE,
        "payload": payload,
        "status": "processing",
    }


def _asset_payload(path: Path) -> dict[str, object]:
    return {
        "asset_id": "gguf:registered",
        "kind": "gguf",
        "identity_basis": "resolved_path",
        "path": str(path),
        "resolved_path": str(path),
        "display_name": path.name,
        "source": "registered_path",
        "metadata": {},
        "capabilities": ["unknown"],
        "mmproj_asset_ids": [],
        "base_model_asset_ids": [],
        "warnings": [],
    }


def _allow_destination(monkeypatch: pytest.MonkeyPatch, worker: Any, models_dir: Path) -> None:
    monkeypatch.setattr(
        worker.acquisition_service,
        "_read_saved_config",
        lambda: {"models_dir": str(models_dir), "allowed_paths": []},
    )


@pytest.mark.asyncio
async def test_worker_downloads_validates_promotes_registers_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    content = b"llama-bytes"
    digest = hashlib.sha256(content).hexdigest()
    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)
    registered: list[Path] = []

    def _register(path: Path):
        registered.append(path)
        return _asset_payload(path)

    monkeypatch.setattr(worker.acquisition_service, "register_completed_download", _register)

    def _stream_factory(url: str, *, timeout_seconds: float):
        assert url == "https://example.com/releases/model.gguf"
        assert timeout_seconds > 0
        return _FakeDownloadStream([content[:5], content[5:]], total_bytes=len(content))

    job_manager = _RecordingJobManager()
    progress = worker._ProgressState()

    result = await worker.process_llamacpp_acquisition_job(
        _job_for(
            final_path,
            payload_overrides={
                "expected_sha256": digest,
                "expected_size_bytes": len(content),
            },
        ),
        job_manager=job_manager,
        progress=progress,
        stream_factory=_stream_factory,
    )

    assert final_path.read_bytes() == content
    assert list(final_path.parent.glob("*.partial")) == []
    assert registered == [final_path]
    assert result["status"] == "ready"
    assert result["asset_id"] == "gguf:registered"
    assert result["bytes"] == len(content)
    assert result["warnings"] == ["dns warning"]
    assert result["progress"]["bytes_downloaded"] == len(content)
    assert result["progress"]["total_bytes"] == len(content)
    assert "source_url" not in result
    assert progress.bytes_downloaded == len(content)
    assert progress.total_bytes == len(content)
    assert any(update["progress_message"] == "downloading" for update in job_manager.progress_updates)


@pytest.mark.asyncio
async def test_worker_checksum_mismatch_removes_partial_and_final_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)
    registered: list[Path] = []
    monkeypatch.setattr(
        worker.acquisition_service,
        "register_completed_download",
        lambda path: registered.append(path) or _asset_payload(path),
    )

    def _stream_factory(_url: str, *, timeout_seconds: float):
        del timeout_seconds
        return _FakeDownloadStream([b"wrong-bytes"], total_bytes=len(b"wrong-bytes"))

    with pytest.raises(worker.LlamaCppAcquisitionJobError) as exc_info:
        await worker.process_llamacpp_acquisition_job(
            _job_for(final_path, payload_overrides={"expected_sha256": "0" * 64}),
            job_manager=_RecordingJobManager(),
            progress=worker._ProgressState(),
            stream_factory=_stream_factory,
        )

    assert exc_info.value.retryable is False
    assert "checksum" in str(exc_info.value).lower()
    assert not final_path.exists()
    assert list(final_path.parent.glob("*.partial")) == []
    assert registered == []


@pytest.mark.asyncio
async def test_worker_cancellation_deletes_partial_and_skips_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)
    registered: list[Path] = []
    monkeypatch.setattr(
        worker.acquisition_service,
        "register_completed_download",
        lambda path: registered.append(path) or _asset_payload(path),
    )

    def _stream_factory(_url: str, *, timeout_seconds: float):
        del timeout_seconds
        return _FakeDownloadStream([b"first", b"second"], total_bytes=len(b"firstsecond"))

    calls = 0

    def _cancel_check() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    job_manager = _RecordingJobManager()
    result = await worker.process_llamacpp_acquisition_job(
        _job_for(final_path),
        job_manager=job_manager,
        progress=worker._ProgressState(),
        stream_factory=_stream_factory,
        cancel_check=_cancel_check,
    )

    assert result == {}
    assert not final_path.exists()
    assert list(final_path.parent.glob("*.partial")) == []
    assert registered == []
    assert job_manager.cancelled == [{"job_id": 41, "reason": "cancel requested during download"}]


@pytest.mark.asyncio
async def test_worker_cancel_check_error_falls_back_to_jobs_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)

    def _stream_factory(_url: str, *, timeout_seconds: float):
        del timeout_seconds
        raise AssertionError("download should not start when Jobs status is cancelled")

    def _cancel_check() -> bool:
        raise RuntimeError("transient cancel callback failure")

    job_manager = _RecordingJobManager(status="cancelled")
    result = await worker.process_llamacpp_acquisition_job(
        _job_for(final_path),
        job_manager=job_manager,
        progress=worker._ProgressState(),
        stream_factory=_stream_factory,
        cancel_check=_cancel_check,
    )

    assert result == {}
    assert not final_path.exists()
    assert list(final_path.parent.glob("*.partial")) == []
    assert job_manager.cancelled == [{"job_id": 41, "reason": "cancel requested before download"}]


@pytest.mark.asyncio
async def test_worker_throttles_small_chunk_progress_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)
    chunks = [b"x"] * 20

    def _stream_factory(_url: str, *, timeout_seconds: float):
        del timeout_seconds
        return _FakeDownloadStream(chunks, total_bytes=1_000_000_000)

    job_manager = _RecordingJobManager()
    progress = worker._ProgressState()
    result = await worker.process_llamacpp_acquisition_job(
        _job_for(final_path, payload_overrides={"register_asset": False}),
        job_manager=job_manager,
        progress=progress,
        stream_factory=_stream_factory,
    )

    downloading_updates = [
        update for update in job_manager.progress_updates if update["progress_message"] == "downloading"
    ]
    assert len(downloading_updates) == 1
    assert job_manager.progress_updates[-1]["progress_message"] == "completed"
    assert progress.bytes_downloaded == len(chunks)
    assert result["bytes"] == len(chunks)


@pytest.mark.asyncio
async def test_worker_existing_destination_without_overwrite_fails_terminally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)
    final_path.write_bytes(b"existing")

    def _stream_factory(_url: str, *, timeout_seconds: float):
        del timeout_seconds
        raise AssertionError("download should not start when destination exists")

    with pytest.raises(worker.LlamaCppAcquisitionJobError) as exc_info:
        await worker.process_llamacpp_acquisition_job(
            _job_for(final_path),
            job_manager=_RecordingJobManager(),
            progress=worker._ProgressState(),
            stream_factory=_stream_factory,
        )

    assert exc_info.value.retryable is False
    assert "already exists" in str(exc_info.value).lower()
    assert final_path.read_bytes() == b"existing"
    assert list(final_path.parent.glob("*.partial")) == []


@pytest.mark.asyncio
async def test_worker_errors_never_echo_source_url_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import llamacpp_acquisition_jobs_worker as worker

    final_path = tmp_path / "models" / "model.gguf"
    final_path.parent.mkdir()
    _allow_destination(monkeypatch, worker, final_path.parent)

    with pytest.raises(worker.LlamaCppAcquisitionJobError) as exc_info:
        await worker.process_llamacpp_acquisition_job(
            _job_for(
                final_path,
                payload_overrides={
                    "source_url": "https://user:pass@example.com/releases/model.gguf?token=secret",
                    "source_label": "bad source",
                },
            ),
            job_manager=_RecordingJobManager(),
            progress=worker._ProgressState(),
            stream_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("download should not start")
            ),
        )

    message = str(exc_info.value)
    assert "user" not in message
    assert "pass" not in message
    assert "secret" not in message
    assert "token=secret" not in message
    assert not final_path.exists()
    assert list(final_path.parent.glob("*.partial")) == []
