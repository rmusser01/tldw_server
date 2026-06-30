"""Unit tests for Audio Studio Jobs worker handlers."""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Audio_Studio.jobs import JOB_TYPE_EXPORT, JOB_TYPE_GENERATE, JOB_TYPE_RENDER
from tldw_Server_API.app.core.Audio_Studio.jobs_worker import (
    AudioStudioJobError,
    build_audio_studio_job_handler,
    handle_audio_studio_job,
)
from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationResult


pytestmark = pytest.mark.unit


@dataclass
class _Project:
    id: int = 7
    project_id: str = "ast_worker"
    title: str = "Worker Project"
    workflow: str = "narration"
    current_revision_id: str = "rev_001"


@dataclass
class _Artifact:
    artifact_id: str
    artifact_type: str
    provider: str | None
    storage_path: str | None
    mime_type: str | None
    size_bytes: int | None
    source_resource_kind: str | None
    source_resource_id: str | None
    source_revision_id: str | None
    content_hash: str
    metadata_json: str | None = None


class _FakeCollectionsDb:
    def __init__(self) -> None:
        self.user_id = "42"
        self.project = _Project()
        self.artifacts: list[Any] = []
        self.updated_jobs: list[dict[str, Any]] = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    def get_audio_studio_project_by_project_id(self, project_id: str) -> _Project:
        assert project_id == self.project.project_id
        return self.project

    def get_audio_studio_revision(self, revision_id: str) -> object:
        if revision_id != self.project.current_revision_id:
            raise KeyError("audio_studio_revision_not_found")
        return object()

    def create_audio_studio_artifact(self, **kwargs: Any) -> object:
        artifact = _Artifact(
            artifact_id=kwargs["artifact_id"],
            artifact_type=kwargs["artifact_type"],
            provider=kwargs.get("provider"),
            storage_path=kwargs.get("storage_path"),
            mime_type=kwargs.get("mime_type"),
            size_bytes=kwargs.get("size_bytes"),
            source_resource_kind=kwargs.get("source_resource_kind"),
            source_resource_id=kwargs.get("source_resource_id"),
            source_revision_id=kwargs.get("source_revision_id"),
            content_hash=kwargs["content_hash"],
            metadata_json=kwargs.get("metadata_json"),
        )
        self.artifacts.append(artifact)
        return artifact

    def list_audio_studio_artifacts(
        self,
        *,
        project_row_id: int,
        limit: int = 100,
        offset: int = 0,
        artifact_id: str | None = None,
    ) -> list[_Artifact]:
        assert project_row_id == self.project.id
        rows = [row for row in self.artifacts if artifact_id is None or row.artifact_id == artifact_id]
        return rows[offset : offset + limit]

    def update_audio_studio_generation_job(self, **kwargs: Any) -> object:
        self.updated_jobs.append(kwargs)
        return object()


class _FakeAdapter:
    provider_id = "tts"
    supported_kinds = frozenset({"speech"})

    async def generate(self, request, **kwargs):
        assert request.text == "Hello from section"
        assert kwargs["user_id"] == 42
        return AudioGenerationResult(
            mime_type="audio/mpeg",
            content_bytes=b"audio-bytes",
            provider="tts",
            metadata={"duration_ms": 1000},
        )


class _FakeRegistry:
    def get_adapter(self, provider: str, kind: str) -> _FakeAdapter:
        assert (provider, kind) == ("tts", "speech")
        return _FakeAdapter()


class _FailingAdapter:
    provider_id = "tts"
    supported_kinds = frozenset({"speech"})

    async def generate(self, request, **kwargs):
        raise AudioStudioJobError("provider_unavailable", retryable=True)


class _FailingRegistry:
    def get_adapter(self, provider: str, kind: str) -> _FailingAdapter:
        assert (provider, kind) == ("tts", "speech")
        return _FailingAdapter()


class _SecretLeakingAdapter:
    provider_id = "tts"
    supported_kinds = frozenset({"speech"})

    async def generate(self, request, **kwargs):
        raise RuntimeError("provider failed with secret-token-123")


class _SecretLeakingRegistry:
    def get_adapter(self, provider: str, kind: str) -> _SecretLeakingAdapter:
        assert (provider, kind) == ("tts", "speech")
        return _SecretLeakingAdapter()


def _generation_job() -> dict[str, Any]:
    return {
        "id": 99,
        "uuid": "job-uuid-99",
        "job_type": JOB_TYPE_GENERATE,
        "owner_user_id": "42",
        "payload": {
            "project_id": "ast_worker",
            "workflow": "narration",
            "kind": "speech",
            "provider": "tts",
            "text": "Hello from section",
            "prompt": None,
            "target_resource_kind": "section",
            "target_resource_id": "sec_001",
            "target_revision_id": "rev_001",
            "provider_options": {"voice": "af_heart"},
        },
    }


def _write_wav(path: Path, *, frames: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x01\x00" * frames)


def _render_job(artifact: _Artifact) -> dict[str, Any]:
    return {
        "id": 100,
        "uuid": "job-render-100",
        "job_type": JOB_TYPE_RENDER,
        "owner_user_id": "42",
        "payload": {
            "project_id": "ast_worker",
            "workflow": "narration",
            "kind": None,
            "provider": None,
            "target_resource_kind": "render",
            "target_resource_id": "render_001",
            "target_revision_id": "rev_001",
            "provider_options": {
                "render_type": "preview_mix",
                "output_format": "wav",
                "artifact_refs": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "source_revision_id": artifact.source_revision_id,
                        "content_hash": artifact.content_hash,
                    }
                ],
            },
        },
    }


def _export_job(artifact: _Artifact) -> dict[str, Any]:
    return {
        "id": 101,
        "uuid": "job-export-101",
        "job_type": JOB_TYPE_EXPORT,
        "owner_user_id": "42",
        "payload": {
            "project_id": "ast_worker",
            "workflow": "narration",
            "kind": None,
            "provider": None,
            "target_resource_kind": "export",
            "target_resource_id": "export_001",
            "target_revision_id": "rev_001",
            "provider_options": {
                "export_type": "zip_package",
                "artifact_refs": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "source_revision_id": artifact.source_revision_id,
                        "content_hash": artifact.content_hash,
                    }
                ],
            },
        },
    }


@pytest.mark.asyncio
async def test_generation_handler_dispatches_provider_and_records_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeCollectionsDb()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio_Studio.jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path / "outputs" / str(user_id),
    )

    result = await handle_audio_studio_job(
        _generation_job(),
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )

    assert result["status"] == "completed"
    assert result["artifact_id"].startswith("art_")
    artifact = db.artifacts[0]
    assert artifact.artifact_type == "generated_audio"
    assert artifact.provider == "tts"
    assert artifact.mime_type == "audio/mpeg"
    assert artifact.size_bytes == len(b"audio-bytes")
    assert artifact.storage_path is not None
    assert Path(artifact.storage_path).read_bytes() == b"audio-bytes"
    assert json.loads(artifact.metadata_json or "{}")["job_id"] == "job-uuid-99"
    updated = db.updated_jobs[0]
    assert updated["job_id"] == "job-uuid-99"
    assert updated["status"] == "completed"


@pytest.mark.asyncio
async def test_generation_handler_skips_stale_revision_without_provider_call() -> None:
    db = _FakeCollectionsDb()
    job = _generation_job()
    job["payload"]["target_revision_id"] = "rev_old"

    result = await handle_audio_studio_job(
        job,
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )

    assert result == {"status": "skipped", "reason": "stale_target_revision"}
    assert db.artifacts == []
    updated = db.updated_jobs[0]
    assert updated["job_id"] == "job-uuid-99"
    assert updated["status"] == "skipped"
    assert json.loads(updated["result_json"]) == {
        "status": "skipped",
        "reason": "stale_target_revision",
    }


@pytest.mark.asyncio
async def test_generation_handler_updates_generation_row_before_provider_failure() -> None:
    db = _FakeCollectionsDb()

    with pytest.raises(AudioStudioJobError, match="provider_unavailable"):
        await handle_audio_studio_job(
            _generation_job(),
            collections_db=db,
            provider_registry=_FailingRegistry(),
        )

    assert db.artifacts == []
    updated = db.updated_jobs[0]
    assert updated["job_id"] == "job-uuid-99"
    assert updated["status"] == "failed"
    assert json.loads(updated["result_json"]) == {
        "status": "failed",
        "reason": "provider_unavailable",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_generation_handler_redacts_known_secrets_from_failure_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_KEY", "secret-token-123")
    db = _FakeCollectionsDb()

    with pytest.raises(RuntimeError, match="provider failed"):
        await handle_audio_studio_job(
            _generation_job(),
            collections_db=db,
            provider_registry=_SecretLeakingRegistry(),
        )

    updated = db.updated_jobs[0]
    result = json.loads(updated["result_json"])
    assert result["status"] == "failed"
    assert result["reason"] == "provider failed with [REDACTED]"


@pytest.mark.asyncio
async def test_build_audio_studio_job_handler_closes_context_managed_db() -> None:
    db = _FakeCollectionsDb()
    handler = build_audio_studio_job_handler(
        collections_db_factory=lambda owner_user_id: db,
        provider_registry_factory=lambda: _FakeRegistry(),
    )

    result = await handler(_generation_job())

    assert result["status"] == "completed"
    assert db.closed is True


@pytest.mark.asyncio
async def test_render_handler_records_mix_and_manifest_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeCollectionsDb()
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)
    source = db.create_audio_studio_artifact(
        project_row_id=db.project.id,
        artifact_id="art_clip",
        artifact_type="clip_audio",
        provider="tts",
        output_id=None,
        storage_path=str(wav_path),
        mime_type="audio/wav",
        size_bytes=wav_path.stat().st_size,
        source_resource_kind="clip",
        source_resource_id="clip_001",
        source_revision_id="rev_001",
        content_hash="source-hash",
        metadata_json=json.dumps({"duration_ms": 10}),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio_Studio.jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path / "outputs" / str(user_id),
    )

    result = await handle_audio_studio_job(
        _render_job(source),
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )

    assert result["status"] == "completed"
    assert result["render_id"] == "render_001"
    assert result["mix_artifact_id"].endswith("_mix")
    assert result["manifest_artifact_id"].endswith("_manifest")
    assert {artifact.artifact_type for artifact in db.artifacts} >= {
        "clip_audio",
        "preview_mix",
        "render_manifest",
    }
    assert result["mix_artifact_id"].startswith("art_job-render-100_")
    assert result["manifest_artifact_id"].startswith("art_job-render-100_")


@pytest.mark.asyncio
async def test_export_handler_records_job_scoped_package_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeCollectionsDb()
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)
    source = db.create_audio_studio_artifact(
        project_row_id=db.project.id,
        artifact_id="art_clip",
        artifact_type="clip_audio",
        provider="tts",
        output_id=None,
        storage_path=str(wav_path),
        mime_type="audio/wav",
        size_bytes=wav_path.stat().st_size,
        source_resource_kind="clip",
        source_resource_id="clip_001",
        source_revision_id="rev_001",
        content_hash="source-hash",
        metadata_json=json.dumps({"duration_ms": 10}),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio_Studio.jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path / "outputs" / str(user_id),
    )

    result = await handle_audio_studio_job(
        _export_job(source),
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )

    assert result["status"] == "completed"
    assert result["package_artifact_id"].startswith("art_job-export-101_")
    assert result["manifest_artifact_id"].startswith("art_job-export-101_")
    assert {artifact.artifact_type for artifact in db.artifacts} >= {
        "clip_audio",
        "package",
        "export_manifest",
    }


@pytest.mark.asyncio
async def test_render_and_export_handlers_skip_stale_revisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeCollectionsDb()
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)
    source = db.create_audio_studio_artifact(
        project_row_id=db.project.id,
        artifact_id="art_clip",
        artifact_type="clip_audio",
        provider="tts",
        output_id=None,
        storage_path=str(wav_path),
        mime_type="audio/wav",
        size_bytes=wav_path.stat().st_size,
        source_resource_kind="clip",
        source_resource_id="clip_001",
        source_revision_id="rev_001",
        content_hash="source-hash",
        metadata_json=json.dumps({"duration_ms": 10}),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio_Studio.jobs_worker.DatabasePaths.get_user_outputs_dir",
        lambda user_id: tmp_path / "outputs" / str(user_id),
    )
    render_job = _render_job(source)
    render_job["payload"]["target_revision_id"] = "rev_old"
    export_job = _export_job(source)
    export_job["payload"]["target_revision_id"] = "rev_old"

    render_result = await handle_audio_studio_job(
        render_job,
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )
    export_result = await handle_audio_studio_job(
        export_job,
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )

    assert render_result == {"status": "skipped", "reason": "stale_target_revision"}
    assert export_result == {"status": "skipped", "reason": "stale_target_revision"}
