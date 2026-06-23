"""Unit tests for Audio Studio Jobs worker handlers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.Audio_Studio.jobs import JOB_TYPE_GENERATE, JOB_TYPE_RENDER
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
    workflow: str = "narration"
    current_revision_id: str = "rev_001"


class _FakeCollectionsDb:
    def __init__(self) -> None:
        self.project = _Project()
        self.artifacts: list[dict[str, Any]] = []
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
        self.artifacts.append(kwargs)
        return type("Artifact", (), kwargs)

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


@pytest.mark.asyncio
async def test_generation_handler_dispatches_provider_and_records_artifact() -> None:
    db = _FakeCollectionsDb()

    result = await handle_audio_studio_job(
        _generation_job(),
        collections_db=db,
        provider_registry=_FakeRegistry(),
    )

    assert result["status"] == "completed"
    assert result["artifact_id"].startswith("art_")
    artifact = db.artifacts[0]
    assert artifact["project_row_id"] == db.project.id
    assert artifact["artifact_type"] == "generated_audio"
    assert artifact["provider"] == "tts"
    assert artifact["mime_type"] == "audio/mpeg"
    assert artifact["size_bytes"] == len(b"audio-bytes")
    assert json.loads(artifact["metadata_json"])["job_id"] == "job-uuid-99"
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
async def test_deferred_job_types_return_clear_result() -> None:
    result = await handle_audio_studio_job(
        {
            "uuid": "job-render-1",
            "job_type": JOB_TYPE_RENDER,
            "owner_user_id": "42",
            "payload": {"project_id": "ast_worker"},
        },
        collections_db=_FakeCollectionsDb(),
        provider_registry=_FakeRegistry(),
    )

    assert result == {
        "status": "deferred",
        "reason": "audio_studio_render_not_implemented",
    }
