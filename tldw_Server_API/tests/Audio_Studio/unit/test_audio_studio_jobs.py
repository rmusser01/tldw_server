"""Unit tests for Audio Studio Jobs enqueue helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.Audio_Studio.jobs import (
    AUDIO_STUDIO_DOMAIN,
    AUDIO_STUDIO_QUEUE,
    JOB_TYPE_EXPORT,
    JOB_TYPE_GENERATE,
    JOB_TYPE_MIGRATE,
    JOB_TYPE_RENDER,
    build_audio_studio_idempotency_key,
    enqueue_audio_studio_export_job,
    enqueue_audio_studio_generation_job,
    enqueue_audio_studio_migration_job,
    enqueue_audio_studio_render_job,
)


pytestmark = pytest.mark.unit


@dataclass
class _Project:
    id: int = 7
    project_id: str = "ast_jobs"
    workflow: str = "narration"
    current_revision_id: str = "rev_001"


class _FakeJobsManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.rows: dict[str, dict[str, Any]] = {}

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        key = kwargs["idempotency_key"]
        row = self.rows.setdefault(
            key,
            {
                "id": len(self.rows) + 1,
                "uuid": f"job-{len(self.rows) + 1}",
                "status": "queued",
                "payload": kwargs["payload"],
            },
        )
        return row


class _FakeCollectionsDb:
    def __init__(self) -> None:
        self.project = _Project()
        self.recorded_jobs: list[dict[str, Any]] = []

    def get_audio_studio_project_by_project_id(self, project_id: str) -> _Project:
        assert project_id == self.project.project_id
        return self.project

    def get_audio_studio_revision(self, revision_id: str) -> object:
        if revision_id != self.project.current_revision_id:
            raise KeyError("audio_studio_revision_not_found")
        return object()

    def record_audio_studio_generation_job(self, **kwargs: Any) -> object:
        self.recorded_jobs.append(kwargs)
        return object()


def test_generation_enqueue_builds_secret_free_idempotent_job() -> None:
    jm = _FakeJobsManager()
    db = _FakeCollectionsDb()

    accepted = enqueue_audio_studio_generation_job(
        jm=jm,
        collections_db=db,
        user_id="42",
        project_id="ast_jobs",
        workflow="narration",
        kind="speech",
        provider="tts",
        target_resource_kind="section",
        target_resource_id="sec_001",
        target_revision_id="rev_001",
        idempotency_key="client-key-123456",
        options={"voice": "af_heart", "api_key": "must-not-store"},
    )
    duplicate = enqueue_audio_studio_generation_job(
        jm=jm,
        collections_db=db,
        user_id="42",
        project_id="ast_jobs",
        workflow="narration",
        kind="speech",
        provider="tts",
        target_resource_kind="section",
        target_resource_id="sec_001",
        target_revision_id="rev_001",
        idempotency_key="client-key-123456",
        options={"voice": "af_heart", "api_key": "must-not-store"},
    )

    assert accepted.job_id == duplicate.job_id == "job-1"
    assert accepted.status == "queued"
    call = jm.calls[0]
    assert call["domain"] == AUDIO_STUDIO_DOMAIN
    assert call["queue"] == AUDIO_STUDIO_QUEUE
    assert call["job_type"] == JOB_TYPE_GENERATE
    assert call["owner_user_id"] == "42"
    assert call["project_id"] == db.project.id
    assert call["idempotency_key"] == build_audio_studio_idempotency_key(
        user_id="42",
        project_id="ast_jobs",
        job_type=JOB_TYPE_GENERATE,
        target_resource_kind="section",
        target_resource_id="sec_001",
        target_revision_id="rev_001",
        caller_idempotency_key="client-key-123456",
    )
    assert "must-not-store" not in json.dumps(call["payload"])
    assert db.recorded_jobs[0]["job_id"] == "job-1"
    assert db.recorded_jobs[0]["request_json"]


def test_generation_enqueue_rejects_stale_revision_before_job_create() -> None:
    jm = _FakeJobsManager()
    db = _FakeCollectionsDb()

    with pytest.raises(ValueError, match="stale_target_revision"):
        enqueue_audio_studio_generation_job(
            jm=jm,
            collections_db=db,
            user_id="42",
            project_id="ast_jobs",
            workflow="narration",
            kind="speech",
            provider="tts",
            target_resource_kind="section",
            target_resource_id="sec_001",
            target_revision_id="rev_old",
            idempotency_key="client-key-123456",
            options={},
        )

    assert jm.calls == []


def test_deferred_job_helpers_enqueue_supported_job_types() -> None:
    jm = _FakeJobsManager()
    db = _FakeCollectionsDb()

    render = enqueue_audio_studio_render_job(
        jm=jm,
        collections_db=db,
        user_id="42",
        project_id="ast_jobs",
        target_resource_kind="render",
        target_resource_id="render_001",
        target_revision_id="rev_001",
        idempotency_key="render-key-123456",
        options={},
    )
    export = enqueue_audio_studio_export_job(
        jm=jm,
        collections_db=db,
        user_id="42",
        project_id="ast_jobs",
        target_resource_kind="export",
        target_resource_id="export_001",
        target_revision_id="rev_001",
        idempotency_key="export-key-123456",
        options={},
    )
    migration = enqueue_audio_studio_migration_job(
        jm=jm,
        collections_db=db,
        user_id="42",
        project_id="ast_jobs",
        target_resource_kind="section",
        target_resource_id="legacy_001",
        target_revision_id="rev_001",
        idempotency_key="migration-key-123456",
        options={},
    )

    assert [call["job_type"] for call in jm.calls] == [JOB_TYPE_RENDER, JOB_TYPE_EXPORT, JOB_TYPE_MIGRATE]
    assert [render.job_type, export.job_type, migration.job_type] == [
        JOB_TYPE_RENDER,
        JOB_TYPE_EXPORT,
        JOB_TYPE_MIGRATE,
    ]
