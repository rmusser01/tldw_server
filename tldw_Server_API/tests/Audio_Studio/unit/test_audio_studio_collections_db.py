"""Unit tests for Audio Studio Collections DB persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def isolated_collections_base(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base_dir = tmp_path / "user_dbs"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    try:
        yield base_dir
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


@pytest.fixture()
def db_user_1(isolated_collections_base: Path) -> CollectionsDatabase:
    return CollectionsDatabase.for_user(user_id=101)


def test_audio_studio_project_crud_and_owner_isolation(
    db_user_1: CollectionsDatabase,
    isolated_collections_base: Path,
) -> None:
    created = db_user_1.create_audio_studio_project(
        project_id="ast_shared",
        title="Narration Project",
        workflow="narration",
        settings_json=json.dumps({"format": "mp3"}),
    )

    assert created.project_id == "ast_shared"
    assert created.status == "draft"
    assert created.deleted == 0
    assert db_user_1.get_audio_studio_project_by_project_id("ast_shared").id == created.id
    assert [row.project_id for row in db_user_1.list_audio_studio_projects()] == ["ast_shared"]

    db_user_2 = CollectionsDatabase.for_user(user_id=202)
    with pytest.raises(KeyError, match="audio_studio_project_not_found"):
        db_user_2.get_audio_studio_project_by_project_id("ast_shared")
    assert db_user_2.list_audio_studio_projects() == []

    other = db_user_2.create_audio_studio_project(
        project_id="ast_shared",
        title="Other User Project",
        workflow="podcast",
    )
    assert other.project_id == created.project_id
    assert other.user_id == "202"

    updated = db_user_1.update_audio_studio_project(
        created.id,
        title="Renamed",
        status="active",
        settings_json=json.dumps({"format": "wav"}),
    )
    assert updated.title == "Renamed"
    assert updated.status == "active"
    assert json.loads(updated.settings_json)["format"] == "wav"

    archived = db_user_1.archive_audio_studio_project(created.id)
    assert archived.status == "archived"
    assert archived.archived_at is not None
    assert archived.deleted == 0
    assert db_user_1.list_audio_studio_projects() == []
    assert db_user_1.get_audio_studio_project(created.id, include_archived=True).project_id == "ast_shared"


def test_audio_studio_revisions_and_resource_upserts(db_user_1: CollectionsDatabase) -> None:
    project = db_user_1.create_audio_studio_project(
        project_id="ast_revisions",
        title="Revision Project",
        workflow="briefing",
    )

    revision = db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_001",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="section",
        resource_id="sec_001",
        content_hash="hash_001",
        payload_json=json.dumps({"title": "Intro"}),
    )
    assert revision.revision_id == "rev_001"
    assert db_user_1.get_audio_studio_project(project.id).current_revision_id == "rev_001"

    with pytest.raises(ValueError, match="stale_base_revision"):
        db_user_1.upsert_audio_studio_section(
            project_row_id=project.id,
            section_id="sec_001",
            base_revision_id="rev_missing",
            workflow="briefing",
            title="Intro",
            body_text="Hello",
            speaker_id=None,
            order_index=0,
            settings_json=json.dumps({}),
            current_revision_id="rev_002",
        )

    section = db_user_1.upsert_audio_studio_section(
        project_row_id=project.id,
        section_id="sec_001",
        base_revision_id="rev_001",
        workflow="briefing",
        title="Intro",
        body_text="Hello",
        speaker_id=None,
        order_index=0,
        settings_json=json.dumps({}),
        current_revision_id="rev_002",
    )
    assert section.section_id == "sec_001"
    assert section.current_revision_id == "rev_002"

    track = db_user_1.upsert_audio_studio_track(
        project_row_id=project.id,
        track_id="trk_001",
        base_revision_id="rev_002",
        name="Narration",
        kind="speech",
        order_index=0,
        muted=False,
        solo=False,
        volume=1.0,
        settings_json=json.dumps({}),
        current_revision_id="rev_003",
    )
    assert track.track_id == "trk_001"
    assert track.muted == 0

    clip = db_user_1.upsert_audio_studio_clip(
        project_row_id=project.id,
        clip_id="clip_001",
        base_revision_id="rev_003",
        section_id="sec_001",
        track_id="trk_001",
        title="Intro clip",
        clip_type="speech",
        start_ms=0,
        duration_ms=None,
        volume=1.0,
        fade_in_ms=0,
        fade_out_ms=0,
        muted=False,
        artifact_id=None,
        settings_json=json.dumps({}),
        current_revision_id="rev_004",
    )
    assert clip.clip_id == "clip_001"
    assert db_user_1.get_audio_studio_project(project.id).current_revision_id == "rev_004"


def test_audio_studio_artifacts_generation_jobs_and_idempotency(db_user_1: CollectionsDatabase) -> None:
    project = db_user_1.create_audio_studio_project(
        project_id="ast_jobs",
        title="Jobs Project",
        workflow="music",
    )

    artifact = db_user_1.create_audio_studio_artifact(
        project_row_id=project.id,
        artifact_id="art_001",
        artifact_type="clip_audio",
        provider="tts",
        output_id=99,
        storage_path="outputs/art_001.mp3",
        mime_type="audio/mpeg",
        size_bytes=1234,
        source_resource_kind="clip",
        source_resource_id="clip_001",
        source_revision_id="rev_001",
        content_hash="hash_001",
        metadata_json=json.dumps({"duration_ms": 1200}),
    )
    assert artifact.artifact_id == "art_001"
    assert [row.artifact_id for row in db_user_1.list_audio_studio_artifacts(project_row_id=project.id)] == ["art_001"]

    job = db_user_1.record_audio_studio_generation_job(
        project_row_id=project.id,
        job_id="job_001",
        provider="tts",
        operation="speech.synthesize.v1",
        target_resource_kind="section",
        target_resource_id="sec_001",
        target_revision_id="rev_001",
        idempotency_key="client-key-123456",
        status="queued",
        request_json=json.dumps({"voice": "af_heart"}),
        result_json=None,
    )
    assert job.job_id == "job_001"

    assert db_user_1.get_audio_studio_idempotency_record("audio_studio", "missing") is None
    db_user_1.put_audio_studio_idempotency_record(
        namespace="audio_studio",
        key="client-key-123456",
        project_row_id=project.id,
        request_hash="hash_001",
        response_json=json.dumps({"job_id": "job_001"}),
    )
    record = db_user_1.get_audio_studio_idempotency_record("audio_studio", "client-key-123456")
    assert record is not None
    assert record.project_row_id == project.id
    assert json.loads(record.response_json)["job_id"] == "job_001"
