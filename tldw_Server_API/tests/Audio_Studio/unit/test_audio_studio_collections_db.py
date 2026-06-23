"""Unit tests for Audio Studio Collections DB persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


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


def test_audio_studio_revision_ids_are_owner_scoped_on_shared_database(tmp_path: Path) -> None:
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "shared_collections.db"),
        )
    )
    db_user_1 = CollectionsDatabase.from_backend(user_id=101, backend=backend)
    db_user_2 = CollectionsDatabase.from_backend(user_id=202, backend=backend)

    project_1 = db_user_1.create_audio_studio_project(
        project_id="ast_user_1",
        title="User 1 Project",
        workflow="narration",
    )
    project_2 = db_user_2.create_audio_studio_project(
        project_id="ast_user_2",
        title="User 2 Project",
        workflow="podcast",
    )

    revision_1 = db_user_1.create_audio_studio_revision(
        project_row_id=project_1.id,
        revision_id="rev_shared",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project_1.project_id,
        content_hash="hash_user_1",
        payload_json=json.dumps({"title": project_1.title}),
    )
    revision_2 = db_user_2.create_audio_studio_revision(
        project_row_id=project_2.id,
        revision_id="rev_shared",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project_2.project_id,
        content_hash="hash_user_2",
        payload_json=json.dumps({"title": project_2.title}),
    )

    assert revision_1.revision_id == revision_2.revision_id == "rev_shared"
    assert db_user_1.get_audio_studio_revision("rev_shared").resource_id == "ast_user_1"
    assert db_user_2.get_audio_studio_revision("rev_shared").resource_id == "ast_user_2"
    assert db_user_1.get_audio_studio_project(project_1.id).current_revision_id == "rev_shared"
    assert db_user_2.get_audio_studio_project(project_2.id).current_revision_id == "rev_shared"


def test_audio_studio_project_mutation_rolls_back_when_revision_insert_fails(
    db_user_1: CollectionsDatabase,
) -> None:
    project = db_user_1.create_audio_studio_project(
        project_id="ast_rollback",
        title="Rollback Project",
        workflow="narration",
        settings_json=json.dumps({"description": "Original"}),
    )
    db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_001",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project.project_id,
        content_hash="hash_001",
        payload_json=json.dumps({"title": "Rollback Project"}),
    )

    with pytest.raises(DatabaseError):
        db_user_1.mutate_audio_studio_project(
            project_row_id=project.id,
            base_revision_id="rev_001",
            revision_id="rev_001",
            mutation_kind="project.update",
            resource_kind="project",
            resource_id=project.project_id,
            content_hash="hash_duplicate",
            payload_json=json.dumps({"title": "Should Roll Back"}),
            title="Should Roll Back",
        )

    persisted = db_user_1.get_audio_studio_project(project.id)
    assert persisted.title == "Rollback Project"
    assert persisted.current_revision_id == "rev_001"
    assert json.loads(persisted.settings_json)["description"] == "Original"


def test_audio_studio_archive_creates_revision_and_advances_project(
    db_user_1: CollectionsDatabase,
) -> None:
    project = db_user_1.create_audio_studio_project(
        project_id="ast_archive_revision",
        title="Archive Revision Project",
        workflow="briefing",
    )
    db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_001",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project.project_id,
        content_hash="hash_001",
        payload_json=json.dumps({"title": project.title}),
    )

    archived = db_user_1.archive_audio_studio_project_with_revision(
        project_row_id=project.id,
        base_revision_id="rev_001",
        revision_id="rev_archive",
        content_hash="hash_archive",
        payload_json=json.dumps({"archived": True}),
    )

    assert archived.status == "archived"
    assert archived.archived_at is not None
    assert archived.current_revision_id == "rev_archive"
    revision = db_user_1.get_audio_studio_revision("rev_archive")
    assert revision.parent_revision_id == "rev_001"
    assert revision.mutation_kind == "project.archive"
    assert revision.resource_kind == "project"
    assert revision.resource_id == project.project_id


def test_audio_studio_resource_mutation_rolls_back_when_revision_insert_fails(
    db_user_1: CollectionsDatabase,
) -> None:
    project = db_user_1.create_audio_studio_project(
        project_id="ast_resource_rollback",
        title="Resource Rollback Project",
        workflow="briefing",
    )
    db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_001",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project.project_id,
        content_hash="hash_001",
        payload_json=json.dumps({"title": project.title}),
    )

    with pytest.raises(DatabaseError):
        db_user_1.upsert_audio_studio_section_with_revision(
            project_row_id=project.id,
            section_id="sec_rollback",
            base_revision_id="rev_001",
            revision_id="rev_001",
            workflow="briefing",
            title="Should Roll Back",
            body_text="This section should not persist",
            speaker_id=None,
            order_index=0,
            settings_json=json.dumps({}),
            content_hash="hash_duplicate",
            payload_json=json.dumps({"section_id": "sec_rollback"}),
        )

    persisted = db_user_1.get_audio_studio_project(project.id)
    assert persisted.current_revision_id == "rev_001"
    row = db_user_1.backend.execute(
        "SELECT section_id FROM audio_studio_sections WHERE project_row_id = ? AND section_id = ?",
        (project.id, "sec_rollback"),
    ).first
    assert row is None


def test_audio_studio_clip_upsert_rejects_dangling_references(
    db_user_1: CollectionsDatabase,
) -> None:
    project = db_user_1.create_audio_studio_project(
        project_id="ast_clip_refs",
        title="Clip References",
        workflow="narration",
    )
    db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_001",
        parent_revision_id=None,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project.project_id,
        content_hash="hash_001",
        payload_json=json.dumps({"title": project.title}),
    )

    with pytest.raises(ValueError, match="audio_studio_track_not_found"):
        db_user_1.upsert_audio_studio_clip_with_revision(
            project_row_id=project.id,
            clip_id="clip_bad_track",
            base_revision_id="rev_001",
            revision_id="rev_bad_track",
            section_id=None,
            track_id="trk_missing",
            title="Bad track",
            clip_type="speech",
            start_ms=0,
            duration_ms=None,
            volume=1.0,
            fade_in_ms=0,
            fade_out_ms=0,
            muted=False,
            artifact_id=None,
            settings_json=json.dumps({}),
            content_hash="hash_bad_track",
            payload_json=json.dumps({"track_id": "trk_missing"}),
        )
    assert db_user_1.get_audio_studio_project(project.id).current_revision_id == "rev_001"

    db_user_1.upsert_audio_studio_track(
        project_row_id=project.id,
        track_id="trk_001",
        base_revision_id="rev_001",
        name="Narration",
        kind="speech",
        order_index=0,
        muted=False,
        solo=False,
        volume=1.0,
        settings_json=json.dumps({}),
        current_revision_id="rev_002",
    )
    db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_002",
        parent_revision_id="rev_001",
        mutation_kind="track.upsert",
        resource_kind="track",
        resource_id="trk_001",
        content_hash="hash_002",
        payload_json=json.dumps({"track_id": "trk_001"}),
    )

    with pytest.raises(ValueError, match="audio_studio_section_not_found"):
        db_user_1.upsert_audio_studio_clip_with_revision(
            project_row_id=project.id,
            clip_id="clip_bad_section",
            base_revision_id="rev_002",
            revision_id="rev_bad_section",
            section_id="sec_missing",
            track_id="trk_001",
            title="Bad section",
            clip_type="speech",
            start_ms=0,
            duration_ms=None,
            volume=1.0,
            fade_in_ms=0,
            fade_out_ms=0,
            muted=False,
            artifact_id=None,
            settings_json=json.dumps({}),
            content_hash="hash_bad_section",
            payload_json=json.dumps({"section_id": "sec_missing"}),
        )
    assert db_user_1.get_audio_studio_project(project.id).current_revision_id == "rev_002"

    db_user_1.upsert_audio_studio_section(
        project_row_id=project.id,
        section_id="sec_001",
        base_revision_id="rev_002",
        workflow="narration",
        title="Intro",
        body_text="Hello",
        speaker_id=None,
        order_index=0,
        settings_json=json.dumps({}),
        current_revision_id="rev_003",
    )
    db_user_1.create_audio_studio_revision(
        project_row_id=project.id,
        revision_id="rev_003",
        parent_revision_id="rev_002",
        mutation_kind="section.upsert",
        resource_kind="section",
        resource_id="sec_001",
        content_hash="hash_003",
        payload_json=json.dumps({"section_id": "sec_001"}),
    )

    with pytest.raises(ValueError, match="audio_studio_artifact_not_found"):
        db_user_1.upsert_audio_studio_clip_with_revision(
            project_row_id=project.id,
            clip_id="clip_bad_artifact",
            base_revision_id="rev_003",
            revision_id="rev_bad_artifact",
            section_id="sec_001",
            track_id="trk_001",
            title="Bad artifact",
            clip_type="speech",
            start_ms=0,
            duration_ms=None,
            volume=1.0,
            fade_in_ms=0,
            fade_out_ms=0,
            muted=False,
            artifact_id="art_missing",
            settings_json=json.dumps({}),
            content_hash="hash_bad_artifact",
            payload_json=json.dumps({"artifact_id": "art_missing"}),
        )
    assert db_user_1.get_audio_studio_project(project.id).current_revision_id == "rev_003"


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
