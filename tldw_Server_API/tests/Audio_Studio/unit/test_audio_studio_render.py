"""Unit tests for Audio Studio render services."""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Audio_Studio.render import (
    build_render_plan,
    record_audio_studio_render_artifact,
    render_audio_studio_mix,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def isolated_collections_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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


def _write_wav(path: Path, *, frames: int = 1200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x01\x00" * frames)


def _create_project(db: CollectionsDatabase, *, project_id: str = "ast_render", revision_id: str = "rev_001"):
    return db.create_audio_studio_project(
        project_id=project_id,
        title="Render Project",
        workflow="narration",
        revision_id=revision_id,
        mutation_kind="project.create",
        resource_kind="project",
        resource_id=project_id,
        content_hash=f"hash_{revision_id}",
        payload_json=json.dumps({"title": "Render Project"}),
    )


def _create_clip_artifact(
    db: CollectionsDatabase,
    *,
    project_row_id: int,
    artifact_id: str,
    wav_path: Path,
    source_revision_id: str = "rev_001",
):
    content = wav_path.read_bytes()
    return db.create_audio_studio_artifact(
        project_row_id=project_row_id,
        artifact_id=artifact_id,
        artifact_type="clip_audio",
        provider="tts",
        output_id=None,
        storage_path=str(wav_path),
        mime_type="audio/wav",
        size_bytes=len(content),
        source_resource_kind="clip",
        source_resource_id=f"clip_{artifact_id}",
        source_revision_id=source_revision_id,
        content_hash=hashlib.sha256(content).hexdigest(),
        metadata_json=json.dumps({"duration_ms": 50}),
    )


def test_render_plan_rejects_cross_project_artifacts(db_user_1: CollectionsDatabase, tmp_path: Path) -> None:
    project = _create_project(db_user_1, project_id="ast_owner")
    other = _create_project(db_user_1, project_id="ast_other", revision_id="rev_other")
    wav_path = tmp_path / "other.wav"
    _write_wav(wav_path)
    _create_clip_artifact(
        db_user_1,
        project_row_id=other.id,
        artifact_id="art_other",
        wav_path=wav_path,
        source_revision_id="rev_001",
    )

    with pytest.raises(ValueError, match="audio_studio_artifact_not_found"):
        build_render_plan(
            collections_db=db_user_1,
            project=project,
            render_id="render_001",
            target_revision_id="rev_001",
            artifact_refs=[{"artifact_id": "art_other", "source_revision_id": "rev_001"}],
            output_format="wav",
        )


def test_render_plan_rejects_stale_artifact_revision(db_user_1: CollectionsDatabase, tmp_path: Path) -> None:
    project = _create_project(db_user_1)
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)
    _create_clip_artifact(
        db_user_1,
        project_row_id=project.id,
        artifact_id="art_001",
        wav_path=wav_path,
        source_revision_id="rev_old",
    )

    with pytest.raises(ValueError, match="stale_artifact_revision"):
        build_render_plan(
            collections_db=db_user_1,
            project=project,
            render_id="render_001",
            target_revision_id="rev_001",
            artifact_refs=[{"artifact_id": "art_001", "source_revision_id": "rev_001"}],
            output_format="wav",
        )


@pytest.mark.asyncio
async def test_render_records_mix_and_manifest_artifacts_separately(
    db_user_1: CollectionsDatabase,
    tmp_path: Path,
) -> None:
    project = _create_project(db_user_1)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_wav(first, frames=400)
    _write_wav(second, frames=800)
    first_artifact = _create_clip_artifact(
        db_user_1,
        project_row_id=project.id,
        artifact_id="art_first",
        wav_path=first,
    )
    second_artifact = _create_clip_artifact(
        db_user_1,
        project_row_id=project.id,
        artifact_id="art_second",
        wav_path=second,
    )

    plan = build_render_plan(
        collections_db=db_user_1,
        project=project,
        render_id="render_001",
        target_revision_id="rev_001",
        artifact_refs=[
            {"artifact_id": first_artifact.artifact_id, "source_revision_id": "rev_001"},
            {"artifact_id": second_artifact.artifact_id, "source_revision_id": "rev_001"},
        ],
        output_format="wav",
    )
    rendered = await render_audio_studio_mix(plan, output_dir=tmp_path / "renders")
    recorded = record_audio_studio_render_artifact(
        collections_db=db_user_1,
        project=project,
        plan=plan,
        render_result=rendered,
    )

    artifacts = db_user_1.list_audio_studio_artifacts(project_row_id=project.id)
    artifact_types = {row.artifact_id: row.artifact_type for row in artifacts}
    assert artifact_types[recorded.mix_artifact_id] == "preview_mix"
    assert artifact_types[recorded.manifest_artifact_id] == "render_manifest"
    assert recorded.mix_artifact_id not in {first_artifact.artifact_id, second_artifact.artifact_id}
    assert rendered.manifest["source_artifacts"][0]["content_hash"] == first_artifact.content_hash
