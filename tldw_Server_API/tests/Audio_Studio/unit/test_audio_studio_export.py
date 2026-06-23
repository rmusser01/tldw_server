"""Unit tests for Audio Studio export services."""

from __future__ import annotations

import hashlib
import json
import shutil
import wave
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Audio_Studio.export import (
    create_audio_studio_export_manifest,
    package_audio_studio_export,
    record_audio_studio_export_artifact,
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


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x01\x00" * 1200)


def _create_project(db: CollectionsDatabase, *, workflow: str = "narration"):
    return db.create_audio_studio_project(
        project_id="ast_export",
        title="Export Project",
        workflow=workflow,
        revision_id="rev_001",
        mutation_kind="project.create",
        resource_kind="project",
        resource_id="ast_export",
        content_hash="hash_rev_001",
        payload_json=json.dumps({"title": "Export Project", "workflow": workflow}),
    )


def _create_mix_artifact(db: CollectionsDatabase, *, project_row_id: int, wav_path: Path):
    content = wav_path.read_bytes()
    return db.create_audio_studio_artifact(
        project_row_id=project_row_id,
        artifact_id="art_mix",
        artifact_type="final_mix",
        provider="audio_studio",
        output_id=None,
        storage_path=str(wav_path),
        mime_type="audio/wav",
        size_bytes=len(content),
        source_resource_kind="render",
        source_resource_id="render_001",
        source_revision_id="rev_001",
        content_hash=hashlib.sha256(content).hexdigest(),
        metadata_json=json.dumps({"render_id": "render_001"}),
    )


def test_export_manifest_includes_source_hashes_without_storage_paths(
    db_user_1: CollectionsDatabase,
    tmp_path: Path,
) -> None:
    project = _create_project(db_user_1)
    wav_path = tmp_path / "mix.wav"
    _write_wav(wav_path)
    artifact = _create_mix_artifact(db_user_1, project_row_id=project.id, wav_path=wav_path)

    manifest = create_audio_studio_export_manifest(
        collections_db=db_user_1,
        project=project,
        export_id="export_001",
        export_type="zip_package",
        target_revision_id="rev_001",
        artifact_refs=[{"artifact_id": artifact.artifact_id, "source_revision_id": "rev_001"}],
    )

    assert manifest["source_artifacts"][0]["artifact_id"] == artifact.artifact_id
    assert manifest["source_artifacts"][0]["content_hash"] == artifact.content_hash
    assert manifest["source_artifacts"][0]["source_resource_kind"] == "render"
    assert "storage_path" not in json.dumps(manifest)


def test_zip_and_narration_packages_include_manifest_and_audio(
    db_user_1: CollectionsDatabase,
    tmp_path: Path,
) -> None:
    project = _create_project(db_user_1)
    wav_path = tmp_path / "mix.wav"
    _write_wav(wav_path)
    artifact = _create_mix_artifact(db_user_1, project_row_id=project.id, wav_path=wav_path)
    manifest = create_audio_studio_export_manifest(
        collections_db=db_user_1,
        project=project,
        export_id="export_001",
        export_type="narration_package",
        target_revision_id="rev_001",
        artifact_refs=[{"artifact_id": artifact.artifact_id, "source_revision_id": "rev_001"}],
    )

    package = package_audio_studio_export(
        manifest=manifest,
        source_artifacts=[artifact],
        export_type="narration_package",
        output_dir=tmp_path / "exports",
    )

    with zipfile.ZipFile(package.path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "audiobook.json" in names
        assert any(name.startswith("audio/") and name.endswith(".wav") for name in names)


def test_single_audio_export_copies_source_audio(
    db_user_1: CollectionsDatabase,
    tmp_path: Path,
) -> None:
    project = _create_project(db_user_1)
    wav_path = tmp_path / "mix.wav"
    _write_wav(wav_path)
    artifact = _create_mix_artifact(db_user_1, project_row_id=project.id, wav_path=wav_path)
    manifest = create_audio_studio_export_manifest(
        collections_db=db_user_1,
        project=project,
        export_id="export_001",
        export_type="single_audio",
        target_revision_id="rev_001",
        artifact_refs=[{"artifact_id": artifact.artifact_id, "source_revision_id": "rev_001"}],
    )

    package = package_audio_studio_export(
        manifest=manifest,
        source_artifacts=[artifact],
        export_type="single_audio",
        output_dir=tmp_path / "exports",
    )

    assert package.path.suffix == ".wav"
    assert package.path.read_bytes() == wav_path.read_bytes()


def test_export_records_package_and_manifest_artifacts_separately(
    db_user_1: CollectionsDatabase,
    tmp_path: Path,
) -> None:
    project = _create_project(db_user_1)
    wav_path = tmp_path / "mix.wav"
    _write_wav(wav_path)
    artifact = _create_mix_artifact(db_user_1, project_row_id=project.id, wav_path=wav_path)
    manifest = create_audio_studio_export_manifest(
        collections_db=db_user_1,
        project=project,
        export_id="export_001",
        export_type="zip_package",
        target_revision_id="rev_001",
        artifact_refs=[{"artifact_id": artifact.artifact_id, "source_revision_id": "rev_001"}],
    )
    package = package_audio_studio_export(
        manifest=manifest,
        source_artifacts=[artifact],
        export_type="zip_package",
        output_dir=tmp_path / "exports",
    )

    recorded = record_audio_studio_export_artifact(
        collections_db=db_user_1,
        project=project,
        manifest=manifest,
        package_result=package,
    )

    artifacts = db_user_1.list_audio_studio_artifacts(project_row_id=project.id)
    artifact_types = {row.artifact_id: row.artifact_type for row in artifacts}
    assert artifact_types[recorded.package_artifact_id] == "package"
    assert artifact_types[recorded.manifest_artifact_id] == "export_manifest"
    assert recorded.package_artifact_id != artifact.artifact_id
