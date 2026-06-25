"""Integration tests for Audio Studio artifact media playback."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


pytestmark = pytest.mark.integration

AUDIO_BYTES = b"0123456789abcdefghijklmnopqrstuvwxyz"


@pytest.fixture()
def client_audio_studio_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()


def _create_project(client: TestClient, *, title: str = "Narration") -> dict:
    response = client.post(
        "/api/v1/audio-studio/projects",
        json={"title": title, "workflow": "narration"},
    )
    assert response.status_code == 200  # nosec B101
    return response.json()


def _outputs_dir(user_id: int = 1) -> Path:
    outputs_dir = DatabasePaths.get_user_base_directory(user_id) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _write_user_output(filename: str, data: bytes = AUDIO_BYTES, *, user_id: int = 1) -> Path:
    path = _outputs_dir(user_id) / filename
    path.write_bytes(data)
    return path


def _create_artifact(
    project: dict,
    *,
    artifact_id: str = "artifact_media",
    storage_path: str = "clip.wav",
    mime_type: str | None = "audio/wav",
    size_bytes: int | None = None,
    content_hash: str | None = None,
    user_id: int = 1,
    normalize_storage_path: bool = True,
) -> dict:
    db = CollectionsDatabase.for_user(user_id=user_id)
    project_row = db.get_audio_studio_project_by_project_id(project["project_id"])
    normalized_storage_path = (
        db.resolve_output_storage_path(storage_path)
        if normalize_storage_path and not Path(storage_path).is_absolute() and os.sep not in storage_path
        else storage_path
    )
    row = db.create_audio_studio_artifact(
        project_row_id=project_row.id,
        artifact_id=artifact_id,
        artifact_type="clip_audio",
        provider="tts",
        output_id=None,
        storage_path=normalized_storage_path,
        mime_type=mime_type,
        size_bytes=len(AUDIO_BYTES) if size_bytes is None else size_bytes,
        source_resource_kind="clip",
        source_resource_id="clip_001",
        source_revision_id=project["current_revision_id"],
        content_hash=content_hash or hashlib.sha256(AUDIO_BYTES).hexdigest(),
        metadata_json=json.dumps({"duration_ms": 50}),
    )
    return {"artifact_id": row.artifact_id, "storage_path": row.storage_path}


def _media_url(project_id: str, artifact_id: str) -> str:
    return f"/api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/media"


def test_artifact_media_full_response_serves_audio_without_path_leak(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    wav_path = _write_user_output("clip.wav")
    storage_path = CollectionsDatabase.for_user(user_id=1).resolve_output_storage_path("clip.wav")
    artifact = _create_artifact(
        project,
        artifact_id="artifact_full",
        storage_path=storage_path,
        content_hash=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
    )

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 200  # nosec B101
    assert response.content == AUDIO_BYTES  # nosec B101
    assert response.headers["content-type"].split(";")[0] == "audio/wav"  # nosec B101
    assert response.headers["accept-ranges"] == "bytes"  # nosec B101
    assert response.headers["x-content-type-options"] == "nosniff"  # nosec B101
    assert "clip.wav" not in response.headers.get("content-disposition", "")  # nosec B101
    assert str(wav_path.parent) not in str(response.headers)  # nosec B101
    assert str(wav_path) not in response.text  # nosec B101


def test_artifact_media_range_response_serves_requested_bytes(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    response = client.get(
        _media_url(project["project_id"], artifact["artifact_id"]),
        headers={"Range": "bytes=0-9"},
    )

    assert response.status_code == 206  # nosec B101
    assert response.content == AUDIO_BYTES[:10]  # nosec B101
    assert response.headers["content-range"] == f"bytes 0-9/{len(AUDIO_BYTES)}"  # nosec B101
    assert response.headers["accept-ranges"] == "bytes"  # nosec B101
    assert response.headers["x-content-type-options"] == "nosniff"  # nosec B101


def test_artifact_media_suffix_range_response_serves_last_bytes(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    response = client.get(
        _media_url(project["project_id"], artifact["artifact_id"]),
        headers={"Range": "bytes=-10"},
    )

    assert response.status_code == 206  # nosec B101
    assert response.content == AUDIO_BYTES[-10:]  # nosec B101
    assert response.headers["content-range"] == f"bytes {len(AUDIO_BYTES) - 10}-{len(AUDIO_BYTES) - 1}/{len(AUDIO_BYTES)}"  # nosec B101
    assert response.headers["accept-ranges"] == "bytes"  # nosec B101


def test_artifact_media_download_query_uses_attachment_disposition(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    response = client.get(f"{_media_url(project['project_id'], artifact['artifact_id'])}?download=true")

    assert response.status_code == 200  # nosec B101
    assert response.headers["content-disposition"].startswith("attachment;")  # nosec B101


def test_artifact_media_does_not_leak_across_users_or_projects(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client, title="Owner project")
    other_project = _create_project(client, title="Other project")
    _write_user_output("clip.wav")
    artifact = _create_artifact(other_project, artifact_id="artifact_other_project")

    attached_to_other_project = client.get(_media_url(project["project_id"], artifact["artifact_id"]))
    assert attached_to_other_project.status_code == 404  # nosec B101

    client.app.dependency_overrides[get_request_user] = (
        lambda: User(id=2, username="other", email="o@e.com", is_active=True, is_admin=False)
    )
    other_user_project = _create_project(client, title="Other user project")
    _write_user_output("other.wav", user_id=2)
    other_user_artifact = _create_artifact(
        other_user_project,
        artifact_id="artifact_other_user",
        storage_path="other.wav",
        user_id=2,
    )

    client.app.dependency_overrides[get_request_user] = (
        lambda: User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)
    )
    assert client.get(_media_url(other_user_project["project_id"], other_user_artifact["artifact_id"])).status_code == 404  # nosec B101


@pytest.mark.parametrize("mime_type", ["text/html", "application/x-msdownload"])
def test_artifact_media_rejects_unsupported_mime(
    client_audio_studio_media,
    mime_type: str,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project, artifact_id=f"artifact_{mime_type.replace('/', '_')}", mime_type=mime_type)

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 415  # nosec B101


@pytest.mark.parametrize(
    ("storage_path", "expected_status"),
    [
        ("https://example.invalid/audio.wav", 400),
        ("../clip.wav", 400),
    ],
)
def test_artifact_media_rejects_invalid_relative_or_url_storage_paths(
    client_audio_studio_media,
    storage_path: str,
    expected_status: int,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    artifact = _create_artifact(project, storage_path=storage_path)

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == expected_status  # nosec B101
    assert storage_path not in response.text  # nosec B101


def test_artifact_media_rejects_absolute_path_outside_user_output_roots(
    client_audio_studio_media,
    tmp_path: Path,
) -> None:
    client, _fixture_tmp_path = client_audio_studio_media
    project = _create_project(client)
    outside_path = tmp_path / "outside.wav"
    outside_path.write_bytes(AUDIO_BYTES)
    artifact = _create_artifact(project, storage_path=str(outside_path))

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 400  # nosec B101
    assert str(outside_path) not in response.text  # nosec B101


def test_artifact_media_rejects_symlink_escape_from_output_root(
    client_audio_studio_media,
    tmp_path: Path,
) -> None:
    client, _fixture_tmp_path = client_audio_studio_media
    project = _create_project(client)
    outside_target = tmp_path / "outside-target.wav"
    outside_target.write_bytes(AUDIO_BYTES)
    symlink_path = _outputs_dir() / "link.wav"
    try:
        symlink_path.symlink_to(outside_target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unsupported on this platform: {exc}")
    artifact = _create_artifact(project, storage_path="link.wav", normalize_storage_path=False)

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 400  # nosec B101
    assert str(outside_target) not in response.text  # nosec B101


def test_artifact_media_disambiguates_duplicate_relative_filename_by_hash(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    output_bytes = b"output-root-audio"
    temp_bytes = b"temp-root-audio"
    _write_user_output("dupe.wav", output_bytes)
    temp_path = DatabasePaths.get_user_temp_outputs_dir(1) / "dupe.wav"
    temp_path.write_bytes(temp_bytes)
    artifact = _create_artifact(
        project,
        artifact_id="artifact_temp_match",
        storage_path="dupe.wav",
        size_bytes=len(temp_bytes),
        content_hash=hashlib.sha256(temp_bytes).hexdigest(),
    )

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 200  # nosec B101
    assert response.content == temp_bytes  # nosec B101


def test_artifact_media_rejects_ambiguous_duplicate_relative_filename_without_hash_match(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("dupe.wav", b"output-root-audio")
    temp_path = DatabasePaths.get_user_temp_outputs_dir(1) / "dupe.wav"
    temp_path.write_bytes(b"temp-root-audio")
    artifact = _create_artifact(
        project,
        artifact_id="artifact_no_hash_match",
        storage_path="dupe.wav",
        size_bytes=len(b"temp-root-audio"),
        content_hash="0" * 64,
    )

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 409  # nosec B101


def test_artifact_media_rejects_extension_mime_mismatch(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.html")
    artifact = _create_artifact(project, storage_path="clip.html", mime_type="audio/wav")

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 415  # nosec B101


def test_artifact_media_rejects_size_mismatch(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(AUDIO_BYTES) + 1)

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 409  # nosec B101


def test_artifact_media_returns_404_for_missing_backing_file(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    artifact = _create_artifact(project, storage_path="missing.wav")

    response = client.get(_media_url(project["project_id"], artifact["artifact_id"]))

    assert response.status_code == 404  # nosec B101


def test_artifact_media_rejects_unsatisfiable_range(
    client_audio_studio_media,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    response = client.get(
        _media_url(project["project_id"], artifact["artifact_id"]),
        headers={"Range": "bytes=999999-999999"},
    )

    assert response.status_code == 416  # nosec B101
    assert response.headers["content-range"] == f"bytes */{len(AUDIO_BYTES)}"  # nosec B101


@pytest.mark.parametrize(
    "range_header",
    ["items=0-10", "bytes=10-1", "bytes=0-1,2-3", "bytes=", "bytes=-0"],
)
def test_artifact_media_rejects_malformed_ranges(
    client_audio_studio_media,
    range_header: str,
) -> None:
    client, _tmp_path = client_audio_studio_media
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    response = client.get(
        _media_url(project["project_id"], artifact["artifact_id"]),
        headers={"Range": range_header},
    )

    assert response.status_code == 416  # nosec B101
    assert response.headers["content-range"] == f"bytes */{len(AUDIO_BYTES)}"  # nosec B101


def test_artifact_media_auth_smoke_requires_single_user_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "test-audio-studio-key-12345"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", api_key)
    reset_settings()

    try:
        app = FastAPI()
        app.include_router(audio_studio_router, prefix="/api/v1")
        with TestClient(app) as client:
            db = CollectionsDatabase.for_user(user_id=1)
            project = db.create_audio_studio_project(
                project_id="ast_auth_smoke",
                title="Auth smoke",
                workflow="narration",
                revision_id="rev_auth_smoke",
                mutation_kind="project.create",
                resource_kind="project",
                resource_id="ast_auth_smoke",
                content_hash=hashlib.sha256(b"auth").hexdigest(),
                payload_json=json.dumps({"title": "Auth smoke"}),
                settings_json=json.dumps({"settings": {}, "metadata": {}, "description": None}),
            )
            _write_user_output("clip.wav")
            db.create_audio_studio_artifact(
                project_row_id=project.id,
                artifact_id="artifact_auth",
                artifact_type="clip_audio",
                provider="tts",
                output_id=None,
                storage_path=db.resolve_output_storage_path("clip.wav"),
                mime_type="audio/wav",
                size_bytes=len(AUDIO_BYTES),
                source_resource_kind="clip",
                source_resource_id="clip_001",
                source_revision_id="rev_auth_smoke",
                content_hash=hashlib.sha256(AUDIO_BYTES).hexdigest(),
                metadata_json="{}",
            )

            missing_credentials = client.get(_media_url("ast_auth_smoke", "artifact_auth"))
            authenticated = client.get(
                _media_url("ast_auth_smoke", "artifact_auth"),
                headers={"X-API-KEY": api_key},
            )
    finally:
        reset_settings()

    assert missing_credentials.status_code == 401  # nosec B101
    assert authenticated.status_code == 200  # nosec B101
    assert authenticated.content == AUDIO_BYTES  # nosec B101
