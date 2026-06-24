"""Integration tests for Audio Studio artifact media tickets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.Audio_Studio.media_tickets import hash_media_ticket_token
from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

pytestmark = pytest.mark.integration

MEDIA_BYTES = b"0123456789abcdefghijklmnopqrstuvwxyz"


@pytest.fixture()
def client_audio_studio_tickets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    reset_settings()
    AuthDatabaseConfig().reset()

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@example.test", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()
    AuthDatabaseConfig().reset()
    reset_settings()


def _outputs_dir(user_id: int = 1) -> Path:
    outputs_dir = DatabasePaths.get_user_base_directory(user_id) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _write_user_output(filename: str, data: bytes = MEDIA_BYTES, *, user_id: int = 1) -> Path:
    path = _outputs_dir(user_id) / filename
    path.write_bytes(data)
    return path


def _create_project(client: TestClient, *, title: str = "Ticket project") -> dict:
    response = client.post("/api/v1/audio-studio/projects", json={"title": title, "workflow": "narration"})
    assert response.status_code == 200  # nosec B101
    return response.json()


def _create_artifact(
    project: dict,
    *,
    artifact_id: str = "artifact_ticket",
    storage_path: str = "clip.wav",
    mime_type: str | None = "audio/wav",
    artifact_type: str = "clip_audio",
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
        artifact_type=artifact_type,
        provider="audio_studio",
        output_id=None,
        storage_path=normalized_storage_path,
        mime_type=mime_type,
        size_bytes=len(MEDIA_BYTES) if size_bytes is None else size_bytes,
        source_resource_kind="clip",
        source_resource_id="clip-1",
        source_revision_id=project["current_revision_id"],
        content_hash=content_hash or hashlib.sha256(MEDIA_BYTES).hexdigest(),
        metadata_json=json.dumps({"filename": storage_path}),
    )
    return {"artifact_id": row.artifact_id, "storage_path": row.storage_path}


def _mint_url(project_id: str, artifact_id: str) -> str:
    return f"/api/v1/audio-studio/projects/{project_id}/artifacts/{artifact_id}/tickets"


def test_mint_playback_ticket_and_redeem_range(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project)

    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})

    assert mint.status_code == 200  # nosec B101
    payload = mint.json()
    assert payload["ticket_path"].startswith("/api/v1/audio-studio/media-tickets/")  # nosec B101
    assert payload["purpose"] == "playback"  # nosec B101
    assert payload["artifact_id"] == artifact["artifact_id"]  # nosec B101
    token = payload["ticket_path"].rsplit("/", 1)[1]

    users_db = AuthDatabaseConfig().get_user_database(client_id="ticket-test")
    rows = users_db.backend.execute("SELECT token_hash FROM audio_studio_media_tickets").rows
    assert rows == [{"token_hash": hash_media_ticket_token(token)}]  # nosec B101
    assert token not in str(rows)  # nosec B101

    redeemed = client.get(payload["ticket_path"], headers={"Range": "bytes=0-9"})

    assert redeemed.status_code == 206  # nosec B101
    assert redeemed.content == MEDIA_BYTES[:10]  # nosec B101
    assert redeemed.headers["content-range"] == f"bytes 0-9/{len(MEDIA_BYTES)}"  # nosec B101
    assert redeemed.headers["content-disposition"].startswith("inline;")  # nosec B101
    assert redeemed.headers["cache-control"] == "private, no-store"  # nosec B101
    assert redeemed.headers["referrer-policy"] == "no-referrer"  # nosec B101
    assert redeemed.headers["x-content-type-options"] == "nosniff"  # nosec B101
    assert redeemed.headers.get("cross-origin-resource-policy") != "same-origin"  # nosec B101


def test_rejects_playback_ticket_for_non_audio_artifact(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    _write_user_output("manifest.json", b'{"ok": true}')
    artifact = _create_artifact(
        project,
        artifact_id="artifact_json",
        storage_path="manifest.json",
        mime_type="application/json",
        artifact_type="export_manifest",
        size_bytes=len(b'{"ok": true}'),
        content_hash=hashlib.sha256(b'{"ok": true}').hexdigest(),
    )

    response = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})

    assert response.status_code == 415  # nosec B101


def test_download_ticket_for_non_audio_is_single_use_and_ignores_range(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    data = b'{"chapters": []}'
    _write_user_output("manifest.json", data)
    artifact = _create_artifact(
        project,
        artifact_id="artifact_manifest",
        storage_path="manifest.json",
        mime_type="application/json",
        artifact_type="export_manifest",
        size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
    )

    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "download"})
    assert mint.status_code == 200  # nosec B101
    ticket_path = mint.json()["ticket_path"]

    first = client.get(ticket_path, headers={"Range": "bytes=0-3"})
    second = client.get(ticket_path)

    assert first.status_code == 200  # nosec B101
    assert first.content == data  # nosec B101
    assert first.headers["content-disposition"].startswith("attachment;")  # nosec B101
    assert "accept-ranges" not in {key.lower(): value for key, value in first.headers.items()}  # nosec B101
    assert second.status_code == 410  # nosec B101
    assert second.json()["detail"] == "audio_studio_media_ticket_consumed"  # nosec B101


def test_ticket_mint_does_not_cross_users_or_projects(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client, title="owner project")
    other_project = _create_project(client, title="other project")
    _write_user_output("clip.wav")
    other_project_artifact = _create_artifact(other_project, artifact_id="artifact_other_project")

    wrong_project = client.post(
        _mint_url(project["project_id"], other_project_artifact["artifact_id"]),
        json={"purpose": "playback"},
    )
    assert wrong_project.status_code == 404  # nosec B101

    client.app.dependency_overrides[get_request_user] = (
        lambda: User(id=2, username="other", email="o@example.test", is_active=True, is_admin=False)
    )
    assert client.post(_mint_url(project["project_id"], other_project_artifact["artifact_id"]), json={"purpose": "playback"}).status_code == 404  # nosec B101


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [
        ("page.html", "text/html"),
        ("vector.svg", "image/svg+xml"),
        ("script.js", "application/javascript"),
        ("binary.exe", "application/x-msdownload"),
        ("runner.sh", "text/x-shellscript"),
    ],
)
def test_download_ticket_blocks_dangerous_content(client_audio_studio_tickets, filename: str, mime_type: str) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    data = b"unsafe"
    _write_user_output(filename, data)
    artifact = _create_artifact(
        project,
        artifact_id=f"artifact_{filename.replace('.', '_')}",
        storage_path=filename,
        mime_type=mime_type,
        artifact_type="export_artifact",
        size_bytes=len(data),
        content_hash=hashlib.sha256(data).hexdigest(),
    )

    response = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "download"})

    assert response.status_code == 415  # nosec B101


def test_ticket_redemption_returns_404_when_backing_file_is_removed(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    path = _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))
    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert mint.status_code == 200  # nosec B101
    path.unlink()

    response = client.get(mint.json()["ticket_path"])

    assert response.status_code == 404  # nosec B101
    assert str(path) not in response.text  # nosec B101


def test_ticket_redemption_revalidates_size_mismatch(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    path = _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))
    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert mint.status_code == 200  # nosec B101
    path.write_bytes(MEDIA_BYTES + b"changed")

    response = client.get(mint.json()["ticket_path"])

    assert response.status_code == 409  # nosec B101
    assert response.json()["detail"] == "audio_studio_artifact_size_mismatch"  # nosec B101


def test_ticket_redemption_rejects_symlink_escape_after_mint(client_audio_studio_tickets) -> None:
    client, tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    path = _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))
    mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert mint.status_code == 200  # nosec B101
    outside = tmp_path / "outside.wav"
    outside.write_bytes(MEDIA_BYTES)
    path.unlink()
    path.symlink_to(outside)

    response = client.get(mint.json()["ticket_path"])

    assert response.status_code == 400  # nosec B101
    assert response.json()["detail"] == "invalid_audio_studio_artifact_path"  # nosec B101
    assert str(outside) not in response.text  # nosec B101


def test_expired_and_revoked_tickets_return_stable_gone_errors(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets
    project = _create_project(client)
    _write_user_output("clip.wav")
    artifact = _create_artifact(project, size_bytes=len(MEDIA_BYTES))

    expired_mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    revoked_mint = client.post(_mint_url(project["project_id"], artifact["artifact_id"]), json={"purpose": "playback"})
    assert expired_mint.status_code == 200  # nosec B101
    assert revoked_mint.status_code == 200  # nosec B101
    expired_token = expired_mint.json()["ticket_path"].rsplit("/", 1)[1]
    revoked_token = revoked_mint.json()["ticket_path"].rsplit("/", 1)[1]

    users_db = AuthDatabaseConfig().get_user_database(client_id="ticket-test")
    users_db.backend.execute(
        "UPDATE audio_studio_media_tickets SET expires_at = ? WHERE token_hash = ?",
        ("2000-01-01T00:00:00Z", hash_media_ticket_token(expired_token)),
    )
    users_db.backend.execute(
        "UPDATE audio_studio_media_tickets SET revoked_at = ? WHERE token_hash = ?",
        ("2026-06-24T00:00:00Z", hash_media_ticket_token(revoked_token)),
    )

    expired_response = client.get(expired_mint.json()["ticket_path"])
    revoked_response = client.get(revoked_mint.json()["ticket_path"])

    assert expired_response.status_code == 410  # nosec B101
    assert expired_response.json()["detail"] == "audio_studio_media_ticket_expired"  # nosec B101
    assert revoked_response.status_code == 410  # nosec B101
    assert revoked_response.json()["detail"] == "audio_studio_media_ticket_revoked"  # nosec B101


def test_unknown_and_malformed_ticket_paths_are_generic_not_found(client_audio_studio_tickets) -> None:
    client, _tmp_path = client_audio_studio_tickets

    malformed = client.get("/api/v1/audio-studio/media-tickets/not valid")
    unknown = client.get("/api/v1/audio-studio/media-tickets/unknown-ticket-token")

    assert malformed.status_code == 404  # nosec B101
    assert unknown.status_code == 404  # nosec B101
