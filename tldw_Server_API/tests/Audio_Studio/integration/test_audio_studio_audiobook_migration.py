"""Integration tests for legacy Audiobook Studio migration endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_audio_studio_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _legacy_payload() -> dict:
    return {
        "id": "legacy-book-1",
        "title": "Legacy Book",
        "description": "Local Dexie audiobook project",
        "voice": "af_heart",
        "speed": 1.05,
        "updated_at": "2026-06-23T12:00:00Z",
        "chapters": [
            {
                "id": "chapter-1",
                "title": "Chapter One",
                "text": "Opening chapter text.",
                "voice": "af_heart",
                "audio_upload_ref": "upload_chapter_1",
                "audio_sha256": "a" * 64,
            },
            {
                "id": "chapter-2",
                "title": "Chapter Two",
                "text": "Second chapter text.",
            },
        ],
    }


def test_audiobook_migration_preview_reports_counts_without_writing_projects(
    client_audio_studio_migration: TestClient,
) -> None:
    response = client_audio_studio_migration.post(
        "/api/v1/audio-studio/migrations/audiobook/preview",
        json={"legacy_project_id": "legacy-book-1", "project_payload": _legacy_payload()},
    )

    assert response.status_code == 200
    preview = response.json()
    assert preview["workflow"] == "narration"
    assert preview["project_count"] == 1
    assert preview["section_count"] == 2
    assert preview["audio_reference_count"] == 1
    assert preview["needs_regeneration_count"] == 1

    db = CollectionsDatabase.for_user(user_id=1)
    assert db.list_audio_studio_projects() == []


def test_audiobook_migration_commit_creates_narration_project_and_is_idempotent(
    client_audio_studio_migration: TestClient,
) -> None:
    payload = {
        "idempotency_key": "client-migration-key-123456",
        "project_payload": _legacy_payload(),
    }

    response = client_audio_studio_migration.post(
        "/api/v1/audio-studio/migrations/audiobook/commit",
        json=payload,
    )
    duplicate = client_audio_studio_migration.post(
        "/api/v1/audio-studio/migrations/audiobook/commit",
        json=payload,
    )

    assert response.status_code == 201
    assert duplicate.status_code == 200
    committed = response.json()
    assert duplicate.json()["project"]["project_id"] == committed["project"]["project_id"]
    assert committed["project"]["workflow"] == "narration"
    assert committed["imported_section_count"] == 2
    assert committed["audio_reference_count"] == 1

    db = CollectionsDatabase.for_user(user_id=1)
    project = db.get_audio_studio_project_by_project_id(committed["project"]["project_id"])
    sections = db.backend.execute(
        "SELECT title, body_text FROM audio_studio_sections WHERE project_row_id = ? ORDER BY order_index",
        (project.id,),
    ).rows
    assert [row["title"] for row in sections] == ["Chapter One", "Chapter Two"]
    assert sections[0]["body_text"] == "Opening chapter text."


def test_audiobook_migration_rejects_external_urls_in_payload(
    client_audio_studio_migration: TestClient,
) -> None:
    payload = _legacy_payload()
    payload["source_url"] = "https://example.invalid/book"

    response = client_audio_studio_migration.post(
        "/api/v1/audio-studio/migrations/audiobook/preview",
        json={"legacy_project_id": "legacy-book-1", "project_payload": payload},
    )

    assert response.status_code == 422
