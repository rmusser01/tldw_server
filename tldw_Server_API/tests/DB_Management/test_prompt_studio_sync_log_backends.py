"""Prompt Studio mutations retain sync events on the real content backends."""

import json

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def studio_db(request, tmp_path):
    """Bootstrap the production shared content schema on official PostgreSQL."""
    backend = media = db = None
    try:
        if request.param == "postgres":
            backend = DatabaseBackendFactory.create_backend(
                request.getfixturevalue("pg_database_config")
            )
            media = MediaDatabase(":memory:", client_id="2", backend=backend)
        db = PromptStudioDatabase(tmp_path / "studio.db", client_id="2", backend=backend)
        yield db
    finally:
        if db is not None:
            db.close()
        if media is not None:
            media.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def _persisted_events(db, entity, entity_uuid):
    """Read after releasing the writer, so pending writes cannot satisfy the test."""
    db.close_connection()
    rows = db.get_connection().execute(
        "SELECT change_id, operation, client_id, payload FROM sync_log "
        "WHERE entity = ? AND entity_uuid = ? ORDER BY change_id",
        (entity, entity_uuid),
    ).fetchall()
    return [
        (row[0], row[1], row[2], json.loads(row[3]))
        for row in rows
    ]


def test_project_create_and_update_persist_sync_events(studio_db):
    project = studio_db.create_project(name="Rowan source project", user_id="2")
    studio_db.update_project(project["id"], description="Public source prompts")

    events = _persisted_events(studio_db, "prompt_studio_project", project["uuid"])

    assert [event[1:3] for event in events] == [("create", "2"), ("update", "2")]
    assert 0 < events[0][0] < events[1][0]
    assert events[0][3]["name"] == "Rowan source project"
    assert events[1][3] == {"description": "Public source prompts"}
    assert studio_db.get_project(project["id"])["description"] == "Public source prompts"


def test_prompt_create_and_new_version_persist_sync_events(studio_db):
    project = studio_db.create_project(name="Pirate project", user_id="2")
    prompt = studio_db.create_prompt(
        project_id=project["id"], name="Pirate", system_prompt="Speak like a pirate."
    )
    updated = studio_db.create_prompt_version(
        prompt["id"], change_description="Keep answers brief", system_prompt="Reply briefly as a pirate."
    )

    created_events = _persisted_events(studio_db, "prompt_studio_prompt", prompt["uuid"])
    updated_events = _persisted_events(studio_db, "prompt_studio_prompt", updated["uuid"])

    assert [event[1:3] for event in created_events] == [("create", "2")]
    assert created_events[0][3]["name"] == "Pirate"
    assert [event[1:3] for event in updated_events] == [("create", "2")]
    assert updated_events[0][3] == {
        "prompt_id": prompt["id"],
        "new_version": 2,
        "change_description": "Keep answers brief",
        "version_operation": "create",
    }
    assert studio_db.get_prompt(updated["id"])["system_prompt"] == "Reply briefly as a pirate."
