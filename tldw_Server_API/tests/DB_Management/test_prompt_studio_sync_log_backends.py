"""Prompt Studio mutations retain sync events on the real content backends."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def studio_db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[PromptStudioDatabase]:
    """Bootstrap the production shared content schema on official PostgreSQL."""
    backend = media = db = None
    try:
        if request.param == "postgres":
            # The shared PostgreSQL plugin owns database provisioning and teardown;
            # this fixture closes only the content adapters and their connection pool.
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


def _persisted_events(
    db: PromptStudioDatabase, entity: str, entity_uuid: str,
) -> list[tuple[int, str, str, dict[str, Any]]]:
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


def test_project_create_and_update_persist_sync_events(studio_db: PromptStudioDatabase) -> None:
    """Create/update through the fixture DB and require durable ordered project events."""
    project = studio_db.create_project(name="Rowan source project", user_id="2")
    studio_db.update_project(project["id"], description="Public source prompts")

    events = _persisted_events(studio_db, "prompt_studio_project", project["uuid"])

    assert [event[1:3] for event in events] == [("create", "2"), ("update", "2")]
    assert 0 < events[0][0] < events[1][0]
    assert events[0][3]["name"] == "Rowan source project"
    assert events[1][3] == {"description": "Public source prompts"}
    assert studio_db.get_project(project["id"])["description"] == "Public source prompts"


def test_prompt_create_and_new_version_persist_sync_events(studio_db: PromptStudioDatabase) -> None:
    """Create a prompt and revision, then read their exact events after releasing the writer."""
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


@pytest.mark.parametrize("studio_db", ["postgres"], indirect=True)
def test_restricted_sync_ownership_is_separate_from_audit_client(
    studio_db: PromptStudioDatabase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web audit ID must neither prevent writes nor expose another tenant's events."""
    monkeypatch.setenv("TLDW_CONTENT_PG_ROLE_SWITCH", "1")
    backend = studio_db.backend
    studio_db.close_connection()
    owners = {
        tenant: PromptStudioDatabase(
            tmp_path / f"studio-{tenant}.db", client_id="web", tenant_user_id=tenant, backend=backend
        )
        for tenant in ("2", "3")
    }
    role_name = f"uat304_sync_{uuid4().hex[:12]}"
    role = backend.escape_identifier(role_name)
    created = False
    try:
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            backend.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}",
                connection=conn,
            )
            backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        created = True
        prompts = {}
        for tenant, db in owners.items():
            with scoped_context(user_id=int(tenant), org_ids=[], team_ids=[], is_admin=False, session_role=role_name):
                flags = db.get_connection().execute(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user"
                ).fetchone()
                assert not flags["rolsuper"] and not flags["rolbypassrls"]
                project = db.create_project(f"Owner {tenant}", user_id=tenant)
                prompt = db.create_prompt(project_id=project["id"], name="Pirate", system_prompt="Speak as a pirate.")
                project_events = _persisted_events(db, "prompt_studio_project", project["uuid"])
                events = _persisted_events(db, "prompt_studio_prompt", prompt["uuid"])
                assert [event[1:3] for event in project_events] == [("create", tenant)]
                assert [event[1:3] for event in events] == [("create", tenant)]
                assert db.get_project(project["id"])["client_id"] == "web"
                assert db.get_prompt(prompt["id"])["client_id"] == "web"
                prompts[tenant] = prompt
                db.close_connection()
        for tenant, db in owners.items():
            foreign = prompts["3" if tenant == "2" else "2"]
            with scoped_context(user_id=int(tenant), org_ids=[], team_ids=[], is_admin=False, session_role=role_name):
                assert _persisted_events(db, "prompt_studio_prompt", foreign["uuid"]) == []
                db.close_connection()
    finally:
        for db in owners.values():
            db.close()
        if created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {role}", connection=conn)
                backend.execute(f"DROP ROLE {role}", connection=conn)
