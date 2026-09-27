"""Integration tests for Notes Studio API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


async def _test_generation_adapter(request: dict[str, object], _context: dict[str, object]) -> dict[str, object]:
    excerpt = str(request.get("excerpt_text") or "").strip()
    title = str(request.get("derived_title") or "Study Notes")
    source_note_id = str(request.get("source_note_id") or "").strip()
    template_type = str(request.get("template_type") or "lined")
    cue_item = "Recall prompt: What is the key idea?" if template_type == "cornell" else "What is the key idea?"
    return {
        "payload": {
            "meta": {"title": title, "source_note_id": source_note_id},
            "sections": [
                {"id": "cue-1", "kind": "cue", "title": "Key Questions", "items": [cue_item]},
                {"id": "notes-1", "kind": "notes", "title": "Notes", "content": excerpt},
                {"id": "summary-1", "kind": "summary", "title": "Summary", "content": excerpt},
            ],
        }
    }


async def _test_diagram_adapter(_request: dict[str, object], _context: dict[str, object]) -> dict[str, object]:
    return {"diagram": "graph TD; A-->B", "format": "mermaid"}


@pytest.fixture()
def client_with_notes_studio_db(tmp_path, monkeypatch):
    db_path = tmp_path / "notes_studio_integration.db"
    db = CharactersRAGDB(str(db_path), client_id="notes_studio_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    class _NoopRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            return True, {}

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return db

    service_cls = notes_endpoint.NotesStudioService

    def _service_factory(*, db, user_id):
        return service_cls(
            db=db,
            user_id=user_id,
            generation_adapter=_test_generation_adapter,
            diagram_adapter=_test_diagram_adapter,
        )

    monkeypatch.setattr(notes_endpoint, "NotesStudioService", _service_factory)

    fastapi_app = FastAPI()
    fastapi_app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep
    fastapi_app.dependency_overrides[get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    fastapi_app.state.notes_studio_db = db

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


def _create_source_note(client: TestClient, *, title: str, content: str) -> str:
    response = client.post("/api/v1/notes/", json={"title": title, "content": content})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_notes_studio_derive_fetch_and_regenerate_flow(client_with_notes_studio_db: TestClient):
    client = client_with_notes_studio_db
    excerpt = "The mitochondrion is the powerhouse of the cell."
    source_note_id = _create_source_note(
        client,
        title="Biology",
        content=(
            "Cells need energy.\n"
            "The mitochondrion is the powerhouse of the cell.\n"
            "ATP supports cellular work."
        ),
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": excerpt,
            "template_type": "cornell",
            "handwriting_mode": "accented",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    derived = derive_response.json()
    note_id = derived["note"]["id"]

    assert derived["note"]["title"] == "Biology Study Notes"
    assert derived["studio_document"]["source_note_id"] == source_note_id
    assert derived["studio_document"]["excerpt_snapshot"] == excerpt
    assert derived["studio_document"]["excerpt_hash"].startswith("sha256:")
    assert derived["studio_document"]["payload_json"]["layout"] == {
        "template_type": "cornell",
        "handwriting_mode": "accented",
        "render_version": 1,
    }
    assert derived["is_stale"] is False

    note_response = client.get(f"/api/v1/notes/{note_id}")
    assert note_response.status_code == 200
    note_payload = note_response.json()
    assert note_payload["studio"]["source_note_id"] == source_note_id

    fetch_response = client.get(f"/api/v1/notes/{note_id}/studio")
    assert fetch_response.status_code == 200
    fetched = fetch_response.json()
    assert fetched["studio_document"]["note_id"] == note_id
    assert fetched["is_stale"] is False

    manual_markdown = (
        "# Biology Refined Study Notes\n\n"
        "## Key Questions\n\n"
        "- Why do cells need ATP?\n"
        "- Which organelle supports cellular energy production?\n\n"
        "## Notes\n\n"
        "The mitochondrion is the powerhouse of the cell.\n"
        "ATP supports cellular work.\n\n"
        "## Summary\n\n"
        "Mitochondria help cells access usable energy."
    )
    drift_response = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"content": manual_markdown},
    )
    assert drift_response.status_code == 200, drift_response.text

    stale_response = client.get(f"/api/v1/notes/{note_id}/studio")
    assert stale_response.status_code == 200
    assert stale_response.json()["is_stale"] is True
    assert stale_response.json()["stale_reason"] == "companion_content_hash_mismatch"

    regenerate_response = client.post(
        f"/api/v1/notes/{note_id}/studio/regenerate",
        json={"expected_version": int(drift_response.json()["version"])},
    )
    assert regenerate_response.status_code == 200, regenerate_response.text
    regenerated = regenerate_response.json()
    assert regenerated["is_stale"] is False
    assert regenerated["note"]["title"] == "Biology Refined Study Notes"
    assert regenerated["note"]["content"] == manual_markdown
    assert regenerated["studio_document"]["payload_json"]["meta"]["title"] == "Biology Refined Study Notes"
    assert regenerated["studio_document"]["payload_json"]["layout"] == {
        "template_type": "cornell",
        "handwriting_mode": "accented",
        "render_version": 1,
    }
    assert regenerated["studio_document"]["payload_json"]["sections"] == [
        {
            "id": "cue-1",
            "kind": "cue",
            "title": "Key Questions",
            "items": [
                "Why do cells need ATP?",
                "Which organelle supports cellular energy production?",
            ],
        },
        {
            "id": "notes-1",
            "kind": "notes",
            "title": "Notes",
            "content": "The mitochondrion is the powerhouse of the cell.\nATP supports cellular work.",
        },
        {
            "id": "summary-1",
            "kind": "summary",
            "title": "Summary",
            "content": "Mitochondria help cells access usable energy.",
        },
    ]
    note_after_regenerate = client.get(f"/api/v1/notes/{note_id}")
    assert note_after_regenerate.status_code == 200
    assert note_after_regenerate.json()["title"] == "Biology Refined Study Notes"


def test_notes_studio_regenerate_accepts_current_markdown_override(client_with_notes_studio_db: TestClient):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Cells use mitochondria to produce ATP.",
            "template_type": "cornell",
            "handwriting_mode": "accented",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    derived_note = derive_response.json()["note"]
    note_id = derived_note["id"]

    override_markdown = (
        "# Biology Refined Study Notes\n\n"
        "## Key Questions\n\n"
        "- What organelle helps produce ATP?\n\n"
        "## Notes\n\n"
        "Cells use mitochondria to produce ATP.\n\n"
        "## Summary\n\n"
        "Mitochondria support cellular energy."
    )

    regenerate_response = client.post(
        f"/api/v1/notes/{note_id}/studio/regenerate",
        json={
            "expected_version": int(derived_note["version"]),
            "current_markdown": override_markdown,
        },
    )
    assert regenerate_response.status_code == 200, regenerate_response.text
    regenerated = regenerate_response.json()

    assert regenerated["is_stale"] is False
    assert regenerated["note"]["title"] == "Biology Refined Study Notes"
    assert regenerated["note"]["content"] == override_markdown
    assert regenerated["studio_document"]["payload_json"]["meta"]["title"] == "Biology Refined Study Notes"

    note_response = client.get(f"/api/v1/notes/{note_id}")
    assert note_response.status_code == 200
    assert note_response.json()["content"] == override_markdown


def test_notes_studio_regenerate_treats_empty_current_markdown_as_an_explicit_override(
    client_with_notes_studio_db: TestClient,
):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Cells use mitochondria to produce ATP.",
            "template_type": "cornell",
            "handwriting_mode": "accented",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    derived_note = derive_response.json()["note"]
    note_id = derived_note["id"]

    regenerate_response = client.post(
        f"/api/v1/notes/{note_id}/studio/regenerate",
        json={"expected_version": int(derived_note["version"]), "current_markdown": ""},
    )
    assert regenerate_response.status_code == 200, regenerate_response.text
    regenerated = regenerate_response.json()

    assert regenerated["is_stale"] is False
    assert regenerated["note"]["title"] == "Biology Study Notes"
    assert regenerated["note"]["content"] == "# Biology Study Notes"
    assert regenerated["studio_document"]["payload_json"]["sections"] == []


def test_notes_studio_regenerate_requires_expected_version(client_with_notes_studio_db: TestClient):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Cells use mitochondria to produce ATP.",
            "template_type": "cornell",
            "handwriting_mode": "accented",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    note_id = derive_response.json()["note"]["id"]

    response = client.post(f"/api/v1/notes/{note_id}/studio/regenerate")

    assert response.status_code == 422


def test_notes_studio_regenerate_rejects_stale_expected_version(client_with_notes_studio_db: TestClient):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Cells use mitochondria to produce ATP.",
            "template_type": "cornell",
            "handwriting_mode": "accented",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    derived_note = derive_response.json()["note"]
    note_id = derived_note["id"]

    drift_response = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"content": "# Edited elsewhere"},
    )
    assert drift_response.status_code == 200, drift_response.text

    response = client.post(
        f"/api/v1/notes/{note_id}/studio/regenerate",
        json={
            "expected_version": int(derived_note["version"]),
            "current_markdown": "# Stale editor body",
        },
    )

    assert response.status_code == 409


def test_notes_studio_derive_rolls_back_note_when_sidecar_persistence_fails(
    client_with_notes_studio_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = client_with_notes_studio_db
    db = client.app.state.notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Physics",
        content="Velocity describes speed with direction.",
    )

    def _raise_sidecar_failure(**_kwargs):
        raise CharactersRAGDBError("sidecar write failed")

    monkeypatch.setattr(db, "create_note_studio_document", _raise_sidecar_failure)

    response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Velocity describes speed with direction.",
            "template_type": "lined",
            "handwriting_mode": "accented",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "A database error occurred while processing your request for note studio."
    assert [note["id"] for note in db.list_notes()] == [source_note_id]


def test_notes_studio_regenerate_rolls_back_note_update_when_sidecar_upsert_fails(
    client_with_notes_studio_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = client_with_notes_studio_db
    db = client.app.state.notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Astronomy",
        content="Stars form inside dense molecular clouds.",
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Stars form inside dense molecular clouds.",
            "template_type": "lined",
            "handwriting_mode": "accented",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    note_id = derive_response.json()["note"]["id"]

    draft_markdown = (
        "# Astronomy Refined Study Notes\n\n"
        "## Key Questions\n\n"
        "* Where do stars form?\n"
        "* What is dense inside the cloud?\n\n"
        "## Notes\n\n"
        "Stars form inside dense molecular clouds.\n"
        "Gravity compresses the gas over time.\n\n"
        "## Summary\n\n"
        "Dense clouds can collapse into new stars."
    )
    drift_response = client.patch(
        f"/api/v1/notes/{note_id}",
        json={"content": draft_markdown},
    )
    assert drift_response.status_code == 200, drift_response.text

    def _raise_sidecar_failure(**_kwargs):
        raise CharactersRAGDBError("sidecar upsert failed")

    monkeypatch.setattr(db, "upsert_note_studio_document", _raise_sidecar_failure)

    regenerate_response = client.post(
        f"/api/v1/notes/{note_id}/studio/regenerate",
        json={"expected_version": int(drift_response.json()["version"])},
    )
    assert regenerate_response.status_code == 500
    assert regenerate_response.json()["detail"] == "A database error occurred while processing your request for note studio."

    note_after_failure = db.get_note_by_id(note_id)
    assert note_after_failure is not None
    assert note_after_failure["title"] == "Astronomy Study Notes"
    assert note_after_failure["content"] == draft_markdown


def test_notes_studio_diagram_manifest_round_trip(client_with_notes_studio_db: TestClient):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="History",
        content="The printing press accelerated the spread of written knowledge across Europe.",
    )

    derive_response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "The printing press accelerated the spread of written knowledge across Europe.",
            "template_type": "lined",
            "handwriting_mode": "off",
        },
    )
    assert derive_response.status_code == 201, derive_response.text
    payload = derive_response.json()
    note_id = payload["note"]["id"]
    sections = payload["studio_document"]["payload_json"]["sections"]

    diagram_response = client.post(
        f"/api/v1/notes/{note_id}/studio/diagrams",
        json={
            "diagram_type": "flowchart",
            "source_section_ids": [sections[0]["id"], sections[1]["id"]],
        },
    )
    assert diagram_response.status_code == 200, diagram_response.text
    manifest = diagram_response.json()["studio_document"]["diagram_manifest_json"]

    assert manifest["diagram_type"] == "flowchart"
    assert manifest["source_section_ids"] == [sections[0]["id"], sections[1]["id"]]
    assert manifest["source_graph"]
    assert manifest["cached_svg"].startswith("<svg")
    assert manifest["render_hash"].startswith("sha256:")
    assert manifest["generation_status"] == "ready"


@pytest.mark.parametrize("excerpt_text", ["   ", "Excerpt missing from source"])
def test_notes_studio_rejects_invalid_excerpt_requests(
    client_with_notes_studio_db: TestClient,
    excerpt_text: str,
):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Source",
        content="Useful content for excerpt validation.",
    )

    response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": excerpt_text,
            "template_type": "lined",
            "handwriting_mode": "accented",
        },
    )

    assert response.status_code == 400


def test_notes_studio_derive_rejects_oversized_provider_override(
    client_with_notes_studio_db: TestClient,
):
    client = client_with_notes_studio_db
    source_note_id = _create_source_note(
        client,
        title="Source",
        content="Useful content for provider validation.",
    )

    response = client.post(
        "/api/v1/notes/studio/derive",
        json={
            "source_note_id": source_note_id,
            "excerpt_text": "Useful content for provider validation.",
            "template_type": "lined",
            "handwriting_mode": "accented",
            "provider": "p" * 101,
        },
    )

    assert response.status_code == 422
