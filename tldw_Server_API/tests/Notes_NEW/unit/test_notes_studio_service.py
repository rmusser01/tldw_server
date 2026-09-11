"""Unit tests for Notes Studio service orchestration."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes.studio_markdown import stable_content_hash
from tldw_Server_API.app.core.Notes.studio_service import NotesStudioService
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    StudioSectionsV1,
    diagram_render_hash,
)

pytestmark = pytest.mark.unit


async def _test_generation_adapter(request: dict[str, object], _context: dict[str, object]) -> dict[str, object]:
    excerpt = str(request.get("excerpt_text") or "").strip()
    title = str(request.get("derived_title") or "Study Notes")
    source_note_id = str(request.get("source_note_id") or "").strip()
    template_type = str(request.get("template_type") or "lined")
    cue_item = "Recall prompt: What is the key idea?" if template_type == "cornell" else "What is the key idea?"
    return {
        "payload": {
            "meta": {
                "title": title,
                "source_note_id": source_note_id,
            },
            "sections": [
                {
                    "id": "cue-1",
                    "kind": "cue",
                    "title": "Key Questions",
                    "items": [cue_item],
                },
                {
                    "id": "notes-1",
                    "kind": "notes",
                    "title": "Notes",
                    "content": excerpt,
                },
                {
                    "id": "summary-1",
                    "kind": "summary",
                    "title": "Summary",
                    "content": excerpt,
                },
            ],
        }
    }


def _service(db: CharactersRAGDB, **kwargs) -> NotesStudioService:
    kwargs.setdefault("generation_adapter", _test_generation_adapter)
    kwargs.setdefault("user_id", "notes_studio_unit")
    return NotesStudioService(db=db, **kwargs)


@pytest.fixture()
def studio_db(tmp_path: Path):
    db = CharactersRAGDB(str(tmp_path / "notes_studio_unit.db"), client_id="notes_studio_unit")
    yield db


def _derive_note(
    service: NotesStudioService,
    *,
    source_note_id: str,
    excerpt_text: str,
    template_type: str = "lined",
    handwriting_mode: str = "accented",
) -> dict:
    return asyncio.run(
        service.derive_from_excerpt(
            source_note_id=source_note_id,
            excerpt_text=excerpt_text,
            template_type=template_type,
            handwriting_mode=handwriting_mode,
        )
    )


def _studio_ensure_fields(
    *,
    note_id: str,
    source_note_id: str,
    content: str = "Accepted canonical content",
) -> dict[str, object]:
    return {
        "note_id": note_id,
        "payload_json": {
            "meta": {"title": "Study", "source_note_id": source_note_id},
            "layout": {
                "template_type": "lined",
                "handwriting_mode": "accented",
                "render_version": 1,
            },
            "sections": [
                {
                    "id": "notes-custom",
                    "kind": "notes",
                    "title": "Notes",
                    "content": content,
                }
            ],
        },
        "template_type": "lined",
        "handwriting_mode": "accented",
        "source_note_id": source_note_id,
        "excerpt_snapshot": "Accepted excerpt",
        "excerpt_hash": stable_content_hash("Accepted excerpt"),
        "diagram_manifest_json": None,
        "companion_content_hash": stable_content_hash(
            "# Study\n\nAccepted companion"
        ),
        "render_version": 1,
        "provenance_kind": "derive",
        "provenance_provider": "openai",
        "provenance_model": "gpt-test",
    }


def test_derive_execution_identity_distinguishes_local_and_llm_runs() -> None:
    """Return truthful derive identities and reject incomplete LLM metadata."""
    assert NotesStudioService._derive_execution_identity(
        {"source": "deterministic_fallback"},
        provider=None,
        model=None,
    ) == ("tldw", "notes-studio-deterministic-v1")
    assert NotesStudioService._derive_execution_identity(
        {"source": "llm"},
        provider="provider-a",
        model="model-a",
    ) == ("provider-a", "model-a")
    with pytest.raises(InputError, match="identity is incomplete"):
        NotesStudioService._derive_execution_identity(
            {"source": "llm"},
            provider="provider-a",
            model=None,
        )


def test_diagram_execution_identity_validates_reported_execution_source() -> None:
    """Return truthful diagram identities and reject invalid adapter metadata."""
    assert NotesStudioService._diagram_execution_identity({}) == (
        "tldw",
        "diagram-deterministic-v1",
    )
    assert NotesStudioService._diagram_execution_identity(
        {"source": "llm", "provider": "provider-b", "model": "model-b"}
    ) == ("provider-b", "model-b")
    with pytest.raises(InputError, match="source is invalid"):
        NotesStudioService._diagram_execution_identity({"source": "unknown"})
    with pytest.raises(InputError, match="identity is incomplete"):
        NotesStudioService._diagram_execution_identity(
            {"source": "llm", "provider": "provider-b"}
        )


def test_derive_creates_derived_note_and_sidecar(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Source Note",
        content=(
            "Cells need energy to function.\n"
            "The mitochondrion is the powerhouse of the cell.\n"
            "ATP stores usable energy."
        ),
    )
    assert source_note_id is not None

    service = _service(db)
    excerpt = "The mitochondrion is the powerhouse of the cell."

    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text=excerpt,
    )

    note = result["note"]
    studio_document = result["studio_document"]

    assert note["id"] != str(source_note_id)
    assert note["title"] == "Source Note Study Notes"
    assert note["content"].startswith("# Source Note Study Notes")
    assert "## Key Questions" in note["content"]
    assert "## Notes" in note["content"]
    assert "## Summary" in note["content"]
    assert "Template:" not in note["content"]
    assert "handwriting_mode" not in note["content"]
    assert result["is_stale"] is False
    assert result["stale_reason"] is None

    assert studio_document["source_note_id"] == str(source_note_id)
    assert studio_document["excerpt_snapshot"] == excerpt
    assert studio_document["excerpt_hash"].startswith("sha256:")
    assert studio_document["companion_content_hash"].startswith("sha256:")
    assert studio_document["payload_json"]["meta"]["source_note_id"] == str(source_note_id)
    assert studio_document["payload_json"]["layout"] == {
        "template_type": "lined",
        "handwriting_mode": "accented",
        "render_version": 1,
    }
    assert studio_document["accepted_provenance_json"] == {
        **studio_document["accepted_provenance_json"],
        "kind": "derive",
        "provider": "tldw",
        "model": "notes-studio-deterministic-v1",
    }


def test_derive_uses_canonical_payload_title_for_note_row_and_markdown(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Source Note",
        content="A source excerpt that yields a canonical title from the generator.",
    )
    assert source_note_id is not None

    async def _generation_adapter(request: dict[str, object], _context: dict[str, object]) -> dict[str, object]:
        assert request["derived_title"] == "Source Note Study Notes"
        return {
            "payload": {
                "meta": {
                    "title": "Canonical Study Notes",
                    "source_note_id": str(source_note_id),
                },
                "sections": [
                    {
                        "id": "notes-1",
                        "kind": "notes",
                        "title": "Notes",
                        "content": "A source excerpt that yields a canonical title from the generator.",
                    }
                ],
            }
        }

    service = NotesStudioService(
        db=db,
        user_id="notes_studio_unit",
        generation_adapter=_generation_adapter,
    )

    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="A source excerpt that yields a canonical title from the generator.",
    )

    note = result["note"]
    studio_document = result["studio_document"]

    assert note["title"] == "Canonical Study Notes"
    assert note["content"].startswith("# Canonical Study Notes")
    assert studio_document["payload_json"]["meta"]["title"] == "Canonical Study Notes"
    assert result["is_stale"] is False
    assert result["stale_reason"] is None


def test_cornell_generation_includes_explicit_recall_prompt(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Biology",
        content="Photosynthesis converts light energy into chemical energy for plants.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="Photosynthesis converts light energy into chemical energy for plants.",
        template_type="cornell",
    )

    note_content = result["note"]["content"]
    cue_section = result["studio_document"]["payload_json"]["sections"][0]

    assert cue_section["kind"] == "cue"
    assert any(
        "Recall prompt:" in item or "Fill in the blank:" in item
        for item in cue_section["items"]
    )
    assert "Recall prompt:" in note_content or "Fill in the blank:" in note_content


def test_derive_rolls_back_note_when_sidecar_persistence_fails(studio_db, monkeypatch: pytest.MonkeyPatch):
    db = studio_db
    source_note_id = db.add_note(
        title="Physics",
        content="Velocity describes speed with direction.",
    )
    assert source_note_id is not None

    service = _service(db)
    original_note_ids = [note["id"] for note in db.list_notes()]

    def _raise_sidecar_failure(**_kwargs):
        raise CharactersRAGDBError("sidecar write failed")

    monkeypatch.setattr(db, "create_note_studio_document", _raise_sidecar_failure)

    with pytest.raises(CharactersRAGDBError, match="sidecar write failed"):
        _derive_note(
            service,
            source_note_id=str(source_note_id),
            excerpt_text="Velocity describes speed with direction.",
        )

    assert [note["id"] for note in db.list_notes()] == original_note_ids


def test_ensure_studio_document_accepts_identical_storage_normalized_retry(
    studio_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )

    first = service._ensure_studio_document(**fields)
    first_accepted_at = first["accepted_provenance_json"]["accepted_at"]
    monkeypatch.setattr(
        db,
        "_get_current_utc_timestamp_iso",
        lambda: "2031-01-02T03:04:05.000000Z",
    )
    second = service._ensure_studio_document(**fields)

    assert second == first
    assert second["accepted_provenance_json"]["accepted_at"] == first_accepted_at
    assert second["version"] == 1
    assert second["canonical_revision"] == 1


@pytest.mark.parametrize(
    "companion_hash_hint",
    (None, stable_content_hash("caller supplied non-authoritative content")),
    ids=("none", "non-authoritative"),
)
def test_ensure_studio_document_normalizes_companion_hash_hint_on_retry(
    studio_db,
    companion_hash_hint: str | None,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    companion = "# Study\n\nAccepted companion"
    note_id = db.add_note(title="Study", content=companion)
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id), source_note_id=str(source_note_id)
    )
    fields["companion_content_hash"] = companion_hash_hint

    first = service._ensure_studio_document(**fields)
    second = service._ensure_studio_document(**fields)

    assert second == first
    assert second["companion_content_hash"] == stable_content_hash(companion)


def test_ensure_studio_document_returns_exact_stored_state_after_parent_edit(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    original_companion = "# Study\n\nAccepted companion"
    note_id = db.add_note(title="Study", content=original_companion)
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    fields["excerpt_hash"] = stable_content_hash("Accepted excerpt")
    fields["companion_content_hash"] = stable_content_hash(original_companion)
    stored = service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.update_note(
        note_id=str(note_id),
        update_data={"title": "Study edited", "content": "Ordinary later edit"},
        expected_version=int(parent["version"]),
    )

    replayed = service._ensure_studio_document(**fields)

    assert replayed == stored


def test_ensure_studio_document_returns_exact_stored_state_after_source_tombstone(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    original_companion = "# Study\n\nAccepted companion"
    note_id = db.add_note(title="Study", content=original_companion)
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    fields["excerpt_hash"] = stable_content_hash("Accepted excerpt")
    fields["companion_content_hash"] = stable_content_hash(original_companion)
    stored = service._ensure_studio_document(**fields)
    source = db.get_note_by_id(str(source_note_id))
    assert source is not None
    db.soft_delete_note(str(source_note_id), expected_version=int(source["version"]))

    replayed = service._ensure_studio_document(**fields)

    assert replayed == stored


def test_ensure_studio_document_rejects_different_payload_after_parent_edit(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    original_companion = "# Study\n\nAccepted companion"
    note_id = db.add_note(title="Study", content=original_companion)
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    fields["excerpt_hash"] = stable_content_hash("Accepted excerpt")
    fields["companion_content_hash"] = stable_content_hash(original_companion)
    service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.update_note(
        note_id=str(note_id),
        update_data={"content": "Ordinary later edit"},
        expected_version=int(parent["version"]),
    )
    changed = dict(fields)
    changed["payload_json"] = {
        "sections": [
            {
                "id": "notes-custom",
                "kind": "notes",
                "title": "Notes",
                "content": "Different canonical content",
            }
        ]
    }

    with pytest.raises(ConflictError, match="captured retry"):
        service._ensure_studio_document(**changed)


@pytest.mark.parametrize(
    "difference",
    ("source", "excerpt", "manifest", "provenance"),
)
def test_ensure_studio_document_rejects_other_different_accepted_semantics(
    studio_db,
    difference: str,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    service._ensure_studio_document(**fields)
    changed = dict(fields)
    if difference == "source":
        other_source_id = db.add_note(title="Other source", content="Other excerpt")
        assert other_source_id
        changed["source_note_id"] = str(other_source_id)
        changed["excerpt_snapshot"] = "Other excerpt"
        changed["excerpt_hash"] = stable_content_hash("Other excerpt")
        changed["payload_json"] = {
            **dict(fields["payload_json"]),
            "meta": {"title": "Study", "source_note_id": str(other_source_id)},
        }
    elif difference == "excerpt":
        changed["excerpt_snapshot"] = "Accepted"
        changed["excerpt_hash"] = stable_content_hash("Accepted")
    elif difference == "manifest":
        diagram = "graph TD; A-->B"
        context = "Notes\nAccepted canonical content"
        changed["diagram_manifest_json"] = {
            "diagram_type": "flowchart",
            "source_section_ids": ["notes-custom"],
            "source_graph": [
                {
                    "id": "notes-custom",
                    "title": "Notes",
                    "kind": "notes",
                    "content": "Accepted canonical content",
                }
            ],
            "diagram": diagram,
            "format": "mermaid",
            "status": "ready",
            "render_hash": diagram_render_hash(
                diagram_type="flowchart", context=context, diagram=diagram
            ),
        }
    elif difference == "provenance":
        changed["provenance_model"] = "gpt-different"

    with pytest.raises(ConflictError, match="captured retry"):
        service._ensure_studio_document(**changed)


def test_ensure_studio_document_returns_exact_parent_tombstone_lifecycle(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    live = service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.soft_delete_note(str(note_id), expected_version=int(parent["version"]))
    tombstone = db.get_note_studio_document(str(note_id))
    assert tombstone is not None and tombstone["deleted"] == 1

    replayed = service._ensure_studio_document(**fields)

    assert replayed == tombstone
    assert replayed["canonical_hash"] != live["canonical_hash"]


@pytest.mark.parametrize(
    "companion_hash_hint",
    (None, stable_content_hash("caller supplied non-authoritative content")),
    ids=("none", "non-authoritative"),
)
def test_ensure_studio_document_normalizes_companion_hint_for_tombstone_replay(
    studio_db,
    companion_hash_hint: str | None,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id), source_note_id=str(source_note_id)
    )
    fields["companion_content_hash"] = companion_hash_hint
    service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.soft_delete_note(str(note_id), expected_version=int(parent["version"]))
    before = dict(
        db.execute_query(
            "SELECT * FROM note_studio_documents WHERE note_id=?", (note_id,)
        ).fetchone()
    )

    replayed = service._ensure_studio_document(**fields)

    after = dict(
        db.execute_query(
            "SELECT * FROM note_studio_documents WHERE note_id=?", (note_id,)
        ).fetchone()
    )
    assert replayed["deleted"] == 1
    assert after == before


def test_upsert_studio_document_rejects_grouped_tombstone_mutation(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id), source_note_id=str(source_note_id)
    )
    service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.soft_delete_note(str(note_id), expected_version=int(parent["version"]))
    before = dict(
        db.execute_query(
            "SELECT * FROM note_studio_documents WHERE note_id=?", (note_id,)
        ).fetchone()
    )
    changed = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
        content="Changed while parent is deleted",
    )
    changed.update(
        provenance_kind="regenerate",
        provenance_provider=None,
        provenance_model=None,
    )

    with pytest.raises(ConflictError, match="Studio parent note not found or not live"):
        db.upsert_note_studio_document(**changed)

    after = dict(
        db.execute_query(
            "SELECT * FROM note_studio_documents WHERE note_id=?", (note_id,)
        ).fetchone()
    )
    assert after == before


def test_ensure_studio_document_rejects_changed_grouped_tombstone_retry(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id), source_note_id=str(source_note_id)
    )
    service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.soft_delete_note(str(note_id), expected_version=int(parent["version"]))
    before = db.get_note_studio_document(str(note_id))
    assert before is not None and before["deleted"] == 1
    changed = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
        content="Changed while parent is deleted",
    )

    with pytest.raises(ConflictError, match="captured retry"):
        service._ensure_studio_document(**changed)

    assert db.get_note_studio_document(str(note_id)) == before


def test_restored_studio_parent_allows_changed_upsert(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id), source_note_id=str(source_note_id)
    )
    service._ensure_studio_document(**fields)
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.soft_delete_note(str(note_id), expected_version=int(parent["version"]))
    tombstone = db.get_note_studio_document(str(note_id))
    deleted_parent = db.get_note_by_id(str(note_id), include_deleted=True)
    assert tombstone is not None and tombstone["deleted"] == 1
    assert deleted_parent is not None
    db.restore_note(str(note_id), expected_version=int(deleted_parent["version"]))
    restored = db.get_note_studio_document(str(note_id))
    assert restored is not None and restored["deleted"] == 0
    changed = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
        content="Changed after grouped restore",
    )
    changed.update(
        provenance_kind="regenerate",
        provenance_provider=None,
        provenance_model=None,
    )

    saved = db.upsert_note_studio_document(**changed)

    assert saved["deleted"] == 0
    assert saved["payload_json"]["sections"][0]["content"] == (
        "Changed after grouped restore"
    )
    assert saved["canonical_revision"] == restored["canonical_revision"] + 1


def test_ensure_new_studio_document_rejects_tombstoned_parent(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    parent = db.get_note_by_id(str(note_id))
    assert parent is not None
    db.soft_delete_note(str(note_id), expected_version=int(parent["version"]))

    with pytest.raises(ConflictError, match="Studio parent note not found or not live"):
        _service(db)._ensure_studio_document(
            **_studio_ensure_fields(
                note_id=str(note_id), source_note_id=str(source_note_id)
            )
        )

    assert db.get_note_studio_document(str(note_id)) is None


def test_ensure_new_studio_document_creates_live_state_for_live_parent(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id

    document = _service(db)._ensure_studio_document(
        **_studio_ensure_fields(
            note_id=str(note_id), source_note_id=str(source_note_id)
        )
    )

    assert document["deleted"] == 0
    assert document["canonical_revision"] == 1


def test_ensure_new_studio_document_still_rejects_tombstoned_source(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    source = db.get_note_by_id(str(source_note_id))
    assert source is not None
    db.soft_delete_note(str(source_note_id), expected_version=int(source["version"]))

    with pytest.raises(ConflictError, match="Source note not found or not live"):
        _service(db)._ensure_studio_document(
            **_studio_ensure_fields(
                note_id=str(note_id), source_note_id=str(source_note_id)
            )
        )
    assert db.get_note_studio_document(str(note_id)) is None


def test_ensure_studio_document_concurrent_identical_creators_converge(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    barrier = threading.Barrier(2)

    def create() -> dict[str, object]:
        barrier.wait(timeout=10)
        return _service(db)._ensure_studio_document(**fields)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=30) for future in [pool.submit(create), pool.submit(create)]]

    assert results[0] == results[1]
    rows = db.execute_query(
        "SELECT COUNT(*) AS total FROM note_studio_documents WHERE note_id=?",
        (note_id,),
    ).fetchone()
    assert rows["total"] == 1


@pytest.mark.parametrize(
    "companion_hash_hint",
    (None, stable_content_hash("caller supplied non-authoritative content")),
    ids=("none", "non-authoritative"),
)
def test_ensure_studio_document_concurrent_companion_hash_hints_converge(
    studio_db,
    companion_hash_hint: str | None,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    fields = _studio_ensure_fields(
        note_id=str(note_id), source_note_id=str(source_note_id)
    )
    fields["companion_content_hash"] = companion_hash_hint
    barrier = threading.Barrier(2)

    def create() -> dict[str, object]:
        barrier.wait(timeout=10)
        return _service(db)._ensure_studio_document(**fields)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create), pool.submit(create)]
        results = [future.result(timeout=30) for future in futures]

    assert results[0] == results[1]
    assert results[0]["companion_content_hash"] == stable_content_hash(
        "# Study\n\nAccepted companion"
    )


def test_ensure_studio_document_rejects_semantically_different_retry(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    service._ensure_studio_document(**fields)
    changed = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
        content="Different canonical content",
    )

    with pytest.raises(ConflictError, match="captured retry"):
        service._ensure_studio_document(**changed)


@pytest.mark.parametrize(
    ("update_sql", "params"),
    [
        (
            "UPDATE note_studio_documents SET canonical_hash=? WHERE note_id=?",
            ("sha256:" + ("0" * 64),),
        ),
        (
            "UPDATE note_studio_documents SET canonical_revision=canonical_revision+1 "
            "WHERE note_id=?",
            (),
        ),
        (
            "UPDATE note_studio_documents SET note_hash=? WHERE note_id=?",
            ("sha256:" + ("0" * 64),),
        ),
        (
            "UPDATE note_studio_documents SET deleted=1 WHERE note_id=?",
            (),
        ),
        (
            "UPDATE note_studio_documents SET source_diagnostic_code=?,"
            "source_diagnostic_hash=? WHERE note_id=?",
            ("stored_studio_state_invalid", "sha256:" + ("0" * 64)),
        ),
        (
            "UPDATE note_studio_documents SET payload_json=? WHERE note_id=?",
            (json.dumps({"unexpected": True}),),
        ),
        (
            "UPDATE note_studio_documents SET diagram_manifest_json=? WHERE note_id=?",
            (json.dumps({"unexpected": True}),),
        ),
    ],
    ids=(
        "canonical-hash",
        "canonical-revision-lineage",
        "note-hash-lineage",
        "parent-lifecycle",
        "source-diagnostic",
        "malformed-payload",
        "malformed-manifest",
    ),
)
def test_ensure_studio_document_rejects_corrupted_identical_retry(
    studio_db,
    update_sql: str,
    params: tuple[object, ...],
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    service._ensure_studio_document(**fields)
    with db.transaction() as conn:
        conn.execute(update_sql, (*params, note_id))

    with pytest.raises(ConflictError, match="stored state"):
        service._ensure_studio_document(**fields)


def test_ensure_studio_document_rejects_corrupted_accepted_result_hash(
    studio_db,
) -> None:
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    note_id = db.add_note(title="Study", content="# Study\n\nAccepted companion")
    assert source_note_id and note_id
    service = _service(db)
    fields = _studio_ensure_fields(
        note_id=str(note_id),
        source_note_id=str(source_note_id),
    )
    document = service._ensure_studio_document(**fields)
    provenance = dict(document["accepted_provenance_json"])
    provenance["result_hash"] = "sha256:" + ("0" * 64)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE note_studio_documents SET accepted_provenance_json=? WHERE note_id=?",
            (json.dumps(provenance), note_id),
        )

    with pytest.raises(ConflictError, match="stored state"):
        service._ensure_studio_document(**fields)


def test_get_state_detects_markdown_drift_and_regenerate_rebuilds_payload_from_current_markdown(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Chemistry",
        content="Atoms form bonds by sharing or transferring electrons.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="Atoms form bonds by sharing or transferring electrons.",
    )
    note_id = result["note"]["id"]

    initial_state = asyncio.run(service.get_note_studio_state(note_id=note_id))
    assert initial_state["is_stale"] is False

    current_note = db.get_note_by_id(note_id=note_id)
    assert current_note is not None
    manual_markdown = (
        "# Chemistry Refined Study Notes\n\n"
        "## Key Questions\n\n"
        "- Which particles are shared in covalent bonds?\n"
        "- What changes during electron transfer?\n\n"
        "## Notes\n\n"
        "Atoms form bonds by sharing or transferring electrons.\n"
        "Electron transfer can create ions.\n\n"
        "## Summary\n\n"
        "Bonding changes electron stability."
    )
    db.update_note(
        note_id=note_id,
        update_data={"content": manual_markdown},
        expected_version=int(current_note["version"]),
    )

    stale_state = asyncio.run(service.get_note_studio_state(note_id=note_id))
    assert stale_state["is_stale"] is True
    assert stale_state["stale_reason"] == "companion_content_hash_mismatch"

    latest_note = db.get_note_by_id(note_id=note_id)
    assert latest_note is not None

    regenerated = asyncio.run(
        service.regenerate_note_markdown(
            note_id=note_id,
            expected_version=int(latest_note["version"]),
        )
    )
    assert regenerated["is_stale"] is False
    assert regenerated["stale_reason"] is None
    assert regenerated["note"]["title"] == "Chemistry Refined Study Notes"
    assert regenerated["note"]["content"] == manual_markdown
    assert regenerated["studio_document"]["companion_content_hash"].startswith("sha256:")
    assert regenerated["studio_document"]["payload_json"]["meta"]["title"] == "Chemistry Refined Study Notes"
    assert regenerated["studio_document"]["payload_json"]["layout"] == {
        "template_type": "lined",
        "handwriting_mode": "accented",
        "render_version": 1,
    }
    assert regenerated["studio_document"]["payload_json"]["sections"] == [
        {
            "id": "cue-1",
            "kind": "cue",
            "title": "Key Questions",
            "items": [
                "Which particles are shared in covalent bonds?",
                "What changes during electron transfer?",
            ],
        },
        {
            "id": "notes-1",
            "kind": "notes",
            "title": "Notes",
            "content": "Atoms form bonds by sharing or transferring electrons.\nElectron transfer can create ions.",
        },
        {
            "id": "summary-1",
            "kind": "summary",
            "title": "Summary",
            "content": "Bonding changes electron stability.",
        },
    ]
    assert regenerated["studio_document"]["accepted_provenance_json"]["kind"] == "regenerate"
    assert regenerated["studio_document"]["accepted_provenance_json"]["provider"] is None
    assert regenerated["studio_document"]["accepted_provenance_json"]["model"] is None
    persisted_note = db.get_note_by_id(note_id=note_id)
    assert persisted_note is not None
    assert persisted_note["title"] == "Chemistry Refined Study Notes"


def test_regenerate_uses_current_markdown_override_without_persisting_drift_first(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="Cells use mitochondria to produce ATP.",
    )
    note_id = result["note"]["id"]

    override_markdown = (
        "# Biology Refined Study Notes\n\n"
        "## Key Questions\n\n"
        "- What organelle helps produce ATP?\n\n"
        "## Notes\n\n"
        "Cells use mitochondria to produce ATP.\n\n"
        "## Summary\n\n"
        "Mitochondria support cellular energy."
    )

    regenerated = asyncio.run(
        service.regenerate_note_markdown(
            note_id=note_id,
            expected_version=int(result["note"]["version"]),
            current_markdown=override_markdown,
        )
    )

    assert regenerated["is_stale"] is False
    assert regenerated["note"]["title"] == "Biology Refined Study Notes"
    assert regenerated["note"]["content"] == override_markdown
    assert regenerated["studio_document"]["payload_json"]["meta"]["title"] == "Biology Refined Study Notes"

    persisted_note = db.get_note_by_id(note_id=note_id)
    assert persisted_note is not None
    assert persisted_note["content"] == override_markdown


def test_regenerate_treats_empty_current_markdown_as_an_explicit_override(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="Cells use mitochondria to produce ATP.",
    )
    note_id = result["note"]["id"]

    regenerated = asyncio.run(
        service.regenerate_note_markdown(
            note_id=note_id,
            expected_version=int(result["note"]["version"]),
            current_markdown="",
        )
    )

    assert regenerated["is_stale"] is False
    assert regenerated["note"]["title"] == "Biology Study Notes"
    assert regenerated["note"]["content"] == "# Biology Study Notes"
    assert regenerated["studio_document"]["payload_json"]["sections"] == []


def test_regenerate_rolls_back_note_update_when_sidecar_upsert_fails(studio_db, monkeypatch: pytest.MonkeyPatch):
    db = studio_db
    source_note_id = db.add_note(
        title="Astronomy",
        content="Stars form inside dense molecular clouds.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="Stars form inside dense molecular clouds.",
    )
    note_id = result["note"]["id"]

    current_note = db.get_note_by_id(note_id=note_id)
    assert current_note is not None
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
    db.update_note(
        note_id=note_id,
        update_data={"content": draft_markdown},
        expected_version=int(current_note["version"]),
    )

    def _raise_sidecar_failure(**_kwargs):
        raise CharactersRAGDBError("sidecar upsert failed")

    monkeypatch.setattr(db, "upsert_note_studio_document", _raise_sidecar_failure)

    with pytest.raises(CharactersRAGDBError, match="sidecar upsert failed"):
        latest_note = db.get_note_by_id(note_id=note_id)
        assert latest_note is not None
        asyncio.run(
            service.regenerate_note_markdown(
                note_id=note_id,
                expected_version=int(latest_note["version"]),
            )
        )

    note_after_failure = db.get_note_by_id(note_id=note_id)
    assert note_after_failure is not None
    assert note_after_failure["title"] == "Astronomy Study Notes"
    assert note_after_failure["content"] == draft_markdown


def test_regenerate_rejects_stale_expected_version(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="Biology",
        content="Cells use mitochondria to produce ATP.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="Cells use mitochondria to produce ATP.",
    )
    note_id = result["note"]["id"]

    db.update_note(
        note_id=note_id,
        update_data={"content": "# Edited elsewhere"},
        expected_version=int(result["note"]["version"]),
    )

    with pytest.raises(ConflictError, match="version mismatch"):
        asyncio.run(
            service.regenerate_note_markdown(
                note_id=note_id,
                expected_version=int(result["note"]["version"]),
                current_markdown="# Stale editor body",
            )
        )


def test_update_diagram_manifest_persists_notebook_diagram_metadata(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="History",
        content="The printing press accelerated the spread of written knowledge.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="The printing press accelerated the spread of written knowledge.",
    )
    note_id = result["note"]["id"]
    section_ids = [section["id"] for section in result["studio_document"]["payload_json"]["sections"]]

    updated = asyncio.run(
        service.update_diagram_manifest(
            note_id=note_id,
            diagram_type="flowchart",
            source_section_ids=section_ids[:2],
        )
    )
    manifest = updated["studio_document"]["diagram_manifest_json"]

    assert manifest["diagram_type"] == "flowchart"
    assert manifest["source_section_ids"] == section_ids[:2]
    assert manifest["source_graph"]
    assert manifest["cached_svg"].startswith("<svg")
    assert manifest["render_hash"].startswith("sha256:")
    assert manifest["generation_status"] == "ready"
    assert updated["studio_document"]["accepted_provenance_json"]["kind"] == "diagram"
    assert updated["studio_document"]["accepted_provenance_json"]["provider"] == "tldw"
    assert updated["studio_document"]["accepted_provenance_json"]["model"] == "diagram-deterministic-v1"


@pytest.mark.parametrize(
    ("adapter_source", "provider", "model", "expected"),
    [
        ("llm", "openai", "gpt-test", ("openai", "gpt-test")),
        ("deterministic_fallback", "openai", "gpt-test", ("tldw", "notes-studio-deterministic-v1")),
        (None, "openai", "gpt-test", ("tldw", "notes-studio-deterministic-v1")),
        ("deterministic_fallback", "openai", None, ("tldw", "notes-studio-deterministic-v1")),
    ],
)
def test_derive_stamps_the_engine_that_actually_executed(
    studio_db,
    adapter_source,
    provider,
    model,
    expected,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None

    async def adapter(request, context):
        result = await _test_generation_adapter(request, context)
        if adapter_source is not None:
            result["source"] = adapter_source
        return result

    service = _service(db, generation_adapter=adapter)
    result = asyncio.run(
        service.derive_from_excerpt(
            source_note_id=source_note_id,
            excerpt_text="Accepted excerpt",
            template_type="lined",
            handwriting_mode="accented",
            provider=provider,
            model=model,
        )
    )
    provenance = result["studio_document"]["accepted_provenance_json"]
    assert (provenance["provider"], provenance["model"]) == expected


@pytest.mark.parametrize(
    ("response_text", "expected_identity", "expected_content"),
    (
        (
            '{"unexpected":"not a Studio payload"}',
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        (
            '{"sections":[{"bogus":1}]}',
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        (
            '{"sections":[{"id":"bad","kind":"invalid","title":"Bad","content":"bad"}]}',
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        (
            '{"sections":[{"id":"dup","kind":"notes","title":"One","content":"one"},'
            '{"id":"dup","kind":"summary","title":"Two","content":"two"}]}',
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        (
            '{"sections":[{"id":"cue-1","kind":"cue","title":"Cue",'
            '"content":"wrong authority"}]}',
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        (
            '{"sections":[{"id":"cue-1","kind":"cue","title":"Cue",'
            '"items":"not-an-array"}]}',
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        (
            json.dumps(
                {
                    "sections": [
                        {
                            "id": "notes-1",
                            "kind": "notes",
                            "title": "Notes",
                            "content": "x" * 65_537,
                        }
                    ]
                }
            ),
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        ("[]", ("tldw", "notes-studio-deterministic-v1"), "Accepted excerpt"),
        ("{malformed", ("tldw", "notes-studio-deterministic-v1"), "Accepted excerpt"),
        (
            "The model returned prose instead of JSON.",
            ("tldw", "notes-studio-deterministic-v1"),
            "Accepted excerpt",
        ),
        ('{"sections":[]}', ("openai", "gpt-test"), None),
        (
            '{"sections":[{"id":"notes-1","kind":"notes","title":"Notes",'
            '"content":"Accepted LLM content"}]}',
            ("openai", "gpt-test"),
            "Accepted LLM content",
        ),
        (
            '{"meta":{"title":"Provider Study Notes","source_note_id":"source-placeholder",'
            '"provider_debug":{"secret":"drop"}},"layout":{"provider_layout":true},'
            '"provider_response":{"trace":"drop"},"sections":[{"id":"notes-1",'
            '"kind":"notes","title":"Notes","content":"Accepted LLM content"}]}',
            ("openai", "gpt-test"),
            "Accepted LLM content",
        ),
    ),
)
def test_real_generation_adapter_persists_only_valid_llm_sections_with_truthful_identity(
    studio_db,
    response_text,
    expected_identity,
    expected_content,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None
    service = NotesStudioService(db=db, user_id="notes_studio_unit")

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=response_text,
    ):
        result = asyncio.run(
            service.derive_from_excerpt(
                source_note_id=str(source_note_id),
                excerpt_text="Accepted excerpt",
                template_type="lined",
                handwriting_mode="accented",
                provider="openai",
                model="gpt-test",
            )
        )

    document = result["studio_document"]
    provenance = document["accepted_provenance_json"]
    assert (provenance["provider"], provenance["model"]) == expected_identity
    assert set(document["payload_json"]["meta"]) == {"title", "source_note_id"}
    assert document["payload_json"]["meta"]["source_note_id"] == str(source_note_id)
    assert set(document["payload_json"]) == {"meta", "layout", "sections"}
    serialized = json.dumps(document["payload_json"], sort_keys=True)
    assert "provider_debug" not in serialized
    assert "provider_response" not in serialized
    assert "provider_layout" not in serialized
    if expected_content is None:
        assert document["payload_json"]["sections"] == []
    else:
        assert any(
            section.get("content") == expected_content
            for section in document["payload_json"]["sections"]
        )


def test_real_generation_adapter_no_provider_fallback_persists_canonical_aliases(
    studio_db,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None
    service = NotesStudioService(db=db, user_id="notes_studio_unit")

    result = asyncio.run(
        service.derive_from_excerpt(
            source_note_id=str(source_note_id),
            excerpt_text="Accepted excerpt",
            template_type="cornell",
            handwriting_mode="accented",
        )
    )

    document = result["studio_document"]
    provenance = document["accepted_provenance_json"]
    assert (provenance["provider"], provenance["model"]) == (
        "tldw",
        "notes-studio-deterministic-v1",
    )
    assert set(document["payload_json"]["meta"]) == {"title", "source_note_id"}
    assert document["payload_json"]["layout"] == {
        "template_type": "cornell",
        "handwriting_mode": "accented",
        "render_version": 1,
    }


def test_real_generation_adapter_preserves_exact_contract_sections_and_llm_identity(
    studio_db,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None
    service = NotesStudioService(db=db, user_id="notes_studio_unit")
    raw_sections = [
        {
            "id": "questions-custom-α",
            "kind": "cue",
            "title": "  Révision 🌿  ",
            "items": ["  leading cue  ", "Cafe\u0301 and 漢字  "],
        },
        {
            "id": "summary-custom",
            "kind": "summary",
            "title": "  Non-default summary  ",
            "content": "  leading summary\ntrailing summary  ",
        },
        {
            "id": "notes-custom",
            "kind": "notes",
            "title": "Notes Δ",
            "content": "  leading notes\ntrailing notes  ",
        },
    ]
    expected_sections = StudioSectionsV1.model_validate(
        {"sections": raw_sections}
    ).model_dump(mode="json")["sections"]

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=json.dumps(
            {
                "meta": {"title": "Provider Study Notes"},
                "sections": raw_sections,
            },
            ensure_ascii=False,
        ),
    ):
        result = asyncio.run(
            service.derive_from_excerpt(
                source_note_id=str(source_note_id),
                excerpt_text="Accepted excerpt",
                template_type="lined",
                handwriting_mode="accented",
                provider="openai",
                model="gpt-test",
            )
        )

    document = result["studio_document"]
    assert document["payload_json"]["sections"] == expected_sections
    assert document["accepted_provenance_json"]["provider"] == "openai"
    assert document["accepted_provenance_json"]["model"] == "gpt-test"


def test_custom_generation_adapter_preserves_valid_contract_sections_before_provenance(
    studio_db,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None
    raw_sections = [
        {
            "id": "custom-cue",
            "kind": "cue",
            "title": "  Prompt  ",
            "items": ["  exact item  "],
        },
        {
            "id": "custom-summary",
            "kind": "summary",
            "title": "Summary",
            "content": "  exact summary  ",
        },
    ]

    async def adapter(_request, _context):
        return {
            "source": "llm",
            "payload": {
                "meta": {"title": "Custom Study Notes", "provider_only": "drop"},
                "layout": {"provider_only": True},
                "sections": raw_sections,
                "provider_only": {"trace": "drop"},
            },
        }

    result = asyncio.run(
        NotesStudioService(
            db=db,
            user_id="notes_studio_unit",
            generation_adapter=adapter,
        ).derive_from_excerpt(
            source_note_id=str(source_note_id),
            excerpt_text="Accepted excerpt",
            template_type="lined",
            handwriting_mode="accented",
            provider="openai",
            model="gpt-test",
        )
    )

    document = result["studio_document"]
    assert document["payload_json"]["sections"] == raw_sections
    assert set(document["payload_json"]) == {"meta", "layout", "sections"}
    assert set(document["payload_json"]["meta"]) == {"title", "source_note_id"}
    assert document["accepted_provenance_json"]["provider"] == "openai"
    assert document["accepted_provenance_json"]["model"] == "gpt-test"


def test_custom_llm_adapter_with_invalid_sections_is_rejected_before_provenance(
    studio_db,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None

    async def adapter(_request, _context):
        return {
            "source": "llm",
            "payload": {
                "sections": [
                    {
                        "id": "invalid",
                        "kind": "provider-kind",
                        "title": "Provider section",
                        "content": "Provider content",
                    }
                ]
            },
        }

    with pytest.raises(InputError, match="canonical sections"):
        asyncio.run(
            NotesStudioService(
                db=db,
                user_id="notes_studio_unit",
                generation_adapter=adapter,
            ).derive_from_excerpt(
                source_note_id=str(source_note_id),
                excerpt_text="Accepted excerpt",
                template_type="lined",
                handwriting_mode="accented",
                provider="openai",
                model="gpt-test",
            )
        )


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-test", ("tldw", "diagram-deterministic-v1")),
        (None, None, ("tldw", "diagram-deterministic-v1")),
        ("openai", None, ("tldw", "diagram-deterministic-v1")),
    ],
)
def test_diagram_stamps_actual_or_deterministic_execution_identity(
    studio_db,
    provider,
    model,
    expected,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None

    async def diagram_adapter(_request, _context):
        return {"diagram": "flowchart TD\nA --> B", "format": "mermaid"}

    service = _service(db, diagram_adapter=diagram_adapter)
    derived = _derive_note(
        service,
        source_note_id=source_note_id,
        excerpt_text="Accepted excerpt",
    )
    result = asyncio.run(
        service.update_diagram_manifest(
            note_id=derived["note"]["id"],
            diagram_type="flowchart",
            provider=provider,
            model=model,
        )
    )
    provenance = result["studio_document"]["accepted_provenance_json"]
    assert (provenance["provider"], provenance["model"]) == expected


@pytest.mark.parametrize(
    ("adapter_result", "expected", "raises"),
    [
        (
            {
                "diagram": "flowchart TD\nA --> B",
                "format": "mermaid",
                "source": "llm",
                "provider": "anthropic",
                "model": "claude-test",
            },
            ("anthropic", "claude-test"),
            None,
        ),
        (
            {
                "diagram": "flowchart TD\nA --> B",
                "format": "mermaid",
                "source": "deterministic_fallback",
                "provider": "tldw",
                "model": "diagram-deterministic-v1",
            },
            ("tldw", "diagram-deterministic-v1"),
            None,
        ),
        (
            {
                "diagram": "flowchart TD\nA --> B",
                "format": "mermaid",
                "source": "llm",
                "provider": "openai",
            },
            None,
            InputError,
        ),
    ],
)
def test_diagram_stamps_only_returned_execution_identity(
    studio_db,
    adapter_result,
    expected,
    raises,
):
    db = studio_db
    source_note_id = db.add_note(title="Source", content="Accepted excerpt")
    assert source_note_id is not None

    async def diagram_adapter(_request, _context):
        return adapter_result

    service = _service(db, diagram_adapter=diagram_adapter)
    derived = _derive_note(
        service,
        source_note_id=source_note_id,
        excerpt_text="Accepted excerpt",
    )
    call = service.update_diagram_manifest(
        note_id=derived["note"]["id"],
        diagram_type="flowchart",
        provider="openai",
        model="requested-but-not-executed",
    )
    if raises is not None:
        with pytest.raises(raises, match="execution identity"):
            asyncio.run(call)
        return
    result = asyncio.run(call)
    provenance = result["studio_document"]["accepted_provenance_json"]
    assert (provenance["provider"], provenance["model"]) == expected


def test_update_diagram_manifest_rejects_unknown_section_ids(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="History",
        content="The printing press accelerated the spread of written knowledge.",
    )
    assert source_note_id is not None

    service = _service(db)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="The printing press accelerated the spread of written knowledge.",
    )

    with pytest.raises(InputError, match="Unknown Studio section"):
        asyncio.run(
            service.update_diagram_manifest(
                note_id=result["note"]["id"],
                diagram_type="flowchart",
                source_section_ids=["missing-section"],
            )
        )


def test_update_diagram_manifest_rejects_concurrent_sidecar_change(studio_db):
    db = studio_db
    source_note_id = db.add_note(
        title="History",
        content="The printing press accelerated the spread of written knowledge.",
    )
    assert source_note_id is not None

    note_id_holder: dict[str, str] = {}

    async def _diagram_adapter(_request: dict[str, object], _context: dict[str, object]) -> dict[str, object]:
        note_id = note_id_holder["note_id"]
        current_document = db.get_note_studio_document(note_id)
        assert current_document is not None
        db.upsert_note_studio_document(
            note_id=note_id,
            payload_json=current_document["payload_json"],
            template_type=current_document["template_type"],
            handwriting_mode=current_document["handwriting_mode"],
            source_note_id=current_document.get("source_note_id"),
            excerpt_snapshot=current_document.get("excerpt_snapshot"),
            excerpt_hash=current_document.get("excerpt_hash"),
            diagram_manifest_json=None,
            companion_content_hash="sha256:concurrent",
            render_version=int(current_document["render_version"]),
            provenance_kind="manual",
        )
        return {"diagram": "graph TD; A-->B", "format": "mermaid"}

    service = _service(db, diagram_adapter=_diagram_adapter)
    result = _derive_note(
        service,
        source_note_id=str(source_note_id),
        excerpt_text="The printing press accelerated the spread of written knowledge.",
    )
    note_id_holder["note_id"] = result["note"]["id"]

    with pytest.raises(ConflictError, match="changed concurrently"):
        asyncio.run(
            service.update_diagram_manifest(
                note_id=result["note"]["id"],
                diagram_type="flowchart",
            )
        )

    persisted_document = db.get_note_studio_document(result["note"]["id"])
    assert persisted_document is not None
    assert persisted_document["companion_content_hash"] != "sha256:concurrent"
    assert persisted_document["diagram_manifest_json"] is None


@pytest.mark.parametrize(
    ("excerpt_text", "expected_message"),
    [
        ("   ", "excerpt_text cannot be empty."),
        ("Not present in source note", "excerpt_text must match content from the source note."),
    ],
)
def test_derive_rejects_invalid_excerpt_requests(studio_db, excerpt_text, expected_message):
    db = studio_db
    source_note_id = db.add_note(
        title="Source",
        content="Useful content for excerpt validation.",
    )
    assert source_note_id is not None

    service = _service(db)

    with pytest.raises(InputError, match=expected_message):
        _derive_note(
            service,
            source_note_id=str(source_note_id),
            excerpt_text=excerpt_text,
        )
