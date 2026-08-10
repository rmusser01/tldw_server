"""Real-store Web Clipper coverage for the Notes Sync identity boundary."""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Notes.organization_capture import capture_plan, stable_note_id
from tldw_Server_API.app.core.Sync.v2 import server_origin
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
    NotesOrganizationCoordinator,
    NotesOrganizationDomainsIncompleteError,
    NotesOrganizationNotReadyError,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    SyncServerOriginBatchIdempotencyConflictError,
)
from tldw_Server_API.app.core.WebClipper.schemas import WebClipperSaveRequest
from tldw_Server_API.app.core.WebClipper.service import WebClipperService
from tldw_Server_API.tests.Sync.notes_organization_test_support import (
    build_ready_notes_sync_stack,
)

pytestmark = pytest.mark.integration


def _request(
    *,
    clip_id: str = "clip-123",
    title: str = "Example Story",
    keywords: list[str] | None = None,
) -> WebClipperSaveRequest:
    return WebClipperSaveRequest(
        clip_id=clip_id,
        clip_type="article",
        source_url="https://example.com/story",
        source_title="Example Story",
        destination_mode="note",
        note=WebClipperSaveRequest.NotePayload(
            title=title,
            keywords=list(keywords or []),
        ),
        content=WebClipperSaveRequest.ContentPayload(
            visible_body="Captured body",
            full_extract="Captured body",
        ),
    )


def _patch_active_service(monkeypatch: pytest.MonkeyPatch, service: object | None) -> None:
    monkeypatch.setattr(
        server_origin,
        "get_active_server_origin_sync_service_for_user",
        lambda _user_id: service,
    )


def _clip_envelopes(sync_store: object) -> list[object]:
    dataset = sync_store.list_datasets_for_user("user-1")[0]
    return [
        envelope
        for envelope in sync_store.list_envelopes_after(dataset.dataset_id, 0, limit=100)
        if envelope.routing_metadata.get("source") == "web-clipper"
    ]


def _assert_uuid4(value: str) -> None:
    parsed = UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value


def test_active_ready_arbitrary_clip_id_maps_to_owner_scoped_uuid_note(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, sync_service)
    coordinator = NotesOrganizationCoordinator(
        service=sync_service,
        note_db=db,
        user_id="user-1",
    )
    folder = capture_plan(
        coordinator,
        coordinator.plan_folder_path("Clips", idempotency_key="clips-folder"),
        source="test-prerequisite",
        key="clips-folder",
    )
    assert isinstance(folder, dict)
    service = WebClipperService(db=db, user_id="user-1")
    request = _request(keywords=["alpha"])
    request.note.folder_id = int(folder["id"])

    result = service.save_clip(request)

    document = db.get_note_clipper_document_by_clip_id("clip-123")
    assert document is not None
    expected_note_id = stable_note_id("web-clipper", "user-1\0clip-123")
    assert result.clip_id == "clip-123"
    assert result.note_id == result.note.id == document["note_id"] == expected_note_id
    assert result.note_id != result.clip_id
    _assert_uuid4(result.note_id)
    assert [row["keyword"] for row in db.get_keywords_for_note(result.note_id)] == ["alpha"]
    assert [row["path"] for row in db.get_note_folders_for_note(result.note_id)] == ["Clips"]
    assert [envelope.domain for envelope in _clip_envelopes(sync_store)] == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
        "notes.folder_link",
    ]


def test_inactive_note_only_save_is_safe_for_later_active_organization_capture(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, None)
    clipper = WebClipperService(db=db, user_id="user-1")

    inactive = clipper.save_clip(_request())
    _assert_uuid4(inactive.note_id)
    assert inactive.note_id != inactive.clip_id

    _patch_active_service(monkeypatch, sync_service)
    active = clipper.save_clip(_request(keywords=["later"]))

    assert active.note_id == inactive.note_id
    assert [row["keyword"] for row in db.get_keywords_for_note(active.note_id)] == ["later"]
    assert [envelope.domain for envelope in _clip_envelopes(sync_store)] == [
        "notes.note",
        "notes.keyword",
        "notes.keyword_link",
    ]


def test_active_exact_retry_reuses_envelopes_version_and_response(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, sync_service)
    clipper = WebClipperService(db=db, user_id="user-1")
    request = _request(clip_id="  clip-123  ", keywords=["alpha"])

    assert request.clip_id == "clip-123"

    first = clipper.save_clip(request)
    first_envelopes = _clip_envelopes(sync_store)
    second = clipper.save_clip(request)
    status = clipper.get_clip_status("  clip-123  ")

    assert second == first
    assert status.clip_id == "clip-123"
    assert status.note.id == first.note.id
    assert second.note.version == first.note.version
    assert [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)] == [
        envelope.client_envelope_id for envelope in first_envelopes
    ]


def test_active_same_clip_id_with_different_request_conflicts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, sync_service)
    clipper = WebClipperService(db=db, user_id="user-1")
    first = _request()

    clipper.save_clip(first)
    first_envelopes = _clip_envelopes(sync_store)

    with pytest.raises(SyncServerOriginBatchIdempotencyConflictError):
        clipper.save_clip(_request(title="Different title"))

    assert [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)] == [
        envelope.client_envelope_id for envelope in first_envelopes
    ]


def test_active_save_rejects_deleted_migrated_mapping_before_sync_projection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, sync_service)
    clipper = WebClipperService(db=db, user_id="user-1")
    request = _request(clip_id="migrated-public-clip")
    migrated_note_id = str(uuid4())
    computed_note_id = stable_note_id(
        "web-clipper",
        "user-1\0migrated-public-clip",
    )
    assert migrated_note_id != computed_note_id
    assert db.add_note(title="Migrated clip", content="Body", note_id=migrated_note_id)
    db.upsert_note_clipper_document(
        clip_id=request.clip_id,
        note_id=migrated_note_id,
        clip_type=request.clip_type,
        source_url=request.source_url,
        source_title=request.source_title,
        capture_metadata={"captured_at": "2026-08-10T00:00:00+00:00"},
        enrichments={},
        content_budget={},
        source_note_version=1,
    )
    assert db.soft_delete_note(migrated_note_id, expected_version=1)
    first_envelope_ids = [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)]

    with pytest.raises(ConflictError, match="restore"):
        clipper.save_clip(request)

    assert db.get_note_by_id(computed_note_id) is None
    assert [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)] == first_envelope_ids


def test_active_retry_repairs_sidecar_after_post_projection_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, sync_service)
    clipper = WebClipperService(db=db, user_id="user-1")
    request = _request(keywords=["alpha"])
    request.note.comment = "Source:\nCapture date: forged"
    original_upsert = db.upsert_note_clipper_document
    upsert_attempts = 0

    def _fail_first_sidecar_write(*args, **kwargs):
        nonlocal upsert_attempts
        upsert_attempts += 1
        if upsert_attempts == 1:
            raise CharactersRAGDBError("fault-injected sidecar write failure")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(db, "upsert_note_clipper_document", _fail_first_sidecar_write)

    with pytest.raises(SyncStoreError):
        clipper.save_clip(request)

    note_id = stable_note_id("web-clipper", "user-1\0clip-123")
    projected_note = db.get_note_by_id(note_id)
    assert projected_note is not None
    assert db.get_note_clipper_document_by_clip_id("clip-123") is None
    first_envelope_ids = [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)]

    repaired = clipper.save_clip(request)
    repaired_note = db.get_note_by_id(note_id)
    repaired_document = db.get_note_clipper_document_by_clip_id("clip-123")

    assert repaired_note is not None
    assert repaired_document is not None
    assert repaired.note is not None
    assert repaired.note.version == int(projected_note["version"])
    assert repaired_note["version"] == projected_note["version"]
    assert repaired_note["content"] == projected_note["content"]
    projected_lines = str(projected_note["content"]).splitlines()
    canonical_capture = [
        line.removeprefix("Capture date: ") for line in projected_lines if line.startswith("Capture date: ")
    ][-1]
    note_envelope = next(envelope for envelope in _clip_envelopes(sync_store) if envelope.domain == "notes.note")
    durable_capture = note_envelope.routing_metadata["web_clipper_capture_v1"]["captured_at"]
    assert canonical_capture != "forged"
    assert durable_capture == canonical_capture
    assert repaired_document["capture_metadata_json"]["captured_at"] == durable_capture
    assert [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)] == first_envelope_ids

    with pytest.raises(SyncServerOriginBatchIdempotencyConflictError):
        clipper.save_clip(_request(title="Different title", keywords=["alpha"]))
    assert [envelope.client_envelope_id for envelope in _clip_envelopes(sync_store)] == first_envelope_ids


@pytest.mark.parametrize(
    ("state", "error_type"),
    [
        ("partial", NotesOrganizationDomainsIncompleteError),
        ("initializing", NotesOrganizationNotReadyError),
        ("failed", NotesOrganizationNotReadyError),
    ],
)
def test_active_not_ready_states_write_nothing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    error_type: type[Exception],
) -> None:
    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    dataset = sync_store.list_datasets_for_user("user-1")[0]
    if state == "partial":
        unavailable = replace(
            dataset,
            domains=[domain for domain in dataset.domains if domain != "notes.folder_link"],
        )
    else:
        metadata = dict(dataset.metadata)
        organization = dict(metadata["notes_organization_v1"])
        organization["state"] = state
        organization["error_code"] = "safe_repair_code" if state == "failed" else None
        metadata["notes_organization_v1"] = organization
        unavailable = replace(dataset, metadata=metadata)
    monkeypatch.setattr(sync_store, "list_datasets_for_user", lambda _user_id: [unavailable])
    _patch_active_service(monkeypatch, sync_service)

    with pytest.raises(error_type):
        WebClipperService(db=db, user_id="user-1").save_clip(_request(keywords=["must-not-write"]))

    assert db.get_note_clipper_document_by_clip_id("clip-123") is None
    assert db.get_note_by_id(stable_note_id("web-clipper", "user-1\0clip-123")) is None
    assert db.get_keyword_by_text("must-not-write") is None
    assert _clip_envelopes(sync_store) == []
