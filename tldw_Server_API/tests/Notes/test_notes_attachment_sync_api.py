"""Boundary contracts for canonical Notes attachment Sync APIs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.notes_attachments import (
    NotesAttachmentFromUploadRequest,
    NotesAttachmentItem,
    NotesAttachmentMutationResponse,
    NotesAttachmentPage,
    NotesAttachmentReasonRequest,
    NotesAttachmentRenameRequest,
    format_notes_attachment_etag,
    parse_notes_attachment_if_match,
    validate_notes_attachment_idempotency_key,
    validate_notes_attachment_keyset_cursor,
)

ATTACHMENT_ID = "2c4cb609-c4db-44f9-8e35-f078bd36d6b2"
NOTE_ID = "a1677eb1-1f41-4c86-a8dd-1eaa14b014e2"
OBJECT_HASH = "sha256:" + "a" * 64
BLOB_HASH = "sha256:" + "b" * 64


def _item_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": "dataset-1",
        "note_id": NOTE_ID,
        "attachment_id": ATTACHMENT_ID,
        "file_name": "report.pdf",
        "original_file_name": "report.pdf",
        "content_type": "application/pdf",
        "size_bytes": 42,
        "blob_hash": BLOB_HASH,
        "version": 3,
        "object_hash": OBJECT_HASH,
        "state": "live",
        "deleted_at": None,
        "delete_reason": None,
        "created_at": "2026-08-11T12:00:00.000000Z",
        "last_modified": "2026-08-11T12:01:00.000000Z",
        "created_by": "device-1",
        "source_kind": "upload",
        "availability": "available",
        "etag": format_notes_attachment_etag(ATTACHMENT_ID, 3, OBJECT_HASH),
    }
    payload.update(overrides)
    return payload


def test_notes_attachment_schema_accepts_canonical_item_page_and_mutation_response():
    item = NotesAttachmentItem.model_validate(_item_payload())
    page = NotesAttachmentPage.model_validate(
        {"items": [item.model_dump()], "next_cursor": "cursor-1", "has_more": True}
    )
    mutation = NotesAttachmentMutationResponse.model_validate(
        {**item.model_dump(), "idempotent_replay": False}
    )

    assert item.etag == f'"att-{ATTACHMENT_ID}-v3-{OBJECT_HASH.removeprefix("sha256:")}"'
    assert page.items == [item]
    assert mutation.idempotent_replay is False


@pytest.mark.parametrize(
    ("payload", "model"),
    [
        ({"file_name": "renamed.pdf", "content_type": "application/pdf"}, NotesAttachmentRenameRequest),
        ({"upload_id": "upload-1", "attachment_id": ATTACHMENT_ID}, NotesAttachmentFromUploadRequest),
        ({"reason": "ok", "extra": True}, NotesAttachmentReasonRequest),
        ({**_item_payload(), "storage_key": "secret"}, NotesAttachmentItem),
    ],
)
def test_notes_attachment_schema_rejects_extra_fields(payload: dict[str, object], model: type):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_notes_attachment_schema_enforces_rename_only_and_reason_bounds():
    rename = NotesAttachmentRenameRequest.model_validate({"file_name": " Quarterly Report.PDF "})
    assert rename.file_name == "Quarterly_Report.pdf"

    with pytest.raises(ValidationError):
        NotesAttachmentReasonRequest.model_validate({"reason": "x" * 257})

    with pytest.raises(ValidationError):
        NotesAttachmentRenameRequest.model_validate({"file_name": "payload.exe"})


def test_notes_attachment_schema_validates_from_upload_identifier():
    request = NotesAttachmentFromUploadRequest.model_validate({"upload_id": "upload-1"})
    assert request.upload_id == "upload-1"

    for upload_id in ("", "x" * 129, "bad\nvalue", "café"):
        with pytest.raises(ValidationError):
            NotesAttachmentFromUploadRequest.model_validate({"upload_id": upload_id})


def test_notes_attachment_etag_grammar_is_exact():
    etag = format_notes_attachment_etag(ATTACHMENT_ID, 3, OBJECT_HASH)
    parsed = parse_notes_attachment_if_match(etag)
    assert parsed == (ATTACHMENT_ID, 3, OBJECT_HASH)

    for value in (
        None,
        "*",
        f'W/{etag}',
        f"{etag}, {etag}",
        f'"att-{ATTACHMENT_ID}-v0-{OBJECT_HASH.removeprefix("sha256:")}"',
        f'"att-{ATTACHMENT_ID.upper()}-v3-{OBJECT_HASH.removeprefix("sha256:")}"',
        f'"att-{ATTACHMENT_ID}-v3-{OBJECT_HASH.removeprefix("sha256:").upper()}"',
    ):
        with pytest.raises(ValueError):
            parse_notes_attachment_if_match(value)


def test_notes_attachment_idempotency_key_and_keyset_cursor_boundaries():
    assert validate_notes_attachment_idempotency_key("request-1") == "request-1"
    assert validate_notes_attachment_keyset_cursor("opaque.cursor") == "opaque.cursor"

    for value in ("", "x" * 129, "bad\nkey", "café"):
        with pytest.raises(ValueError):
            validate_notes_attachment_idempotency_key(value)
    for value in ("", "x" * 513, "bad\ncursor", "café"):
        with pytest.raises(ValueError):
            validate_notes_attachment_keyset_cursor(value)


def test_notes_attachment_item_rejects_inconsistent_lifecycle_and_etag():
    with pytest.raises(ValidationError):
        NotesAttachmentItem.model_validate(_item_payload(state="tombstoned"))
    with pytest.raises(ValidationError):
        NotesAttachmentItem.model_validate(_item_payload(etag=f'"att-{ATTACHMENT_ID}-v2-{OBJECT_HASH}"'))


@pytest.mark.parametrize(
    "overrides",
    [
        {"size_bytes": "42"},
        {"size_bytes": True},
        {"version": "3"},
        {"dataset_id": " dataset-1"},
        {"created_by": " device-1"},
    ],
)
def test_notes_attachment_item_rejects_coerced_or_unnormalized_authority_fields(overrides):
    with pytest.raises(ValidationError):
        NotesAttachmentItem.model_validate(_item_payload(**overrides))
