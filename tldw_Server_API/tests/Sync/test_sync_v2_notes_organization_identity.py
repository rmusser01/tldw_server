import uuid
from typing import cast

import pytest

from tldw_Server_API.app.core.Sync.v2.notes_organization import (
    NotesOrganizationValidationError,
    new_organization_sync_id,
    organization_link_id,
    parse_notes_organization_payload,
    validate_organization_object_id,
    validate_resource_sync_id,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncDomain

CANONICAL_SYNC_ID = "123e4567-e89b-42d3-a456-426614174000"


def test_organization_link_ids_use_canonical_domain_tagged_vectors() -> None:
    assert organization_link_id(
        "notes.keyword_link", ["note", "note-123", "kw-456"]
    ) == "notes.keyword_link:sha256:10f9eab3be80b6e439ce1bcf8fae952527bde7d7e026d0e227f0a87ada963be0"
    assert organization_link_id(
        "notes.keyword_collection_link", ["collection-123", "kw-456"]
    ) == "notes.keyword_collection_link:sha256:e9427c2d8bc4cfa8586130bc1fcc54cf432ca6dbb3df77bab3e65033b6148199"
    assert organization_link_id(
        "notes.folder_link", ["note-123", "folder-456"]
    ) == "notes.folder_link:sha256:9076b60d9d8476f852736928ef3661cb06d9ba55696dd4504657c753f414b670"


@pytest.mark.parametrize(
    ("domain", "members"),
    [
        ("notes.keyword_link", ["note", "note-123"]),
        ("notes.keyword_collection_link", ["collection-123"]),
        ("notes.folder_link", ["note-123", "folder-456", "extra"]),
        ("notes.keyword", ["not", "a", "relationship"]),
    ],
)
def test_organization_link_id_rejects_invalid_domain_or_member_count(
    domain: str,
    members: list[str],
) -> None:
    with pytest.raises(ValueError):
        organization_link_id(cast(SyncDomain, domain), members)


def test_organization_resource_sync_ids_are_canonical_uuid4() -> None:
    sync_id = new_organization_sync_id()

    assert sync_id == str(uuid.UUID(sync_id))
    assert uuid.UUID(sync_id).version == 4
    assert validate_resource_sync_id(sync_id) == sync_id

    with pytest.raises(ValueError):
        validate_resource_sync_id(sync_id.upper())
    with pytest.raises(ValueError):
        validate_resource_sync_id("550e8400-e29b-11d4-a716-446655440000")


@pytest.mark.parametrize(
    "object_id",
    [
        "notes.keyword_link:sha256:5C135A053523A5C90CF764D8263FBF13014FAAC6713E9F756B9AC64AE44560EF",
        "notes.keyword_link:sha256:5c135a053523a5c90cf764d8263fbf13014faac6713e9f756b9ac64ae44560e",
    ],
)
def test_organization_object_id_rejects_noncanonical_link_digest(object_id: str) -> None:
    with pytest.raises(ValueError):
        validate_organization_object_id(
            "notes.keyword_link",
            object_id,
            {"subject_type": "note", "subject_id": "note-123", "keyword_sync_id": "kw-456"},
        )


def test_organization_object_id_rejects_link_payload_that_does_not_reproduce_identity() -> None:
    object_id = organization_link_id("notes.keyword_link", ["note", "note-123", "kw-456"])

    with pytest.raises(ValueError):
        validate_organization_object_id(
            "notes.keyword_link",
            object_id,
            {"subject_type": "note", "subject_id": "note-123", "keyword_sync_id": "kw-789"},
        )


def test_notes_organization_payloads_are_strict_and_normalized() -> None:
    assert parse_notes_organization_payload(
        "notes.keyword", "upsert", {"keyword": "  research  "}
    ) == {"keyword": "research"}
    assert parse_notes_organization_payload("notes.keyword", "tombstone", {}) == {}
    assert parse_notes_organization_payload(
        "notes.folder_link",
        "tombstone",
        {"note_id": CANONICAL_SYNC_ID, "folder_sync_id": CANONICAL_SYNC_ID},
    ) == {"note_id": CANONICAL_SYNC_ID, "folder_sync_id": CANONICAL_SYNC_ID}

    with pytest.raises(ValueError):
        parse_notes_organization_payload("notes.keyword", "upsert", {"keyword": "research", "extra": True})
    with pytest.raises(ValueError):
        parse_notes_organization_payload("notes.keyword", "tombstone", {"keyword": "research"})
    with pytest.raises(ValueError):
        parse_notes_organization_payload("notes.folder_link", "tombstone", {})


@pytest.mark.parametrize(
    ("domain", "payload"),
    [
        (
            "notes.keyword_link",
            {"subject_type": "note", "subject_id": "1", "keyword_sync_id": CANONICAL_SYNC_ID},
        ),
        (
            "notes.keyword_link",
            {"subject_type": "conversation", "subject_id": "conversation-1", "keyword_sync_id": "1"},
        ),
        ("notes.keyword_collection", {"name": "Research", "parent_sync_id": "1"}),
        (
            "notes.keyword_collection_link",
            {"collection_sync_id": "1", "keyword_sync_id": CANONICAL_SYNC_ID},
        ),
        (
            "notes.keyword_collection_link",
            {"collection_sync_id": CANONICAL_SYNC_ID, "keyword_sync_id": "1"},
        ),
        ("notes.folder", {"name": "Research", "parent_sync_id": "1"}),
        ("notes.folder_link", {"note_id": "1", "folder_sync_id": CANONICAL_SYNC_ID}),
        ("notes.folder_link", {"note_id": CANONICAL_SYNC_ID, "folder_sync_id": "1"}),
    ],
)
def test_notes_organization_payload_rejects_noncanonical_resource_references(
    domain: SyncDomain,
    payload: dict[str, str],
) -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        parse_notes_organization_payload(domain, "upsert", payload)

    assert exc_info.value.error_code == "notes_organization_resource_sync_id_invalid"


def test_notes_organization_validation_errors_expose_stable_error_codes() -> None:
    with pytest.raises(NotesOrganizationValidationError) as exc_info:
        parse_notes_organization_payload("notes.keyword", "upsert", {"keyword": "research", "extra": True})

    assert exc_info.value.error_code == "notes_organization_payload_invalid"
