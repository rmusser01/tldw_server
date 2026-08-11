from __future__ import annotations

import math

import pytest

from tldw_Server_API.app.core.Sync.v2.notes_link import (
    NOTES_LINK_LABEL_MAX_CHARS,
    NOTES_LINK_PROPERTIES_MAX_BYTES,
    NOTES_LINK_PROPERTIES_MAX_DEPTH,
    NOTES_LINK_PROPERTIES_MAX_KEYS,
    NOTES_LINK_REASON_MAX_CHARS,
    NOTES_LINK_WEIGHT_MAX,
    NotesLinkValidationError,
    parse_notes_link_payload,
    validate_notes_link_object_id,
    validate_notes_link_provenance,
)

EDGE_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
CREATED_AT = "2026-08-10T12:00:00+00:00"
UPDATED_AT = "2026-08-10T13:00:00+00:00"
DEVICE_ID = "device-notes-link-1"


def _upsert_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_note_id": SOURCE_ID,
        "target_note_id": TARGET_ID,
        "type": "manual",
        "directed": False,
        "weight": 1.0,
        "label": "Related",
        "properties": {"context": "research"},
        "created_at": CREATED_AT,
        "last_modified": CREATED_AT,
        "created_by": DEVICE_ID,
    }
    payload.update(overrides)
    return payload


def test_notes_link_upsert_normalizes_one_strict_canonical_payload() -> None:
    parsed = parse_notes_link_payload("upsert", _upsert_payload())

    assert parsed == _upsert_payload()
    assert validate_notes_link_object_id(EDGE_ID) == EDGE_ID
    validate_notes_link_provenance(
        parsed,
        envelope_created_at_client=CREATED_AT,
        authenticated_device_id=DEVICE_ID,
        prior_payload=None,
    )


def test_notes_link_update_copies_creation_provenance_and_uses_envelope_timestamp() -> None:
    prior = parse_notes_link_payload("upsert", _upsert_payload())
    updated = parse_notes_link_payload(
        "upsert",
        _upsert_payload(label="Updated", last_modified=UPDATED_AT),
    )

    validate_notes_link_provenance(
        updated,
        envelope_created_at_client=UPDATED_AT,
        authenticated_device_id="device-notes-link-2",
        prior_payload=prior,
    )


def test_notes_link_tombstone_keeps_snapshot_and_bounded_reason() -> None:
    tombstone = parse_notes_link_payload(
        "tombstone",
        _upsert_payload(
            last_modified=UPDATED_AT,
            deleted_at=UPDATED_AT,
            reason="manual unlink",
        ),
    )

    assert tombstone["deleted_at"] == UPDATED_AT
    assert tombstone["reason"] == "manual unlink"
    validate_notes_link_provenance(
        tombstone,
        envelope_created_at_client=UPDATED_AT,
        authenticated_device_id="device-notes-link-2",
        prior_payload=_upsert_payload(),
    )


@pytest.mark.parametrize(
    "object_id",
    [
        "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        "aaaaaaaa-aaaa-1aaa-8aaa-aaaaaaaaaaaa",
        "not-a-uuid",
    ],
)
def test_notes_link_object_id_requires_canonical_uuid4(object_id: str) -> None:
    with pytest.raises(NotesLinkValidationError, match="canonical UUIDv4"):
        validate_notes_link_object_id(object_id)


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_note_id": SOURCE_ID},
        {"source_note_id": TARGET_ID, "target_note_id": SOURCE_ID},
        {"type": "wikilink"},
        {"weight": -1},
        {"weight": NOTES_LINK_WEIGHT_MAX + 1},
        {"weight": math.inf},
        {"directed": 1},
        {"label": "x" * (NOTES_LINK_LABEL_MAX_CHARS + 1)},
        {"extra": "forbidden"},
    ],
)
def test_notes_link_rejects_invalid_identity_and_scalar_bounds(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(NotesLinkValidationError):
        parse_notes_link_payload("upsert", _upsert_payload(**overrides))


def test_notes_link_directed_payload_preserves_endpoint_order() -> None:
    parsed = parse_notes_link_payload(
        "upsert",
        _upsert_payload(
            source_note_id=TARGET_ID,
            target_note_id=SOURCE_ID,
            directed=True,
        ),
    )

    assert parsed["source_note_id"] == TARGET_ID
    assert parsed["target_note_id"] == SOURCE_ID


@pytest.mark.parametrize(
    "properties",
    [
        {str(index): index for index in range(NOTES_LINK_PROPERTIES_MAX_KEYS + 1)},
        {"a": {"b": {"c": {"d": {"e": True}}}}},
        {"payload": "x" * NOTES_LINK_PROPERTIES_MAX_BYTES},
        {"bad": math.nan},
        {"nested": {1: "non-string key"}},
        ["not", "an", "object"],
    ],
)
def test_notes_link_rejects_noncanonical_or_oversized_properties(properties: object) -> None:
    with pytest.raises(NotesLinkValidationError):
        parse_notes_link_payload("upsert", _upsert_payload(properties=properties))


def test_notes_link_accepts_maximum_properties_depth() -> None:
    properties = {"a": {"b": {"c": {"d": True}}}}

    parsed = parse_notes_link_payload("upsert", _upsert_payload(properties=properties))

    assert parsed["properties"] == properties
    assert NOTES_LINK_PROPERTIES_MAX_DEPTH == 4


def test_notes_link_rejects_oversized_tombstone_reason() -> None:
    with pytest.raises(NotesLinkValidationError):
        parse_notes_link_payload(
            "tombstone",
            _upsert_payload(
                last_modified=UPDATED_AT,
                deleted_at=UPDATED_AT,
                reason="x" * (NOTES_LINK_REASON_MAX_CHARS + 1),
            ),
        )


def test_notes_link_provenance_rejects_post_submit_enrichment_or_rewrite() -> None:
    parsed = parse_notes_link_payload("upsert", _upsert_payload())

    with pytest.raises(NotesLinkValidationError, match="created_at"):
        validate_notes_link_provenance(
            parsed,
            envelope_created_at_client=UPDATED_AT,
            authenticated_device_id=DEVICE_ID,
            prior_payload=None,
        )

    with pytest.raises(NotesLinkValidationError, match="created_by"):
        validate_notes_link_provenance(
            parsed,
            envelope_created_at_client=CREATED_AT,
            authenticated_device_id="different-device",
            prior_payload=None,
        )


def test_notes_link_trusted_bootstrap_preserves_legacy_creation_provenance() -> None:
    parsed = parse_notes_link_payload(
        "upsert",
        _upsert_payload(created_by="user:legacy", created_at="2025-01-01T00:00:00+00:00"),
    )

    validate_notes_link_provenance(
        parsed,
        envelope_created_at_client=CREATED_AT,
        authenticated_device_id="server-origin",
        prior_payload=None,
        trusted_bootstrap=True,
    )
