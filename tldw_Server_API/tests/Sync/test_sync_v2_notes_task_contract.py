from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Sync.v2.notes_task_contract import (
    NotesTaskActivityTombstoneV1,
    NotesTaskContractError,
    NotesTaskV1Payload,
    canonical_json_bytes,
    convert_legacy_task_event,
    notes_task_activity_object_hash,
    notes_task_object_hash,
    parse_notes_task_activity_tombstone_v1,
    parse_notes_task_activity_v1,
    parse_notes_task_tombstone_v1,
    parse_notes_task_v1,
)

OWNER_ID = "owner-user-1"
TASK_ID = "11111111-1111-4111-8111-111111111111"
NOTE_ID = "22222222-2222-4222-8222-222222222222"
ACTIVITY_ID = "33333333-3333-4333-8333-333333333333"
DEVICE_ID = "44444444-4444-4444-8444-444444444444"
CORRECTED_ID = "55555555-5555-4555-8555-555555555555"
OCCURRED_AT = "2026-08-13T10:00:00+00:00"


def valid_task_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task_id": TASK_ID,
        "note_id": NOTE_ID,
        "title": "Prepare launch notes",
        "description": "Confirm the final release checklist.\nOwner approved.",
        "status": "open",
        "completed_at": None,
        "priority": "high",
        "due_date": "2026-08-31",
        "estimate": "90m",
        "recurrence": {
            "frequency": "weekly",
            "interval": 2,
            "by_weekday": ["mo", "we", "fr"],
            "until": "2026-12-31",
            "state": "active",
            "occurrence_index": 7,
        },
        "assignee_id": OWNER_ID,
        "tags": ["Zulu", "alpha", "Résumé"],
        "custom": {"board.column": "Next", "score": 1, "nested": {"ok": True}},
    }
    payload.update(overrides)
    return payload


def canonical_task_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "description": None,
        "priority": None,
        "due_date": None,
        "estimate": None,
        "recurrence": None,
        "assignee_id": None,
        "tags": [],
        "custom": {},
    }
    metadata.update(overrides)
    return metadata


def valid_activity_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "activity_id": ACTIVITY_ID,
        "note_id": NOTE_ID,
        "task_id": TASK_ID,
        "event_type": "updated",
        "actor_type": "user",
        "actor_id": OWNER_ID,
        "source_device_id": DEVICE_ID,
        "client_occurred_at": OCCURRED_AT,
        "source_kind": "client",
        "corrects_activity_id": None,
        "old_value": {"metadata": canonical_task_metadata(priority="low")},
        "new_value": {"metadata": canonical_task_metadata(priority="high")},
        "metadata": {"mutation_group_step": 2},
    }
    payload.update(overrides)
    return payload


def parse_activity(payload: dict[str, object]):
    return parse_notes_task_activity_v1(
        payload,
        owner_user_id=OWNER_ID,
        bound_actor_type=str(payload["actor_type"]),
        bound_actor_id=payload["actor_id"],
        authenticated_device_id=DEVICE_ID,
        trusted_server_origin=False,
    )


def legacy_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "id": ACTIVITY_ID,
        "task_id": TASK_ID,
        "note_id": NOTE_ID,
        "event_type": "status_changed",
        "actor_type": "user",
        "actor_id": OWNER_ID,
        "tool_name": None,
        "policy_mode": None,
        "approval_id": None,
        "old_value": {"status": "open"},
        "new_value": {"status": "done"},
        "created_at": "2026-08-13T10:00:00Z",
        "client_id": "legacy-client",
    }
    event.update(overrides)
    return event


def convert_legacy(event: dict[str, object]):
    return convert_legacy_task_event(
        event,
        owner_user_id=OWNER_ID,
        resolved_task_note_id=NOTE_ID,
    )


def test_notes_task_v1_exact_valid_vector_is_typed_sorted_and_frozen() -> None:
    parsed = parse_notes_task_v1(valid_task_payload(), owner_user_id=OWNER_ID)

    assert isinstance(parsed, NotesTaskV1Payload)
    assert parsed.model_dump(mode="json") == {
        **valid_task_payload(),
        "tags": ["alpha", "Résumé", "Zulu"],
    }
    with pytest.raises(ValidationError):
        parsed.title = "changed"  # type: ignore[misc]


def test_notes_task_tombstone_v1_is_the_same_complete_payload_contract() -> None:
    parsed = parse_notes_task_tombstone_v1(
        valid_task_payload(), owner_user_id=OWNER_ID
    )

    assert isinstance(parsed, NotesTaskV1Payload)
    assert parsed == parse_notes_task_v1(valid_task_payload(), owner_user_id=OWNER_ID)


@pytest.mark.parametrize(
    "field",
    [
        "description",
        "completed_at",
        "priority",
        "due_date",
        "estimate",
        "recurrence",
        "assignee_id",
    ],
)
def test_notes_task_v1_requires_every_nullable_field(field: str) -> None:
    payload = valid_task_payload()
    payload.pop(field)

    with pytest.raises(NotesTaskContractError):
        parse_notes_task_v1(payload, owner_user_id=OWNER_ID)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"}, "UUIDv4"),
        ({"note_id": "00000000-0000-1000-8000-000000000000"}, "UUIDv4"),
        ({"title": ""}, "title"),
        ({"title": " leading"}, "stripped"),
        ({"title": "line\nbreak"}, "control"),
        ({"title": "x" * 2_001}, "title"),
        ({"description": "x" * 16_001}, "description"),
        ({"description": "bad\x00control"}, "control"),
        ({"status": "open", "completed_at": OCCURRED_AT}, "completion"),
        ({"status": "done", "completed_at": None}, "completion"),
        ({"status": "done", "completed_at": "2026-08-13T10:00:00Z"}, "canonical"),
        ({"due_date": "2026-02-29"}, "date"),
        ({"due_date": "2026-8-3"}, "date"),
        ({"estimate": "1.5h"}, "estimate"),
        ({"estimate": "1000000m"}, "estimate"),
        ({"assignee_id": "another-owner"}, "assignee"),
        ({"tags": ["Tag", "tag"]}, "casefold"),
        ({"tags": ["e\u0301"]}, "NFKC"),
        ({"tags": [" trailing "]}, "trimmed"),
        ({"tags": [str(index) for index in range(33)]}, "32"),
        ({"custom": {"title": "shadow"}}, "reserved"),
        ({"custom": {"unsafe key": True}}, "safe"),
        ({"custom": {f"k{index}": index for index in range(33)}}, "32"),
        ({"custom": {"a": {"b": {"c": {"d": {"e": 1}}}}}}, "depth"),
        ({"custom": {"large": "x" * (16 * 1_024)}}, "16 KiB"),
        ({"custom": {"bad": {1, 2}}}, "JSON"),
    ],
)
def test_notes_task_v1_rejects_each_payload_boundary(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(NotesTaskContractError, match=message):
        parse_notes_task_v1(valid_task_payload(**overrides), owner_user_id=OWNER_ID)


@pytest.mark.parametrize(
    "value",
    [1.0, 1e-7, -0.0, 9_007_199_254_740_992, -9_007_199_254_740_992],
)
def test_arbitrary_json_rejects_cross_runtime_numeric_values(value: object) -> None:
    with pytest.raises(NotesTaskContractError):
        canonical_json_bytes({"value": value})
    with pytest.raises(NotesTaskContractError):
        parse_notes_task_v1(
            valid_task_payload(custom={"value": value}),
            owner_user_id=OWNER_ID,
        )
    with pytest.raises(NotesTaskContractError):
        parse_activity(valid_activity_payload(metadata={"value": value}))


def test_arbitrary_json_accepts_js_safe_integer_endpoints() -> None:
    minimum = -9_007_199_254_740_991
    maximum = 9_007_199_254_740_991

    assert canonical_json_bytes({"minimum": minimum, "maximum": maximum}) == (
        b'{"maximum":9007199254740991,"minimum":-9007199254740991}'
    )
    assert parse_notes_task_v1(
        valid_task_payload(custom={"minimum": minimum, "maximum": maximum}),
        owner_user_id=OWNER_ID,
    ).custom == {"minimum": minimum, "maximum": maximum}
    assert parse_activity(
        valid_activity_payload(metadata={"minimum": minimum, "maximum": maximum})
    ).metadata == {"minimum": minimum, "maximum": maximum}


@pytest.mark.parametrize(
    "recurrence",
    [
        {
            "frequency": "weekly",
            "interval": 0,
            "by_weekday": [],
            "until": None,
            "state": "active",
            "occurrence_index": 0,
        },
        {
            "frequency": "daily",
            "interval": 1,
            "by_weekday": ["mo"],
            "until": None,
            "state": "active",
            "occurrence_index": 0,
        },
        {
            "frequency": "weekly",
            "interval": 1,
            "by_weekday": ["we", "mo"],
            "until": None,
            "state": "active",
            "occurrence_index": 0,
        },
        {
            "frequency": "weekly",
            "interval": 1,
            "by_weekday": ["mo", "mo"],
            "until": None,
            "state": "active",
            "occurrence_index": 0,
        },
        {
            "frequency": "weekly",
            "interval": 1,
            "by_weekday": [],
            "until": "2026-02-29",
            "state": "active",
            "occurrence_index": 0,
        },
        {
            "frequency": "weekly",
            "interval": 1,
            "by_weekday": [],
            "until": None,
            "state": "active",
            "occurrence_index": 0,
            "timezone": "UTC",
        },
    ],
)
def test_notes_task_v1_rejects_invalid_recurrence_combinations(
    recurrence: dict[str, object],
) -> None:
    with pytest.raises(NotesTaskContractError, match="recurrence"):
        parse_notes_task_v1(
            valid_task_payload(recurrence=recurrence), owner_user_id=OWNER_ID
        )


def test_notes_task_v1_rejects_extra_top_level_fields() -> None:
    with pytest.raises(NotesTaskContractError, match="extra"):
        parse_notes_task_v1(
            valid_task_payload(projection_row_version=99), owner_user_id=OWNER_ID
        )


def test_notes_task_hash_has_exact_stable_vector_and_excludes_projection_version() -> None:
    parsed = parse_notes_task_v1(valid_task_payload(), owner_user_id=OWNER_ID)
    projection_row_version_before = 8
    projection_row_version_after = 9

    before = notes_task_object_hash(parsed, revision=7, deleted=False)
    after = notes_task_object_hash(parsed, revision=7, deleted=False)

    assert projection_row_version_before != projection_row_version_after
    assert before == after == "sha256:8b0692a0287e324c733176bb8199e073f1f526bc4fbd153151e8c09e4e73352b"
    assert notes_task_object_hash(parsed, revision=8, deleted=False) != before
    assert notes_task_object_hash(parsed, revision=7, deleted=True) != before


def test_notes_task_nested_custom_is_immutable_and_hash_stable() -> None:
    raw = valid_task_payload(
        custom={"nested": {"items": [{"value": 1}]}},
    )
    parsed = parse_notes_task_v1(raw, owner_user_id=OWNER_ID)
    before = notes_task_object_hash(parsed, revision=1, deleted=False)

    with pytest.raises(TypeError):
        parsed.custom["nested"]["items"][0]["value"] = 2
    with pytest.raises(TypeError):
        parsed.custom["nested"]["items"].append({"value": 2})

    raw_custom = raw["custom"]
    assert isinstance(raw_custom, dict)
    raw_custom["nested"]["items"].append({"value": 3})
    assert notes_task_object_hash(parsed, revision=1, deleted=False) == before


def test_deepcopy_preserves_parsed_model_values_and_hashes() -> None:
    task = parse_notes_task_v1(valid_task_payload(), owner_user_id=OWNER_ID)
    activity = parse_activity(valid_activity_payload())

    task_copy = deepcopy(task)
    activity_copy = deepcopy(activity)

    assert task_copy == task
    assert activity_copy == activity
    assert notes_task_object_hash(
        task_copy, revision=1, deleted=False
    ) == notes_task_object_hash(task, revision=1, deleted=False)
    assert notes_task_activity_object_hash(
        activity_copy, revision=1, deleted=False
    ) == notes_task_activity_object_hash(activity, revision=1, deleted=False)


@pytest.mark.parametrize(
    ("event_type", "old_value", "new_value", "corrects_activity_id"),
    [
        (
            "created",
            None,
            {
                "title": "Created task",
                "status": "open",
                "completed_at": None,
                "metadata": canonical_task_metadata(),
            },
            None,
        ),
        (
            "updated",
            {"metadata": canonical_task_metadata(priority="low")},
            {"metadata": canonical_task_metadata(priority="high")},
            None,
        ),
        ("completed", {"status": "open"}, {"status": "done"}, None),
        ("reopened", {"status": "done"}, {"status": "open"}, None),
        (
            "deleted",
            {"deleted": False, "projection_status": "ambiguous"},
            {"deleted": True, "projection_status": "deleted"},
            None,
        ),
        (
            "restored",
            {"deleted": True, "projection_status": "deleted"},
            {"deleted": False, "projection_status": "live"},
            None,
        ),
        (
            "projection_linked",
            {"projection_status": "unlinked"},
            {"projection_status": "live"},
            None,
        ),
        (
            "projection_unlinked",
            {"projection_status": "live"},
            {"projection_status": "unlinked"},
            None,
        ),
        (
            "projection_drift",
            None,
            {"reason_code": "both_changed"},
            None,
        ),
        (
            "corrected",
            {"title": "Old", "metadata": canonical_task_metadata()},
            {"title": "New", "metadata": canonical_task_metadata()},
            CORRECTED_ID,
        ),
    ],
)
def test_notes_task_activity_v1_accepts_every_exact_event_schema(
    event_type: str,
    old_value: dict[str, object] | None,
    new_value: dict[str, object] | None,
    corrects_activity_id: str | None,
) -> None:
    parsed = parse_activity(
        valid_activity_payload(
            event_type=event_type,
            old_value=old_value,
            new_value=new_value,
            corrects_activity_id=corrects_activity_id,
        )
    )

    assert parsed.event_type == event_type


@pytest.mark.parametrize(
    ("event_type", "old_value", "new_value", "corrects_activity_id"),
    [
        ("unknown", None, None, None),
        ("completed", {"status": "open", "extra": True}, {"status": "done"}, None),
        ("reopened", {"status": "done"}, {"status": "done"}, None),
        ("updated", {"title": "Old"}, {"title": "New"}, None),
        ("projection_drift", None, {"reason_code": "raw_markdown"}, None),
        ("created", None, {"title": "missing snapshot"}, None),
        ("corrected", {"status": "open"}, {"status": "done"}, None),
        ("completed", {"status": "open"}, {"status": "done"}, CORRECTED_ID),
    ],
)
def test_notes_task_activity_v1_rejects_noncanonical_event_shapes(
    event_type: str,
    old_value: dict[str, object] | None,
    new_value: dict[str, object] | None,
    corrects_activity_id: str | None,
) -> None:
    with pytest.raises(NotesTaskContractError):
        parse_activity(
            valid_activity_payload(
                event_type=event_type,
                old_value=old_value,
                new_value=new_value,
                corrects_activity_id=corrects_activity_id,
            )
        )


@pytest.mark.parametrize(
    ("event_type", "old_value", "new_value"),
    [
        (
            "deleted",
            {"deleted": 0, "projection_status": "live"},
            {"deleted": True, "projection_status": "deleted"},
        ),
        (
            "deleted",
            {"deleted": False, "projection_status": "live"},
            {"deleted": 1, "projection_status": "deleted"},
        ),
        (
            "restored",
            {"deleted": 1, "projection_status": "deleted"},
            {"deleted": False, "projection_status": "live"},
        ),
        (
            "restored",
            {"deleted": True, "projection_status": "deleted"},
            {"deleted": 0, "projection_status": "live"},
        ),
    ],
)
def test_lifecycle_activity_rejects_integer_deleted_flags(
    event_type: str,
    old_value: dict[str, object],
    new_value: dict[str, object],
) -> None:
    with pytest.raises(NotesTaskContractError):
        parse_activity(
            valid_activity_payload(
                event_type=event_type,
                old_value=old_value,
                new_value=new_value,
            )
        )


@pytest.mark.parametrize(
    ("old_value", "new_value"),
    [
        (
            {"status": "open", "completed_at": OCCURRED_AT},
            {"status": "done", "completed_at": OCCURRED_AT},
        ),
        (
            {"status": "open", "completed_at": None},
            {"status": "done", "completed_at": None},
        ),
        (
            {"deleted": False, "projection_status": "deleted"},
            {"deleted": True, "projection_status": "deleted"},
        ),
        (
            {"deleted": True, "projection_status": "deleted"},
            {"deleted": False, "projection_status": "deleted"},
        ),
    ],
)
def test_corrected_activity_rejects_incompatible_coupled_states(
    old_value: dict[str, object],
    new_value: dict[str, object],
) -> None:
    with pytest.raises(NotesTaskContractError):
        parse_activity(
            valid_activity_payload(
                event_type="corrected",
                old_value=old_value,
                new_value=new_value,
                corrects_activity_id=CORRECTED_ID,
            )
        )


@pytest.mark.parametrize(
    ("old_value", "new_value"),
    [
        (
            {"status": "open", "completed_at": None},
            {"status": "done", "completed_at": OCCURRED_AT},
        ),
        (
            {"deleted": True, "projection_status": "deleted"},
            {"deleted": False, "projection_status": "live"},
        ),
        (
            {"projection_status": "ambiguous"},
            {"projection_status": "live"},
        ),
    ],
)
def test_corrected_activity_accepts_compatible_target_event_subsets(
    old_value: dict[str, object],
    new_value: dict[str, object],
) -> None:
    parsed = parse_activity(
        valid_activity_payload(
            event_type="corrected",
            old_value=old_value,
            new_value=new_value,
            corrects_activity_id=CORRECTED_ID,
        )
    )

    assert parsed.old_value == old_value
    assert parsed.new_value == new_value


def test_notes_task_activity_v1_binds_actor_and_device_provenance() -> None:
    payload = valid_activity_payload()

    with pytest.raises(NotesTaskContractError, match="actor"):
        parse_notes_task_activity_v1(
            payload,
            owner_user_id=OWNER_ID,
            bound_actor_type="agent",
            bound_actor_id=OWNER_ID,
            authenticated_device_id=DEVICE_ID,
            trusted_server_origin=False,
        )
    with pytest.raises(NotesTaskContractError, match="device"):
        parse_notes_task_activity_v1(
            payload,
            owner_user_id=OWNER_ID,
            bound_actor_type="user",
            bound_actor_id=OWNER_ID,
            authenticated_device_id=CORRECTED_ID,
            trusted_server_origin=False,
        )
    with pytest.raises(NotesTaskContractError, match="owner"):
        parse_notes_task_activity_v1(
            valid_activity_payload(actor_id="another-owner"),
            owner_user_id=OWNER_ID,
            bound_actor_type="user",
            bound_actor_id="another-owner",
            authenticated_device_id=DEVICE_ID,
            trusted_server_origin=False,
        )


def test_notes_task_activity_v1_trusted_server_provenance_has_no_device() -> None:
    payload = valid_activity_payload(
        actor_type="system",
        actor_id=None,
        source_device_id=None,
        source_kind="repair",
    )
    parsed = parse_notes_task_activity_v1(
        payload,
        owner_user_id=OWNER_ID,
        bound_actor_type="system",
        bound_actor_id=None,
        authenticated_device_id=None,
        trusted_server_origin=True,
    )

    assert parsed.source_device_id is None


@pytest.mark.parametrize(
    "metadata",
    [
        {f"k{index}": index for index in range(17)},
        {"a": {"b": {"c": {"d": 1}}}},
        {"large": "x" * (8 * 1_024)},
        {"api_key": "secret"},
        {"raw_markdown": "- [ ] secret"},
    ],
)
def test_notes_task_activity_v1_rejects_metadata_boundaries(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(NotesTaskContractError, match="metadata"):
        parse_activity(valid_activity_payload(metadata=metadata))


def test_notes_task_activity_v1_rejects_extra_fields_and_noncanonical_ids() -> None:
    with pytest.raises(NotesTaskContractError, match="extra"):
        parse_activity(valid_activity_payload(server_cursor=1))
    with pytest.raises(NotesTaskContractError, match="UUIDv4"):
        parse_activity(
            valid_activity_payload(
                activity_id="AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
            )
        )


def test_notes_task_activity_create_hash_is_revision_one_and_stable_id_bound() -> None:
    parsed = parse_activity(valid_activity_payload())
    digest = notes_task_activity_object_hash(parsed, revision=1, deleted=False)
    changed_id = parse_activity(valid_activity_payload(activity_id=CORRECTED_ID))
    changed_content = parse_activity(
        valid_activity_payload(metadata={"mutation_group_step": 3})
    )

    assert digest == "sha256:163c01a2ec395d36c3769b2f3c91f9579576afc15483f84cd2a4e3cabe3f4c83"
    assert notes_task_activity_object_hash(changed_id, revision=1, deleted=False) != digest
    assert (
        notes_task_activity_object_hash(
            changed_content, revision=1, deleted=False
        )
        != digest
    )
    with pytest.raises(NotesTaskContractError, match="revision 1"):
        notes_task_activity_object_hash(parsed, revision=2, deleted=False)


def test_notes_task_activity_nested_json_is_immutable_and_hash_stable() -> None:
    raw = valid_activity_payload(
        old_value={
            "metadata": canonical_task_metadata(
                priority="low",
                custom={"nested": {"value": 1}},
            )
        },
        new_value={
            "metadata": canonical_task_metadata(
                priority="high",
                custom={"nested": {"value": 2}},
            )
        },
        metadata={"nested": {"items": [3]}},
    )
    parsed = parse_activity(raw)
    before = notes_task_activity_object_hash(parsed, revision=1, deleted=False)
    assert parsed.old_value is not None
    assert parsed.new_value is not None

    with pytest.raises(TypeError):
        parsed.old_value["metadata"]["custom"]["nested"]["value"] = 4
    with pytest.raises(TypeError):
        parsed.new_value["metadata"]["custom"]["nested"]["value"] = 4
    with pytest.raises(TypeError):
        parsed.metadata["nested"]["items"].append(4)

    raw_metadata = raw["metadata"]
    assert isinstance(raw_metadata, dict)
    raw_metadata["nested"]["items"].append(5)
    assert notes_task_activity_object_hash(parsed, revision=1, deleted=False) == before


def test_notes_task_activity_tombstone_binds_parents_timestamp_and_revision_two() -> None:
    original = parse_activity(valid_activity_payload())
    original_hash = notes_task_activity_object_hash(
        original, revision=1, deleted=False
    )
    tombstone = parse_notes_task_activity_tombstone_v1(
        {
            "note_id": NOTE_ID,
            "task_id": TASK_ID,
            "deleted_at": OCCURRED_AT,
            "delete_reason": "user_request",
        },
        envelope_created_at_client="2026-08-13T10:00:00Z",
        original_activity=original,
    )

    assert isinstance(tombstone, NotesTaskActivityTombstoneV1)
    assert notes_task_activity_object_hash(
        tombstone,
        revision=2,
        deleted=True,
        activity_id=ACTIVITY_ID,
        original_create_hash=original_hash,
    ) == "sha256:0f6af02bfdeee072cac6e8c902b082d2ff2d27ce0586b2de5027858f29e17519"

    for changed in (
        {"deleted_at": "2026-08-13T10:00:01+00:00"},
        {"note_id": CORRECTED_ID},
        {"task_id": None},
    ):
        payload = tombstone.model_dump(mode="json")
        payload.update(changed)
        with pytest.raises(NotesTaskContractError):
            parse_notes_task_activity_tombstone_v1(
                payload,
                envelope_created_at_client="2026-08-13T10:00:00Z",
                original_activity=original,
            )

    with pytest.raises(NotesTaskContractError, match="revision 2"):
        notes_task_activity_object_hash(
            tombstone,
            revision=1,
            deleted=True,
            activity_id=ACTIVITY_ID,
            original_create_hash=original_hash,
        )


@pytest.mark.parametrize(
    ("event", "canonical_type", "old_value", "new_value"),
    [
        (
            legacy_event(
                event_type="created",
                old_value=None,
                new_value={"text": "New", "status": "open", "metadata": {}},
            ),
            "created",
            None,
            {
                "title": "New",
                "status": "open",
                "completed_at": None,
                "metadata": canonical_task_metadata(),
            },
        ),
        (
            legacy_event(
                event_type="updated",
                old_value={"metadata": {"priority": "low"}},
                new_value={"metadata": {"priority": "high"}},
            ),
            "updated",
            {"metadata": canonical_task_metadata(priority="low")},
            {"metadata": canonical_task_metadata(priority="high")},
        ),
        (
            legacy_event(
                event_type="updated",
                old_value={"text": "Old", "metadata": {}},
                new_value={"text": "New", "metadata": {"estimate": "2h"}},
            ),
            "updated",
            {"title": "Old", "metadata": canonical_task_metadata()},
            {
                "title": "New",
                "metadata": canonical_task_metadata(estimate="2h"),
            },
        ),
        (
            legacy_event(),
            "completed",
            {"status": "open"},
            {"status": "done"},
        ),
        (
            legacy_event(
                old_value={"status": "done"}, new_value={"status": "open"}
            ),
            "reopened",
            {"status": "done"},
            {"status": "open"},
        ),
        (
            legacy_event(
                event_type="unlinked",
                old_value={"projection_status": "live"},
                new_value={"projection_status": "unlinked"},
            ),
            "projection_unlinked",
            {"projection_status": "live"},
            {"projection_status": "unlinked"},
        ),
        (
            legacy_event(
                event_type="deleted",
                old_value={"deleted": False, "projection_status": "ambiguous"},
                new_value={"deleted": True, "projection_status": "deleted"},
            ),
            "deleted",
            {"deleted": False, "projection_status": "ambiguous"},
            {"deleted": True, "projection_status": "deleted"},
        ),
    ],
)
def test_convert_legacy_task_event_maps_every_approved_source_family(
    event: dict[str, object],
    canonical_type: str,
    old_value: dict[str, object] | None,
    new_value: dict[str, object] | None,
) -> None:
    converted = convert_legacy(event)

    assert converted.event_type == canonical_type
    assert converted.old_value == old_value
    assert converted.new_value == new_value
    assert converted.source_kind == "trusted_bootstrap_v1"
    assert converted.source_device_id is None
    assert converted.client_occurred_at == OCCURRED_AT
    assert converted.metadata["legacy_source_verified"] is True


def test_convert_legacy_created_done_uses_event_time_and_removes_idempotency_key() -> None:
    event = legacy_event(
        event_type="created",
        old_value=None,
        new_value={
            "text": "Done",
            "status": "done",
            "metadata": {"due_date": "2026-08-31"},
            "idempotency_key": "Request-Key-1",
        },
        tool_name="notes-tool",
        policy_mode="review",
        approval_id="approval-1",
    )

    converted = convert_legacy(event)

    assert converted.new_value == {
        "title": "Done",
        "status": "done",
        "completed_at": OCCURRED_AT,
        "metadata": canonical_task_metadata(due_date="2026-08-31"),
    }
    assert converted.metadata == {
        "legacy_source_verified": True,
        "origin_request_fingerprint": "sha256:"
        + hashlib.sha256(b"Request-Key-1").hexdigest(),
        "legacy_context": {
            "tool_name": "notes-tool",
            "policy_mode": "review",
            "approval_id": "approval-1",
        },
    }


def test_convert_legacy_created_expands_missing_metadata() -> None:
    converted = convert_legacy(
        legacy_event(
            event_type="created",
            old_value=None,
            new_value={"text": "New", "status": "open", "metadata": None},
        )
    )

    assert converted.new_value == {
        "title": "New",
        "status": "open",
        "completed_at": None,
        "metadata": canonical_task_metadata(),
    }
    assert converted.metadata == {"legacy_source_verified": True}


def test_convert_legacy_task_event_derives_only_a_verified_missing_note_parent() -> None:
    converted = convert_legacy(legacy_event(note_id=None))
    assert str(converted.note_id) == NOTE_ID

    with pytest.raises(NotesTaskContractError, match="parent"):
        convert_legacy_task_event(
            legacy_event(note_id=None),
            owner_user_id=OWNER_ID,
            resolved_task_note_id=None,
        )
    with pytest.raises(NotesTaskContractError, match="parent"):
        convert_legacy_task_event(
            legacy_event(),
            owner_user_id=OWNER_ID,
            resolved_task_note_id=CORRECTED_ID,
        )


@pytest.mark.parametrize(
    "event",
    [
        legacy_event(event_type="renamed"),
        legacy_event(old_value={"status": "open"}, new_value={"status": "paused"}),
        legacy_event(
            event_type="updated",
            old_value={"metadata": {"unknown": 1}},
            new_value={"metadata": {}},
        ),
        legacy_event(new_value={"status": "done", "idempotency_key": ""}),
        legacy_event(new_value={"status": "done", "idempotency_key": 7}),
        legacy_event(unexpected="data"),
    ],
)
def test_convert_legacy_task_event_fails_closed_on_unknown_or_malformed_data(
    event: dict[str, object],
) -> None:
    with pytest.raises(NotesTaskContractError):
        convert_legacy(event)


def test_convert_legacy_task_event_bounds_deep_json_before_schema_matching() -> None:
    deep: dict[str, object] = {}
    for _ in range(1_500):
        deep = {"nested": deep}

    with pytest.raises(NotesTaskContractError, match="depth"):
        convert_legacy(
            legacy_event(
                event_type="updated",
                old_value=deep,
                new_value=deep,
            )
        )


def test_convert_legacy_task_event_wraps_invalid_ids_as_contract_errors() -> None:
    with pytest.raises(NotesTaskContractError, match="UUIDv4"):
        convert_legacy(
            legacy_event(id="AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA")
        )


def test_convert_legacy_idempotency_fingerprint_changes_activity_hash() -> None:
    first = convert_legacy(
        legacy_event(new_value={"status": "done", "idempotency_key": "first"})
    )
    second = convert_legacy(
        legacy_event(new_value={"status": "done", "idempotency_key": "second"})
    )

    assert first.new_value == second.new_value == {"status": "done"}
    assert notes_task_activity_object_hash(
        first, revision=1, deleted=False
    ) != notes_task_activity_object_hash(second, revision=1, deleted=False)


def test_hash_inputs_do_not_accept_server_cursor_read_state_or_projection_cache() -> None:
    task = parse_notes_task_v1(valid_task_payload(), owner_user_id=OWNER_ID)
    activity = parse_activity(valid_activity_payload())
    task_before = notes_task_object_hash(task, revision=3, deleted=False)
    activity_before = notes_task_activity_object_hash(
        activity, revision=1, deleted=False
    )
    excluded_local_state = deepcopy(
        {
            "projection_row_version": 99,
            "server_cursor": 321,
            "read_at": OCCURRED_AT,
            "dismissed_at": None,
        }
    )
    excluded_local_state.update(
        projection_row_version=100,
        server_cursor=322,
        read_at=None,
        dismissed_at=OCCURRED_AT,
    )

    assert notes_task_object_hash(task, revision=3, deleted=False) == task_before
    assert (
        notes_task_activity_object_hash(activity, revision=1, deleted=False)
        == activity_before
    )
