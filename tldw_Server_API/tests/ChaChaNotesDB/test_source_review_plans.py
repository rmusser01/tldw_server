import json
from datetime import date, datetime, timezone

import pytest

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackSourceSelection
from tldw_Server_API.app.core.Flashcards.source_review import (
    build_source_review_launch_metadata,
    compute_source_review_due_at,
    compute_source_review_schedule,
    normalize_source_review_bundle,
)


def test_source_review_day_offset_uses_local_midnight():
    due_at = compute_source_review_due_at(
        starts_on=date(2026, 7, 9),
        timezone_name="America/Los_Angeles",
        offset_value=1,
        offset_unit="day",
    )

    assert due_at == datetime(2026, 7, 10, 7, 0, tzinfo=timezone.utc)  # nosec B101


def test_source_review_month_offset_clamps_to_month_end():
    due_at = compute_source_review_due_at(
        starts_on=date(2026, 1, 31),
        timezone_name="America/Los_Angeles",
        offset_value=1,
        offset_unit="month",
    )

    assert due_at == datetime(2026, 2, 28, 8, 0, tzinfo=timezone.utc)  # nosec B101


@pytest.mark.parametrize(
    ("offset_value", "offset_unit"),
    [
        (0, "day"),
        (-1, "month"),
        (1.5, "day"),
        (True, "day"),
        (3651, "day"),
        (121, "month"),
    ],
)
def test_source_review_offset_rejects_non_positive_and_over_cap_values(
    offset_value,
    offset_unit,
):
    with pytest.raises(ValueError, match="offset_value"):
        compute_source_review_due_at(
            starts_on=date(2026, 1, 1),
            timezone_name="UTC",
            offset_value=offset_value,
            offset_unit=offset_unit,
        )


def test_source_review_due_at_rejects_unsupported_unit():
    with pytest.raises(ValueError, match="offset_unit"):
        compute_source_review_due_at(
            starts_on=date(2026, 1, 1),
            timezone_name="UTC",
            offset_value=1,
            offset_unit="week",
        )


def test_source_review_due_at_rejects_invalid_timezone():
    with pytest.raises(ValueError, match="timezone"):
        compute_source_review_due_at(
            starts_on=date(2026, 1, 1),
            timezone_name="Not/A_Timezone",
            offset_value=1,
            offset_unit="day",
        )


def test_source_review_schedule_rejects_duplicate_computed_due_activity():
    with pytest.raises(ValueError, match="Duplicate"):
        compute_source_review_schedule(
            starts_on=date(2026, 1, 31),
            timezone_name="UTC",
            schedule=[
                {
                    "offset_value": 28,
                    "offset_unit": "day",
                    "activity_type": "quiz",
                },
                {
                    "offset_value": 1,
                    "offset_unit": "month",
                    "activity_type": "quiz",
                },
            ],
        )


def test_source_review_schedule_allows_different_activities_at_same_due_at():
    rows = compute_source_review_schedule(
        starts_on=date(2026, 1, 31),
        timezone_name="UTC",
        schedule=[
            {
                "offset_value": 28,
                "offset_unit": "day",
                "activity_type": "quiz",
            },
            {
                "offset_value": 1,
                "offset_unit": "month",
                "activity_type": "flashcards",
            },
        ],
    )

    assert rows[0]["due_at"] == rows[1]["due_at"]  # nosec B101


@pytest.mark.parametrize(
    ("activity_type", "expected_launch_fields"),
    [
        (
            "reread",
            (
                "/flashcards",
                "source_review_due_panel",
                "show_reread_snapshot",
                "source_bundle",
            ),
        ),
        (
            "quiz",
            (
                "/quiz",
                "quiz_generation",
                "prefill_generation_sources",
                "source_items",
            ),
        ),
        (
            "flashcards",
            (
                "/flashcards",
                "flashcard_generation",
                "prefill_generation_sources",
                "source_items",
            ),
        ),
        (
            "cloze",
            (
                "/flashcards",
                "cloze_flashcard_generation",
                "prefill_generation_sources",
                "source_items",
            ),
        ),
    ],
)
def test_source_review_launch_metadata_maps_activity_to_existing_surface(
    activity_type,
    expected_launch_fields,
):
    metadata = build_source_review_launch_metadata(
        activity_type=activity_type,
        plan_id=7,
        occurrence_id=11,
        created_at="2026-07-09T12:00:00Z",
    )

    assert (
        metadata["target_route"],
        metadata["target_surface"],
        metadata["action"],
        metadata["source_payload_field"],
    ) == expected_launch_fields  # nosec B101
    assert metadata["activity_type"] == activity_type  # nosec B101
    assert metadata["plan_id"] == 7  # nosec B101
    assert metadata["occurrence_id"] == 11  # nosec B101
    assert metadata["completion_required"] is True  # nosec B101
    assert metadata["created_at"] == "2026-07-09T12:00:00Z"  # nosec B101


def test_source_review_launch_metadata_is_thin_and_under_size_cap():
    metadata = build_source_review_launch_metadata(
        activity_type="quiz",
        plan_id=7,
        occurrence_id=11,
        created_at="2026-07-09T12:00:00Z",
    )

    assert set(metadata) == {  # nosec B101
        "activity_type",
        "plan_id",
        "occurrence_id",
        "target_route",
        "target_surface",
        "action",
        "source_payload_field",
        "completion_required",
        "created_at",
    }
    assert len(json.dumps(metadata).encode("utf-8")) <= 16 * 1024  # nosec B101


def test_source_review_launch_metadata_rejects_unsupported_activity():
    with pytest.raises(ValueError, match="activity_type"):
        build_source_review_launch_metadata(
            activity_type="matching",
            plan_id=7,
            occurrence_id=11,
            created_at="2026-07-09T12:00:00Z",
        )


def test_source_review_launch_metadata_rejects_payload_over_size_cap():
    with pytest.raises(ValueError, match="16 KiB"):
        build_source_review_launch_metadata(
            activity_type="reread",
            plan_id=7,
            occurrence_id=11,
            created_at="x" * (16 * 1024),
        )


def test_source_review_launch_metadata_rejects_default_json_over_size_cap():
    baseline = build_source_review_launch_metadata(
        activity_type="quiz",
        plan_id=7,
        occurrence_id=11,
        created_at="",
    )
    created_at = "x" * (16 * 1024 - len(json.dumps(baseline).encode("utf-8")) + 1)
    oversized_metadata = {**baseline, "created_at": created_at}

    assert len(json.dumps(oversized_metadata).encode("utf-8")) == 16 * 1024 + 1  # nosec B101
    with pytest.raises(ValueError, match="16 KiB"):
        build_source_review_launch_metadata(
            activity_type="quiz",
            plan_id=7,
            occurrence_id=11,
            created_at=created_at,
        )


def test_source_review_bundle_normalizes_models_and_source_title_alias():
    bundle = normalize_source_review_bundle(
        [
            {
                "source_type": "media",
                "source_id": " 42 ",
                "source_title": " Lecture 42 ",
                "excerpt_text": " Additive increase. ",
                "locator": {"page": 12, "empty": None},
            },
            StudyPackSourceSelection(
                source_type="note",
                source_id="note-1",
                label="Notes",
            ),
        ]
    )

    assert bundle == {  # nosec B101
        "items": [
            {
                "source_type": "media",
                "source_id": "42",
                "label": "Lecture 42",
                "excerpt_text": "Additive increase.",
                "locator": {"page": 12},
            },
            {
                "source_type": "note",
                "source_id": "note-1",
                "label": "Notes",
                "locator": {},
            },
        ]
    }
    assert "source_title" not in bundle["items"][0]  # nosec B101
