from __future__ import annotations

from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.unit

from tldw_Server_API.app.api.v1.schemas.osce import (
    OsceStationCreateContent,
    OsceStationStoredContent,
    OsceStationUpdateContent,
)
from tldw_Server_API.app.core.exceptions import OsceStationIdentityError
from tldw_Server_API.app.services.osce_practice import (
    evidence_fingerprint,
    materialize_station_content,
    project_station_summary,
    reconcile_station_update,
)


def _create_content() -> OsceStationCreateContent:
    return OsceStationCreateContent.model_validate(
        {
            "title": "Discuss safe anticoagulant use",
            "candidate_instructions": "You are speaking with a simulated patient.",
            "candidate_task": "Explain key safety advice and respond to concerns.",
            "patient_context": {
                "text": "A fictional adult has recently started warfarin.",
                "citations": [
                    {
                        "source_type": "url",
                        "source_id": "guidance-1",
                        "source_url": "https://example.test/guidance",
                    }
                ],
            },
            "recommended_duration_seconds": 480,
            "checklist_items": [
                {
                    "label": "Explains the purpose of treatment",
                    "rationale": "Understanding the indication supports safe use.",
                    "citations": [],
                },
                {
                    "label": "Discusses monitoring",
                    "rationale": "Monitoring helps keep treatment within a safe range.",
                    "citations": [],
                },
            ],
            "rubric_domains": [
                {
                    "label": "Communication",
                    "levels": [
                        {"label": "Needs development", "description": "Explanation is unclear."},
                        {"label": "Developing", "description": "Explanation is partly clear."},
                        {"label": "Effective", "description": "Explanation is clear."},
                    ],
                },
                {
                    "label": "Safety",
                    "levels": [
                        {"label": "Needs development", "description": "Important risks are omitted."},
                        {"label": "Effective", "description": "Important risks are covered."},
                    ],
                },
            ],
            "expected_key_points": [
                {
                    "text": "Discusses monitoring and clinically important warning signs.",
                    "citations": [],
                }
            ],
        }
    )


@pytest.fixture
def stored_station() -> OsceStationStoredContent:
    return materialize_station_content(_create_content())


def _all_nested_ids(content: OsceStationStoredContent) -> list[UUID]:
    ids = [item.id for item in content.checklist_items]
    for domain in content.rubric_domains:
        ids.append(domain.id)
        ids.extend(level.id for level in domain.levels)
    ids.extend(point.id for point in content.expected_key_points)
    return ids


def test_materialize_assigns_unique_server_owned_nested_ids() -> None:
    first = materialize_station_content(_create_content())
    second = materialize_station_content(_create_content())

    first_ids = _all_nested_ids(first)
    assert len(first_ids) == len(set(first_ids))
    assert set(first_ids).isdisjoint(_all_nested_ids(second))


def test_update_preserves_recognized_nested_ids(stored_station: OsceStationStoredContent) -> None:
    update = OsceStationUpdateContent.model_validate(
        {
            **stored_station.model_dump(mode="json"),
            "title": "Updated title",
        }
    )

    result = reconcile_station_update(stored_station, update, "manually_authored")

    assert result.content.title == "Updated title"
    assert _all_nested_ids(result.content) == _all_nested_ids(stored_station)


def test_unknown_nested_id_is_rejected(stored_station: OsceStationStoredContent) -> None:
    update = OsceStationUpdateContent.model_validate(
        {
            "checklist_items": [
                {
                    **stored_station.checklist_items[0].model_dump(mode="json"),
                    "id": str(uuid4()),
                }
            ]
        }
    )

    with pytest.raises(OsceStationIdentityError, match="checklist item"):
        reconcile_station_update(stored_station, update, "manually_authored")


def test_duplicate_nested_id_is_rejected(stored_station: OsceStationStoredContent) -> None:
    first = stored_station.checklist_items[0]
    duplicate = stored_station.checklist_items[1].model_copy(update={"id": first.id})
    update = OsceStationUpdateContent.model_construct(checklist_items=[first, duplicate])

    with pytest.raises(OsceStationIdentityError, match="duplicate"):
        reconcile_station_update(stored_station, update, "manually_authored")


def test_moving_existing_level_to_another_domain_is_rejected(
    stored_station: OsceStationStoredContent,
) -> None:
    first_domain, second_domain = stored_station.rubric_domains
    moved_level = first_domain.levels[0]
    update = OsceStationUpdateContent.model_construct(
        rubric_domains=[
            first_domain.model_copy(update={"levels": first_domain.levels[1:]}),
            second_domain.model_copy(update={"levels": [*second_domain.levels, moved_level]}),
        ]
    )

    with pytest.raises(OsceStationIdentityError, match="rubric level"):
        reconcile_station_update(stored_station, update, "manually_authored")


def test_supplied_collection_fully_replaces_and_assigns_new_ids(
    stored_station: OsceStationStoredContent,
) -> None:
    retained = stored_station.checklist_items[1]
    update = OsceStationUpdateContent.model_validate(
        {
            "checklist_items": [
                retained.model_dump(mode="json"),
                {
                    "label": "Checks understanding",
                    "rationale": "Checking understanding supports safe use.",
                    "citations": [],
                },
            ]
        }
    )

    result = reconcile_station_update(stored_station, update, "manually_authored")

    assert [item.id for item in result.content.checklist_items][:1] == [retained.id]
    assert result.content.checklist_items[1].id not in _all_nested_ids(stored_station)
    assert stored_station.checklist_items[0].id not in _all_nested_ids(result.content)


def test_evidence_edit_invalidates_source_verification(
    stored_station: OsceStationStoredContent,
) -> None:
    updated = stored_station.model_copy(
        update={
            "patient_context": stored_station.patient_context.model_copy(
                update={"text": "Changed fact"}
            )
        }
    )

    result = reconcile_station_update(stored_station, updated, "source_verified")

    assert result.verification_state == "modified_after_verification"


def test_presentation_only_edits_preserve_source_verification(
    stored_station: OsceStationStoredContent,
) -> None:
    first_domain = stored_station.rubric_domains[0]
    update = OsceStationUpdateContent.model_construct(
        title="A clearer title",
        candidate_task="Explain the same evidence in a clearer order.",
        rubric_domains=[
            first_domain.model_copy(update={"label": "Communication skills"}),
            *stored_station.rubric_domains[1:],
        ],
    )

    result = reconcile_station_update(stored_station, update, "source_verified")

    assert result.verification_state == "source_verified"


def test_identical_evidence_with_new_nested_ids_preserves_source_verification(
    stored_station: OsceStationStoredContent,
) -> None:
    update = OsceStationUpdateContent.model_validate(
        {
            "checklist_items": [
                item.model_dump(mode="json", exclude={"id"})
                for item in stored_station.checklist_items
            ],
            "expected_key_points": [
                point.model_dump(mode="json", exclude={"id"})
                for point in stored_station.expected_key_points
            ],
        }
    )

    result = reconcile_station_update(stored_station, update, "source_verified")

    assert {item.id for item in result.content.checklist_items}.isdisjoint(
        item.id for item in stored_station.checklist_items
    )
    assert {point.id for point in result.content.expected_key_points}.isdisjoint(
        point.id for point in stored_station.expected_key_points
    )
    assert result.verification_state == "source_verified"


def test_evidence_fingerprint_excludes_presentation_fields(
    stored_station: OsceStationStoredContent,
) -> None:
    presentation_edit = stored_station.model_copy(
        update={
            "title": "New title",
            "candidate_instructions": "Reworded instructions.",
            "recommended_duration_seconds": 600,
        }
    )
    evidence_edit = stored_station.model_copy(
        update={
            "expected_key_points": [
                stored_station.expected_key_points[0].model_copy(update={"text": "Changed evidence"})
            ]
        }
    )

    assert evidence_fingerprint(presentation_edit) == evidence_fingerprint(stored_station)
    assert evidence_fingerprint(evidence_edit) != evidence_fingerprint(stored_station)


def test_project_station_summary_omits_marking_guide(
    stored_station: OsceStationStoredContent,
) -> None:
    summary = project_station_summary(
        {
            "id": 7,
            "quiz_id": 3,
            "content": stored_station,
            "order_index": 2,
            "version": 4,
            "verification_state": "source_verified",
            "created_at": "2026-09-10T12:00:00.000Z",
            "updated_at": "2026-09-10T12:05:00.000Z",
        }
    )

    assert summary.model_dump(mode="json") == {
        "id": 7,
        "quiz_id": 3,
        "title": stored_station.title,
        "recommended_duration_seconds": stored_station.recommended_duration_seconds,
        "order_index": 2,
        "version": 4,
        "checklist_count": 2,
        "rubric_domain_count": 2,
        "verification_state": "source_verified",
        "created_at": "2026-09-10T12:00:00.000Z",
        "updated_at": "2026-09-10T12:05:00.000Z",
    }
