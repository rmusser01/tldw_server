"""OSCE attempt projection, privacy, and assessment contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.api.v1.schemas.osce import OsceStationStoredContent
from tldw_Server_API.app.services.osce_practice import (
    materialize_station_content,
    project_candidate_attempt,
    project_revealed_attempt,
    summarize_osce_attempt,
    validate_assessment_selections,
)


def stored_content(*, checklist_count: int = 2) -> OsceStationStoredContent:
    return materialize_station_content(
        {
            "title": "Warfarin counselling",
            "candidate_instructions": "You are speaking with a simulated patient.",
            "candidate_task": "Explain safe medicine use.",
            "patient_context": {
                "text": "A fictional adult has started warfarin.",
                "citations": [
                    {
                        "source_type": "document",
                        "source_id": "doc-1",
                        "label": "Anticoagulation guide",
                        "quote": "Private source quote",
                        "chunk_id": "chunk-1",
                        "page_number": 4,
                    }
                ],
            },
            "recommended_duration_seconds": 480,
            "checklist_items": [
                {
                    "label": f"Checklist item {index}",
                    "rationale": f"Private rationale {index}",
                    "citations": [],
                }
                for index in range(checklist_count)
            ],
            "rubric_domains": [
                {
                    "label": "Communication",
                    "levels": [
                        {"label": "Developing", "description": "The explanation is incomplete."},
                        {"label": "Effective", "description": "The explanation is clear."},
                    ],
                },
                {
                    "label": "Safety",
                    "levels": [
                        {"label": "Unsafe", "description": "Important advice is missing."},
                        {"label": "Safe", "description": "Important advice is complete."},
                    ],
                },
            ],
            "expected_key_points": [
                {"text": "Private expected key point", "citations": []}
            ],
        }
    )


def attempt_row(
    content: OsceStationStoredContent,
    *,
    state: str = "in_progress",
    version: int = 1,
    checklist_selections: Mapping[str, str] | None = None,
    rubric_selections: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "id": 11,
        "quiz_id": 7,
        "station_id": 5,
        "client_attempt_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "station_snapshot": {
            "title": content.title,
            "content": content.model_dump(mode="json"),
            "provenance": {"private notes": "Never expose provenance to the candidate."},
            "source_bundle": [{"quote": "Never expose source bundle quotes."}],
        },
        "state": state,
        "candidate_notes": "private notes",
        "checklist_selections": dict(checklist_selections or {}),
        "rubric_selections": dict(rubric_selections or {}),
        "started_at": "2026-09-10T12:00:00.000Z",
        "self_assessment_started_at": (
            None if state == "in_progress" else "2026-09-10T12:05:00.000Z"
        ),
        "completed_at": (
            "2026-09-10T12:06:00.000Z" if state == "completed" else None
        ),
        "frozen_elapsed_seconds": None if state == "in_progress" else 300,
        "elapsed_seconds": None if state == "in_progress" else 300,
        "version": version,
        "last_modified_at": "2026-09-10T12:06:00.000Z",
    }


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


def test_candidate_projection_is_built_from_a_structural_allowlist() -> None:
    payload = project_candidate_attempt(attempt_row(stored_content())).model_dump(mode="json")

    forbidden = {
        "checklist_items",
        "rubric_domains",
        "expected_key_points",
        "rationale",
        "quote",
        "chunk_id",
        "page_number",
        "provenance",
        "source_bundle",
    }
    assert forbidden.isdisjoint(nested_keys(payload))
    assert payload["station"]["patient_context"]["citations"] == [
        {
            "source_type": "document",
            "source_id": "doc-1",
            "label": "Anticoagulation guide",
        }
    ]
    assert "private notes" not in json.dumps(payload["station"])
    assert payload["notes"] == "private notes"


def test_revealed_projection_includes_snapshot_guide_and_quotes() -> None:
    content = stored_content()
    row = attempt_row(content, state="self_assessment", version=2)

    payload = project_revealed_attempt(row).model_dump(mode="json")

    assert payload["station"]["checklist_items"][0]["rationale"] == "Private rationale 0"
    assert payload["station"]["patient_context"]["citations"][0]["quote"] == (
        "Private source quote"
    )
    assert payload["elapsed_seconds"] == 300


def test_selection_validation_rejects_unknown_ids_and_cross_domain_levels() -> None:
    content = stored_content()
    checklist = {str(content.checklist_items[0].id): "met"}
    first_domain, second_domain = content.rubric_domains

    with pytest.raises(ValueError, match="unknown checklist"):
        validate_assessment_selections(
            content,
            {"bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb": "met"},
            {},
        )
    with pytest.raises(ValueError, match="does not belong"):
        validate_assessment_selections(
            content,
            checklist,
            {str(first_domain.id): str(second_domain.levels[0].id)},
        )


def test_completion_validation_requires_every_checklist_item_and_rubric_domain() -> None:
    content = stored_content()
    checklist = {str(item.id): "met" for item in content.checklist_items}
    rubric = {
        str(domain.id): str(domain.levels[0].id) for domain in content.rubric_domains
    }

    with pytest.raises(ValueError, match="checklist selections are incomplete"):
        validate_assessment_selections(
            content,
            dict(list(checklist.items())[:-1]),
            rubric,
            require_complete=True,
        )
    with pytest.raises(ValueError, match="rubric selections are incomplete"):
        validate_assessment_selections(
            content,
            checklist,
            dict(list(rubric.items())[:-1]),
            require_complete=True,
        )


@given(checklist_states=st.lists(st.sampled_from(["met", "not_met"]), min_size=1, max_size=20))
def test_complete_summary_accepts_domain_valid_selections_without_deriving_score(
    checklist_states: list[str],
) -> None:
    content = stored_content(checklist_count=len(checklist_states))
    checklist = {
        str(item.id): selection
        for item, selection in zip(content.checklist_items, checklist_states, strict=True)
    }
    rubric = {
        str(domain.id): str(domain.levels[index % len(domain.levels)].id)
        for index, domain in enumerate(content.rubric_domains)
    }

    normalized_checklist, normalized_rubric = validate_assessment_selections(
        content,
        checklist,
        rubric,
        require_complete=True,
    )
    summary = summarize_osce_attempt(
        attempt_row(
            content,
            state="completed",
            version=3,
            checklist_selections=normalized_checklist,
            rubric_selections=normalized_rubric,
        )
    ).model_dump(mode="json")

    assert summary["checklist_met_count"] == checklist_states.count("met")
    assert summary["checklist_total"] == len(checklist_states)
    assert [result["domain_label"] for result in summary["rubric_results"]] == [
        "Communication",
        "Safety",
    ]
    assert "score" not in summary
    assert "passed" not in summary
    assert "notes" not in summary


def test_in_progress_summary_omits_results_and_notes() -> None:
    summary = summarize_osce_attempt(attempt_row(stored_content())).model_dump(mode="json")

    assert summary["checklist_met_count"] is None
    assert summary["checklist_total"] is None
    assert summary["rubric_results"] == []
    assert "notes" not in summary


def test_selection_validation_returns_json_safe_uuid_keys() -> None:
    content = stored_content()
    checklist_id: UUID = content.checklist_items[0].id
    domain = content.rubric_domains[0]

    checklist, rubric = validate_assessment_selections(
        content,
        {checklist_id: "not_met"},
        {domain.id: domain.levels[1].id},
    )

    assert checklist == {str(checklist_id): "not_met"}
    assert rubric == {str(domain.id): str(domain.levels[1].id)}
