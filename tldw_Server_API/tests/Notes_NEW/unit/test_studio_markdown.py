"""Focused tests for Notes Studio payload normalization."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Notes.studio_markdown import (
    _try_canonical_studio_sections,
    build_derived_studio_payload,
    normalize_studio_payload,
)
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    StudioSectionsV1,
)

pytestmark = pytest.mark.unit


def test_normalize_studio_payload_preserves_contract_valid_sections_exactly() -> None:
    """Preserve exact contract-valid section content during normalization."""
    sections = [
        {
            "id": "custom-cue-α",
            "kind": "cue",
            "title": "  Cues 🌿  ",
            "items": ["  first cue  ", "Cafe\u0301 cue"],
        },
        {
            "id": "custom-summary",
            "kind": "summary",
            "title": "  Summary  ",
            "content": "  leading\ntrailing  ",
        },
    ]
    expected = StudioSectionsV1.model_validate(
        {"sections": sections}
    ).model_dump(mode="json")["sections"]

    normalized = normalize_studio_payload(
        {"meta": {"title": "Title"}, "sections": sections},
        template_type="lined",
        handwriting_mode="accented",
    )

    assert normalized["sections"] == expected


def test_normalize_studio_payload_preserves_valid_empty_sections() -> None:
    """Preserve an explicitly empty valid section list over legacy content."""
    normalized = normalize_studio_payload(
        {"meta": {"title": "Title"}, "sections": []},
        template_type="lined",
        handwriting_mode="accented",
        existing_payload={
            "sections": [
                {
                    "id": "legacy-notes",
                    "kind": "notes",
                    "title": "Notes",
                    "content": "Legacy content",
                }
            ]
        },
    )

    assert normalized["sections"] == []


def test_build_derived_studio_payload_normalizes_layout_and_validates_sections() -> None:
    """Build derived state with fallback metadata and canonical sections."""
    sections = [
        {
            "id": "notes-1",
            "kind": "notes",
            "title": "Notes",
            "content": "Accepted content",
        }
    ]

    derived = build_derived_studio_payload(
        {"meta": {"title": "   "}, "sections": sections},
        template_type="  ",
        handwriting_mode="  ",
        fallback_title=" Study Notes ",
        source_note_id="source-note",
    )

    assert derived == {
        "meta": {"title": "Study Notes", "source_note_id": "source-note"},
        "layout": {
            "template_type": "lined",
            "handwriting_mode": "accented",
            "render_version": 1,
        },
        "sections": sections,
    }


def test_try_canonical_studio_sections_returns_valid_state_or_none() -> None:
    """Return canonical sections for valid input and None for invalid input."""
    sections = [
        {
            "id": "summary-1",
            "kind": "summary",
            "title": "Summary",
            "content": "Accepted content",
        }
    ]

    assert _try_canonical_studio_sections(sections) == sections
    assert _try_canonical_studio_sections({"sections": sections}) is None
