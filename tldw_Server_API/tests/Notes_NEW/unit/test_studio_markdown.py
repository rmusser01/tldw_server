"""Focused tests for Notes Studio payload normalization."""

from __future__ import annotations

from tldw_Server_API.app.core.Notes.studio_markdown import normalize_studio_payload
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    StudioSectionsV1,
)


def test_normalize_studio_payload_preserves_contract_valid_sections_exactly():
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


def test_normalize_studio_payload_preserves_valid_empty_sections():
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
