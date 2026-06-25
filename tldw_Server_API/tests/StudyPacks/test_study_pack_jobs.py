from __future__ import annotations

import pytest

from tldw_Server_API.app.core.StudyPacks.jobs import extract_study_pack_source_items


pytestmark = pytest.mark.unit


def test_extract_study_pack_source_items_preserves_locator_label_and_evidence_hint():
    source_items = extract_study_pack_source_items(
        {
            "items": [
                {
                    "source_type": "media",
                    "source_id": "42",
                    "label": "Lecture 42",
                    "locator": {"chunk_index": 4, "timestamp_seconds": 61},
                    "evidence_text": "The selected chunk explains slow start.",
                }
            ]
        }
    )

    assert source_items == [  # nosec B101
        {
            "source_type": "media",
            "source_id": "42",
            "label": "Lecture 42",
            "locator": {"chunk_index": 4, "timestamp_seconds": 61},
            "excerpt_text": "The selected chunk explains slow start.",
        }
    ]


def test_extract_study_pack_source_items_bounds_large_evidence_hints(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STUDY_PACK_MAX_EVIDENCE_CHARS_PER_SOURCE", "12")

    source_items = extract_study_pack_source_items(
        {
            "items": [
                {
                    "source_type": "note",
                    "source_id": "note-large",
                    "locator": {"note_id": "note-large"},
                    "evidence_text": "abcdefghijklmnopqrstuvwxyz",
                }
            ]
        }
    )

    assert source_items == [  # nosec B101
        {
            "source_type": "note",
            "source_id": "note-large",
            "locator": {
                "note_id": "note-large",
                "excerpt_truncated": True,
                "excerpt_original_chars": 26,
            },
            "excerpt_text": "abcdefghijkl",
        }
    ]
