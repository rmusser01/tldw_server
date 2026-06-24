from __future__ import annotations

from tldw_Server_API.app.core.StudyPacks.jobs import extract_study_pack_source_items


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
