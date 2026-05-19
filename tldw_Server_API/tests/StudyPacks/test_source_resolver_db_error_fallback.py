from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.StudyPacks.source_resolver import StudySourceResolver
from tldw_Server_API.app.core.StudyPacks.types import StudySourceSelection


def _expect_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        pytest.fail(f"{message}: expected {expected!r}, got {actual!r}")


def _expect_contains(text: str, needle: str, message: str) -> None:
    if needle not in text:
        pytest.fail(f"{message}: expected {needle!r} in {text!r}")


def test_media_source_falls_back_to_transcript_when_chunk_lookup_raises_database_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_get_media_by_id(db: Any, media_id: int, **_: Any):
        calls.append("get_media_by_id")
        return {"id": media_id, "title": "Packet Capture Walkthrough"}

    def fake_get_unvectorized_chunks_in_range(db: Any, media_id: int, start_index: int, end_index: int):
        calls.append("get_unvectorized_chunks_in_range")
        raise DatabaseError("chunk lookup failed")

    def fake_get_latest_transcription(db: Any, media_id: int):
        calls.append("get_latest_transcription")
        return "Transcript fallback remains available."

    monkeypatch.setattr(
        "tldw_Server_API.app.core.StudyPacks.source_resolver.get_media_by_id",
        fake_get_media_by_id,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.StudyPacks.source_resolver.get_unvectorized_chunks_in_range",
        fake_get_unvectorized_chunks_in_range,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.StudyPacks.source_resolver.get_latest_transcription",
        fake_get_latest_transcription,
    )

    resolver = StudySourceResolver(media_db=SimpleNamespace(name="media-db"))
    bundle = resolver.resolve(
        [
            StudySourceSelection(
                source_type="media",
                source_id="9",
                locator={"chunk_index": 4, "timestamp_seconds": 61},
            )
        ]
    )

    item = bundle.items[0]
    _expect_equal(calls, ["get_media_by_id", "get_unvectorized_chunks_in_range", "get_latest_transcription"], "expected transcript fallback sequence")
    _expect_equal(item.locator["media_id"], 9, "expected normalized media id")
    _expect_equal(item.locator["timestamp_seconds"], 61.0, "expected timestamp fallback locator")
    _expect_contains(item.evidence_text, "Transcript fallback", "expected transcript evidence")
