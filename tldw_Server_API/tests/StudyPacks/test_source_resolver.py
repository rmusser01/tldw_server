from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackSourceSelection as ApiStudyPackSourceSelection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _load_study_modules():
    try:
        types_mod = import_module("tldw_Server_API.app.core.StudyPacks.types")
        resolver_mod = import_module("tldw_Server_API.app.core.StudyPacks.source_resolver")
    except ModuleNotFoundError as exc:
        pytest.fail(f"StudyPacks resolver modules are missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"StudyPacks resolver imports are not yet usable: {exc}")
    return types_mod, resolver_mod


def _selection(**kwargs: Any):
    types_mod, _ = _load_study_modules()
    return types_mod.StudySourceSelection(**kwargs)


def _resolver(*, db: Any | None = None, media_db: Any | None = None):
    _, resolver_mod = _load_study_modules()
    return resolver_mod.StudySourceResolver(db=db, media_db=media_db)


def test_note_source_resolves_via_get_note_by_id():
    db = MagicMock(spec=CharactersRAGDB)
    db.get_note_by_id.return_value = {
        "id": "note-1",
        "title": "TCP Handshake",
        "content": "A SYN starts the exchange and the ACK completes it.",
    }

    bundle = _resolver(db=db).resolve(
        [_selection(source_type="note", source_id="note-1")]
    )

    db.get_note_by_id.assert_called_once_with("note-1")
    item = bundle.items[0]
    assert item.source_type == "note"  # nosec B101
    assert item.source_id == "note-1"  # nosec B101
    assert item.locator["note_id"] == "note-1"  # nosec B101
    assert "SYN" in item.evidence_text  # nosec B101


def test_note_source_keeps_extra_locator_fields_but_canonical_note_id_wins():
    db = MagicMock(spec=CharactersRAGDB)
    db.get_note_by_id.return_value = {
        "id": "note-1",
        "title": "TCP Handshake",
        "content": "A SYN starts the exchange and the ACK completes it.",
    }

    bundle = _resolver(db=db).resolve(
        [
            _selection(
                source_type="note",
                source_id="note-1",
                locator={"note_id": "caller-note", "anchor": "section-2"},
            )
        ]
    )

    item = bundle.items[0]
    assert item.locator["note_id"] == "note-1"  # nosec B101
    assert item.locator["anchor"] == "section-2"  # nosec B101


def test_resolver_accepts_api_schema_source_selection_models():
    db = MagicMock(spec=CharactersRAGDB)
    db.get_note_by_id.return_value = {
        "id": "note-1",
        "title": "Congestion control",
        "content": "Slow start increases the congestion window aggressively.",
    }

    bundle = _resolver(db=db).resolve(
        [
            ApiStudyPackSourceSelection(
                source_type="note",
                source_id="note-1",
                source_title="Congestion control",
                excerpt_text="Slow start increases the congestion window aggressively.",
            )
        ]
    )

    item = bundle.items[0]
    assert item.label == "Congestion control"  # nosec B101
    assert item.evidence_text == "Slow start increases the congestion window aggressively."  # nosec B101


def test_media_source_resolves_chunks_via_package_native_helpers(monkeypatch: pytest.MonkeyPatch):
    _, resolver_mod = _load_study_modules()
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def fake_get_media_by_id(db: Any, media_id: int, **_: Any):
        calls.append(("get_media_by_id", (db, media_id)))
        return {"id": media_id, "title": "Lecture 7"}

    def fake_get_unvectorized_chunks_in_range(db: Any, media_id: int, start_index: int, end_index: int):
        calls.append(("get_unvectorized_chunks_in_range", (db, media_id, start_index, end_index)))
        return [
            {
                "uuid": "chunk-7",
                "chunk_index": 7,
                "chunk_text": "The retransmission timer doubles after each loss.",
            }
        ]

    def fake_get_latest_transcription(db: Any, media_id: int):
        calls.append(("get_latest_transcription", (db, media_id)))
        return "This fallback should not be used."

    monkeypatch.setattr(resolver_mod, "get_media_by_id", fake_get_media_by_id)
    monkeypatch.setattr(resolver_mod, "get_unvectorized_chunks_in_range", fake_get_unvectorized_chunks_in_range)
    monkeypatch.setattr(resolver_mod, "get_latest_transcription", fake_get_latest_transcription)

    media_db = SimpleNamespace(name="media-db")
    bundle = _resolver(media_db=media_db).resolve(
        [
            _selection(
                source_type="media",
                source_id="7",
                locator={"chunk_index": 7},
            )
        ]
    )

    item = bundle.items[0]
    assert item.source_type == "media"  # nosec B101
    assert item.source_id == "7"  # nosec B101
    assert item.label == "Lecture 7"  # nosec B101
    assert item.locator["media_id"] == 7  # nosec B101
    assert item.locator["chunk_id"] == "chunk-7"  # nosec B101
    assert "retransmission timer" in item.evidence_text  # nosec B101
    assert [name for name, _ in calls] == [  # nosec B101
        "get_media_by_id",
        "get_unvectorized_chunks_in_range",
    ]


def test_media_source_resolves_zero_based_chunk_index(monkeypatch: pytest.MonkeyPatch):
    _, resolver_mod = _load_study_modules()
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def fake_get_media_by_id(db: Any, media_id: int, **_: Any):
        calls.append(("get_media_by_id", (db, media_id)))
        return {"id": media_id, "title": "Lecture 0"}

    def fake_get_unvectorized_chunks_in_range(db: Any, media_id: int, start_index: int, end_index: int):
        calls.append(("get_unvectorized_chunks_in_range", (db, media_id, start_index, end_index)))
        return [
            {
                "uuid": "chunk-0",
                "chunk_index": 0,
                "chunk_text": "Opening definitions live in the first chunk.",
            }
        ]

    monkeypatch.setattr(resolver_mod, "get_media_by_id", fake_get_media_by_id)
    monkeypatch.setattr(resolver_mod, "get_unvectorized_chunks_in_range", fake_get_unvectorized_chunks_in_range)

    bundle = _resolver(media_db=SimpleNamespace(name="media-db")).resolve(
        [
            _selection(
                source_type="media",
                source_id="7",
                locator={"chunk_index": 0},
            )
        ]
    )

    item = bundle.items[0]
    assert item.locator["media_id"] == 7  # nosec B101
    assert item.locator["chunk_index"] == 0  # nosec B101
    assert item.locator["chunk_id"] == "chunk-0"  # nosec B101
    assert "first chunk" in item.evidence_text  # nosec B101
    assert [name for name, _ in calls] == [  # nosec B101
        "get_media_by_id",
        "get_unvectorized_chunks_in_range",
    ]


def test_media_source_falls_back_to_transcript_and_timestamp(monkeypatch: pytest.MonkeyPatch):
    _, resolver_mod = _load_study_modules()
    calls: list[str] = []

    def fake_get_media_by_id(db: Any, media_id: int, **_: Any):
        calls.append("get_media_by_id")
        return {"id": media_id, "title": "Packet Capture Walkthrough"}

    def fake_get_unvectorized_chunks_in_range(db: Any, media_id: int, start_index: int, end_index: int):
        calls.append("get_unvectorized_chunks_in_range")
        return []

    def fake_get_latest_transcription(db: Any, media_id: int):
        calls.append("get_latest_transcription")
        return "At sixty one seconds the speaker explains slow start recovery."

    monkeypatch.setattr(resolver_mod, "get_media_by_id", fake_get_media_by_id)
    monkeypatch.setattr(resolver_mod, "get_unvectorized_chunks_in_range", fake_get_unvectorized_chunks_in_range)
    monkeypatch.setattr(resolver_mod, "get_latest_transcription", fake_get_latest_transcription)

    bundle = _resolver(media_db=SimpleNamespace()).resolve(
        [
            _selection(
                source_type="media",
                source_id="9",
                locator={"timestamp_seconds": 61},
            )
        ]
    )

    item = bundle.items[0]
    assert item.label == "Packet Capture Walkthrough"  # nosec B101
    assert item.locator["media_id"] == 9  # nosec B101
    assert item.locator["timestamp_seconds"] == 61  # nosec B101
    assert "slow start recovery" in item.evidence_text  # nosec B101
    assert calls == [  # nosec B101
        "get_media_by_id",
        "get_latest_transcription",
    ]


def test_media_source_accepts_fractional_timestamp_strings(monkeypatch: pytest.MonkeyPatch):
    _, resolver_mod = _load_study_modules()

    monkeypatch.setattr(
        resolver_mod,
        "get_media_by_id",
        lambda db, media_id, **_: {"id": media_id, "title": "Packet Capture Walkthrough"},
    )
    monkeypatch.setattr(
        resolver_mod,
        "get_latest_transcription",
        lambda db, media_id: "At sixty one point five seconds the speaker explains fast recovery.",
    )

    bundle = _resolver(media_db=SimpleNamespace()).resolve(
        [
            _selection(
                source_type="media",
                source_id="9",
                locator={"timestamp_seconds": "61.5"},
            )
        ]
    )

    item = bundle.items[0]
    assert item.locator["timestamp_seconds"] == 61.5  # nosec B101
    assert "fast recovery" in item.evidence_text  # nosec B101


def test_media_source_falls_back_to_transcript_without_locator(monkeypatch: pytest.MonkeyPatch):
    _, resolver_mod = _load_study_modules()

    monkeypatch.setattr(
        resolver_mod,
        "get_media_by_id",
        lambda db, media_id, **_: {"id": media_id, "title": "Packet Capture Walkthrough"},
    )
    monkeypatch.setattr(
        resolver_mod,
        "get_latest_transcription",
        lambda db, media_id: "The transcript remains usable even without a specific timestamp.",
    )

    bundle = _resolver(media_db=SimpleNamespace()).resolve(
        [
            _selection(
                source_type="media",
                source_id="9",
            )
        ]
    )

    item = bundle.items[0]
    assert item.locator == {"media_id": 9}  # nosec B101
    assert "without a specific timestamp" in item.evidence_text  # nosec B101


def test_message_source_requires_stable_message_and_conversation_identity():
    db = MagicMock(spec=CharactersRAGDB)
    db.get_message_by_id.return_value = {
        "id": "msg-1",
        "conversation_id": "conv-1",
        "sender": "user",
        "content": "Why does the sender back off after packet loss?",
    }

    bundle = _resolver(db=db).resolve(
        [_selection(source_type="message", source_id="msg-1")]
    )

    db.get_message_by_id.assert_called_once_with("msg-1")
    item = bundle.items[0]
    assert item.source_type == "message"  # nosec B101
    assert item.locator["conversation_id"] == "conv-1"  # nosec B101
    assert item.locator["message_id"] == "msg-1"  # nosec B101
    assert item.evidence_text  # nosec B101


def test_message_source_keeps_extra_locator_fields_but_canonical_ids_win():
    db = MagicMock(spec=CharactersRAGDB)
    db.get_message_by_id.return_value = {
        "id": "msg-1",
        "conversation_id": "conv-1",
        "sender": "user",
        "content": "Why does the sender back off after packet loss?",
    }

    bundle = _resolver(db=db).resolve(
        [
            _selection(
                source_type="message",
                source_id="msg-1",
                locator={
                    "conversation_id": "caller-conv",
                    "message_id": "caller-msg",
                    "anchor": "reply-1",
                },
            )
        ]
    )

    item = bundle.items[0]
    assert item.locator["conversation_id"] == "conv-1"  # nosec B101
    assert item.locator["message_id"] == "msg-1"  # nosec B101
    assert item.locator["anchor"] == "reply-1"  # nosec B101


def test_study_source_bundle_normalizes_items_and_rejects_non_bundle_items():
    types_mod, _ = _load_study_modules()
    item = types_mod.StudySourceBundleItem(
        source_type="note",
        source_id="note-1",
        label="Example note",
        evidence_text="Example evidence",
        locator={"note_id": "note-1"},
    )

    bundle = types_mod.StudySourceBundle(items=(item,))

    assert bundle.items == [item]  # nosec B101

    with pytest.raises(ValueError, match="StudySourceBundleItem"):
        types_mod.StudySourceBundle(
            items=[{"source_type": "note", "source_id": "note-1"}]
        )


@pytest.mark.parametrize(
    ("selection_kwargs", "db_record", "expected_error"),
    [
        (
            {"source_type": "unsupported", "source_id": "x-1"},
            None,
            "Unsupported study source type",
        ),
        (
            {"source_type": "message", "source_id": "msg-2"},
            {"id": "msg-2", "conversation_id": "", "content": "orphan"},
            "requires both stable message identity and conversation identity",
        ),
        (
            {"source_type": "message", "source_id": "msg-3"},
            {"id": "", "conversation_id": "conv-3", "content": "missing id"},
            "requires both stable message identity and conversation identity",
        ),
    ],
)
def test_unsupported_or_incomplete_sources_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
    selection_kwargs: dict[str, Any],
    db_record: dict[str, Any] | None,
    expected_error: str,
):
    db = MagicMock(spec=CharactersRAGDB)
    db.get_message_by_id.return_value = db_record

    _, resolver_mod = _load_study_modules()
    monkeypatch.setattr(
        resolver_mod,
        "get_media_by_id",
        lambda *_args, **_kwargs: {"id": 5, "title": "Unlocated media"},
    )
    monkeypatch.setattr(
        resolver_mod,
        "get_unvectorized_chunks_in_range",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        resolver_mod,
        "get_latest_transcription",
        lambda *_args, **_kwargs: "Transcript exists but no locator should still fail.",
    )

    with pytest.raises(ValueError, match=expected_error):
        _resolver(db=db, media_db=SimpleNamespace()).resolve(
            [_selection(**selection_kwargs)]
        )


def test_excerpt_text_is_allowed_only_when_the_parent_source_still_exists():
    db = MagicMock(spec=CharactersRAGDB)
    db.get_note_by_id.return_value = {
        "id": "note-2",
        "title": "Congestion Control",
        "content": "Full note content that is longer than the selected excerpt.",
    }
    selection = _selection(
        source_type="note",
        source_id="note-2",
        excerpt_text="Fast retransmit reacts before the timeout expires.",
    )

    bundle = _resolver(db=db).resolve([selection])

    assert bundle.items[0].evidence_text == "Fast retransmit reacts before the timeout expires."  # nosec B101

    db.get_note_by_id.return_value = None
    with pytest.raises(ValueError, match="Note 'note-2' not found"):
        _resolver(db=db).resolve([selection])
