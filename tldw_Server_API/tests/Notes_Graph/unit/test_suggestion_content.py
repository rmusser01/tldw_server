from __future__ import annotations

import hashlib
import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Notes_Graph.suggestion_content import (
    EvidenceReference,
    canonical_content_bytes,
    canonicalize_note_content,
    content_fingerprint,
    reconstruct_evidence,
    split_evidence_windows,
    stored_text_utf8_bytes,
)

pytestmark = pytest.mark.unit


def test_canonical_bytes_normalize_line_endings_nfc_and_preserve_astral_unicode() -> None:
    canonical = canonicalize_note_content("Cafe\u0301\r\nTitle", "Body\r\U0001f642\u0301")

    assert canonical.title == "Caf\u00e9\nTitle"
    assert canonical.content == "Body\n\U0001f642\u0301"
    assert canonical_content_bytes(canonical.title, canonical.content) == (
        b"notes-graph-content-v1\0Caf\xc3\xa9\nTitle\0Body\n\xf0\x9f\x99\x82\xcc\x81"
    )


def test_content_fingerprint_matches_exact_versioned_byte_sequence() -> None:
    expected_bytes = b"notes-graph-content-v1\0Title\0Body"

    assert content_fingerprint("Title", "Body") == f"sha256:{hashlib.sha256(expected_bytes).hexdigest()}"


def test_stored_utf8_byte_count_uses_original_text_not_code_points() -> None:
    assert stored_text_utf8_bytes("\U0001f642", "e\u0301") == 7


def test_reconstruct_evidence_uses_canonical_field_offsets_and_matching_fingerprint() -> None:
    fingerprint = content_fingerprint("A\r\n\U0001f642", "Cafe\u0301")
    evidence = EvidenceReference(
        note_id="note-1",
        field="title",
        fingerprint=fingerprint,
        start_offset=2,
        end_offset=3,
    )

    assert reconstruct_evidence(evidence, title="A\n\U0001f642", content="Caf\u00e9") == "\U0001f642"


@pytest.mark.parametrize(
    ("field", "start_offset", "end_offset"),
    [
        ("title", -1, 1),
        ("title", 1, 1),
        ("title", 0, 4),
        ("content", 0, 5),
    ],
)
def test_reconstruct_evidence_rejects_invalid_or_cross_field_offsets(
    field: str,
    start_offset: int,
    end_offset: int,
) -> None:
    fingerprint = content_fingerprint("One", "Two")
    evidence = EvidenceReference(
        note_id="note-1",
        field=field,
        fingerprint=fingerprint,
        start_offset=start_offset,
        end_offset=end_offset,
    )

    assert reconstruct_evidence(evidence, title="One", content="Two") is None


def test_reconstruct_evidence_rejects_stale_fingerprint() -> None:
    evidence = EvidenceReference(
        note_id="note-1",
        field="content",
        fingerprint=content_fingerprint("Title", "Old"),
        start_offset=0,
        end_offset=3,
    )

    assert reconstruct_evidence(evidence, title="Title", content="New") is None


@given(
    title=st.text(max_size=1_200),
    content=st.text(max_size=1_200),
    max_windows=st.integers(min_value=1, max_value=4),
    max_code_points=st.integers(min_value=1, max_value=480),
)
def test_evidence_windows_never_cross_canonical_fields_or_exceed_limits(
    title: str,
    content: str,
    max_windows: int,
    max_code_points: int,
) -> None:
    oracle_title = unicodedata.normalize("NFC", title.replace("\r\n", "\n").replace("\r", "\n"))
    oracle_content = unicodedata.normalize("NFC", content.replace("\r\n", "\n").replace("\r", "\n"))
    windows = split_evidence_windows(
        note_id="note-1",
        title=title,
        content=content,
        max_windows=max_windows,
        max_code_points=max_code_points,
    )

    assert len(windows) <= max_windows
    prior_ends = {"title": 0, "content": 0}
    for window in windows:
        field_text = oracle_title if window.field == "title" else oracle_content
        assert 0 <= window.start_offset < window.end_offset <= len(field_text)
        assert window.end_offset - window.start_offset <= max_code_points
        assert prior_ends[window.field] <= window.start_offset
        assert reconstruct_evidence(window, title=title, content=content) == field_text[
            window.start_offset : window.end_offset
        ]
        prior_ends[window.field] = window.end_offset
