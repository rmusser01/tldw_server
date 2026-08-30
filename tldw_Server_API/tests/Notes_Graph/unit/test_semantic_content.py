from __future__ import annotations

import unicodedata

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Notes_Graph.semantic_content import (
    SEMANTIC_CHUNKER_VERSION,
    SEMANTIC_NORMALIZATION_VERSION,
    SemanticContentError,
    build_semantic_chunks,
    reconstruct_semantic_chunk,
    semantic_content_fingerprint,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings

pytestmark = pytest.mark.unit


def _settings(**overrides: int) -> SemanticIndexSettings:
    values = {
        "max_stored_note_bytes": 10_000,
        "max_canonical_field_code_points": 10_000,
        "max_chunk_code_points": 4,
        "max_chunks_per_note": 100,
        "max_provider_input_bytes": 100,
    }
    values.update(overrides)
    return SemanticIndexSettings(**values)


def test_semantic_fingerprint_binds_canonical_fields_version_and_normalization() -> None:
    normalized = semantic_content_fingerprint("Cafe\u0301\r\nTitle", "Body\r\n", 7)

    assert normalized == semantic_content_fingerprint("Caf\u00e9\nTitle", "Body\n", 7)
    assert normalized != semantic_content_fingerprint("Caf\u00e9\nTitle", "Body\n", 8)
    assert normalized != semantic_content_fingerprint(
        "Caf\u00e9\nTitle",
        "Body\n",
        7,
        normalization_version="notes-semantic-normalization-v2",
    )


def test_body_chunks_use_content_offsets_and_title_only_notes_use_title_offsets() -> None:
    body_chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="Context title",
        content="abcd\U0001f642ef",
        content_version=3,
        settings=_settings(),
    )
    title_chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-2",
        title="Title only",
        content=" \r\n ",
        content_version=1,
        settings=_settings(),
    )

    assert [(chunk.field, chunk.start_offset, chunk.end_offset) for chunk in body_chunks] == [
        ("content", 0, 4),
        ("content", 4, 7),
    ]
    assert all(chunk.provider_input.text.startswith("Context title\n\n") for chunk in body_chunks)
    assert all(chunk.field == "title" for chunk in title_chunks)
    assert "\r" not in "".join(chunk.provider_input.text for chunk in title_chunks)


def test_whitespace_only_note_is_excluded() -> None:
    with pytest.raises(SemanticContentError, match="note_empty"):
        build_semantic_chunks(
            generation_id="generation-1",
            note_id="note-1",
            title=" \t",
            content="\r\n ",
            content_version=1,
            settings=_settings(),
        )


@pytest.mark.parametrize(
    ("overrides", "title", "content", "code"),
    [
        ({"max_stored_note_bytes": 3}, "ab", "cd", "stored_note_bytes_exceeded"),
        (
            {"max_canonical_field_code_points": 3, "max_chunk_code_points": 3},
            "title",
            "",
            "canonical_field_code_points_exceeded",
        ),
        ({"max_chunks_per_note": 1}, "", "abcdefgh", "chunks_per_note_exceeded"),
        ({"max_provider_input_bytes": 3}, "title", "body", "provider_input_bytes_exceeded"),
    ],
)
def test_chunking_rejects_caps_without_truncation(
    overrides: dict[str, int],
    title: str,
    content: str,
    code: str,
) -> None:
    with pytest.raises(SemanticContentError, match=code):
        build_semantic_chunks(
            generation_id="generation-1",
            note_id="note-1",
            title=title,
            content=content,
            content_version=1,
            settings=_settings(**overrides),
        )


def test_chunk_ids_and_fingerprints_bind_versions_and_source_coordinates() -> None:
    common = {
        "generation_id": "generation-1",
        "note_id": "note-1",
        "title": "Title",
        "content": "abcdefgh",
        "content_version": 2,
        "settings": _settings(),
    }
    baseline = build_semantic_chunks(**common)

    assert baseline == build_semantic_chunks(**common)
    assert baseline != build_semantic_chunks(**{**common, "content_version": 3})
    assert baseline != build_semantic_chunks(
        **{**common, "normalization_version": "notes-semantic-normalization-v2"}
    )
    assert baseline != build_semantic_chunks(
        **{**common, "chunker_version": "notes-semantic-chunker-v2"}
    )
    assert baseline[0].normalization_version == SEMANTIC_NORMALIZATION_VERSION
    assert baseline[0].chunker_version == SEMANTIC_CHUNKER_VERSION


@given(
    title=st.text(max_size=40),
    content=st.text(min_size=1, max_size=80).filter(str.strip),
    first_version=st.integers(min_value=1, max_value=1_000_000),
    second_version=st.integers(min_value=1, max_value=1_000_000),
)
def test_fingerprints_and_chunks_are_deterministic_and_version_bound(
    title: str,
    content: str,
    first_version: int,
    second_version: int,
) -> None:
    if first_version == second_version:
        return
    first = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title=title,
        content=content,
        content_version=first_version,
        settings=_settings(),
    )
    repeated = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title=title,
        content=content,
        content_version=first_version,
        settings=_settings(),
    )
    second = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title=title,
        content=content,
        content_version=second_version,
        settings=_settings(),
    )

    assert first == repeated
    assert [chunk.chunk_fingerprint for chunk in first] != [
        chunk.chunk_fingerprint for chunk in second
    ]
    assert [chunk.vector_id for chunk in first] != [chunk.vector_id for chunk in second]


@given(
    content=st.text(
        alphabet=st.sampled_from(["a", "\u00e9", "\U0001f642"]),
        min_size=2,
        max_size=40,
    )
)
def test_stored_note_byte_cap_rejects_the_whole_note(content: str) -> None:
    byte_length = len(content.encode("utf-8"))

    with pytest.raises(SemanticContentError, match="stored_note_bytes_exceeded"):
        build_semantic_chunks(
            generation_id="generation-1",
            note_id="note-1",
            title="",
            content=content,
            content_version=1,
            settings=_settings(max_stored_note_bytes=byte_length - 1),
        )


@given(
    title=st.text(max_size=80),
    content=st.text(max_size=160),
    max_code_points=st.integers(min_value=1, max_value=20),
)
def test_chunks_never_cross_fields_and_reconstruct_exact_canonical_slices(
    title: str,
    content: str,
    max_code_points: int,
) -> None:
    canonical_title = unicodedata.normalize("NFC", title.replace("\r\n", "\n").replace("\r", "\n"))
    canonical_content = unicodedata.normalize(
        "NFC", content.replace("\r\n", "\n").replace("\r", "\n")
    )
    if not canonical_title.strip() and not canonical_content.strip():
        return
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title=title,
        content=content,
        content_version=11,
        settings=_settings(
            max_chunk_code_points=max_code_points,
            max_provider_input_bytes=1_000,
        ),
    )
    expected_field = "content" if canonical_content.strip() else "title"
    expected_text = canonical_content if expected_field == "content" else canonical_title

    assert all(chunk.field == expected_field for chunk in chunks)
    assert "".join(
        reconstruct_semantic_chunk(
            chunk,
            title=title,
            content=content,
            content_version=11,
        )
        or ""
        for chunk in chunks
    ) == expected_text
    for chunk in chunks:
        assert 0 <= chunk.start_offset < chunk.end_offset <= len(expected_text)
        assert chunk.end_offset - chunk.start_offset <= max_code_points
        assert reconstruct_semantic_chunk(
            chunk,
            title=title,
            content=content,
            content_version=11,
        ) == expected_text[chunk.start_offset : chunk.end_offset]
