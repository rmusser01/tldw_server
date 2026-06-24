from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking.base import ChunkMetadata, ChunkResult
from tldw_Server_API.app.core.Chunking.exceptions import ChunkingError, InvalidInputError


def test_process_text_normal_path_uses_chunk_text(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker()
    calls: list[str] = []

    def fake_chunk_text(*args, **kwargs):
        calls.append("chunk_text")
        return [{"text": "alpha", "metadata": {"start_offset": 0, "end_offset": 5}}]

    def forbidden_metadata_path(*args, **kwargs):
        raise AssertionError("normal process_text path must not call chunk_text_with_metadata")

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(chunker, "chunk_text_with_metadata", forbidden_metadata_path)

    rows = chunker.process_text("alpha beta", options={"method": "words", "max_size": 10, "overlap": 0})

    assert calls == ["chunk_text"]
    assert rows[0]["text"] == "alpha"
    assert rows[0]["metadata"]["start_offset"] == 0


def test_process_text_normal_path_stringifies_custom_chunk_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker()

    class CustomChunk:
        text = "attribute text"
        metadata = {"start_offset": 99, "copied_from_attribute": True}

        def __str__(self) -> str:
            return "custom object text"

    def fake_chunk_text(*args, **kwargs):
        return [CustomChunk()]

    def forbidden_metadata_path(*args, **kwargs):
        raise AssertionError("normal process_text path must not call chunk_text_with_metadata")

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)
    monkeypatch.setattr(chunker, "chunk_text_with_metadata", forbidden_metadata_path)

    rows = chunker.process_text("alpha beta", options={"method": "words", "max_size": 10, "overlap": 0})

    assert rows[0]["text"] == "custom object text"
    assert "copied_from_attribute" not in rows[0]["metadata"]
    assert rows[0]["metadata"]["chunk_index"] == 1
    assert rows[0]["metadata"]["chunk_method"] == "words"


def test_process_text_multi_level_fallback_uses_chunk_text_and_clamps_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[tuple[str, str]] = []
    text = "First paragraph.\n\nSecond paragraph."

    def fake_chunk_text_with_metadata(segment, *args, **kwargs):
        calls.append(("chunk_text_with_metadata", segment))
        raise ChunkingError("force fallback")

    def fake_chunk_text(segment, *args, **kwargs):
        calls.append(("chunk_text", segment))
        return [f"{segment} beyond the paragraph span"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_chunk_text_with_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    rows = chunker.process_text(
        text,
        options={"method": "words", "max_size": 10, "overlap": 0, "multi_level": True},
    )

    assert [name for name, _segment in calls] == [
        "chunk_text_with_metadata",
        "chunk_text",
        "chunk_text_with_metadata",
        "chunk_text",
    ]
    first_segment = calls[0][1]
    second_segment = calls[2][1]
    assert [row["metadata"]["paragraph_index"] for row in rows] == [0, 1]
    assert rows[0]["metadata"]["start_offset"] == 0
    assert rows[0]["metadata"]["end_offset"] == len(first_segment)
    assert rows[0]["metadata"]["end_offset"] <= text.index("Second")
    assert rows[1]["metadata"]["start_offset"] == text.index(second_segment)
    assert rows[1]["metadata"]["end_offset"] == rows[1]["metadata"]["start_offset"] + len(second_segment)
    assert rows[1]["metadata"]["end_offset"] <= len(text)


def test_process_text_hierarchical_template_path_uses_instance_flat_chunker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[dict] = []

    def fake_chunk_text_hierarchical_flat(*args, **kwargs):
        calls.append(kwargs)
        return [{"text": "Title", "metadata": {"start_offset": 0, "end_offset": 5}}]

    def forbidden_normal_path(*args, **kwargs):
        raise AssertionError("hierarchical process_text path must not call chunk_text")

    monkeypatch.setattr(chunker, "chunk_text_hierarchical_flat", fake_chunk_text_hierarchical_flat)
    monkeypatch.setattr(chunker, "chunk_text", forbidden_normal_path)

    template = {"levels": [{"name": "heading", "pattern": r"^# .+"}]}
    rows = chunker.process_text(
        "# Title\n\nBody text.",
        options={"method": "words", "max_size": 20, "overlap": 0, "hierarchical_template": template},
    )

    assert rows[0]["text"] == "Title"
    assert calls == [
        {
            "text": "# Title\n\nBody text.",
            "method": "words",
            "max_size": 20,
            "overlap": 0,
            "language": "en",
            "template": template,
            "method_options": {},
        }
    ]


def test_process_text_frontmatter_offsets_and_timecode_map_are_original_coordinates() -> None:
    chunker = Chunker()
    frontmatter = '{"meta": "x", "__tldw_frontmatter__": true}\n\n'
    body = "Body text."
    payload = frontmatter + body
    timecode_map = [
        {
            "start_offset": len(frontmatter),
            "end_offset": len(payload),
            "start_time": 12.0,
            "end_time": 18.0,
        }
    ]

    rows = chunker.process_text(
        payload,
        options={
            "method": "words",
            "max_size": 10,
            "overlap": 0,
            "hierarchical": True,
            "timecode_map": timecode_map,
        },
    )

    assert rows
    expected_metadata = {
        "start_offset": len(frontmatter),
        "end_offset": len(payload),
        "start_time": 12.0,
        "end_time": 18.0,
        "initial_document_json_metadata": {"meta": "x"},
    }
    metadata = rows[0]["metadata"]
    for key, value in expected_metadata.items():
        assert metadata[key] == value
    assert rows[0]["text"] == body


def test_process_text_string_false_frontmatter_option_remains_truthy() -> None:
    chunker = Chunker()
    payload = '{"meta": "x", "__tldw_frontmatter__": true}\nBody text.'

    rows = chunker.process_text(payload, options={"enable_frontmatter_parsing": "false"})

    assert rows
    assert rows[0]["metadata"]["initial_document_json_metadata"] == {"meta": "x"}
    assert rows[0]["text"].startswith("Body")


def test_process_text_string_false_hierarchical_option_remains_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()

    def forbidden_hierarchical_path(*args, **kwargs):
        raise AssertionError("string 'false' hierarchical option must not enable hierarchical mode")

    monkeypatch.setattr(chunker, "chunk_text_hierarchical_flat", forbidden_hierarchical_path)

    rows = chunker.process_text(
        "# Title\n\nBody text.",
        options={"method": "words", "max_size": 50, "overlap": 0, "hierarchical": "false"},
    )

    assert rows
    assert all("paragraph_kind" not in row["metadata"] for row in rows)
    assert all("ancestry_titles" not in row["metadata"] for row in rows)


def test_process_text_tokenizer_override_reaches_chunk_text_without_mutating_cached_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    original_chunk_text = chunker.chunk_text
    calls: list[dict] = []

    def spy_chunk_text(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_chunk_text(*args, **kwargs)

    monkeypatch.setattr(chunker, "chunk_text", spy_chunk_text)

    rows = chunker.process_text(
        "one two three four five six seven eight nine ten",
        options={"method": "tokens", "max_size": 5},
        tokenizer_name_or_path="test-tokenizer",
    )

    token_strategy = chunker.get_strategy("tokens")
    assert rows
    assert calls[0]["tokenizer_name_or_path"] == "test-tokenizer"
    assert getattr(token_strategy, "tokenizer_name", None) != "test-tokenizer"


def test_process_text_preserves_explicit_zero_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker()
    calls: list[dict] = []

    def fake_chunk_text(*args, **kwargs):
        calls.append(dict(kwargs))
        return ["alpha beta"]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    rows = chunker.process_text("alpha beta", options={"method": "words", "max_size": 10, "overlap": 0})

    assert calls[0]["overlap"] == 0
    assert rows[0]["metadata"]["overlap"] == 0
    assert rows[0]["metadata"]["overlap_setting"] == 0


def test_process_text_clamps_negative_overlap_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker()
    calls: list[dict] = []

    def fake_chunk_text(*args, **kwargs):
        calls.append(dict(kwargs))
        return ["alpha beta"]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    rows = chunker.process_text("alpha beta", options={"method": "words", "max_size": 10, "overlap": -5})

    assert calls[0]["overlap"] == 0
    assert rows[0]["metadata"]["overlap"] == 0
    assert rows[0]["metadata"]["overlap_setting"] == 0


def test_process_text_invalid_max_size_raises_invalid_input_error() -> None:
    with pytest.raises(InvalidInputError):
        Chunker().process_text("alpha beta", options={"method": "words", "max_size": "not-an-int"})


def test_process_text_invalid_non_string_input_raises_invalid_input_error() -> None:
    with pytest.raises(InvalidInputError):
        Chunker().process_text(
            ChunkResult(
                text="alpha",
                metadata=ChunkMetadata(index=0, start_char=0, end_char=5, word_count=1),
            )
        )  # type: ignore[arg-type]


def test_process_text_invalid_input_increments_process_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.Chunking.chunker as chunker_module

    calls: list[tuple[str, dict | None]] = []

    def fake_increment_counter(name, labels=None):
        calls.append((name, labels))

    monkeypatch.setattr(chunker_module, "increment_counter", fake_increment_counter)

    with pytest.raises(InvalidInputError):
        Chunker().process_text(None)

    assert ("chunker_process_total", {"component": "chunker", "op": "process_text"}) in calls
