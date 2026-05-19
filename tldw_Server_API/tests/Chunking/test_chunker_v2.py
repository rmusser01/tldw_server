# test_chunker_v2.py
"""
Direct unit tests for V2 chunking implementation.
Tests the new modular chunker and strategy pattern.
"""

import asyncio
import re
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from tldw_Server_API.app.core.Chunking import (
    Chunker,
    create_chunker,
    ChunkingMethod,
    ChunkResult,
    ChunkMetadata,
    ChunkingError,
    InvalidInputError,
    InvalidChunkingMethodError,
    DEFAULT_CHUNK_OPTIONS,
)
from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


class TestV2Chunker:
    """Test the main V2 Chunker class."""

    def test_chunker_initialization(self):
        """Test that Chunker initializes correctly."""
        chunker = Chunker()
        assert chunker is not None
        assert chunker.config is not None
        # Use public API to check available methods
        assert len(chunker.get_available_methods()) > 0

    def test_chunker_with_custom_config(self):
        """Test Chunker with custom configuration."""
        config = ChunkerConfig(
            default_method=ChunkingMethod.SENTENCES, default_max_size=100, default_overlap=10, language="fr"
        )
        chunker = Chunker(config=config)
        assert chunker.config.default_method == ChunkingMethod.SENTENCES
        assert chunker.config.default_max_size == 100
        assert chunker.config.default_overlap == 10
        assert chunker.config.language == "fr"

    def test_chunk_text_returns_strings(self):
        """Test that chunk_text returns list of strings."""
        chunker = Chunker()
        text = "This is a test. This is another sentence. And a third one."
        chunks = chunker.chunk_text(text, method="sentences", max_size=2)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert len(chunks) > 0

    def test_chunk_text_with_metadata(self):
        """Test that chunk_text_with_metadata returns ChunkResult objects."""
        chunker = Chunker()
        text = "This is a test. This is another sentence. And a third one."
        chunks = chunker.chunk_text_with_metadata(text, method="sentences", max_size=2)

        assert isinstance(chunks, list)
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(isinstance(chunk.metadata, ChunkMetadata) for chunk in chunks)
        assert len(chunks) > 0

    def test_chunk_text_with_metadata_preserves_original_whitespace(self):
        """Returned chunk text should match the original span exactly."""
        chunker = Chunker()
        text = "Hello  world\nThis   is   a test."

        chunks = chunker.chunk_text_with_metadata(
            text,
            method="words",
            max_size=3,
            overlap=1,
        )
        assert chunks  # sanity

        for chunk in chunks:
            md = chunk.metadata
            assert isinstance(md.start_char, int) and isinstance(md.end_char, int)
            original_slice = text[md.start_char : md.end_char]
            assert chunk.text == original_slice

    def test_process_text_multi_level_maps_whitespace_offsets(self):
        """multi_level path should emit text/offsets aligned with the source."""
        chunker = Chunker()
        text = "Hello  world\nThis   is   a test.\n\nAnother   paragraph here."
        options = {"method": "words", "max_size": 3, "overlap": 1, "multi_level": True}

        rows = chunker.process_text(text, options=options)
        assert rows

        for row in rows:
            md = row.get("metadata", {})
            start = md.get("start_offset")
            end = md.get("end_offset")
            assert isinstance(start, int) and isinstance(end, int), md
            assert 0 <= start <= end <= len(text)
            assert row.get("text") == text[start:end]

    def test_process_text_multi_level_includes_headers(self):
        """multi_level chunking should retain header spans in the output."""
        chunker = Chunker()
        text = "# Heading\n\nFirst paragraph sentence one.\nSecond sentence.\n\nAnother block."
        rows = chunker.process_text(
            text,
            options={
                "method": "words",
                "max_size": 20,
                "multi_level": True,
            },
        )
        assert rows
        headers = [row for row in rows if row["metadata"].get("paragraph_kind") == "header_atx"]
        assert headers, "Expected header chunk to be present"
        assert headers[0]["text"].strip().startswith("# Heading")

    def test_flatten_hierarchical_no_space_language_no_extra_space(self):
        """Flattening should not inject spaces for no-space languages."""
        chunker = Chunker()
        tree = {
            "method": "structure_aware",
            "max_size": 2,
            "overlap": 0,
            "root": {
                "kind": "root",
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": None,
                        "children": [
                            {
                                "kind": "paragraph",
                                "chunks": [
                                    {
                                        "text": "你好",
                                        "metadata": {
                                            "language": "zh",
                                            "paragraph_kind": "paragraph",
                                            "start_offset": 0,
                                            "end_offset": 2,
                                        },
                                    }
                                ],
                                "children": [],
                            },
                            {
                                "kind": "paragraph",
                                "chunks": [
                                    {
                                        "text": "世界",
                                        "metadata": {
                                            "language": "zh",
                                            "paragraph_kind": "paragraph",
                                            "start_offset": 2,
                                            "end_offset": 4,
                                        },
                                    }
                                ],
                                "children": [],
                            },
                        ],
                    }
                ],
            },
        }

        rows = chunker.flatten_hierarchical(tree)
        assert len(rows) == 1
        assert rows[0]["text"] == "你好世界"

    def test_hierarchical_respects_method_options(self):
        """Hierarchical chunking should propagate strategy options."""
        chunker = Chunker()
        text = "# Heading\n\none two three four five six seven\n"
        chunks = chunker.chunk_text_hierarchical_flat(
            text,
            method="words",
            max_size=3,
            method_options={"min_chunk_size": 5},
        )
        paragraph_chunks = [c for c in chunks if c["metadata"].get("paragraph_kind") == "paragraph"]
        assert paragraph_chunks
        counts = [len(c["text"].split()) for c in paragraph_chunks]
        assert max(counts) >= 5

    def test_hierarchical_template_bold_subsection_boundary(self):
        """Template-driven bold_subsection boundaries should create nested sections."""
        chunker = Chunker()
        text = "Intro paragraph.\n\n**Bold Heading**\n\nBody paragraph."
        template = {
            "boundaries": [
                {"kind": "bold_subsection", "pattern": r"^\s*\*\*[^*]+\*\*\s*$", "flags": "m"}
            ]
        }

        chunks = chunker.chunk_text_hierarchical_flat(
            text,
            method="words",
            max_size=50,
            overlap=0,
            template=template,
        )
        assert chunks
        # Expect the bold heading itself to be tagged and the body to carry its section path
        assert any(c.get("metadata", {}).get("paragraph_kind") == "bold_subsection" for c in chunks)
        assert any("Bold Heading" in (c.get("metadata", {}).get("section_path") or "") for c in chunks)

    def test_hierarchical_preserves_raw_bidi_override(self):
        """Hierarchical output defaults to sanitized text, with optional raw fidelity."""
        chunker = Chunker()
        text = "Alpha\u202eBeta Gamma"

        plain_chunks = chunker.chunk_text(text, method="words", max_size=2, overlap=0, language="en")
        hier_chunks_sanitized = chunker.chunk_text_hierarchical_flat(
            text,
            method="words",
            max_size=2,
            overlap=0,
            language="en",
        )
        hier_chunks_raw = chunker.chunk_text_hierarchical_flat(
            text,
            method="words",
            max_size=2,
            overlap=0,
            language="en",
            method_options={"sanitize_output": False},
        )

        assert all("\u202e" not in ch for ch in plain_chunks)
        assert all("\u202e" not in row["text"] for row in hier_chunks_sanitized)
        assert any("\u202e" in row["text"] for row in hier_chunks_raw)

    def test_hierarchical_flat_includes_chunk_type(self):
        """Flattened hierarchical chunks should include a normalized chunk_type."""
        chunker = Chunker()
        text = "# Heading\n\nParagraph content.\n\n- list item"
        chunks = chunker.chunk_text_hierarchical_flat(text, method="sentences", max_size=20, overlap=0)

        assert chunks
        for chunk in chunks:
            md = chunk.get("metadata", {})
            chunk_type = md.get("chunk_type")
            assert isinstance(chunk_type, str) and chunk_type

    def test_code_mode_ast_forces_ast_strategy_even_without_language_hint(self):
        """Explicit code_mode='ast' should route to the AST strategy regardless of language hints."""
        chunker = Chunker()
        code_sample = "def foo():\n    return 42\n"
        with patch.object(chunker, "get_strategy", wraps=chunker.get_strategy) as spy:
            chunker.chunk_text(
                code_sample,
                method="code",
                language="en",
                code_mode="ast",
                max_size=128,
            )
        assert any(call.args and call.args[0] == "code_ast" for call in spy.call_args_list)

    def test_hierarchical_code_mode_ast_routes_child_chunks(self):
        """Hierarchical mode must preserve code_mode routing for nested chunking."""
        chunker = Chunker()
        code_sample = "# Heading\n\ndef foo():\n    return 1\n"
        with patch.object(chunker, "get_strategy", wraps=chunker.get_strategy) as spy:
            chunker.chunk_text_hierarchical_flat(
                code_sample,
                method="code",
                language="en",
                method_options={"code_mode": "ast"},
            )
        assert any(call.args and call.args[0] == "code_ast" for call in spy.call_args_list)

    def test_process_text_tokenizer_override(self):
        """tokenizer_name_or_path should use per-call strategy without mutating cached tokens."""
        chunker = Chunker()
        text = "one two three four five six seven eight nine ten"
        chunker.process_text(
            text,
            options={"method": "tokens", "max_size": 5},
            tokenizer_name_or_path="test-tokenizer",
        )
        token_strategy = chunker.get_strategy("tokens")
        assert getattr(token_strategy, "tokenizer_name", None) != "test-tokenizer"

    def test_process_text_rejects_non_string_input(self):
        """process_text should raise on non-string inputs."""
        chunker = Chunker()
        with pytest.raises(InvalidInputError):
            chunker.process_text(None)

    def test_chunk_text_with_metadata_ignores_whitespace_only_input(self):
        """Whitespace-only payloads should return an empty list just like chunk_text."""
        chunker = Chunker()
        result = chunker.chunk_text_with_metadata(" \n\t\r ", method="words", max_size=10)
        assert result == []

    def test_invalid_method_raises_error(self):
        """Test that invalid chunking method raises error."""
        chunker = Chunker()
        with pytest.raises(InvalidChunkingMethodError):
            chunker.chunk_text("test", method="invalid_method")

    def test_all_methods_available(self):
        """Test that all expected methods are available."""
        chunker = Chunker()
        expected_methods = [
            "words",
            "sentences",
            "paragraphs",
            "tokens",
            "semantic",
            "json",
            "xml",
            "ebook_chapters",
            "rolling_summarize",
        ]

        available = set(chunker.get_available_methods())
        for method in expected_methods:
            # Should be advertised by the chunker
            assert method in available

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        chunker = Chunker()
        # V2 returns empty list for empty text rather than raising error
        result = chunker.chunk_text("", method="words")
        assert result == []

    def test_chunk_text_generator(self):
        """Test the generator method for memory efficiency."""
        chunker = Chunker()
        text = " ".join(["word"] * 1000)  # Large text

        generator = chunker.chunk_text_generator(text, method="words", max_size=10)
        assert hasattr(generator, "__iter__")

        chunks = list(generator)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_process_text_respects_max_size_after_frontmatter(self):
        """process_text should trim headers before enforcing size limits."""
        config = ChunkerConfig(max_text_size=64)
        chunker = Chunker(config=config)
        frontmatter = '{"meta": "x"}\n'
        body_ok = "a" * (config.max_text_size - len(frontmatter) - 1)
        result = chunker.process_text(frontmatter + body_ok)
        assert result and isinstance(result, list)
        with pytest.raises(InvalidInputError):
            chunker.process_text(frontmatter + body_ok + "b" * (len(frontmatter) + 2))

    def test_chunk_cache_includes_llm_signature(self):
        """Cache should miss when LLM hook/config changes and hit when restored."""
        config = ChunkerConfig(default_method=ChunkingMethod.WORDS, enable_cache=True, cache_size=4)
        chunker = Chunker(config=config)
        text = "llm cache sanity text"

        class StubStrategy:
            def __init__(self, parent):
                self.parent = parent
                self.language = "en"
                self.calls = 0

            def chunk(self, text, max_size, overlap, **options):

                self.calls += 1
                fn = getattr(self.parent, "llm_call_func", None)
                cfg = getattr(self.parent, "llm_config", {}) or {}
                marker = fn(cfg) if callable(fn) else "no-fn"
                return [f"{marker}|{cfg.get('variant')}"]

        stub = StubStrategy(chunker)
        chunker._strategies["words"] = stub
        chunker._strategy_factories["words"] = lambda: stub

        def llm_stub(cfg=None):

            cfg = cfg or {}
            return f"run-{cfg.get('variant')}"

        chunker.llm_call_func = llm_stub
        chunker.llm_config = {"variant": 1}

        first = chunker.chunk_text(text, method="words", max_size=10, overlap=0)
        assert first == ["run-1|1"]
        assert stub.calls == 1

        chunker.llm_config = {"variant": 2}
        second = chunker.chunk_text(text, method="words", max_size=10, overlap=0)
        assert second == ["run-2|2"]
        assert stub.calls == 2  # cache miss due to new config

        chunker.llm_config = {"variant": 1}
        third = chunker.chunk_text(text, method="words", max_size=10, overlap=0)
        assert third == first
        assert stub.calls == 2  # cache hit reuses previous computation

    def test_process_text_keeps_leading_json_without_frontmatter_option(self):
        """Leading JSON documents must remain intact unless frontmatter parsing is enabled."""
        chunker = Chunker()
        payload = '{"first": 1}\n{"second": 2}\n'

        rows = chunker.process_text(payload)
        assert rows

        combined_text = " ".join(row.get("text", "") for row in rows)
        assert '"first": 1' in combined_text
        assert '"second": 2' in combined_text
        for row in rows:
            metadata = row.get("metadata", {})
            assert "initial_document_json_metadata" not in metadata

    def test_process_text_extracts_frontmatter_with_sentinel(self):
        """Frontmatter extraction should activate only with the sentinel and option."""
        chunker = Chunker()
        payload = '{"meta": "x", "__tldw_frontmatter__": true}\n{"second": 2}\n'

        rows = chunker.process_text(payload)
        assert rows
        primary = rows[0]
        assert primary["text"].strip() == '{"second": 2}'
        assert primary["metadata"]["initial_document_json_metadata"] == {"meta": "x"}

    def test_process_text_custom_frontmatter_sentinel(self):
        """Custom frontmatter sentinel keys must be honored when provided."""
        chunker = Chunker()
        payload = '{"meta": "y", "__custom_frontmatter__": true}\n{"third": 3}\n'
        options = {"frontmatter_sentinel_key": "__custom_frontmatter__"}

        rows = chunker.process_text(payload, options=options)
        assert rows
        primary = rows[0]
        assert primary["text"].strip() == '{"third": 3}'
        assert primary["metadata"]["initial_document_json_metadata"] == {"meta": "y"}

    def test_process_text_frontmatter_offsets_use_original_text(self):
        """Offsets from process_text should refer to the original input even after frontmatter removal."""
        chunker = Chunker()
        frontmatter = '{"meta": "x", "__tldw_frontmatter__": true}\n'
        body = "Hello world."
        payload = frontmatter + body
        timecode_map = [
            {
                "start_offset": len(frontmatter),
                "end_offset": len(payload),
                "start_time": 0.0,
                "end_time": 5.0,
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
        row = rows[0]
        md = row.get("metadata", {})
        assert md.get("start_offset") == len(frontmatter)
        assert row.get("text") == payload[md["start_offset"] : md["end_offset"]]
        assert md.get("start_time") == 0.0
        assert md.get("end_time") == 5.0

    def test_process_text_parses_multiline_frontmatter(self):
        """Frontmatter parsing should handle nested, pretty-printed JSON blocks."""
        chunker = Chunker()
        frontmatter = (
            "{\n" '  "__tldw_frontmatter__": true,\n' '  "meta": {"nested": 1},\n' '  "tags": ["a", "b"]\n' "}\n"
        )
        body = "# Heading\n\nBody text continues here."
        rows = chunker.process_text(frontmatter + body)
        assert rows
        meta = rows[0]["metadata"]
        assert meta.get("initial_document_json_metadata") == {"meta": {"nested": 1}, "tags": ["a", "b"]}

    def test_process_text_can_disable_frontmatter_parsing(self):
        """Frontmatter parsing can be explicitly disabled even when sentinel is present."""
        chunker = Chunker()
        payload = '{"meta": "z", "__tldw_frontmatter__": true}\n{"fourth": 4}\n'

        rows = chunker.process_text(payload, options={"enable_frontmatter_parsing": False})
        assert rows
        combined_text = " ".join(row.get("text", "") for row in rows)
        assert '"meta": "z"' in combined_text
        assert '"fourth": 4' in combined_text
        for row in rows:
            assert "initial_document_json_metadata" not in row.get("metadata", {})

    def test_process_text_forwards_method_options(self):
        """process_text should pass user strategy options to chunking strategies."""
        chunker = Chunker()
        text = "one two three four five six seven"
        rows = chunker.process_text(
            text,
            options={
                "method": "words",
                "max_size": 3,
                "min_chunk_size": 5,
            },
        )
        assert rows
        max_words = max(len(row["text"].split()) for row in rows)
        assert max_words >= 5

    def test_chunk_file_stream_respects_encoding(self, tmp_path):
        """chunk_file_stream should honor the caller-provided encoding."""
        chunker = Chunker()
        content = "café monde\nligne suivante"
        file_path = tmp_path / "latin1.txt"
        file_path.write_bytes(content.encode("cp1252"))

        with pytest.raises(InvalidInputError):
            list(chunker.chunk_file_stream(file_path, method="words", max_size=10))

        chunks = list(chunker.chunk_file_stream(file_path, method="words", max_size=10, encoding="cp1252"))
        assert chunks
        reconstructed = " ".join(chunks)
        assert "café" in reconstructed

    def test_cache_put_metrics_include_reason(self, monkeypatch):
        """chunker_cache_put_total metrics must carry the required reason label."""
        config = ChunkerConfig(
            enable_cache=True,
            cache_size=4,
            min_text_length_to_cache=0,
            max_text_length_to_cache=1_000,
        )
        chunker = Chunker(config=config)

        calls = []

        def fake_increment_counter(name, labels=None):

            calls.append((name, labels or {}))

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chunking.chunker.increment_counter",
            fake_increment_counter,
        )

        chunker.chunk_text("one two three four five", method="words", max_size=2, overlap=0)

        stored = [
            labels for name, labels in calls if name == "chunker_cache_put_total" and labels.get("result") == "stored"
        ]
        assert stored, f"Expected stored metric call, saw {calls}"
        assert all("reason" in lbl for lbl in stored)

    def test_process_text_reports_effective_method(self):
        """process_text metadata should reflect the actual strategy used."""
        chunker = Chunker()
        code_snippet = "def foo():\n    return 42\n"
        rows = chunker.process_text(code_snippet, options={"method": "code", "language": "python"})
        assert rows
        methods = {row["metadata"].get("chunk_method") for row in rows}
        assert methods == {"code_ast"}

    def test_process_text_string_false_does_not_enable_hierarchical(self):
        """Loose string booleans should not accidentally enable hierarchical mode."""
        chunker = Chunker()
        text = "# Title\n\nPara one.\n\nPara two."

        rows = chunker.process_text(
            text,
            options={"method": "words", "max_size": 50, "overlap": 0, "hierarchical": "false"},
        )

        assert rows
        assert all("paragraph_kind" not in row["metadata"] for row in rows)
        assert all("ancestry_titles" not in row["metadata"] for row in rows)

    def test_process_text_uses_tracing_helper_api_correctly(self, monkeypatch):
        """process_text should use the current-span helper API without recording TypeErrors."""
        import tldw_Server_API.app.core.Chunking.chunker as chunker_module

        span_attributes = []
        span_events = []
        recorded_exceptions = []

        class _DummySpanContext:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        def fake_start_span(name, **kwargs):
            return _DummySpanContext()

        def fake_set_span_attribute(key, value):
            span_attributes.append((key, value))

        def fake_add_span_event(name, attributes=None):
            span_events.append((name, attributes))

        def fake_record_span_exception(exception, escaped=True):
            recorded_exceptions.append((exception, escaped))

        monkeypatch.setattr(chunker_module, "start_span", fake_start_span)
        monkeypatch.setattr(chunker_module, "set_span_attribute", fake_set_span_attribute)
        monkeypatch.setattr(chunker_module, "add_span_event", fake_add_span_event)
        monkeypatch.setattr(chunker_module, "record_span_exception", fake_record_span_exception)

        rows = Chunker().process_text("One sentence. Another sentence.")

        assert rows
        assert ("chunk.method", "words") in span_attributes
        assert any(event_name == "chunker.completed" for event_name, _attrs in span_events)
        assert recorded_exceptions == []

    def test_chunk_file_stream_avoids_partial_tokens(self, tmp_path):
        """Streaming chunking should not emit truncated word fragments."""
        text = " ".join(f"word{i}" for i in range(1, 21))
        file_path = tmp_path / "stream_source.txt"
        file_path.write_text(text, encoding="utf-8")

        chunker = Chunker()
        streamed = list(
            chunker.chunk_file_stream(
                file_path,
                method="words",
                max_size=5,
                overlap=2,
                buffer_size=20,  # keep buffers small to force multiple passes
            )
        )

        original_tokens = set(text.split())
        assert streamed  # sanity: got output
        for chunk in streamed:
            for token in chunk.split():
                assert token in original_tokens, f"Unexpected token fragment {token!r} in chunk {chunk!r}"

    def test_chunk_file_stream_preserves_word_boundaries(self, tmp_path):
        """Streaming word chunks must not fuse adjacent tokens without whitespace."""
        text = " ".join(f"word{i}" for i in range(1, 16))
        file_path = tmp_path / "stream_boundaries_source.txt"
        file_path.write_text(text, encoding="utf-8")

        chunker = Chunker()
        streamed = list(
            chunker.chunk_file_stream(
                file_path,
                method="words",
                max_size=3,
                overlap=1,
                buffer_size=12,
            )
        )
        assert streamed, "Expected streamed output"

        combined = " ".join(streamed)
        # If boundary handling regresses, patterns like word2word3 will appear.
        assert re.search(r"word\\d+word\\d+", combined) is None

    def test_chunk_file_stream_respects_word_sized_chunks(self, tmp_path):
        """Streaming chunking should accumulate enough characters for word-sized chunks."""
        chunker = Chunker()
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        file_path = tmp_path / "stream_words_source.txt"
        file_path.write_text(text, encoding="utf-8")

        chunks = list(
            chunker.chunk_file_stream(
                file_path,
                method="words",
                max_size=50,
                overlap=0,
                buffer_size=64,
            )
        )
        assert chunks
        first_chunk_words = len(chunks[0].split())
        assert first_chunk_words >= 40

    def test_process_text_honors_zero_overlap_and_rejects_zero_max_size(self):
        """process_text should keep explicit zero overlap and reject zero max_size."""
        chunker = Chunker()
        text = " ".join(f"word{i}" for i in range(1, 25))

        chunks = chunker.process_text(
            text,
            options={
                "method": "words",
                "max_size": 8,
                "overlap": 0,
            },
        )

        assert chunks, "Expected chunks with zero overlap configuration"
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            assert metadata.get("overlap") == 0
            assert metadata.get("max_size") == 8

        with pytest.raises(InvalidInputError):
            chunker.process_text(
                text,
                options={
                    "method": "words",
                    "max_size": 0,
                },
            )

    def test_chunk_text_enforces_byte_sized_limit(self):
        """Multibyte characters should count against max_text_size in bytes."""
        config = ChunkerConfig(max_text_size=5)
        chunker = Chunker(config=config)
        payload = "aa😀"  # len(text)==3 but UTF-8 bytes == 6
        with pytest.raises(InvalidInputError):
            chunker.chunk_text(payload, method="words", max_size=10, overlap=0)

    def test_chunk_text_accepts_enum_method(self):
        """chunk_text should accept ChunkingMethod enum inputs."""
        chunker = Chunker()
        text = "This is a short test sentence."
        chunks = chunker.chunk_text(text, method=ChunkingMethod.WORDS, max_size=5, overlap=0)
        assert chunks
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestWordsStrategy:
    """Test the words chunking strategy."""

    def test_words_basic_chunking(self):
        """Test basic word-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy()
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = strategy.chunk(text, max_size=5, overlap=2)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Check overlap
        first_chunk_words = chunks[0].split()
        second_chunk_words = chunks[1].split()
        assert first_chunk_words[-2:] == second_chunk_words[:2]  # 2 word overlap

    def test_words_no_overlap(self):
        """Test word chunking without overlap."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy()
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = strategy.chunk(text, max_size=5, overlap=0)

        assert len(chunks) == 4  # 20 words / 5 words per chunk
        # Verify no overlap
        for i in range(len(chunks) - 1):
            assert chunks[i].split()[-1] != chunks[i + 1].split()[0]

    def test_words_with_metadata(self):
        """Test word chunking with metadata."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy()
        text = "One two three four five six seven eight nine ten"
        chunks = strategy.chunk_with_metadata(text, max_size=3, overlap=1)

        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert chunks[0].metadata.word_count == 3
        assert chunks[0].metadata.method == "words"
        assert chunks[0].metadata.index == 0

    def test_words_metadata_preserves_offsets_with_whitespace(self):
        """Chunk text should preserve original spacing for source-aligned spans."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy()
        text = "Alpha  beta\tgamma\n\ndelta epsilon"
        chunks = strategy.chunk_with_metadata(text, max_size=2, overlap=0)

        assert len(chunks) >= 2
        first = chunks[0]
        second = chunks[1]

        first_slice = text[first.metadata.start_char : first.metadata.end_char]
        assert first.text == first_slice
        assert "  " in first.text

        second_slice = text[second.metadata.start_char : second.metadata.end_char]
        assert "gamma" in second_slice
        assert "delta" in second_slice
        assert second.text == second_slice

    def test_thai_tokenizer_fallback_preserves_newlines(self):
        """Fallback Thai tokenization should keep explicit newlines intact."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy(language="th")
        text = "กแรก\nกสอง"
        chunks = strategy.chunk(text, max_size=10, overlap=0)
        assert chunks
        assert text in chunks


class TestSentencesStrategy:
    """Test the sentences chunking strategy."""

    def test_sentences_basic_chunking(self):
        """Test basic sentence-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.sentences import SentenceChunkingStrategy

        strategy = SentenceChunkingStrategy()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = strategy.chunk(text, max_size=2, overlap=1)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert "First sentence. Second sentence." in chunks[0]

    def test_sentences_handles_various_punctuation(self):
        """Test handling of various sentence endings."""
        from tldw_Server_API.app.core.Chunking.strategies.sentences import SentenceChunkingStrategy

        strategy = SentenceChunkingStrategy()
        text = "Question? Exclamation! Statement. Another?"
        chunks = strategy.chunk(text, max_size=2, overlap=0)

        assert len(chunks) == 2
        assert "Question? Exclamation!" in chunks[0]
        assert "Statement. Another?" in chunks[1]

    def test_sentences_metadata_preserves_offsets_with_whitespace(self):
        """Sentence chunks should preserve original spacing for source-aligned spans."""
        from tldw_Server_API.app.core.Chunking.strategies.sentences import SentenceChunkingStrategy

        strategy = SentenceChunkingStrategy()
        text = "First sentence.\n\n   Second sentence!  Third sentence?\nFourth sentence."
        chunks = strategy.chunk_with_metadata(text, max_size=2, overlap=1)

        assert len(chunks) >= 2
        first = chunks[0]
        first_slice = text[first.metadata.start_char : first.metadata.end_char]
        assert first_slice.startswith("First sentence.")
        assert "Second sentence!" in first_slice
        assert first.text == first_slice

        second = chunks[1]
        second_slice = text[second.metadata.start_char : second.metadata.end_char]
        assert "Third sentence?" in second_slice
        assert second.text == second_slice


class TestParagraphsStrategy:
    """Test the paragraphs chunking strategy."""

    def test_paragraphs_basic_chunking(self):
        """Test basic paragraph-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.paragraphs import ParagraphChunkingStrategy

        strategy = ParagraphChunkingStrategy()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph."
        chunks = strategy.chunk(text, max_size=2, overlap=1)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert "First paragraph.\n\nSecond paragraph." in chunks[0]

    def test_paragraphs_single_paragraph(self):
        """Test handling of text without paragraph breaks."""
        from tldw_Server_API.app.core.Chunking.strategies.paragraphs import ParagraphChunkingStrategy

        strategy = ParagraphChunkingStrategy()
        text = "This is all one paragraph without any breaks."
        chunks = strategy.chunk(text, max_size=1, overlap=0)

        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_paragraphs_with_metadata(self):
        """Test paragraph chunking with metadata."""
        from tldw_Server_API.app.core.Chunking.strategies.paragraphs import ParagraphChunkingStrategy

        strategy = ParagraphChunkingStrategy()
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = strategy.chunk_with_metadata(text, max_size=2, overlap=0)

        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert chunks[0].metadata.method == "paragraphs"
        assert chunks[0].metadata.options is not None
        assert "paragraph_count" in chunks[0].metadata.options


class TestTokensStrategy:
    """Test the tokens chunking strategy."""

    def test_tokens_basic_chunking(self):
        """Test basic token-based chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.tokens import TokenChunkingStrategy

        # Test with actual tokenizer or skip if not available
        try:
            strategy = TokenChunkingStrategy()
            text = "Some text to tokenize for testing purposes"
            chunks = strategy.chunk(text, max_size=5, overlap=2)

            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
        except ImportError:
            # Skip if transformers not available
            pytest.skip("transformers library not available")

    def test_tokens_fallback_clamps_minimum_chunk_size(self):
        """Fallback tokenization should still emit chunks for very small max_size."""
        from tldw_Server_API.app.core.Chunking.strategies.tokens import TokenChunkingStrategy, FallbackTokenizer

        strategy = TokenChunkingStrategy()
        # Force fallback mode regardless of available libraries
        strategy._tokenizer = FallbackTokenizer(strategy.tokenizer_name)
        text = "one two three four"

        chunks = strategy.chunk(text, max_size=1, overlap=0)

        assert chunks, "Fallback tokenization should return at least one chunk"
        assert all(chunk.strip() for chunk in chunks)

    def test_tokens_preserve_leading_indentation_when_chunking_mid_block(self):
        """Token chunks must retain leading whitespace to keep code formatting intact."""
        from tldw_Server_API.app.core.Chunking.strategies.tokens import TokenChunkingStrategy

        strategy = TokenChunkingStrategy()
        text = "\n".join(
            [
                "def foo():",
                "    x = 1",
                "    y = 2",
                "    z = 3",
                "    w = 4",
                "    return x + y + z + w",
            ]
        )

        chunks = strategy.chunk(text, max_size=20, overlap=0)

        assert len(chunks) >= 2
        second_chunk = chunks[1]
        assert second_chunk.lstrip().startswith("z = 3")
        assert second_chunk != second_chunk.lstrip()


class TestEbookChaptersStrategy:
    """Test the ebook chapters chunking strategy."""

    def test_ebook_chapters_basic(self):
        """Test basic chapter detection and chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy

        strategy = EbookChapterChunkingStrategy()
        text = """Chapter 1: Introduction
        Some content here.

        Chapter 2: Main Content
        More content here.

        Chapter 3: Conclusion
        Final content."""

        chunks = strategy.chunk(text, max_size=1000)  # Large size to keep chapters intact

        assert len(chunks) == 3
        assert "Chapter 1" in chunks[0]
        assert "Chapter 2" in chunks[1]


class TestChunkerMetrics:
    """Ensure chunker-specific metrics are registered and populated."""

    def test_chunker_cache_metrics_registered_and_incremented(self):
        """Chunker cache metrics should exist and capture miss/hit/store events."""
        from tldw_Server_API.app.core.Chunking.base import ChunkerConfig
        from tldw_Server_API.app.core.Chunking.chunker import Chunker
        from tldw_Server_API.app.core.Metrics import get_metrics_registry

        registry = get_metrics_registry()

        assert "chunker_cache_get_total" in registry.metrics
        assert "chunker_cache_put_total" in registry.metrics

        registry.values["chunker_cache_get_total"].clear()
        registry.values["chunker_cache_put_total"].clear()

        cfg = ChunkerConfig(enable_cache=True, cache_size=2)
        chunker = Chunker(config=cfg)
        text = " ".join(f"word{i}" for i in range(50))

        first_chunks = chunker.chunk_text(text, method="words", max_size=10, overlap=2)
        second_chunks = chunker.chunk_text(text, method="words", max_size=10, overlap=2)

        get_events = list(registry.values["chunker_cache_get_total"])
        put_events = list(registry.values["chunker_cache_put_total"])

        assert any(e.labels.get("result") == "miss" for e in get_events)
        assert any(e.labels.get("result") == "hit" for e in get_events)
        assert any(e.labels.get("result") == "stored" for e in put_events)
        assert first_chunks == second_chunks

    def test_ebook_no_chapters(self):
        """Test handling of text without chapter markers."""
        from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy

        strategy = EbookChapterChunkingStrategy()
        text = "This is text without any chapter markers. " * 20
        chunks = strategy.chunk(text, max_size=50, overlap=10)

        assert len(chunks) > 1  # Should split by size
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_ebook_custom_pattern(self):
        """Test custom chapter pattern."""
        from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy

        strategy = EbookChapterChunkingStrategy()
        text = """Part 1: Beginning
        Content here.

        Part 2: Middle
        More content.

        Part 3: End
        Final content."""

        # Use custom pattern that matches "Part N:"
        chunks = strategy.chunk(text, max_size=1000, custom_chapter_pattern=r"Part \d+:")

        assert len(chunks) == 3
        assert "Part 1" in chunks[0]


class TestSemanticStrategy:
    """Test the semantic chunking strategy."""

    def test_semantic_basic_chunking(self):
        """Test basic semantic chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.semantic import SemanticChunkingStrategy

        # Test with actual model or skip if not available
        try:
            strategy = SemanticChunkingStrategy()
            text = "First sentence. Second sentence. Third sentence. Fourth sentence."
            chunks = strategy.chunk(text, max_size=2, overlap=0)

            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
        except (ImportError, RuntimeError):
            # Skip if sentence-transformers not available or model can't load
            pytest.skip("sentence-transformers library or model not available")


class TestJSONStrategy:
    """Test the JSON chunking strategy."""

    def test_json_list_chunking(self):
        """Test chunking of JSON lists."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy

        strategy = JSONChunkingStrategy()
        json_text = '[{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]'
        chunks = strategy.chunk(json_text, max_size=2, overlap=1)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        # Each chunk should be valid JSON
        import json

        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, list)

    def test_json_dict_chunking(self):
        """Test chunking of JSON objects."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy

        strategy = JSONChunkingStrategy()
        json_text = '{"key1": "value1", "key2": "value2", "key3": "value3"}'
        chunks = strategy.chunk(json_text, max_size=2, overlap=0)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        # Each chunk should be valid JSON
        import json

        for chunk in chunks:
            parsed = json.loads(chunk)
            assert isinstance(parsed, dict)

    def test_json_invalid_input(self):
        """Test handling of invalid JSON."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy

        strategy = JSONChunkingStrategy()
        with pytest.raises(InvalidInputError):
            strategy.chunk("not valid json {", max_size=2)


class TestXMLStrategy:
    """Test the XML chunking strategy."""

    def test_xml_basic_chunking(self):
        """Test basic XML chunking."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import XMLChunkingStrategy

        strategy = XMLChunkingStrategy()
        xml_text = """<root>
            <item>Content 1</item>
            <item>Content 2</item>
            <item>Content 3</item>
        </root>"""
        chunks = strategy.chunk(xml_text, max_size=50, overlap=0)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_xml_invalid_input(self):
        """Test handling of invalid XML."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import XMLChunkingStrategy

        strategy = XMLChunkingStrategy()
        with pytest.raises(InvalidInputError):
            strategy.chunk("not valid xml <", max_size=2)


class TestRollingSummarizeStrategy:
    """Test the rolling summarize strategy."""

    def test_rolling_summarize_without_llm(self):
        """Test that rolling summarize works without LLM (returns raw chunks)."""
        from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import RollingSummarizeStrategy

        strategy = RollingSummarizeStrategy()
        # Without LLM, should return raw chunks (not summarized)
        result = strategy.chunk("Some text to summarize", max_size=100)
        assert isinstance(result, list)
        # Should return the text as-is since it's shorter than max_size
        assert len(result) >= 1

    @patch("tldw_Server_API.app.core.Chunking.strategies.rolling_summarize.RollingSummarizeStrategy._call_llm")
    def test_rolling_summarize_with_llm(self, mock_llm):
        """Test rolling summarize with mocked LLM."""
        from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import RollingSummarizeStrategy

        # Mock LLM responses
        mock_llm.return_value = "Summarized content"

        # Create strategy with mock LLM function
        mock_llm_func = Mock(return_value="Summary")
        strategy = RollingSummarizeStrategy(llm_call_func=mock_llm_func)

        text = "This is a long text. " * 100
        chunks = strategy.chunk(text, max_size=50, overlap=10)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_rolling_summarize_chunk_with_metadata_without_llm(self):
        """chunk_with_metadata should return source-aligned spans when no LLM is used."""
        from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import RollingSummarizeStrategy

        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
        strategy = RollingSummarizeStrategy()
        results = strategy.chunk_with_metadata(text, max_size=2, overlap=1)

        assert results
        for res in results:
            s = res.metadata.start_char
            e = res.metadata.end_char
            assert isinstance(s, int) and isinstance(e, int)
            assert 0 <= s <= e <= len(text)
            assert res.text == text[s:e]

    @patch("tldw_Server_API.app.core.Chunking.strategies.rolling_summarize.RollingSummarizeStrategy._call_llm")
    def test_rolling_summarize_chunk_with_metadata_with_llm(self, mock_llm):
        """chunk_with_metadata should map summaries to source spans when LLM is used."""
        from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import RollingSummarizeStrategy

        mock_llm.return_value = "Summarized content"
        strategy = RollingSummarizeStrategy(llm_call_func=Mock(return_value="Summary"))
        text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4."
        segments = strategy._build_segments_with_spans(text, max_size=2, overlap=1)
        results = strategy.chunk_with_metadata(text, max_size=2, overlap=1)

        assert len(results) == len(segments)
        for res, (_seg, seg_start, seg_end, _count) in zip(results, segments):
            assert res.text == "Summarized content"
            assert res.metadata.start_char == seg_start
            assert res.metadata.end_char == seg_end
            assert text[seg_start:seg_end] != ""


def test_structure_aware_chunk_with_metadata_spans():
    from tldw_Server_API.app.core.Chunking import Chunker

    text = "# Title\n\nParagraph one.\n\n- item\n"
    ck = Chunker()
    results = ck.chunk_text_with_metadata(text, method="structure_aware", max_size=2, overlap=0, language="en")

    assert results
    prev_end = -1
    for res in results:
        s = res.metadata.start_char
        e = res.metadata.end_char
        assert isinstance(s, int) and isinstance(e, int)
        assert 0 <= s <= e <= len(text)
        assert e >= prev_end
        prev_end = e
        assert text[s:e].strip() != ""


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_improved_chunking_process(self):
        """Test the backward compatibility improved_chunking_process function."""
        from tldw_Server_API.app.core.Chunking import improved_chunking_process

        text = "Test text. Another sentence. Third sentence."
        options = {"method": "sentences", "max_size": 2, "overlap": 1}

        chunks = improved_chunking_process(text, options)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)

    def test_chunk_for_embedding(self):
        """Test the backward compatibility chunk_for_embedding function."""
        from tldw_Server_API.app.core.Chunking import chunk_for_embedding

        text = "Test text for embedding. " * 10
        chunks = chunk_for_embedding(text, "test_file.txt", max_size=50)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)

    def test_default_options_exported(self):
        """Test that DEFAULT_CHUNK_OPTIONS is properly exported."""
        from tldw_Server_API.app.core.Chunking import DEFAULT_CHUNK_OPTIONS

        assert isinstance(DEFAULT_CHUNK_OPTIONS, dict)
        assert "method" in DEFAULT_CHUNK_OPTIONS
        assert "max_size" in DEFAULT_CHUNK_OPTIONS
        assert "overlap" in DEFAULT_CHUNK_OPTIONS


class TestErrorHandling:
    """Test error handling across the module."""

    def test_empty_text_handling_strategies(self):
        """Test that strategies handle empty text appropriately."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy()
        # V2 strategies return empty list for empty text
        result = strategy.chunk("", max_size=10)
        assert result == []

    def test_invalid_method_error(self):
        """Test that invalid method raises appropriate error."""
        chunker = Chunker()

        with pytest.raises(InvalidChunkingMethodError) as exc_info:
            chunker.chunk_text("test text", method="nonexistent")

        assert "unknown" in str(exc_info.value).lower()

    def test_method_name_case_insensitivity(self):
        """Chunker should accept method names regardless of casing."""
        chunker = Chunker()
        text = "First sentence. Second sentence. Third sentence."

        sentence_chunks = chunker.chunk_text(text, method="Sentences", max_size=1)
        assert sentence_chunks, "Expected chunks when using mixed-case 'Sentences' method"

        word_chunks_with_meta = chunker.chunk_text_with_metadata(text, method="WORDS", max_size=3)
        assert word_chunks_with_meta, "Expected metadata chunks when using upper-case 'WORDS' method"
        assert word_chunks_with_meta[0].metadata.method == "words"

    def test_invalid_parameters(self):
        """Test that invalid parameters are handled appropriately."""
        from tldw_Server_API.app.core.Chunking.strategies.words import WordChunkingStrategy

        strategy = WordChunkingStrategy()

        # Negative max_size
        with pytest.raises((InvalidInputError, ValueError)):
            strategy.chunk("test text", max_size=-1)

        # Overlap larger than max_size - V2 adjusts this automatically
        result = strategy.chunk("test text", max_size=10, overlap=15)
        # Should still work, overlap adjusted
        assert isinstance(result, list)


class TestPerformance:
    """Test performance-related features."""

    def test_generator_memory_efficiency(self):
        """Test that generator method is memory efficient."""
        chunker = Chunker()
        large_text = "word " * 10000  # Large text

        # Generator should not create all chunks at once
        generator = chunker.chunk_text_generator(large_text, method="words", max_size=100)

        # Get first chunk without generating all
        first_chunk = next(generator)
        assert isinstance(first_chunk, str)
        assert len(first_chunk.split()) <= 100

    def test_caching_disabled_by_default(self):
        """Test that caching can be disabled."""
        config = ChunkerConfig(enable_cache=False)
        chunker = Chunker(config=config)

        assert chunker._cache is None

        # Should work without cache
        text = "Test text for caching"
        chunks1 = chunker.chunk_text(text, method="words", max_size=5)
        chunks2 = chunker.chunk_text(text, method="words", max_size=5)

        assert chunks1 == chunks2


class TestAsyncChunkerConcurrency:
    """Ensure AsyncChunker maintains per-task language selection."""

    @pytest.mark.asyncio
    async def test_async_chunker_preserves_language_per_task(self):
        from tldw_Server_API.app.core.Chunking.async_chunker import AsyncChunker

        english_text = "alpha beta gamma delta epsilon zeta eta"
        japanese_text = "猫は可愛いです犬も可愛いです"

        async with AsyncChunker() as chunker:
            for _ in range(3):
                english_chunks, japanese_chunks = await asyncio.gather(
                    chunker.chunk_text(
                        english_text,
                        method="words",
                        max_size=3,
                        overlap=0,
                        language="en",
                    ),
                    chunker.chunk_text(
                        japanese_text,
                        method="words",
                        max_size=3,
                        overlap=0,
                        language="ja",
                    ),
                )

                assert english_chunks, "Expected English chunks from async chunker"
                assert any(" " in chunk for chunk in english_chunks), "English chunks lost whitespace separation"

                assert japanese_chunks, "Expected Japanese chunks from async chunker"
                assert " " not in "".join(japanese_chunks), "Japanese chunks should not contain inserted spaces"


def test_paragraph_chunk_with_metadata_offsets_match_source():
    """Paragraphs strategy should produce offsets that slice the original text exactly."""
    chunker = Chunker()
    # Leading/trailing/multiple blank lines and indents
    text = (
        "\n\n   First paragraph, with  leading spaces.  \n"
        "\n   \n\nSecond paragraph.\n\n\n"
        "  Third paragraph with trailing spaces.   \n   "
    )
    results = chunker.chunk_text_with_metadata(
        text,
        method="paragraphs",
        max_size=2,
        overlap=1,
    )
    assert results, "Expected paragraph chunks"
    for res in results:
        md = res.metadata
        assert isinstance(md.start_char, int) and isinstance(md.end_char, int)
        assert 0 <= md.start_char <= md.end_char <= len(text)
        # chunker aligns emitted text to original span
        assert res.text == text[md.start_char : md.end_char]


def test_hierarchical_tokens_offsets_map_to_source():
    """Hierarchical tokens path must map local spans to global offsets and preserve exact source slices."""
    chunker = Chunker()
    text = (
        "Header\n\n"
        "First block with tokens. Another line here.\n"
        "\n\n"
        "Second block content 123 and more words here.\n"
    )
    rows = chunker.chunk_text_hierarchical_flat(
        text,
        method="tokens",
        max_size=8,
        overlap=2,
        language="en",
    )
    assert rows, "Expected hierarchical tokens output"
    for row in rows:
        md = row.get("metadata", {})
        s = md.get("start_offset")
        e = md.get("end_offset")
        assert isinstance(s, int) and isinstance(e, int)
        assert 0 <= s <= e <= len(text)
        assert row.get("text") == text[s:e]


def test_paragraph_chunk_with_crlf_offsets_match_source():
    """Paragraphs with Windows CRLF line endings should produce accurate offsets."""
    chunker = Chunker()
    text = "\r\n\r\n  Para A line 1\r\nline 2  \r\n\r\n\r\n" "   Para B\r\n\r\n" "Para C with trailing spaces   \r\n"
    results = chunker.chunk_text_with_metadata(
        text,
        method="paragraphs",
        max_size=2,
        overlap=1,
    )
    assert results, "Expected paragraph chunks with CRLF input"
    for res in results:
        md = res.metadata
        assert isinstance(md.start_char, int) and isinstance(md.end_char, int)
        assert 0 <= md.start_char <= md.end_char <= len(text)
        assert res.text == text[md.start_char : md.end_char]


def test_words_chunk_with_metadata_preserves_offsets_japanese():
    """Japanese word chunking with metadata should slice exact source spans (no spaces inserted)."""
    chunker = Chunker()
    text = "猫は可愛いです。犬も可愛いです。ありがとうございます。"
    chunks = chunker.chunk_text_with_metadata(
        text,
        method="words",
        max_size=6,
        overlap=2,
        language="ja",
    )
    assert chunks, "Expected Japanese word chunks"
    for ch in chunks:
        s = ch.metadata.start_char
        e = ch.metadata.end_char
        assert 0 <= s <= e <= len(text)
        assert ch.text == text[s:e]


def test_sentences_chunk_with_metadata_no_space_thai():
    """Thai sentence chunking should not introduce spaces and should preserve offsets."""
    chunker = Chunker()
    text = "นี่คือการทดสอบ!ขอบคุณ?สวัสดี!"
    chunks = chunker.chunk_text_with_metadata(
        text,
        method="sentences",
        max_size=1,
        overlap=0,
        language="th",
    )
    assert chunks, "Expected Thai sentence chunks"
    for ch in chunks:
        s = ch.metadata.start_char
        e = ch.metadata.end_char
        assert 0 <= s <= e <= len(text)
        assert ch.text == text[s:e]
        # Ensure no spaces are introduced by joining
        assert " " not in ch.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
