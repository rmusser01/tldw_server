from __future__ import annotations

import ast
import json
import time
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking import chunker as chunker_module
from tldw_Server_API.app.core.Chunking.base import ChunkMetadata, ChunkResult
from tldw_Server_API.app.core.Chunking.constants import FRONTMATTER_SENTINEL_KEY
from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.exceptions import (
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
)
from tldw_Server_API.app.core.Chunking.llm_context import _LLM_UNSET, llm_override_scope
from tldw_Server_API.app.core.Chunking.option_utils import _coerce_bool_option
from tldw_Server_API.app.core.Chunking.process_text import options as process_options
from tldw_Server_API.app.core.Chunking.process_text.options import (
    METHOD_OPTION_EXCLUDES,
    resolve_process_options,
)
from tldw_Server_API.app.core.Chunking.process_text import preparation
from tldw_Server_API.app.core.Chunking.process_text.preparation import (
    extract_header,
    prepare_frontmatter,
)
from tldw_Server_API.app.core.Chunking.process_text import dispatch as process_dispatch
from tldw_Server_API.app.core.Chunking.process_text.dispatch import dispatch_chunks
from tldw_Server_API.app.core.Chunking.process_text import metadata as process_metadata
from tldw_Server_API.app.core.Chunking.process_text.metadata import (
    copy_chunks_for_finalization,
    finalize_chunks,
    restore_prefix_offsets_for_finalization,
)
from tldw_Server_API.app.core.Chunking.process_text import pipeline as process_pipeline
from tldw_Server_API.app.core.Chunking.process_text import models
from tldw_Server_API.app.core.Chunking.process_text.models import (
    NormalizedChunk,
    PreparedText,
    ProcessTextContext,
    ResolvedProcessOptions,
    TelemetryHooks,
)


def _assert_no_chunker_imports(module: Any) -> None:
    assert not hasattr(module, "Chunker")

    module_path = Path(module.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "tldw_Server_API.app.core.Chunking.chunker"
                assert not alias.name.endswith(".chunker")
                assert alias.name != "chunker"
                assert alias.asname != "Chunker"
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            assert module_name != "tldw_Server_API.app.core.Chunking.chunker"
            assert not module_name.endswith(".chunker")
            assert not (node.level and module_name == "chunker")
            assert all(alias.name not in {"Chunker", "chunker"} for alias in node.names)


def _resolved_for_dispatch(**overrides: Any) -> ResolvedProcessOptions:
    values: dict[str, Any] = {
        "method": "words",
        "method_lower": "words",
        "max_size": 100,
        "overlap": 0,
        "language": "en",
        "adaptive": False,
        "hierarchical": False,
        "hier_template": None,
        "multi_level": False,
        "code_mode_for_method": None,
        "method_options_for_chunk": {},
    }
    values.update(overrides)
    return ResolvedProcessOptions(**values)


def _prepared_for_finalize(**overrides: Any) -> PreparedText:
    values: dict[str, Any] = {
        "original_text": "alpha beta",
        "processed_text": "alpha beta",
        "prefix_offset": 0,
        "json_meta": {},
        "header_text": "",
        "options": {},
    }
    values.update(overrides)
    return PreparedText(**values)


def test_llm_override_scope_restores_missing_attribute_after_exception() -> None:
    chunker = Chunker()
    assert not hasattr(chunker._thread_local, "llm_overrides")

    with pytest.raises(RuntimeError, match="inside scope"):
        with llm_override_scope(chunker, llm_call_func="call"):
            assert chunker._thread_local.llm_overrides == ("call", _LLM_UNSET)
            raise RuntimeError("inside scope")

    assert not hasattr(chunker._thread_local, "llm_overrides")


def test_llm_override_scope_restores_existing_override_tuple() -> None:
    chunker = Chunker()
    existing = ("previous-call", {"model": "old"})
    chunker._thread_local.llm_overrides = existing

    with llm_override_scope(chunker, llm_config={"model": "new"}):
        assert chunker._thread_local.llm_overrides == (_LLM_UNSET, {"model": "new"})

    assert chunker._thread_local.llm_overrides is existing


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        ("false", True, False),
        ("true", False, True),
        (" yes ", False, True),
        ("0", True, False),
        (None, True, True),
        (None, False, False),
        (True, False, True),
        (False, True, False),
        (1, False, True),
        (0, True, False),
    ],
)
def test_coerce_bool_option_matches_existing_loose_bool_behavior(
    value: Any,
    default: bool,
    expected: bool,
) -> None:
    assert _coerce_bool_option(value, default) is expected


def test_chunker_noncritical_exceptions_include_chunking_errors() -> None:
    assert ChunkingError in CHUNKER_NONCRITICAL_EXCEPTIONS
    assert InvalidChunkingMethodError in CHUNKER_NONCRITICAL_EXCEPTIONS
    assert InvalidInputError in CHUNKER_NONCRITICAL_EXCEPTIONS


def test_chunker_strategy_factory_still_raises_invalid_method_error() -> None:
    chunker = Chunker()

    with pytest.raises(InvalidChunkingMethodError, match="Unknown chunking method"):
        chunker.get_strategy("missing-method")


def test_process_text_internal_dataclasses_expose_expected_attributes() -> None:
    prepared = PreparedText(
        original_text="raw",
        processed_text="processed",
        prefix_offset=3,
        json_meta={"source": "frontmatter"},
        header_text="header",
        options={"method": "words"},
    )
    resolved = ResolvedProcessOptions(
        method="words",
        method_lower="words",
        max_size=100,
        overlap=10,
        language="en",
        adaptive=False,
        hierarchical=False,
        hier_template=None,
        multi_level=False,
        code_mode_for_method=None,
        method_options_for_chunk={"strip": True},
    )
    normalized = NormalizedChunk(text="chunk", metadata={"chunk_index": 1})
    hooks = TelemetryHooks(
        increment_counter=lambda *args, **kwargs: None,
        observe_histogram=lambda *args, **kwargs: None,
        set_gauge=lambda *args, **kwargs: None,
        start_span=lambda *args, **kwargs: None,
        set_span_attribute=lambda *args, **kwargs: None,
        add_span_event=lambda *args, **kwargs: None,
        record_span_exception=lambda *args, **kwargs: None,
    )

    assert prepared.original_text == "raw"
    assert prepared.options["method"] == "words"
    assert resolved.max_size == 100
    assert resolved.method_options_for_chunk == {"strip": True}
    assert resolved.align_text_to_source is True
    assert normalized.metadata["chunk_index"] == 1
    assert hooks.increment_counter("metric") is None


def test_process_text_context_protocol_accepts_chunker_shape() -> None:
    expected_members = {
        "config",
        "_thread_local",
        "_enforce_text_size",
        "_normalize_method_argument",
        "_resolve_method",
        "_compute_paragraph_spans",
        "chunk_text",
        "chunk_text_with_metadata",
        "chunk_text_hierarchical_flat",
    }

    assert expected_members.issubset(ProcessTextContext.__annotations__ | ProcessTextContext.__dict__.keys())
    assert ProcessTextContext.__annotations__["config"] == "ChunkerConfig"


def test_process_text_models_module_does_not_import_chunker() -> None:
    _assert_no_chunker_imports(models)


def test_process_text_preparation_module_does_not_import_chunker() -> None:
    _assert_no_chunker_imports(preparation)


def test_process_text_options_module_does_not_import_chunker() -> None:
    _assert_no_chunker_imports(process_options)


def test_process_text_dispatch_module_does_not_import_chunker() -> None:
    _assert_no_chunker_imports(process_dispatch)


def test_process_text_metadata_module_does_not_import_chunker() -> None:
    _assert_no_chunker_imports(process_metadata)


def test_process_text_pipeline_module_does_not_import_chunker() -> None:
    _assert_no_chunker_imports(process_pipeline)


@pytest.mark.parametrize(
    "source",
    [
        "from tldw_Server_API.app.core.Chunking import chunker\n",
        "from .. import chunker\n",
    ],
)
def test_process_text_import_boundary_rejects_chunker_import_alias(
    tmp_path: Path,
    source: str,
) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(source, encoding="utf-8")
    module = SimpleNamespace(__file__=str(module_path))

    with pytest.raises(AssertionError):
        _assert_no_chunker_imports(module)


def test_resolve_process_options_rejects_invalid_max_size() -> None:
    with pytest.raises(InvalidInputError, match="Invalid max_size value: bad"):
        resolve_process_options(Chunker(), "Body text", {"max_size": "bad"})


def test_resolve_process_options_rejects_nonpositive_max_size() -> None:
    with pytest.raises(InvalidInputError, match="max_size must be positive, got 0"):
        resolve_process_options(Chunker(), "Body text", {"max_size": 0})


def test_resolve_process_options_clamps_negative_overlap() -> None:
    resolved = resolve_process_options(Chunker(), "Body text", {"overlap": -5})

    assert resolved.overlap == 0


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        ("ภาษาไทย", "th"),
        ("これは日本語です", "ja"),
        ("Русский текст", "ru"),
    ],
)
def test_resolve_process_options_autodetects_script_languages(
    text: str,
    expected_language: str,
) -> None:
    resolved = resolve_process_options(Chunker(), text, {})

    assert resolved.language == expected_language


def test_resolve_process_options_default_language_detection_preserves_config_default() -> None:
    chunker = Chunker()

    resolved = resolve_process_options(chunker, "Plain English text", {})

    assert resolved.language == chunker.config.language


def test_resolve_process_options_excludes_process_only_options_and_keeps_tokenizer_overrides() -> None:
    opts: dict[str, Any] = {
        "method": "words",
        "max_size": 100,
        "overlap": 5,
        "language": "en",
        "hierarchical": False,
        "hierarchical_template": {"levels": []},
        "multi_level": False,
        "timecode_map": [],
        "enable_frontmatter_parsing": True,
        "frontmatter_sentinel_key": "sentinel",
        "adaptive": False,
        "base_adaptive_chunk_size": 100,
        "min_adaptive_chunk_size": 50,
        "max_adaptive_chunk_size": 200,
        "adaptive_overlap": False,
        "base_overlap": 5,
        "max_adaptive_overlap": 10,
        "code_mode": "ast",
        "align_text_to_source": True,
    }
    opts.update(
        {
            "custom_option": "kept",
            "tokenizer_name": "explicit-tokenizer",
            "tokenizer_name_or_path": "explicit-tokenizer-path",
        }
    )

    resolved = resolve_process_options(Chunker(), "Body text", opts)

    assert METHOD_OPTION_EXCLUDES == {
        "method",
        "max_size",
        "overlap",
        "language",
        "hierarchical",
        "hierarchical_template",
        "multi_level",
        "timecode_map",
        "enable_frontmatter_parsing",
        "frontmatter_sentinel_key",
        "adaptive",
        "base_adaptive_chunk_size",
        "min_adaptive_chunk_size",
        "max_adaptive_chunk_size",
        "adaptive_overlap",
        "base_overlap",
        "max_adaptive_overlap",
        "code_mode",
        "align_text_to_source",
    }
    assert resolved.method_options_for_chunk == {
        "custom_option": "kept",
        "tokenizer_name": "explicit-tokenizer",
        "tokenizer_name_or_path": "explicit-tokenizer-path",
    }
    assert resolved.align_text_to_source is True


def test_resolve_process_options_sets_align_text_to_source() -> None:
    resolved = resolve_process_options(
        Chunker(),
        "Body text",
        {"method": "words", "align_text_to_source": "false"},
    )

    assert resolved.align_text_to_source is False


@pytest.mark.parametrize(
    ("method", "expected_code_mode"),
    [
        ("code_ast", "ast"),
        ("code", "auto"),
    ],
)
def test_resolve_process_options_defaults_code_mode_for_code_methods(
    method: str,
    expected_code_mode: str,
) -> None:
    resolved = resolve_process_options(Chunker(), "def example():\n    return 1\n", {"method": method})

    assert resolved.code_mode_for_method == expected_code_mode
    assert resolved.method_options_for_chunk["code_mode"] == expected_code_mode


def test_resolve_process_options_adaptive_size_and_overlap() -> None:
    resolved = resolve_process_options(
        Chunker(),
        "x" * 20_000,
        {
            "method": "words",
            "max_size": 80,
            "overlap": 5,
            "adaptive": True,
            "base_adaptive_chunk_size": 100,
            "min_adaptive_chunk_size": 50,
            "max_adaptive_chunk_size": 200,
            "adaptive_overlap": True,
            "base_overlap": 10,
            "max_adaptive_overlap": 25,
        },
    )

    assert resolved.max_size == 140
    assert resolved.overlap == 25


def test_resolve_process_options_hierarchical_false_and_multi_level_exclusion() -> None:
    template = {"levels": [{"name": "heading"}]}

    resolved_without_hierarchy = resolve_process_options(
        Chunker(),
        "Paragraph one.\n\nParagraph two.",
        {"method": "words", "hierarchical": "false", "multi_level": True},
    )
    resolved_with_template = resolve_process_options(
        Chunker(),
        "Paragraph one.\n\nParagraph two.",
        {
            "method": "words",
            "hierarchical": "false",
            "hierarchical_template": template,
            "multi_level": True,
        },
    )

    assert resolved_without_hierarchy.hierarchical is False
    assert resolved_without_hierarchy.multi_level is True
    assert resolved_with_template.hierarchical is False
    assert resolved_with_template.hier_template == template
    assert resolved_with_template.multi_level is False


def test_prepare_frontmatter_extracts_default_sentinel_metadata() -> None:
    text = f'  {{"title": "Example", "{FRONTMATTER_SENTINEL_KEY}": true}}\n\r\nBody text'

    prepared = prepare_frontmatter(text, {"method": "words"}, tokenizer_name_or_path=None)

    assert prepared.original_text == text
    assert prepared.processed_text == "Body text"
    assert prepared.prefix_offset == len(text) - len("Body text")
    assert prepared.json_meta == {"title": "Example"}
    assert prepared.header_text == ""
    assert prepared.options == {"method": "words"}


def test_prepare_frontmatter_extracts_custom_sentinel_metadata() -> None:
    text = '{"title": "Custom", "custom_sentinel": 1}\nBody text'

    prepared = prepare_frontmatter(
        text,
        {"frontmatter_sentinel_key": "custom_sentinel"},
        tokenizer_name_or_path=None,
    )

    assert prepared.processed_text == "Body text"
    assert prepared.prefix_offset == len(text) - len("Body text")
    assert prepared.json_meta == {"title": "Custom"}
    assert "frontmatter_sentinel_key" not in prepared.options


def test_prepare_frontmatter_disabled_false_leaves_frontmatter_in_text() -> None:
    text = f'{{"title": "Example", "{FRONTMATTER_SENTINEL_KEY}": true}}\nBody text'

    prepared = prepare_frontmatter(
        text,
        {"enable_frontmatter_parsing": False},
        tokenizer_name_or_path=None,
    )

    assert prepared.processed_text == text
    assert prepared.prefix_offset == 0
    assert prepared.json_meta == {}
    assert "enable_frontmatter_parsing" not in prepared.options


def test_prepare_frontmatter_string_false_remains_enabled() -> None:
    text = f'{{"title": "Example", "{FRONTMATTER_SENTINEL_KEY}": true}}\nBody text'

    prepared = prepare_frontmatter(
        text,
        {"enable_frontmatter_parsing": "false"},
        tokenizer_name_or_path=None,
    )

    assert prepared.processed_text == "Body text"
    assert prepared.json_meta == {"title": "Example"}


def test_prepare_frontmatter_tokenizer_override_precedence() -> None:
    injected = prepare_frontmatter(
        "Body",
        {"method": "words"},
        tokenizer_name_or_path="fallback-tokenizer",
    )
    path_existing = prepare_frontmatter(
        "Body",
        {"tokenizer_name_or_path": "explicit-path"},
        tokenizer_name_or_path="fallback-tokenizer",
    )
    name_existing = prepare_frontmatter(
        "Body",
        {"tokenizer_name": "explicit-name"},
        tokenizer_name_or_path="fallback-tokenizer",
    )

    assert injected.options["tokenizer_name_or_path"] == "fallback-tokenizer"
    assert path_existing.options["tokenizer_name_or_path"] == "explicit-path"
    assert "tokenizer_name_or_path" not in name_existing.options
    assert name_existing.options["tokenizer_name"] == "explicit-name"


def test_extract_header_removes_legacy_transcription_header_and_updates_offset() -> None:
    header = "This text was transcribed using faster-whisper\nmodel: base\n\n"
    prepared = PreparedText(
        original_text=header + " \tBody text",
        processed_text=header + " \tBody text",
        prefix_offset=5,
        json_meta={"source": "frontmatter"},
        header_text="",
        options={"method": "words"},
    )

    extracted = extract_header(prepared)

    assert extracted is not prepared
    assert extracted.header_text == header
    assert extracted.processed_text == "Body text"
    assert extracted.prefix_offset == 5 + len(header) + 2
    assert extracted.json_meta == prepared.json_meta
    assert extracted.options == prepared.options


def test_prepare_frontmatter_malformed_leading_json_does_not_raise() -> None:
    text = f'{{"title": "Broken", "{FRONTMATTER_SENTINEL_KEY}": true\nBody text'

    prepared = prepare_frontmatter(text, None, tokenizer_name_or_path=None)

    assert prepared.processed_text == text
    assert prepared.prefix_offset == 0
    assert prepared.json_meta == {}


def test_process_text_frontmatter_metric_excludes_option_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, float] = {}
    real_prepare_options = process_pipeline._prepare_frontmatter_options

    def slow_prepare_options(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.05)
        return real_prepare_options(*args, **kwargs)

    def capture_histogram(name: str, value: float, **kwargs: Any) -> None:
        if name == "chunker_frontmatter_duration_seconds":
            observed[name] = value

    monkeypatch.setattr(process_pipeline, "_prepare_frontmatter_options", slow_prepare_options)
    monkeypatch.setattr(chunker_module, "observe_histogram", capture_histogram)

    rows = Chunker().process_text("Body text", options={"method": "words", "max_size": 100})

    assert rows[0]["text"] == "Body text"
    assert observed["chunker_frontmatter_duration_seconds"] < 0.03


def test_dispatch_chunks_normal_path_stringifies_custom_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    chunker = Chunker()

    class CustomChunk:
        text = "attribute text"
        metadata = {"copied": True}

        def __str__(self) -> str:
            return "custom object text"

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[Any]:
        return [CustomChunk()]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    chunks = dispatch_chunks(chunker, "alpha beta", _resolved_for_dispatch())

    assert chunks == [NormalizedChunk(text="custom object text", metadata={})]


def test_dispatch_chunks_normal_path_converts_json_metadata_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    metadata = {"start_offset": 0, "end_offset": 7}

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[Any]:
        return [{"json": {"value": "β"}, "metadata": metadata}]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    chunks = dispatch_chunks(chunker, "ignored", _resolved_for_dispatch())
    metadata["start_offset"] = 99

    assert chunks == [
        NormalizedChunk(
            text=json.dumps({"value": "β"}, ensure_ascii=False),
            metadata={"start_offset": 0, "end_offset": 7},
        )
    ]


def test_dispatch_chunks_normal_path_converts_text_metadata_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    metadata = {"start_offset": 2, "end_offset": 6}

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[Any]:
        return [{"text": "beta", "metadata": metadata}]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    chunks = dispatch_chunks(chunker, "alpha beta", _resolved_for_dispatch())
    metadata["end_offset"] = 99

    assert chunks == [NormalizedChunk(text="beta", metadata={"start_offset": 2, "end_offset": 6})]


def test_dispatch_chunks_normal_path_preserves_mapping_like_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    metadata = UserDict({"start_offset": 1, "end_offset": 5, "source": "mapping"})

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[Any]:
        return [{"text": "beta", "metadata": metadata}]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    chunks = dispatch_chunks(chunker, "alpha beta", _resolved_for_dispatch())
    metadata["source"] = "mutated"

    assert chunks == [
        NormalizedChunk(
            text="beta",
            metadata={"start_offset": 1, "end_offset": 5, "source": "mapping"},
        )
    ]


def test_dispatch_chunks_normal_path_propagates_unexpected_metadata_conversion_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()

    class BrokenMetadata:
        def __iter__(self):
            raise RuntimeError("strategy metadata conversion failed")

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[Any]:
        return [{"text": "beta", "metadata": BrokenMetadata()}]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    with pytest.raises(RuntimeError, match="strategy metadata conversion failed"):
        dispatch_chunks(chunker, "alpha beta", _resolved_for_dispatch())


def test_dispatch_chunks_hierarchical_path_uses_context_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[dict[str, Any]] = []
    template = {"levels": [{"name": "heading"}]}

    def fake_chunk_text_hierarchical_flat(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [{"text": "Heading", "metadata": {"start_offset": 0, "end_offset": 7}}]

    def forbidden_chunk_text(*args: Any, **kwargs: Any) -> list[Any]:
        raise AssertionError("hierarchical dispatch must not call chunk_text")

    monkeypatch.setattr(chunker, "chunk_text_hierarchical_flat", fake_chunk_text_hierarchical_flat)
    monkeypatch.setattr(chunker, "chunk_text", forbidden_chunk_text)

    chunks = dispatch_chunks(
        chunker,
        "# Heading",
        _resolved_for_dispatch(
            hierarchical=True,
            hier_template=template,
            max_size=50,
            overlap=5,
            method_options_for_chunk={"strip": True},
        ),
    )

    assert chunks == [NormalizedChunk(text="Heading", metadata={"start_offset": 0, "end_offset": 7})]
    assert calls == [
        {
            "text": "# Heading",
            "method": "words",
            "max_size": 50,
            "overlap": 5,
            "language": "en",
            "template": template,
            "method_options": {"strip": True},
        }
    ]


def test_dispatch_chunks_multi_level_metadata_result_becomes_dict_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    text = "prefix Alpha beta suffix"

    def fake_spans(*args: Any, **kwargs: Any) -> list[tuple[int, int, str]]:
        return [(7, 17, "paragraph")]

    def fake_chunk_text_with_metadata(*args: Any, **kwargs: Any) -> list[ChunkResult]:
        return [
            ChunkResult(
                text="lpha",
                metadata=ChunkMetadata(index=3, start_char=1, end_char=5, word_count=1),
            )
        ]

    monkeypatch.setattr(chunker, "_compute_paragraph_spans", fake_spans)
    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_chunk_text_with_metadata)

    chunks = dispatch_chunks(chunker, text, _resolved_for_dispatch(multi_level=True))

    assert chunks[0].text == "lpha"
    assert chunks[0].metadata["index"] == 3
    assert chunks[0].metadata["start_char"] == 8
    assert chunks[0].metadata["end_char"] == 12
    assert chunks[0].metadata["start_offset"] == 8
    assert chunks[0].metadata["end_offset"] == 12
    assert chunks[0].metadata["paragraph_index"] == 0
    assert chunks[0].metadata["paragraph_kind"] == "paragraph"
    assert chunks[0].metadata["multi_level"] is True


def test_dispatch_chunks_multi_level_uses_resolved_align_text_to_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[dict[str, Any]] = []

    def fake_spans(*args: Any, **kwargs: Any) -> list[tuple[int, int, str]]:
        return [(0, 10, "paragraph")]

    def fake_chunk_text_with_metadata(*args: Any, **kwargs: Any) -> list[ChunkResult]:
        calls.append(dict(kwargs))
        return [
            ChunkResult(
                text="Alpha",
                metadata=ChunkMetadata(index=0, start_char=0, end_char=5, word_count=1),
            )
        ]

    monkeypatch.setattr(chunker, "_compute_paragraph_spans", fake_spans)
    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_chunk_text_with_metadata)

    dispatch_chunks(
        chunker,
        "Alpha beta",
        _resolved_for_dispatch(multi_level=True, align_text_to_source=False),
    )

    assert calls[0]["align_text_to_source"] is False


def test_dispatch_chunks_multi_level_fallback_clamps_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[str] = []

    def fake_spans(*args: Any, **kwargs: Any) -> list[tuple[int, int, str]]:
        return [(0, 5, "paragraph")]

    def fake_chunk_text_with_metadata(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append("metadata")
        raise ChunkingError("force fallback")

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[str]:
        calls.append("normal")
        return ["First text that extends beyond the paragraph"]

    monkeypatch.setattr(chunker, "_compute_paragraph_spans", fake_spans)
    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_chunk_text_with_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    chunks = dispatch_chunks(chunker, "First paragraph", _resolved_for_dispatch(multi_level=True))

    assert calls == ["metadata", "normal"]
    assert chunks[0].text == "First text that extends beyond the paragraph"
    assert chunks[0].metadata["start_offset"] == 0
    assert chunks[0].metadata["end_offset"] == 5
    assert chunks[0].metadata["paragraph_index"] == 0


def test_copy_chunks_for_finalization_copies_metadata() -> None:
    original = NormalizedChunk(text="alpha", metadata={"start_offset": 0})

    copied = copy_chunks_for_finalization([original])
    copied[0].metadata["start_offset"] = 99

    assert original.metadata["start_offset"] == 0


def test_restore_prefix_offsets_for_finalization_does_not_mutate_input() -> None:
    original = NormalizedChunk(
        text="alpha",
        metadata={"start_offset": 1, "end_offset": 6, "start_char": 2, "end_char": 7},
    )

    restored = restore_prefix_offsets_for_finalization([original], 10)

    assert restored[0].metadata == {
        "start_offset": 11,
        "end_offset": 16,
        "start_char": 12,
        "end_char": 17,
    }
    assert original.metadata == {
        "start_offset": 1,
        "end_offset": 6,
        "start_char": 2,
        "end_char": 7,
    }


def test_process_text_restores_prefix_before_normalization_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_restore = process_pipeline.restore_prefix_offsets_for_finalization
    real_finalize = process_pipeline.finalize_chunks
    events: list[Any] = []

    def tracking_restore(*args: Any, **kwargs: Any) -> Any:
        events.append("restore")
        return real_restore(*args, **kwargs)

    def tracking_finalize(*args: Any, **kwargs: Any) -> Any:
        prepared = kwargs["prepared"]
        chunks = kwargs["chunks"]
        events.append(("finalize", prepared.prefix_offset, chunks[0].metadata["start_offset"]))
        return real_finalize(*args, **kwargs)

    def capture_histogram(name: str, value: float, **kwargs: Any) -> None:
        if name in {"chunker_chunking_duration_seconds", "chunker_normalization_seconds"}:
            events.append(name)

    monkeypatch.setattr(process_pipeline, "restore_prefix_offsets_for_finalization", tracking_restore)
    monkeypatch.setattr(process_pipeline, "finalize_chunks", tracking_finalize)
    monkeypatch.setattr(chunker_module, "observe_histogram", capture_histogram)

    body = "Body text."
    payload = '{"meta": "x", "__tldw_frontmatter__": true}\nBody text.'
    prefix_offset = len(payload) - len(body)
    chunker = Chunker()

    def fake_chunk_text(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"text": body, "metadata": {"start_offset": 0, "end_offset": len(body)}}]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    rows = chunker.process_text(payload, options={"method": "words", "max_size": 100})

    assert rows[0]["metadata"]["initial_document_json_metadata"] == {"meta": "x"}
    assert events.index("chunker_chunking_duration_seconds") < events.index("restore")
    assert events.index("restore") < events.index("chunker_normalization_seconds")
    finalize_event = ("finalize", 0, prefix_offset)
    assert finalize_event in events
    assert events.index(finalize_event) < events.index("chunker_normalization_seconds")


def test_finalize_chunks_restores_prefix_offset_to_all_offset_keys() -> None:
    rows = finalize_chunks(
        original_text="0123456789alpha beta",
        chunks=[
            NormalizedChunk(
                text="alpha",
                metadata={
                    "start_offset": 1,
                    "end_offset": 6,
                    "start_char": 2,
                    "end_char": 7,
                },
            )
        ],
        prepared=_prepared_for_finalize(prefix_offset=10),
        resolved=_resolved_for_dispatch(),
    )

    metadata = rows[0]["metadata"]
    assert metadata["start_offset"] == 11
    assert metadata["end_offset"] == 16
    assert metadata["start_char"] == 12
    assert metadata["end_char"] == 17


def test_finalize_chunks_preserves_strategy_metadata_defaults() -> None:
    rows = finalize_chunks(
        original_text="alpha",
        chunks=[
            NormalizedChunk(
                text="alpha",
                metadata={
                    "chunk_index": 99,
                    "total_chunks": 88,
                    "chunk_method": "strategy-method",
                    "max_size": 1234,
                    "overlap": 42,
                    "language": "strategy-language",
                    "adaptive_chunking_used": True,
                    "relative_position": 0.75,
                    "chunk_content_hash": "strategy-hash",
                    "origin": "strategy-origin",
                },
            )
        ],
        prepared=_prepared_for_finalize(),
        resolved=_resolved_for_dispatch(max_size=100, overlap=0),
    )

    metadata = rows[0]["metadata"]
    assert metadata["chunk_index"] == 99
    assert metadata["total_chunks"] == 88
    assert metadata["chunk_method"] == "strategy-method"
    assert metadata["max_size"] == 1234
    assert metadata["overlap"] == 42
    assert metadata["language"] == "strategy-language"
    assert metadata["adaptive_chunking_used"] is True
    assert metadata["relative_position"] == 0.75
    assert metadata["chunk_content_hash"] == "strategy-hash"
    assert metadata["origin"] == "strategy-origin"


def test_finalize_chunks_maps_missing_start_and_end_times() -> None:
    rows = finalize_chunks(
        original_text="x" * 100,
        chunks=[
            NormalizedChunk(
                text="middle",
                metadata={"start_offset": 25, "end_offset": 75},
            )
        ],
        prepared=_prepared_for_finalize(
            original_text="x" * 100,
            processed_text="x" * 100,
            options={
                "timecode_map": [
                    {"start_offset": 50, "end_offset": 100, "start_time": 5.0, "end_time": 10.0},
                    {"start_offset": 0, "end_offset": 50, "start_time": 0.0, "end_time": 5.0},
                ]
            },
        ),
        resolved=_resolved_for_dispatch(),
    )

    assert rows[0]["metadata"]["start_time"] == 2.5
    assert rows[0]["metadata"]["end_time"] == 7.5


def test_finalize_chunks_does_not_overwrite_existing_times() -> None:
    rows = finalize_chunks(
        original_text="x" * 100,
        chunks=[
            NormalizedChunk(
                text="middle",
                metadata={"start_offset": 25, "end_offset": 75, "start_time": 111.0, "end_time": 222.0},
            )
        ],
        prepared=_prepared_for_finalize(
            original_text="x" * 100,
            processed_text="x" * 100,
            options={
                "timecode_map": [
                    {"start_offset": 0, "end_offset": 100, "start_time": 0.0, "end_time": 10.0},
                ]
            },
        ),
        resolved=_resolved_for_dispatch(),
    )

    assert rows[0]["metadata"]["start_time"] == 111.0
    assert rows[0]["metadata"]["end_time"] == 222.0


def test_finalize_chunks_relative_position_uses_original_input_length() -> None:
    frontmatter = '{"meta": "x", "__tldw_frontmatter__": true}\n\n'
    body = "Body text"
    rows = finalize_chunks(
        original_text=frontmatter + body,
        chunks=[
            NormalizedChunk(
                text=body,
                metadata={"start_offset": 0, "end_offset": len(body)},
            )
        ],
        prepared=_prepared_for_finalize(
            original_text=frontmatter + body,
            processed_text=body,
            prefix_offset=len(frontmatter),
            json_meta={"meta": "x"},
        ),
        resolved=_resolved_for_dispatch(),
    )

    expected_midpoint = len(frontmatter) + (len(body) / 2.0)
    assert rows[0]["metadata"]["relative_position"] == expected_midpoint / len(frontmatter + body)
    assert rows[0]["metadata"]["initial_document_json_metadata"] == {"meta": "x"}


def test_finalize_chunks_adds_content_hash_for_ordinary_text() -> None:
    rows = finalize_chunks(
        original_text="alpha",
        chunks=[NormalizedChunk(text="alpha", metadata={})],
        prepared=_prepared_for_finalize(original_text="alpha", processed_text="alpha"),
        resolved=_resolved_for_dispatch(),
    )

    assert rows[0]["metadata"]["chunk_content_hash"] == "2c1743a391305fbf367df8e4f069f9f9"


def test_finalize_chunks_ignores_invalid_timecode_map_without_raising() -> None:
    rows = finalize_chunks(
        original_text="alpha beta",
        chunks=[
            NormalizedChunk(
                text="alpha",
                metadata={"start_offset": 0, "end_offset": 5},
            )
        ],
        prepared=_prepared_for_finalize(
            options={
                "timecode_map": [
                    "not-a-segment",
                    {"start_offset": "0", "end_offset": 5, "start_time": 0.0, "end_time": 1.0},
                    {"start_offset": 0, "end_offset": 5, "start_time": "0", "end_time": 1.0},
                ]
            },
        ),
        resolved=_resolved_for_dispatch(),
    )

    assert "start_time" not in rows[0]["metadata"]
    assert "end_time" not in rows[0]["metadata"]
