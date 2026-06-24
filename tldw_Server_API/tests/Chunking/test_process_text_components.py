from __future__ import annotations

import inspect
from typing import Any

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking.constants import FRONTMATTER_SENTINEL_KEY
from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.exceptions import (
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
)
from tldw_Server_API.app.core.Chunking.llm_context import _LLM_UNSET, llm_override_scope
from tldw_Server_API.app.core.Chunking.option_utils import _coerce_bool_option
from tldw_Server_API.app.core.Chunking.process_text.preparation import (
    extract_header,
    prepare_frontmatter,
)
from tldw_Server_API.app.core.Chunking.process_text import models
from tldw_Server_API.app.core.Chunking.process_text.models import (
    NormalizedChunk,
    PreparedText,
    ProcessTextContext,
    ResolvedProcessOptions,
    TelemetryHooks,
)


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
    source = inspect.getsource(models)

    assert not hasattr(models, "Chunker")
    assert ".chunker" not in source
    assert "Chunking.chunker" not in source


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
