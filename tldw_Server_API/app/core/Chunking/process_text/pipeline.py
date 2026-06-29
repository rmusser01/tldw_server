"""Orchestration for the internal ``Chunker.process_text`` pipeline."""

from __future__ import annotations

import time
from contextlib import suppress
from dataclasses import replace
from typing import Any

from tldw_Server_API.app.core.Chunking.error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS
from tldw_Server_API.app.core.Chunking.exceptions import InvalidInputError
from tldw_Server_API.app.core.Chunking.llm_context import llm_override_scope
from tldw_Server_API.app.core.Chunking.process_text.dispatch import dispatch_chunks
from tldw_Server_API.app.core.Chunking.process_text.metadata import (
    copy_chunks_for_finalization,
    finalize_chunks,
    restore_prefix_offsets_for_finalization,
)
from tldw_Server_API.app.core.Chunking.process_text.models import (
    ProcessTextContext,
    TelemetryHooks,
)
from tldw_Server_API.app.core.Chunking.process_text.options import resolve_process_options
from tldw_Server_API.app.core.Chunking.process_text.preparation import (
    _parse_frontmatter,
    _prepare_frontmatter_options,
    extract_header,
)


class ProcessTextPipeline:
    """Run text preparation, option resolution, dispatch, and metadata finalization."""

    def __init__(self, context: ProcessTextContext, telemetry: TelemetryHooks) -> None:
        """Store the chunker context and telemetry adapter used by this pipeline."""
        self._context = context
        self._telemetry = telemetry

    def run(
        self,
        text: str,
        options: dict[str, Any] | None = None,
        *,
        tokenizer_name_or_path: str | None = None,
        llm_call_func: Any = None,
        llm_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """End-to-end processing: optional frontmatter extraction, chunking, normalization."""
        overall_start = time.perf_counter()
        labels = {"component": "chunker", "op": "process_text"}
        self._telemetry.increment_counter("chunker_process_total", labels=labels)
        if text is None or not isinstance(text, str):
            raise InvalidInputError(f"Expected string input, got {type(text).__name__}")

        prepared, frontmatter_enabled, sentinel_key = _prepare_frontmatter_options(
            text,
            options,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
        fm_start = time.perf_counter()
        prepared = _parse_frontmatter(
            prepared,
            frontmatter_enabled=frontmatter_enabled,
            sentinel_key=sentinel_key,
        )
        opts = prepared.options
        processed_text = prepared.processed_text
        self._telemetry.observe_histogram(
            "chunker_frontmatter_duration_seconds",
            time.perf_counter() - fm_start,
            labels=labels,
        )

        self._context._enforce_text_size(processed_text, source="process_text")

        hdr_start = time.perf_counter()
        prepared = extract_header(prepared)
        opts = prepared.options
        processed_text = prepared.processed_text
        self._telemetry.observe_histogram(
            "chunker_header_extract_seconds",
            time.perf_counter() - hdr_start,
            labels=labels,
        )

        resolved = resolve_process_options(self._context, processed_text, opts)
        method = resolved.method
        language = resolved.language
        hierarchical = resolved.hierarchical
        hier_template = resolved.hier_template
        multi_level = resolved.multi_level

        chunk_start = time.perf_counter()
        with llm_override_scope(self._context, llm_call_func, llm_config):
            dispatched_chunks = dispatch_chunks(self._context, processed_text, resolved)
        finalization_chunks = copy_chunks_for_finalization(dispatched_chunks)
        self._telemetry.observe_histogram(
            "chunker_chunking_duration_seconds",
            time.perf_counter() - chunk_start,
            labels=labels,
        )

        prepared_for_finalization = prepared
        if prepared.prefix_offset:
            finalization_chunks = restore_prefix_offsets_for_finalization(
                finalization_chunks,
                prepared.prefix_offset,
            )
            prepared_for_finalization = replace(prepared, prefix_offset=0)

        norm_start = time.perf_counter()
        out = finalize_chunks(
            original_text=text,
            chunks=finalization_chunks,
            prepared=prepared_for_finalization,
            resolved=resolved,
        )
        self._telemetry.observe_histogram(
            "chunker_normalization_seconds",
            time.perf_counter() - norm_start,
            labels=labels,
        )

        total_bytes = sum(len(chunk["text"]) for chunk in out)
        self._telemetry.set_gauge("chunker_last_chunk_count", float(len(out)), labels=labels)
        self._telemetry.observe_histogram("chunker_output_bytes", float(total_bytes), labels=labels)
        self._telemetry.observe_histogram("chunker_input_bytes", float(len(text)), labels=labels)
        self._telemetry.observe_histogram(
            "chunker_process_total_seconds",
            time.perf_counter() - overall_start,
            labels={
                **labels,
                "method": method,
                "hierarchical": str(bool(hierarchical or hier_template)).lower(),
            },
        )
        try:
            with self._telemetry.start_span("chunker.process_text"):
                self._telemetry.set_span_attribute("chunk.method", method)
                self._telemetry.set_span_attribute("chunk.lang", language)
                self._telemetry.set_span_attribute("chunk.hierarchical", bool(hierarchical or hier_template))
                self._telemetry.set_span_attribute("chunk.multi_level", multi_level)
                self._telemetry.set_span_attribute("chunk.count", len(out))
                self._telemetry.add_span_event("chunker.completed")
        except CHUNKER_NONCRITICAL_EXCEPTIONS as exc:
            with suppress(CHUNKER_NONCRITICAL_EXCEPTIONS):
                self._telemetry.record_span_exception(exc, escaped=False)

        return out
