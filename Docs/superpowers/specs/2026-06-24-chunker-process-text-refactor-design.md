# Chunker process_text Refactor Design

Backlog: TASK-9935

## Purpose

`Chunker.process_text` has become the main end-to-end path for text preparation, option resolution, chunk dispatch, metadata normalization, timecode mapping, and metrics. The method is difficult to review and test because several responsibilities are interleaved in one large body.

This design extracts `process_text` into smaller internal components while preserving public behavior. The first implementation PR must not redesign the chunking API, change return shapes, or fix unrelated quirks.

## Goals

- Reduce `chunker.py` size and make `process_text` easier to understand.
- Preserve `Chunker.process_text(...)` as the public entry point.
- Preserve the returned `list[{"text": str, "metadata": dict}]` shape.
- Preserve current metadata keys, option aliases, timecode behavior, tracing/metric names, typo alias `processs_text`, and noncritical error suppression.
- Create testable internal boundaries for preparation, option resolution, dispatch, and metadata finalization.
- Keep the first refactor PR reviewable by moving behavior in stages and avoiding broad rewrites.

## Non-Goals

- No public API redesign.
- No changes to strategy implementations except where required to keep moved code working.
- No behavior fixes for questionable existing quirks unless existing tests already require them.
- No README/API documentation churn unless public behavior changes.
- No full telemetry-module extraction in the first PR.

## Current Shape

`Chunker.process_text` currently handles:

- input type validation
- shallow option copy and tokenizer override merge
- sentinel JSON frontmatter extraction
- configured text-size enforcement
- legacy transcription header extraction
- method, size, overlap, language, code-mode, adaptive, hierarchy, and strategy-option resolution
- per-call LLM override setup and restoration
- hierarchical, multi-level, and normal chunk dispatch
- normalization of strategy outputs into dictionaries
- prefix-offset restoration
- timecode mapping
- final metadata defaults
- content hashing
- metrics and tracing

This design keeps those semantics but splits the implementation into an internal `process_text` subsystem.

## Proposed Package

Create:

```text
tldw_Server_API/app/core/Chunking/process_text/
├── __init__.py
├── dispatch.py
├── metadata.py
├── models.py
├── options.py
├── pipeline.py
└── preparation.py
```

Move generic option helpers that are also useful outside this subsystem to a non-circular shared module, for example:

```text
tldw_Server_API/app/core/Chunking/option_utils.py
```

Move the current noncritical exception policy to a non-circular shared module, for example:

```text
tldw_Server_API/app/core/Chunking/error_policy.py
```

`chunker.py` and the new `process_text` package should import the same exception tuple from that module. Do not duplicate `_CHUNKER_NONCRITICAL_EXCEPTIONS` in multiple files, and do not import it from `chunker.py`.

Move the LLM override sentinel and override scope to a non-circular shared module, for example:

```text
tldw_Server_API/app/core/Chunking/llm_context.py
```

`chunker.py` and the new `process_text` package should import `_LLM_UNSET` and `llm_override_scope` from that shared module. Do not make general Chunker helpers such as `_sync_strategy_llm(...)`, `_get_effective_llm_hooks(...)`, or `_get_llm_signature(...)` import from the internal `process_text` package.

`Chunker` imports the pipeline. The new package must not import `Chunker`.

## Internal Models

Use small internal dataclasses:

- `PreparedText`
  - `original_text`
  - `processed_text`
  - `prefix_offset`
  - `json_meta`
  - `header_text`
  - copied `options`

- `ResolvedProcessOptions`
  - effective `method`
  - `method_lower`
  - `max_size`
  - `overlap`
  - `language`
  - `adaptive`
  - `hierarchical`
  - `hier_template`
  - `multi_level`
  - `code_mode_for_method`
  - `method_options_for_chunk`

- `NormalizedChunk`
  - `text`
  - `metadata`

Optionally add a small `TelemetryHooks` dataclass if passing metric/tracing helpers is cleaner than importing a wrapper:

- `TelemetryHooks`
  - `increment_counter`
  - `observe_histogram`
  - `set_gauge`
  - `start_span`
  - `set_span_attribute`
  - `add_span_event`
  - `record_span_exception`

Define a private `ProcessTextContext` protocol in `models.py`. It should include only the attributes and methods that the pipeline actually needs:

- `config`
- `_thread_local`
- `_enforce_text_size(...)`
- `_normalize_method_argument(...)`
- `_resolve_method(...)`
- `_compute_paragraph_spans(...)`
- `chunk_text(...)`
- `chunk_text_with_metadata(...)`
- `chunk_text_hierarchical_flat(...)`

`Chunker` can pass `self` to the pipeline, but the pipeline should be written against this protocol. This keeps the dependency boundary explicit and avoids circular imports.

Do not solve dependency access by importing from `chunker.py` inside the new package. The `_LLM_UNSET` sentinel and LLM override helper must live in a shared non-circular module outside the internal `process_text` package. Telemetry should be passed as `TelemetryHooks` or imported from a self-contained wrapper that does not import `Chunker`.

The noncritical exception tuple should also come from a shared non-circular module. Helpers that preserve suppressed behavior must use that same tuple unless a targeted behavior-change test justifies narrowing it.

## Public Entry Point

`Chunker.process_text(...)` remains the public API:

```python
def process_text(
    self,
    text: str,
    options: Optional[dict[str, Any]] = None,
    *,
    tokenizer_name_or_path: Optional[str] = None,
    llm_call_func: Optional[Any] = None,
    llm_config: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    return ProcessTextPipeline(self).run(
        text,
        options,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_func=llm_call_func,
        llm_config=llm_config,
    )
```

The implementation should prefer this shape: construct the pipeline with the context and pass the original method arguments through unchanged. If implementation discovers a need for a different constructor, the public `Chunker.process_text` wrapper must still remain a straight delegation with no embedded business logic.

## Data Flow

`ProcessTextPipeline.run(...)` owns sequencing and observability:

To preserve current metrics behavior, initialize the duration timer and labels and increment `chunker_process_total` before public input validation.

1. Validate input is a string.
2. Copy options and merge `tokenizer_name_or_path` into options using current precedence.
3. Extract sentinel JSON frontmatter and update `prefix_offset`.
4. Enforce configured text-size limits on post-frontmatter text.
5. Extract the legacy transcription header and update `prefix_offset`.
6. Resolve method, max size, overlap, language, code mode, adaptive settings, hierarchy flags, and method-specific options.
7. Dispatch chunks inside a per-call LLM override scope.
8. Restore prefix offsets to original input coordinates.
9. Finalize metadata, timecodes, content hashes, metrics, and tracing.

The key boundary is:

- dispatch produces `NormalizedChunk` values
- metadata finalization produces public output dictionaries
- dispatch does not know about final metadata defaults
- metadata finalization does not know how chunks were produced

## Preparation

`preparation.py` handles:

- shallow option copy
- tokenizer override merge
- sentinel JSON frontmatter extraction
- legacy transcription header extraction

Input type validation remains owned by `ProcessTextPipeline.run(...)`. Preparation helpers should not duplicate the public non-string validation path.

Size enforcement stays in the pipeline because the current order matters:

1. extract sentinel JSON frontmatter
2. enforce configured text size
3. extract legacy header

Noncritical frontmatter and header parse failures must continue to be suppressed.

Preserve the current `enable_frontmatter_parsing` coercion exactly: missing means enabled, otherwise the current raw `bool(value)` behavior applies. Do not replace it with `_coerce_bool_option` in the refactor PR; for example, a non-empty string such as `"false"` currently remains truthy.

## Option Resolution

`options.py` handles:

- method normalization and fallback to `config.default_method`
- `max_size` parsing and validation
- `overlap` parsing and negative-overlap clamping
- lightweight language detection for auto/detect/missing language
- `_resolve_method(...)` call through the context
- construction of method-specific options
- code mode resolution
- adaptive sizing and adaptive overlap
- hierarchy/template/multi-level flags

Move `_coerce_bool_option` to `option_utils.py` so extracted code does not import from `chunker.py`.

Move the current process-only method option exclusion set into a named constant and preserve it exactly unless a behavior-change test says otherwise:

- `method`
- `max_size`
- `overlap`
- `language`
- `hierarchical`
- `hierarchical_template`
- `multi_level`
- `timecode_map`
- `enable_frontmatter_parsing`
- `frontmatter_sentinel_key`
- `adaptive`
- `base_adaptive_chunk_size`
- `min_adaptive_chunk_size`
- `max_adaptive_chunk_size`
- `adaptive_overlap`
- `base_overlap`
- `max_adaptive_overlap`
- `code_mode`
- `align_text_to_source`

Do not add tokenizer override keys to this exclusion set during the refactor; the current pipeline forwards them into lower-level chunking calls where existing tokenizer override handling lives.

Do not change current boolean quirks in this refactor PR unless existing tests already enforce the change.

## Dispatch

`dispatch.py` handles chunk production:

- hierarchical/template path calls `context.chunk_text_hierarchical_flat(...)`
- multi-level path calls `context._compute_paragraph_spans(...)`
- multi-level metadata path calls `context.chunk_text_with_metadata(...)`
- multi-level fallback path catches `ChunkingError`, calls `context.chunk_text(...)`, and clamps offsets to paragraph spans
- normal path calls `context.chunk_text(...)`

Output normalization must remain path-specific:

- hierarchical/template path forwards the dictionaries returned by `chunk_text_hierarchical_flat(...)`
- multi-level metadata path converts `ChunkResult`/`ChunkMetadata`-style results into dict metadata
- multi-level fallback path preserves existing string/dict fallback offset behavior
- normal `chunk_text` path only special-cases dicts with `json` plus `metadata`, dicts with `text`, and strings; all other objects become `str(obj)` with empty metadata

Do not upgrade normal-path fallback objects into metadata-bearing `ChunkResult`-like outputs during this refactor unless a targeted behavior-change test explicitly requires it.

Dispatch must call through the passed context. This preserves compatibility with tests and downstream code that monkeypatch `chunker.chunk_text`, `chunker.chunk_text_with_metadata`, or other instance methods.

Add an internal LLM override context manager so dispatch cannot forget to restore thread-local state:

```python
with llm_override_scope(context, llm_call_func, llm_config):
    chunks = dispatch_chunks(...)
```

The context manager must preserve current `_LLM_UNSET` behavior.

## Metadata Finalization

`metadata.py` handles:

- applying `prefix_offset` to `start_offset`, `end_offset`, `start_char`, and `end_char`
- parsing/sorting `timecode_map`
- mapping chunk offsets to `start_time` and `end_time`
- setting defaults:
  - `chunk_index`
  - `total_chunks`
  - `chunk_method`
  - `max_size_setting`
  - `overlap_setting`
  - `max_size`
  - `overlap`
  - `language`
  - `adaptive_chunking_used`
  - `code_mode_used`
  - `relative_position`
  - `initial_document_json_metadata`
  - `initial_document_header_text`
  - `chunk_content_hash`
  - `origin`

Use `setdefault` semantics where current code uses them. Existing metadata supplied by strategies must keep precedence.

## Metrics And Tracing

The pipeline should keep existing metric names and labels:

- `chunker_process_total`
- `chunker_frontmatter_duration_seconds`
- `chunker_header_extract_seconds`
- `chunker_chunking_duration_seconds`
- `chunker_normalization_seconds`
- `chunker_last_chunk_count`
- `chunker_output_bytes`
- `chunker_input_bytes`
- `chunker_process_total_seconds`
- tracing span `chunker.process_text`

For the first PR, avoid moving all metric fallback code out of `chunker.py`. Prefer passing a small telemetry bundle from `chunker.py` into `ProcessTextPipeline`. If that proves awkward, use a self-contained `process_text/telemetry.py` wrapper that does not import `Chunker` and does not require other extracted modules to import from `chunker.py`. A later PR can extract telemetry more thoroughly.

Metrics/tracing failures remain noncritical.

## Error Handling

Preserve public error behavior:

- non-string input raises `InvalidInputError`
- invalid `max_size` raises `InvalidInputError`
- negative overlap is adjusted to `0` with warning
- configured size limit raises `InvalidInputError`
- multi-level `ChunkingError` fallback remains unchanged

Preserve current noncritical suppression for:

- frontmatter parsing
- header extraction
- adaptive sizing fallback
- timecode parsing/mapping
- metrics and tracing
- content hash fallback

Do not convert currently swallowed noncritical failures into new user-facing errors.

## Testing Strategy

Add output-equivalence tests for representative `process_text` scenarios:

- plain sentence chunking
- multi-level words/sentences
- hierarchical template path
- frontmatter plus prefix offsets
- timecode map
- code and code_ast metadata
- tokenizer override
- invalid input, invalid max size, and zero/negative overlap behavior
- normal-path dispatch through `chunk_text`, not `chunk_text_with_metadata`
- normal-path fallback object normalization, including a custom or `ChunkResult`-like object becoming `str(obj)` with empty metadata
- metrics counter behavior for invalid input using a monkeypatched telemetry hook

Keep existing focused tests passing:

- `test_chunker_v2.py`
- `test_chunker_process_metrics.py`
- `test_review_hardening.py`
- `test_streaming_overlap.py` process_text coverage
- hierarchy and offset tests that exercise `chunk_text_hierarchical_flat`

`test_review_hardening.py` comes from the Chunking hardening branch. If that branch has not landed in the implementation branch's base yet, either rebase the refactor implementation onto the hardening branch or run the equivalent hardening regression tests after rebasing.

Add deterministic helper tests for:

- frontmatter/header preparation
- option resolution
- metadata finalization and timecode parsing
- LLM override scope cleanup when dispatch or chunking raises

Helper tests should assert stable behavior fields, not incidental implementation details or telemetry side effects.

Write the representative output-equivalence tests before moving production logic. Expected outputs should be static fixtures or explicit expected dictionaries derived from the current implementation, not values produced dynamically by the refactored code under test.

## Implementation Staging

The future implementation plan should keep tests passing after each stage:

1. Add output-equivalence tests against the current implementation.
2. Add `process_text` package, internal models, shared option helper, shared noncritical exception policy, and shared LLM context module.
3. Extract and wire preparation logic into the active `Chunker.process_text` path.
4. Extract and wire option resolution into the active path.
5. Extract and wire dispatch plus LLM override scope into the active path.
6. Extract and wire metadata finalization into the active path.
7. Replace remaining `Chunker.process_text` body with the thin pipeline wrapper.
8. Remove now-unused imports from `chunker.py`.
9. Run focused Chunking tests and Bandit on touched Chunking paths.

Each extraction stage must wire the helper into the active production path before moving to the next stage. Avoid dead helper code that is only exercised after the final wrapper replacement. Each stage should prefer moving code first, then making only local cleanup that improves the extracted unit without changing behavior.

## Review Risks

- Circular imports if helpers import from `chunker.py`.
- Behavior drift if the noncritical exception tuple is duplicated or narrowed.
- Coupling drift if general Chunker LLM helpers import from the internal `process_text` package.
- Behavior drift in frontmatter/header size-enforcement order.
- Metrics drift if the process counter moves after input validation.
- Broken monkeypatch expectations if dispatch stops calling through the `Chunker` instance.
- Lost thread-local LLM override cleanup.
- Metadata differences from replacing `setdefault` with assignment.
- Overly broad changes to metrics imports or public docs.

## Acceptance Criteria For Implementation PR

- `Chunker.process_text` delegates to the pipeline and remains public-compatible.
- Existing process_text behavior is preserved for representative output-equivalence tests.
- New helper tests cover deterministic preparation, option, dispatch, and metadata behavior.
- No unrelated strategy or public API refactors are included.
- Focused Chunking tests pass.
- Bandit runs on touched Chunking paths.
