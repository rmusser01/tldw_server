# Chunker process_text Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `Chunker.process_text` into smaller internal components while preserving public behavior, output shape, metadata semantics, metrics, tracing names, and compatibility with existing monkeypatch-based tests.

**Architecture:** Keep `Chunker.process_text(...)` as a thin public wrapper over a new internal `process_text` pipeline. The new package depends on a small `ProcessTextContext` protocol, shared non-circular modules for option coercion, noncritical exception policy, and LLM override scope, plus focused modules for preparation, option resolution, dispatch, and metadata finalization.

**Tech Stack:** Python dataclasses and protocols, existing Chunking strategies, Loguru, existing Metrics helpers, pytest, Bandit.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-06-24-chunker-process-text-refactor-design.md`
- Backlog design task: `TASK-9935`
- Backlog plan task: `TASK-9936`
- Current public entry point: `tldw_Server_API/app/core/Chunking/chunker.py`

## Stage Map

Stage 1: characterize behavior before moving production logic.

Stage 2: introduce shared non-circular helpers and internal data contracts.

Stage 3: extract and wire preparation, option resolution, dispatch, and metadata finalization one stage at a time.

Stage 4: replace the large method body with the pipeline wrapper, clean imports, and verify.

## File Structure

Create:

- `tldw_Server_API/app/core/Chunking/error_policy.py`
  - Owns `CHUNKER_NONCRITICAL_EXCEPTIONS`.
  - Imports Chunking exceptions but does not import `chunker.py`.
- `tldw_Server_API/app/core/Chunking/option_utils.py`
  - Owns `_coerce_bool_option(...)`.
  - Imports `is_truthy` from `tldw_Server_API.app.core.testing`.
- `tldw_Server_API/app/core/Chunking/llm_context.py`
  - Owns `_LLM_UNSET` and `llm_override_scope(...)`.
  - Does not import `Chunker`.
- `tldw_Server_API/app/core/Chunking/process_text/__init__.py`
  - Exports `ProcessTextPipeline`.
- `tldw_Server_API/app/core/Chunking/process_text/models.py`
  - Defines `PreparedText`, `ResolvedProcessOptions`, `NormalizedChunk`, `TelemetryHooks`, and `ProcessTextContext`.
- `tldw_Server_API/app/core/Chunking/process_text/preparation.py`
  - Copies options, merges tokenizer override, extracts sentinel frontmatter, and extracts legacy transcription header.
- `tldw_Server_API/app/core/Chunking/process_text/options.py`
  - Resolves method, size, overlap, language, code mode, adaptive sizing, hierarchy flags, multi-level flag, and method-specific options.
- `tldw_Server_API/app/core/Chunking/process_text/dispatch.py`
  - Produces normalized internal chunks through the passed context.
- `tldw_Server_API/app/core/Chunking/process_text/metadata.py`
  - Restores prefix offsets, parses timecode maps, applies final metadata defaults, and computes content hashes.
- `tldw_Server_API/app/core/Chunking/process_text/pipeline.py`
  - Sequences validation, preparation, size enforcement, option resolution, dispatch, finalization, metrics, and tracing.
- `tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py`
  - Holds behavior-preservation tests for representative end-to-end paths and monkeypatch-sensitive dispatch behavior.
- `tldw_Server_API/tests/Chunking/test_process_text_components.py`
  - Holds deterministic helper tests for preparation, option resolution, metadata finalization, and LLM override cleanup.

Modify:

- `tldw_Server_API/app/core/Chunking/chunker.py`
  - Imports shared exception policy, option coercion, LLM sentinel, and pipeline.
  - Keeps `process_text(...)` as a straight delegation.
  - Keeps `processs_text(...)` alias unchanged.
  - Keeps all non-`process_text` methods public-compatible.
- `tldw_Server_API/tests/Chunking/test_chunker_v2.py`
  - Only adjust monkeypatch target strings if telemetry helper movement requires it.
- `tldw_Server_API/tests/Chunking/test_chunker_process_metrics.py`
  - Keep metric-name assertions intact.

## Behavioral Contracts

`Chunker.process_text(...)` must keep this public shape:

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
    return ProcessTextPipeline(self, _process_text_telemetry_hooks()).run(
        text,
        options,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_func=llm_call_func,
        llm_config=llm_config,
    )
```

The new package must never import `Chunker`.

`ProcessTextContext` must include only dependencies used by the pipeline:

```python
class ProcessTextContext(Protocol):
    config: ChunkerConfig
    _thread_local: Any

    def _enforce_text_size(self, text: str, *, source: str) -> None: ...
    def _normalize_method_argument(self, method: Any) -> Any: ...
    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any: ...
    def _compute_paragraph_spans(self, text: str, template: Any = None) -> list[tuple[int, int, str]]: ...
    def chunk_text(self, text: str, method: Any = None, max_size: Any = None, overlap: Any = None, language: Any = None, **options: Any) -> list[Any]: ...
    def chunk_text_with_metadata(self, text: str, method: Any = None, max_size: Any = None, overlap: Any = None, language: Any = None, **options: Any) -> list[Any]: ...
    def chunk_text_hierarchical_flat(self, text: str, method: Any = None, max_size: Any = None, overlap: Any = None, language: Any = None, template: Any = None, method_options: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...
```

`METHOD_OPTION_EXCLUDES` in `process_text/options.py` must preserve the current set exactly:

```python
METHOD_OPTION_EXCLUDES = {
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
```

Do not add tokenizer override keys to that exclusion set.

## Task 1: Add Characterization Tests First

**Files:**

- Create `tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py`
- Modify `tldw_Server_API/tests/Chunking/test_chunker_v2.py` only if a duplicate assertion should move into the new focused file.

- [ ] **Step 1: Add dispatch-path characterization tests**

Create tests with these imports:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking.base import ChunkMetadata, ChunkResult
from tldw_Server_API.app.core.Chunking.exceptions import ChunkingError, InvalidInputError
```

Add tests that assert:

- normal `process_text` calls `chunk_text(...)` and does not call `chunk_text_with_metadata(...)`;
- a custom object returned by normal `chunk_text(...)` becomes `{"text": str(obj), "metadata": ...}` with no metadata copied from object attributes before final defaults are applied;
- multi-level `ChunkingError` fallback still calls `chunk_text(...)` and clamps offsets within the paragraph span;
- hierarchical/template path calls `chunk_text_hierarchical_flat(...)` through the `Chunker` instance.

Use instance monkeypatches so the tests fail if dispatch stops calling through the context:

```python
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
```

- [ ] **Step 2: Add metadata and option behavior tests**

Cover these representative current behaviors:

- sentinel frontmatter with prefix offsets and `timecode_map`;
- string `"false"` for `enable_frontmatter_parsing` remains truthy because current raw `bool(value)` semantics are preserved;
- string `"false"` for `hierarchical` remains false through `_coerce_bool_option`;
- tokenizer override is forwarded through the lower-level chunking path and does not mutate the cached token strategy;
- explicit zero overlap is preserved;
- negative overlap is clamped to `0`;
- invalid `max_size` raises `InvalidInputError`;
- invalid non-string input raises `InvalidInputError`.

Use explicit expected dictionaries for stable fields:

```python
def test_process_text_string_false_frontmatter_option_remains_truthy() -> None:
    chunker = Chunker()
    payload = '{"meta": "x", "__tldw_frontmatter__": true}\nBody text.'

    rows = chunker.process_text(payload, options={"enable_frontmatter_parsing": "false"})

    assert rows
    assert rows[0]["metadata"]["initial_document_json_metadata"] == {"meta": "x"}
    assert rows[0]["text"].startswith("Body")
```

- [ ] **Step 3: Add invalid-input metric counter test**

Monkeypatch the current telemetry hook and prove `chunker_process_total` increments before public input validation:

```python
def test_process_text_invalid_input_increments_process_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.Chunking.chunker as chunker_module

    calls: list[tuple[str, dict | None]] = []

    def fake_increment_counter(name, labels=None):
        calls.append((name, labels))

    monkeypatch.setattr(chunker_module, "increment_counter", fake_increment_counter)

    with pytest.raises(InvalidInputError):
        Chunker().process_text(None)

    assert ("chunker_process_total", {"component": "chunker", "op": "process_text"}) in calls
```

If later telemetry movement changes the monkeypatch target, update only this target and keep the assertion that the counter fires before validation.

- [ ] **Step 4: Run the new failing tests against current code**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q
```

Expected result before production edits:

```text
passed
```

If a test fails before refactoring, adjust the expected assertion to match current behavior, then rerun the same command until the characterization suite reflects the current module.

## Task 2: Add Shared Helpers And Internal Models

**Files:**

- Create `tldw_Server_API/app/core/Chunking/error_policy.py`
- Create `tldw_Server_API/app/core/Chunking/option_utils.py`
- Create `tldw_Server_API/app/core/Chunking/llm_context.py`
- Create `tldw_Server_API/app/core/Chunking/process_text/__init__.py`
- Create `tldw_Server_API/app/core/Chunking/process_text/models.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Create `tldw_Server_API/tests/Chunking/test_process_text_components.py`

- [ ] **Step 1: Move the noncritical exception tuple to `error_policy.py`**

Create:

```python
from __future__ import annotations

import json

from .exceptions import ChunkingError, InvalidChunkingMethodError, InvalidInputError

CHUNKER_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
)
```

In `chunker.py`, replace the local tuple with:

```python
from .error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS as _CHUNKER_NONCRITICAL_EXCEPTIONS
```

- [ ] **Step 2: Move `_coerce_bool_option(...)` to `option_utils.py`**

Create:

```python
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.testing import is_truthy


def _coerce_bool_option(value: Any, default: bool = False) -> bool:
    """Normalize loose option values into stable booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return is_truthy(value.strip().lower())
    return bool(value)
```

In `chunker.py`, import `_coerce_bool_option` from the new module.

- [ ] **Step 3: Add LLM override scope**

Create:

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from .error_policy import CHUNKER_NONCRITICAL_EXCEPTIONS

_LLM_UNSET = object()


@contextmanager
def llm_override_scope(context: Any, llm_call_func: Any = None, llm_config: Any = None) -> Iterator[None]:
    previous = getattr(context._thread_local, "llm_overrides", _LLM_UNSET)
    apply_overrides = (llm_call_func is not None) or (llm_config is not None)
    if apply_overrides:
        override_func = llm_call_func if llm_call_func is not None else _LLM_UNSET
        override_config = llm_config if llm_config is not None else _LLM_UNSET
        context._thread_local.llm_overrides = (override_func, override_config)
    try:
        yield
    finally:
        if apply_overrides:
            if previous is _LLM_UNSET:
                with suppress(CHUNKER_NONCRITICAL_EXCEPTIONS):
                    delattr(context._thread_local, "llm_overrides")
            else:
                context._thread_local.llm_overrides = previous
```

In `chunker.py`, import `_LLM_UNSET` from the new module.

- [ ] **Step 4: Add internal models**

Create `models.py` with frozen dataclasses unless a later task needs mutation:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


@dataclass(frozen=True)
class PreparedText:
    original_text: str
    processed_text: str
    prefix_offset: int
    json_meta: dict[str, Any]
    header_text: str
    options: dict[str, Any]


@dataclass(frozen=True)
class ResolvedProcessOptions:
    method: Any
    method_lower: str
    max_size: int
    overlap: int
    language: Any
    adaptive: bool
    hierarchical: bool
    hier_template: dict[str, Any] | None
    multi_level: bool
    code_mode_for_method: str | None
    method_options_for_chunk: dict[str, Any]


@dataclass(frozen=True)
class NormalizedChunk:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TelemetryHooks:
    increment_counter: Callable[..., Any]
    observe_histogram: Callable[..., Any]
    set_gauge: Callable[..., Any]
    start_span: Callable[..., Any]
    set_span_attribute: Callable[..., Any]
    add_span_event: Callable[..., Any]
    record_span_exception: Callable[..., Any]


class ProcessTextContext(Protocol):
    config: ChunkerConfig
    _thread_local: Any
```

Append method protocol definitions from the Behavioral Contracts section.

- [ ] **Step 5: Add component tests for shared helpers**

In `test_process_text_components.py`, add:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking.llm_context import _LLM_UNSET, llm_override_scope
from tldw_Server_API.app.core.Chunking.option_utils import _coerce_bool_option


def test_llm_override_scope_restores_missing_attribute_on_error() -> None:
    chunker = Chunker()

    with pytest.raises(RuntimeError):
        with llm_override_scope(chunker, llm_call_func=lambda *_args, **_kwargs: "x"):
            assert getattr(chunker._thread_local, "llm_overrides")[0] is not _LLM_UNSET
            raise RuntimeError("forced")

    assert not hasattr(chunker._thread_local, "llm_overrides")


def test_coerce_bool_option_preserves_string_false_behavior() -> None:
    assert _coerce_bool_option("false", True) is False
```

- [ ] **Step 6: Run focused checks**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q
```

Expected result:

```text
passed
```

## Task 3: Extract And Wire Preparation

**Files:**

- Create `tldw_Server_API/app/core/Chunking/process_text/preparation.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_components.py`

- [ ] **Step 1: Add `prepare_frontmatter(...)` and `extract_header(...)`**

Create frontmatter preparation with this contract:

```python
def prepare_frontmatter(
    text: str,
    options: dict[str, Any] | None,
    *,
    tokenizer_name_or_path: str | None,
) -> PreparedText:
    ...
```

It must:

- make `opts = dict(options or {})`;
- merge `tokenizer_name_or_path` only when both `tokenizer_name_or_path` and `tokenizer_name` are absent;
- pop `enable_frontmatter_parsing` and `frontmatter_sentinel_key`;
- preserve current frontmatter enablement with `True if value is None else bool(value)`;
- parse only leading JSON objects with the active sentinel key truthy;
- strip leading newlines after the frontmatter exactly as current code does;
- return an empty `header_text` because header extraction runs after size enforcement.

Create header extraction with this contract:

```python
def extract_header(prepared: PreparedText) -> PreparedText:
    ...
```

- [ ] **Step 2: Wire frontmatter preparation into `Chunker.process_text` before extracting the rest**

Inside the current method body, replace only the shallow option copy, tokenizer merge, frontmatter extraction, and `prefix_offset` initialization with:

```python
prepared = prepare_frontmatter(
    text,
    options,
    tokenizer_name_or_path=tokenizer_name_or_path,
)
opts = prepared.options
processed_text = prepared.processed_text
prefix_offset = prepared.prefix_offset
json_meta = prepared.json_meta
header_text = prepared.header_text
```

Keep `self._enforce_text_size(processed_text, source="process_text")` immediately after frontmatter preparation.

After size enforcement, replace the header extraction block with:

```python
prepared = extract_header(prepared)
opts = prepared.options
processed_text = prepared.processed_text
prefix_offset = prepared.prefix_offset
json_meta = prepared.json_meta
header_text = prepared.header_text
```

Keep the existing duration metric calls around the preparation helper calls.

- [ ] **Step 3: Add preparation helper tests**

Cover:

- frontmatter metadata extraction with default sentinel;
- custom sentinel;
- disabled frontmatter with `False`;
- string `"false"` remains enabled;
- tokenizer override precedence;
- legacy header extraction increments `prefix_offset`.

- [ ] **Step 4: Run focused checks**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_extracts_frontmatter_with_sentinel tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_frontmatter_offsets_use_original_text -q
```

Expected result:

```text
passed
```

## Task 4: Extract And Wire Option Resolution

**Files:**

- Create `tldw_Server_API/app/core/Chunking/process_text/options.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_components.py`

- [x] **Step 1: Add `resolve_process_options(...)`**

Create:

```python
def resolve_process_options(
    context: ProcessTextContext,
    processed_text: str,
    options: dict[str, Any],
) -> ResolvedProcessOptions:
    ...
```

Move the current code for:

- `requested_method`;
- default method fallback;
- `max_size` parsing and positive validation;
- `overlap` parsing and negative-overlap clamp;
- language autodetection;
- `context._resolve_method(...)`;
- `method_option_excludes`;
- `method_options_for_chunk`;
- `code_mode_for_method`;
- adaptive size and overlap;
- hierarchical/template flags;
- multi-level flag.

Keep the negative overlap warning message:

```python
logger.warning(f"Negative overlap ({overlap}) adjusted to 0 in process_text")
```

- [x] **Step 2: Wire option resolution into the active method**

Replace local option resolution in `Chunker.process_text` with:

```python
resolved = resolve_process_options(self, processed_text, opts)
method = resolved.method
method_lower = resolved.method_lower
max_size = resolved.max_size
overlap = resolved.overlap
language = resolved.language
adaptive = resolved.adaptive
hierarchical = resolved.hierarchical
hier_template = resolved.hier_template
multi_level = resolved.multi_level
code_mode_for_method = resolved.code_mode_for_method
method_options_for_chunk = resolved.method_options_for_chunk
```

Do not move dispatch in this task.

- [x] **Step 3: Add option-resolution tests**

Use a real `Chunker` as context and assert:

- invalid `max_size="bad"` raises `InvalidInputError`;
- `max_size=0` raises `InvalidInputError`;
- `overlap=-5` resolves to `0`;
- Thai, Japanese, Cyrillic, and default language detection preserve current values;
- `method_options_for_chunk` excludes only the contract set and keeps tokenizer override keys;
- `code_mode` resolves to `ast` for `code_ast` and `auto` for `code`.

- [x] **Step 4: Run focused checks**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_streaming_overlap.py::test_language_autodetect_thai tldw_Server_API/tests/Chunking/test_streaming_overlap.py::test_language_autodetect_japanese_prefers_kana -q
```

Expected result:

```text
passed
```

## Task 5: Extract And Wire Dispatch

**Files:**

- Create `tldw_Server_API/app/core/Chunking/process_text/dispatch.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_components.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py`

- [x] **Step 1: Add dispatch helpers**

Create:

```python
def dispatch_chunks(
    context: ProcessTextContext,
    processed_text: str,
    resolved: ResolvedProcessOptions,
) -> list[NormalizedChunk]:
    ...
```

Move current path-specific behavior:

- hierarchical/template path calls `context.chunk_text_hierarchical_flat(...)` and forwards its dictionaries after wrapping as `NormalizedChunk`;
- multi-level path calls `context._compute_paragraph_spans(...)`;
- multi-level metadata path calls `context.chunk_text_with_metadata(...)`;
- multi-level fallback catches `ChunkingError`, calls `context.chunk_text(...)`, and clamps offsets;
- normal path calls `context.chunk_text(...)`;
- normal path only special-cases dicts with `json` plus `metadata`, dicts with `text`, and strings;
- every other normal-path object becomes `str(obj)` with empty metadata.

Convert metadata objects with current rules:

```python
if isinstance(metadata_obj, ChunkMetadata):
    md = asdict(metadata_obj)
elif isinstance(metadata_obj, dict):
    md = dict(metadata_obj)
else:
    md = {}
```

- [x] **Step 2: Wire dispatch into the active method with LLM override scope**

Replace the current local dispatch block with:

```python
with llm_override_scope(self, llm_call_func, llm_config):
    norm_chunks = dispatch_chunks(self, processed_text, resolved)
```

Keep `observe_histogram("chunker_chunking_duration_seconds", ...)` around the call.

- [x] **Step 3: Add dispatch helper tests**

Cover:

- normal-path custom object fallback;
- normal-path `{"json": ..., "metadata": ...}` conversion;
- normal-path `{"text": ..., "metadata": ...}` conversion;
- hierarchical path calls the context method;
- multi-level metadata result converts `ChunkResult`/`ChunkMetadata` into dict metadata;
- multi-level fallback clamps offsets;
- `llm_override_scope` restores previous override tuple when one already exists.

- [x] **Step 4: Run focused checks**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_chunking_regressions.py::test_process_text_multi_level_fallback_offsets_clamped -q
```

Expected result:

```text
passed
```

## Task 6: Extract And Wire Metadata Finalization

**Files:**

- Create `tldw_Server_API/app/core/Chunking/process_text/metadata.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_components.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py`

- [x] **Step 1: Add finalization helper**

Create:

```python
def finalize_chunks(
    *,
    original_text: str,
    chunks: list[NormalizedChunk],
    prepared: PreparedText,
    resolved: ResolvedProcessOptions,
) -> list[dict[str, Any]]:
    ...
```

Move current finalization behavior:

- add `prefix_offset` to `start_offset`, `end_offset`, `start_char`, and `end_char`;
- parse `timecode_map` from `prepared.options`;
- sort time segments by `start_offset`;
- set `start_time` and `end_time` only when absent;
- use `setdefault` for all final metadata defaults;
- use original text length for `relative_position`;
- add `initial_document_json_metadata` and `initial_document_header_text` only when present;
- compute `chunk_content_hash` with `hashlib.md5(..., usedforsecurity=False)`;
- set `origin` to `unified_chunker`.

- [x] **Step 2: Wire finalization into the active method**

Replace local prefix restoration and final output loop with:

```python
out = finalize_chunks(
    original_text=text,
    chunks=dispatched_chunks,
    prepared=prepared,
    resolved=resolved,
)
```

Keep normalization duration metrics around the helper call.

- [x] **Step 3: Add metadata helper tests**

Cover:

- prefix offset restoration applies to all four offset keys;
- strategy-provided metadata wins over defaults because `setdefault` is used;
- timecode mapping fills missing `start_time` and `end_time`;
- existing `start_time` and `end_time` are not overwritten;
- relative position uses the original input length after frontmatter stripping;
- content hash is present for ordinary text.
- malformed/invalid `timecode_map` is ignored without raising.

- [x] **Step 4: Run focused checks**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_frontmatter_offsets_use_original_text -q
```

Expected result:

```text
passed
```

## Task 7: Add Pipeline And Thin Public Wrapper

**Files:**

- Create `tldw_Server_API/app/core/Chunking/process_text/pipeline.py`
- Modify `tldw_Server_API/app/core/Chunking/process_text/__init__.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py`

- [x] **Step 1: Add `ProcessTextPipeline`**

Create:

```python
class ProcessTextPipeline:
    def __init__(self, context: ProcessTextContext, telemetry: TelemetryHooks) -> None:
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
        ...
```

The sequence inside `run(...)` must be:

1. initialize `overall_start` and labels;
2. increment `chunker_process_total`;
3. validate string input;
4. prepare frontmatter;
5. enforce text size;
6. extract header;
7. resolve options;
8. dispatch inside `llm_override_scope`;
9. finalize metadata;
10. record output metrics;
11. record tracing span `chunker.process_text`;
12. return output.

Keep metric names:

- `chunker_process_total`
- `chunker_frontmatter_duration_seconds`
- `chunker_header_extract_seconds`
- `chunker_chunking_duration_seconds`
- `chunker_normalization_seconds`
- `chunker_last_chunk_count`
- `chunker_output_bytes`
- `chunker_input_bytes`
- `chunker_process_total_seconds`

- [x] **Step 2: Add telemetry bundle factory in `chunker.py`**

Keep telemetry functions in `chunker.py` for the first implementation PR and pass them into the pipeline:

```python
def _process_text_telemetry_hooks() -> TelemetryHooks:
    return TelemetryHooks(
        increment_counter=increment_counter,
        observe_histogram=observe_histogram,
        set_gauge=set_gauge,
        start_span=start_span,
        set_span_attribute=set_span_attribute,
        add_span_event=add_span_event,
        record_span_exception=record_span_exception,
    )
```

This preserves existing monkeypatch tests that patch `tldw_Server_API.app.core.Chunking.chunker.increment_counter` before calling `process_text(...)`.

- [x] **Step 3: Replace `Chunker.process_text` with the wrapper**

Use:

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
    """End-to-end processing: optional frontmatter extraction, chunking, normalization."""
    return ProcessTextPipeline(self, _process_text_telemetry_hooks()).run(
        text,
        options,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_func=llm_call_func,
        llm_config=llm_config,
    )
```

Leave:

```python
def processs_text(self, *args, **kwargs):
    return self.process_text(*args, **kwargs)
```

- [x] **Step 4: Run focused checks**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_chunker_process_metrics.py tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_rejects_non_string_input tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_uses_tracing_helper_api_correctly -q
```

Expected result:

```text
passed
```

## Task 8: Clean Imports And Verify

**Files:**

- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify any new files listed above if import cleanup reveals unused symbols.
- Update `backlog/tasks/task-9936 - Plan-Chunker-process-text-refactor-implementation.md` only if this plan is executed in the same branch.

- [ ] **Step 1: Remove unused imports from `chunker.py`**

After the wrapper replacement, remove imports that are no longer used by `chunker.py`.

Likely candidates after extraction:

- `copy` if only extracted `process_text` used it;
- `hashlib` if only metadata finalization uses it;
- `json` if only frontmatter and normal-path JSON normalization use it;
- `re` only if no other chunker methods use it;
- `asdict` if only dispatch uses it.

Confirm with `rg` before removing each import.

- [ ] **Step 2: Run focused Chunking tests**

Command:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_chunker_v2.py tldw_Server_API/tests/Chunking/test_chunker_process_metrics.py tldw_Server_API/tests/Chunking/test_chunking_regressions.py::test_process_text_multi_level_fallback_offsets_clamped tldw_Server_API/tests/Chunking/test_streaming_overlap.py::test_language_autodetect_thai tldw_Server_API/tests/Chunking/test_streaming_overlap.py::test_language_autodetect_japanese_prefers_kana -q
```

Expected result:

```text
passed
```

- [ ] **Step 3: Compile touched Python modules**

Command:

```bash
source .venv/bin/activate && python -m compileall tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/app/core/Chunking/error_policy.py tldw_Server_API/app/core/Chunking/option_utils.py tldw_Server_API/app/core/Chunking/llm_context.py tldw_Server_API/app/core/Chunking/process_text tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py
```

Expected result:

```text
compileall exits 0
```

- [ ] **Step 4: Run Bandit on touched Chunking code**

Command:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/app/core/Chunking/error_policy.py tldw_Server_API/app/core/Chunking/option_utils.py tldw_Server_API/app/core/Chunking/llm_context.py tldw_Server_API/app/core/Chunking/process_text -f json -o /tmp/bandit_chunker_process_text_refactor.json
```

Expected result:

```text
No issues identified.
```

- [ ] **Step 5: Check diff quality**

Command:

```bash
git diff --check
```

Expected result:

```text
no output
```

## Self-Review Checklist

- [ ] The plan covers every approved spec stage: characterization tests, shared helpers, preparation, option resolution, dispatch, metadata, wrapper, cleanup, tests, and Bandit.
- [ ] No new module in `process_text/` imports `Chunker` or imports from `chunker.py`.
- [ ] The noncritical exception tuple is shared and not duplicated.
- [ ] `_LLM_UNSET` and `llm_override_scope(...)` live outside the internal `process_text` package.
- [ ] `enable_frontmatter_parsing` keeps raw `bool(value)` behavior.
- [ ] Normal path dispatch still calls `context.chunk_text(...)`.
- [ ] Normal-path fallback objects still become `str(obj)` with empty metadata before final defaults.
- [ ] The invalid-input metric counter test proves `chunker_process_total` increments before validation.
- [ ] Verification commands use the project virtual environment.
- [ ] Bandit is included for touched Chunking code.
