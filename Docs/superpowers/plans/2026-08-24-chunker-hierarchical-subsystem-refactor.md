# Chunker Hierarchical Subsystem Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `Chunker`'s hierarchy behavior into focused internal modules while preserving public signatures, tree and flat dictionary contracts, offsets, fallback behavior, logging, call multiplicity, and supported override seams.

**Architecture:** Keep the three public hierarchy methods on `Chunker` as compatibility wrappers. A context-backed `HierarchyService` resolves each call and delegates immutable call data to a tree builder or flattener; leaf construction and grouping are pure-enough focused components, and paragraph spans are shared directly with `process_text` dispatch. Production wiring changes incrementally so every extracted component is active and tested before the next extraction.

**Tech Stack:** Python protocols and frozen dataclasses, existing Chunking strategies and regex-safety helpers, Loguru, pytest, Ruff, Black, mypy, Bandit.

---

## Source References

- Approved spec: `Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md`
- Completed design task: `TASK-13112`
- Implementation task: `TASK-13215`
- Current implementation: `tldw_Server_API/app/core/Chunking/chunker.py`
- Existing process pipeline: `tldw_Server_API/app/core/Chunking/process_text/`

## Execution Rules

- Work only in the isolated worktree and branch for this workstream. Do not modify or reset the user's local `dev` checkout.
- Before the first production edit, reconcile the branch with current `origin/dev` and rerun the focused baseline suite in Task 1.
- Activate the shared project environment before Python tooling:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
```

- Follow red-green-refactor within each task. A characterization test is expected to pass before extraction; a new component test is expected to fail with `ModuleNotFoundError`, `ImportError`, or a missing symbol before its component is created.
- Keep tests behavioral. Do not assert implementation-local helper call graphs except where the approved call multiplicity, public composition, or dependency boundaries are compatibility contracts.
- Use `apply_patch` for manual edits. Use Backlog MCP to update `TASK-13215` after each task with the commit, touched files, test evidence, and any correction-gate finding.
- Do not broaden exception handling, validation, copying, or normalization while moving code.
- A behavior correction is out of the structural commits. If a candidate appears, apply the gate in the approved spec and use a separate red-green commit only after recording the baseline evidence in `TASK-13215`.
- Do not use `--no-verify` on any commit.

## Stage Map

1. Reconcile the implementation baseline and freeze public behavior.
2. Extract shared models and paragraph spans, including the `process_text` dependency migration.
3. Extract leaf construction, then tree construction and service coordination.
4. Extract grouping and flattening, wire the remaining public delegates, and remove stale hierarchy ownership from `Chunker`.
5. Enforce dependency boundaries, run the complete quality gate, and prepare the PR handoff.

## Final File Structure

Create:

```text
tldw_Server_API/app/core/Chunking/hierarchical/
|-- __init__.py
|-- models.py
|-- spans.py
|-- leaves.py
|-- builder.py
|-- grouping.py
|-- flatten.py
`-- service.py
```

Create focused tests:

```text
tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py
tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py
tldw_Server_API/tests/Chunking/test_hierarchical_spans.py
tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py
tldw_Server_API/tests/Chunking/test_hierarchical_builder.py
tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py
tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py
```

Modify:

- `tldw_Server_API/app/core/Chunking/chunker.py`
  - Remove `_compute_paragraph_spans(...)` and `_extract_header_title(...)`.
  - Replace tree and flatten bodies with `HierarchyService` delegates.
  - Keep `chunk_text_hierarchical_flat(...)` composing the two public methods.
- `tldw_Server_API/app/core/Chunking/process_text/models.py`
  - Remove `_compute_paragraph_spans(...)` from `ProcessTextContext`.
- `tldw_Server_API/app/core/Chunking/process_text/dispatch.py`
  - Import and call `hierarchical.spans.compute_paragraph_spans(...)` directly.
- `tldw_Server_API/tests/Chunking/test_process_text_components.py`
  - Remove the private helper from the protocol assertion and patch the dispatch dependency in multi-level tests.
- `Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md`
  - Update only the reconciled baseline hash/counts if Task 1 requires it.
- `backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md`
  - Maintain implementation status, evidence, commits, files, and PR link through Backlog MCP.

## Required Internal Interfaces

`hierarchical/models.py` must define these passive contracts without validation or coercion:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tldw_Server_API.app.core.Chunking.base import ChunkerConfig


class LeafChunkingContext(Protocol):
    def chunk_text(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]: ...

    def chunk_text_with_metadata(
        self,
        text: str,
        method: Any = None,
        max_size: Any = None,
        overlap: Any = None,
        language: Any = None,
        **options: Any,
    ) -> list[Any]: ...


class HierarchyContext(LeafChunkingContext, Protocol):
    config: ChunkerConfig

    def _enforce_text_size(self, text: str, *, source: str) -> None: ...
    def _normalize_method_argument(self, method: Any) -> Any: ...
    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any: ...
    def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str: ...
    def normalize_chunk_type(self, value: Any) -> str | None: ...


@dataclass(frozen=True)
class ResolvedHierarchyOptions:
    method: Any
    max_size: Any
    overlap: Any
    language: Any
    method_options: dict[str, Any]
    sanitize_output: bool


@dataclass(frozen=True)
class HierarchyTextViews:
    original: str
    sanitized: str
    output: str
```

Use these component signatures exactly:

```python
def compute_paragraph_spans(
    text: str,
    template: dict[str, Any] | None = None,
) -> list[tuple[int, int, str]]: ...


def build_leaf_block(
    context: LeafChunkingContext,
    texts: HierarchyTextViews,
    span: tuple[int, int, str],
    options: ResolvedHierarchyOptions,
) -> dict[str, Any] | None: ...


def build_hierarchy_tree(
    context: LeafChunkingContext,
    texts: HierarchyTextViews,
    spans: list[tuple[int, int, str]],
    options: ResolvedHierarchyOptions,
) -> dict[str, Any]: ...


def flatten_tree(
    tree: dict[str, Any],
    normalize_chunk_type: Callable[[Any], str | None],
) -> list[dict[str, Any]]: ...
```

`HierarchyService` may hold only the supplied context. It must resolve fresh call data on every invocation and must not cache options, text views, strategies, or trees.

## Task 1: Reconcile `origin/dev` Before Production Edits

**Files:**

- Modify `Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md` only when the current baseline differs.
- Update `TASK-13215` through Backlog MCP.

- [x] **Step 1: Refresh and inspect the baseline**

Run:

```bash
git fetch origin dev
git rev-parse origin/dev
git diff --name-status 4958cfed65d3c6e9baa43ea47e2b155fed204e13 origin/dev -- tldw_Server_API/app/core/Chunking tldw_Server_API/tests/Chunking
```

Expected: record the current `origin/dev` hash. As of plan creation, local `origin/dev` is `b1d0aed671dcf45bbe4211a9690022c083c99feb` and the scoped diff is empty. If the scoped diff is no longer empty, inspect every listed change and stop for design review if any approved contract is affected.

- [x] **Step 2: Rebase the isolated branch**

Run:

```bash
git rebase origin/dev
git status --short --branch
```

Expected: the branch is based on current `origin/dev`, contains only this workstream's design/planning commits, and has no unresolved changes. Resolve only conflicts in this workstream's documentation or Backlog records; do not discard upstream work.

- [x] **Step 3: Rerun the focused baseline suite**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchical_rewrite_offsets.py \
  tldw_Server_API/tests/Chunking/test_offsets_additional.py \
  tldw_Server_API/tests/Chunking/test_template_classifier.py \
  tldw_Server_API/tests/Chunking/test_thai_tables_spans.py \
  tldw_Server_API/tests/Chunking/test_template_hierarchical_options.py \
  tldw_Server_API/tests/Chunking/test_process_text_components.py \
  tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py \
  -q
```

Expected from the approved baseline: `91 passed, 1 skipped`. Record the fresh counts and any environmental warning. A failure must be investigated before production edits.

- [x] **Step 4: Record the reconciled evidence**

Update the spec's baseline hash and counts with `apply_patch`. Through Backlog MCP, append the same hash, scoped-diff result, test counts, and reconciliation date to `TASK-13215`.

- [x] **Step 5: Commit the baseline update**

```bash
git add Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "docs: reconcile hierarchical refactor baseline"
```

If the spec hash was already current after rebase, include only the Backlog evidence and do not manufacture a spec change.

## Task 2: Freeze Public, Error, Identity, and Call-Trace Contracts

**Files:**

- Create `tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py`
- Create `tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py`
- Create `tldw_Server_API/tests/Chunking/test_hierarchical_spans.py`

- [ ] **Step 1: Add exact public signature and composition tests**

In `test_hierarchy_refactor_contracts.py`, assert the three current signatures with `inspect.signature` and prove the flat wrapper uses public overrides:

```python
def test_public_hierarchy_signatures_are_stable() -> None:
    assert str(inspect.signature(Chunker.chunk_text_hierarchical_tree)) == (
        "(self, text: str, method: str | None = None, max_size: int | None = None, "
        "overlap: int | None = None, language: str | None = None, "
        "template: dict[str, Any] | None = None, "
        "method_options: dict[str, Any] | None = None) -> dict[str, typing.Any]"
    )
    assert str(inspect.signature(Chunker.flatten_hierarchical)) == (
        "(self, tree: dict[str, typing.Any]) -> list[dict[str, typing.Any]]"
    )
    assert str(inspect.signature(Chunker.chunk_text_hierarchical_flat)) == (
        "(self, text: str, method: str | None = None, max_size: int | None = None, "
        "overlap: int | None = None, language: str | None = None, "
        "template: dict[str, Any] | None = None, "
        "method_options: dict[str, Any] | None = None) -> list[dict[str, typing.Any]]"
    )


def test_public_flat_method_composes_overridable_public_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[tuple[str, Any]] = []
    sentinel_tree = {"root": {"kind": "root", "children": []}}
    sentinel_rows = [{"text": "flat", "metadata": {}}]

    def fake_tree(**kwargs: Any) -> dict[str, Any]:
        calls.append(("tree", kwargs))
        return sentinel_tree

    def fake_flatten(tree: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append(("flatten", tree))
        return sentinel_rows

    monkeypatch.setattr(chunker, "chunk_text_hierarchical_tree", fake_tree)
    monkeypatch.setattr(chunker, "flatten_hierarchical", fake_flatten)

    assert chunker.chunk_text_hierarchical_flat("source", method="words") is sentinel_rows
    assert calls[0][0] == "tree"
    assert calls[1] == ("flatten", sentinel_tree)
```

- [ ] **Step 2: Freeze option ordering and shallow-copy behavior**

Use a same-length sanitizer result and `structure_aware` mode so no leaf strategy obscures option behavior. Assert:

- `dict(method_options or {})` creates a new top-level dictionary;
- nested values retain identity;
- `sanitize_output` defaults to `True`;
- falsey values select original output;
- a `__bool__` raising `RuntimeError` falls back to sanitized output;
- `sanitize_output` is absent from both `_resolve_method(...)` options and leaf calls.

The covered-exception fixture is:

```python
class BrokenBool:
    def __bool__(self) -> bool:
        raise RuntimeError("cannot decide")
```

Patch `_resolve_method` to capture its options and return `"structure_aware"`. Patch `_sanitize_input` to return `"SAN"` for input `"raw"`. Assert the emitted structure-aware chunk is `"SAN"` for the default and `BrokenBool`, and `"raw"` for `False`.

- [ ] **Step 3: Freeze leaf call order and multiplicity through the public API**

Add deterministic instance fakes for these per-block traces:

```text
words metadata success                 ["metadata"]
sentences metadata success             ["metadata"]
tokens metadata success                ["metadata"]
metadata result missing integer offset ["metadata"] and no emitted chunks
metadata call raises                   ["metadata", "plain"]
first plain fallback raises            ["metadata", "plain", "plain"]
rewrite method success                 ["plain"]
ordinary method success                ["plain"]
structure_aware success                []
```

For metadata results, use `SimpleNamespace(metadata=SimpleNamespace(start_char=0, end_char=3))`. For the bounded second plain attempt, make the first `chunk_text(...)` call raise `RuntimeError("first plain failure")` and the second return `["raw"]`; assert exactly two plain calls and one output block.

- [ ] **Step 4: Freeze log levels and stable message text**

Capture Loguru records with a temporary sink and always remove it:

```python
records: list[dict[str, Any]] = []
sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
try:
    chunker.chunk_text_hierarchical_tree("raw", method="words", max_size=10)
finally:
    logger.remove(sink_id)
```

Configure the test's `chunk_text_with_metadata` or `chunk_text` fake before this
call so it enters the fallback being asserted.

Cover:

- words/sentences metadata failure: `DEBUG`, `"<method> metadata mapping failed, using fallback:"`;
- token metadata failure: `DEBUG`, `"Token metadata mapping failed, using fallback:"`;
- outer offset failure: `WARNING`, `"Offset mapping failed for method=<method>:"` and `"using naive offsets"`.

Do not assert module, function, path, line, timestamp, or formatted color output.

- [ ] **Step 5: Freeze malformed and aliasing behavior**

In `test_hierarchy_malformed_contracts.py`, use explicit trees and assert this exact matrix:

```text
falsey root + legacy blocks:
  public/package -> one "legacy" row with paragraph_kind="paragraph",
  ancestry_titles=[], chunk_type="text", chunk_index=1, total_chunks=1

truthy root="invalid":
  public/package -> AttributeError("'str' object has no attribute 'get'")

non-dict root children ["skip", 3, None]:
  public/package -> []

root chunks="ab":
  public/package -> rows "a" and "b", each with ancestry_titles=[],
  indexes 1/2 and total_chunks=2

chunk metadata="invalid":
  public -> ValueError containing "dictionary update sequence element #0"
  package helper -> []

structure-aware paragraph weight="invalid":
  public -> ValueError containing "invalid literal for int()"
  package helper -> []
```

Also assert that flattening:

- does not mutate the input tree or source top-level metadata dictionary;
- returns a distinct top-level metadata dictionary;
- retains identity for a nested mutable metadata value;
- shares the same `ancestry_titles` list between sibling rows in one ancestry context;
- preserves preexisting `chunk_index` and `total_chunks` through `setdefault`.

- [ ] **Step 6: Freeze span and regex-safety behavior**

In `test_hierarchical_spans.py`, initially call `Chunker()._compute_paragraph_spans(...)`. Use explicit expected span tuples for:

- empty input, blank lines, paragraphs, ATX headers, horizontal rules, ordered and unordered lists, Markdown table lines;
- closed and unclosed backtick and tilde fences;
- accepted custom kinds, 21-rule truncation, overlong patterns, unsafe patterns, invalid flags, and invalid rule objects;
- warning level and stable text for overlong, unsafe, and invalid rules;
- call-time `safe_search` replacement;
- import-time lookup failure for `safe_search`, where direct compiled `pattern.search(...)` still matches;
- a `safe_search(...)` call that raises a covered exception, where that pattern is skipped rather than directly retried.

For the lookup-failure case, patch `builtins.__import__` only when the requested module is the Chunking `regex_safety` module and `fromlist` contains `safe_search`; delegate every other import to the original function. This distinguishes lookup failure from invocation failure.

- [ ] **Step 7: Run the characterization tests against the unextracted code**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_spans.py \
  -q
```

Expected: all tests pass before any hierarchy production logic moves. If an expectation fails, update it to the observed baseline behavior and record the discrepancy in `TASK-13215`; do not change production code to satisfy the draft expectation.

- [ ] **Step 8: Commit the frozen contracts**

```bash
git add Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py tldw_Server_API/tests/Chunking/test_hierarchical_spans.py "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "test: characterize hierarchical chunking contracts"
```

## Task 3: Extract Models and Shared Paragraph Spans

**Files:**

- Create `tldw_Server_API/app/core/Chunking/hierarchical/__init__.py`
- Create `tldw_Server_API/app/core/Chunking/hierarchical/models.py`
- Create `tldw_Server_API/app/core/Chunking/hierarchical/spans.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/app/core/Chunking/process_text/models.py`
- Modify `tldw_Server_API/app/core/Chunking/process_text/dispatch.py`
- Modify `tldw_Server_API/tests/Chunking/test_hierarchical_spans.py`
- Modify `tldw_Server_API/tests/Chunking/test_process_text_components.py`
- Modify `tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py`

- [ ] **Step 1: Add failing model and span import tests**

Change `test_hierarchical_spans.py` to import:

```python
from tldw_Server_API.app.core.Chunking.hierarchical.spans import compute_paragraph_spans
```

Add protocol/dataclass assertions to `test_hierarchy_refactor_contracts.py`:

```python
def test_hierarchy_models_are_passive_and_frozen() -> None:
    nested: list[str] = []
    options = ResolvedHierarchyOptions("words", "10", 0, None, {"nested": nested}, True)
    views = HierarchyTextViews(original="a", sanitized="bb", output="ccc")

    assert options.method_options["nested"] is nested
    assert (views.original, views.sanitized, views.output) == ("a", "bb", "ccc")
    with pytest.raises(FrozenInstanceError):
        views.output = "changed"  # type: ignore[misc]
```

Run the two files. Expected red: the `hierarchical` package or symbols do not exist.

- [ ] **Step 2: Create passive models and move span logic**

Create the package marker with only a module docstring. Implement `models.py` exactly as specified in Required Internal Interfaces.

Move `_compute_paragraph_spans(...)` into `spans.py` as `compute_paragraph_spans(...)` without restructuring its branches. Preserve:

- local call-time imports of `check_pattern` and `compile_flags` during boundary preparation;
- local call-time import of `safe_search` inside template matching;
- caps of 20 rules and 256 pattern characters;
- warning levels and stable text;
- the direct compiled-search fallback only when `safe_search` lookup fails;
- `_CHUNKER_NONCRITICAL_EXCEPTIONS` breadth;
- exact fence, line-offset, and span ordering behavior.

Use package-relative imports from `..error_policy` and `..regex_safety`. Do not import `Chunker`, `chunker`, or `process_text`.

- [ ] **Step 3: Wire both active span callers and remove the private seam**

In `chunker.py`, import `compute_paragraph_spans`, replace the current tree call with:

```python
spans = compute_paragraph_spans(text, template)
```

Delete `Chunker._compute_paragraph_spans(...)` completely.

In `process_text/dispatch.py`, import the same function and replace the multi-level call with:

```python
spans = compute_paragraph_spans(processed_text, template=None)
```

Delete `_compute_paragraph_spans(...)` from `ProcessTextContext`.

Update the three existing multi-level tests to patch:

```python
monkeypatch.setattr(process_dispatch, "compute_paragraph_spans", fake_spans)
```

Update the protocol member assertion so the private helper is absent.

- [ ] **Step 4: Add import-boundary tests**

Use AST inspection, not substring matching. Assert every module under `hierarchical/` rejects imports whose resolved module contains either:

```text
tldw_Server_API.app.core.Chunking.chunker
tldw_Server_API.app.core.Chunking.process_text
```

Also assert `leaves.py` and `grouping.py`, once present, do not import `service`, `builder`, or `flatten`. Keep this assertion data-driven so missing future files are ignored until their tasks create them.

- [ ] **Step 5: Run focused green tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchical_spans.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_process_text_components.py \
  tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py \
  tldw_Server_API/tests/Chunking/test_thai_tables_spans.py \
  -q
```

Expected: all pass with the same existing skip status.

- [ ] **Step 6: Commit the shared models and spans**

```bash
git add Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md tldw_Server_API/app/core/Chunking/hierarchical tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/app/core/Chunking/process_text/models.py tldw_Server_API/app/core/Chunking/process_text/dispatch.py tldw_Server_API/tests/Chunking/test_hierarchical_spans.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py tldw_Server_API/tests/Chunking/test_process_text_components.py "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "refactor: share hierarchical paragraph spans"
```

## Task 4: Extract Leaf Construction and Activate It

**Files:**

- Create `tldw_Server_API/app/core/Chunking/hierarchical/leaves.py`
- Create `tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py`

- [ ] **Step 1: Add failing direct leaf tests**

Build a small fake implementing only `LeafChunkingContext`. Instantiate `HierarchyTextViews` and `ResolvedHierarchyOptions` directly. Cover:

- `start >= end` returns `None` with zero context calls;
- rewrite methods emit `offsets_valid=False` and `None` offsets;
- words, sentences, and tokens use one metadata call on success;
- missing or non-integer metadata offsets are skipped without plain fallback;
- metadata exceptions trigger one plain fallback;
- a failing first plain fallback allows exactly one outer plain retry;
- structure-aware emits the exact output slice with zero context calls;
- ordinary methods use rolling, block-bounded source search;
- repeated and missing chunk text remains monotonic and bounded;
- raw output mode slices `texts.original`, while sanitized output mode slices `texts.output`;
- returned blocks and child lists are fresh and no input container is mutated.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py -q
```

Expected red: `hierarchical.leaves` or `build_leaf_block` is missing.

- [ ] **Step 2: Move the leaf algorithm without changing branches**

Implement `build_leaf_block(...)` with the required signature. Define the rewrite-method set at module scope using current `ChunkingMethod` values plus `"code_ast"`. Return this shape for non-empty spans:

```python
{
    "kind": kind,
    "start_offset": start,
    "end_offset": end,
    "chunks": out_chunks,
    "children": [],
}
```

Keep `chunks = None` and the current nested/outer `CHUNKER_NONCRITICAL_EXCEPTIONS` structure so the bounded second plain attempt remains possible. Keep exact debug and warning messages. `leaves.py` must not receive a parent/root/stack or mutate `texts`, `span`, or `options`.

- [ ] **Step 3: Replace the active local leaf body**

In the existing `chunk_text_hierarchical_tree(...)` body, create one `HierarchyTextViews` and one `ResolvedHierarchyOptions` after option resolution and sanitization. Replace `_add_block(...)` internals with:

```python
def _add_block(parent: dict[str, Any], start: int, end: int, kind: str) -> None:
    block = build_leaf_block(
        self,
        text_views,
        (start, end, kind),
        resolved_options,
    )
    if block is not None:
        parent.setdefault("children", []).append(block)
```

Remove the duplicate rewrite set and leaf mapping code from `chunker.py`. Keep option resolution and tree mutation there until Task 5.

- [ ] **Step 4: Run component and public contract tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_rewrite_offsets.py \
  tldw_Server_API/tests/Chunking/test_offsets_additional.py \
  tldw_Server_API/tests/Chunking/test_chunking_regressions.py \
  -q
```

Expected: all pass, including exact call traces and logs.

- [ ] **Step 5: Commit the leaf extraction**

```bash
git add Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md tldw_Server_API/app/core/Chunking/hierarchical/leaves.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "refactor: extract hierarchical leaf construction"
```

## Task 5: Extract Tree Builder and Tree Service Coordination

**Files:**

- Create `tldw_Server_API/app/core/Chunking/hierarchical/builder.py`
- Create `tldw_Server_API/app/core/Chunking/hierarchical/service.py`
- Create `tldw_Server_API/tests/Chunking/test_hierarchical_builder.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py`

- [ ] **Step 1: Add failing builder and service tests**

Directly test `build_hierarchy_tree(...)` with a fake leaf context and fixed spans. Use explicit expected dictionaries for:

- minimal root with no spans;
- preface creation and final bounds;
- ATX level nesting and sibling closure;
- bold subsections under the nearest non-bold section;
- repeated bold headings becoming siblings;
- header-only sections retaining their header block;
- blank spans producing no block;
- builder non-mutation of text views, options, and span list;
- every appended block being the fresh object returned by `build_leaf_block(...)`.

Patch `builder.build_leaf_block` for pure tree tests. Separately test the real builder plus real leaf function for one header/preface integration fixture.

Add service tests with a minimal `HierarchyContext` fake. Assert empty input returns exactly:

```python
{
    "type": "hierarchical",
    "schema_version": 1,
    "root": {"kind": "root", "children": []},
}
```

Also assert empty input performs no size enforcement, option resolution, sanitization, or span computation.
Construct one `HierarchyService` around a mutable fake context, change its config
defaults and replace one context method between two calls, and assert the second
call observes both changes. This proves the service does not snapshot context
state or bound methods at construction time.

- [ ] **Step 2: Move all tree mutation into `builder.py`**

Implement `build_hierarchy_tree(...)` with the required interface. Move `_extract_header_title(...)` into this module as a private module function. The builder exclusively owns:

- root creation;
- section stack and current section;
- preface creation;
- section closure and final bounds;
- ATX and bold-subsection parent selection;
- parent `children` mutation.

It calls `build_leaf_block(...)`, appends only non-`None` results, and treats text views/options/spans as read-only. It does not import `Chunker`, `process_text`, `service`, `flatten`, or `grouping`.

- [ ] **Step 3: Implement per-call tree coordination in `service.py`**

Use a `HierarchyService` with `__init__(self, context: HierarchyContext) -> None`
and `build_tree(self, text: str, method: Any = None, max_size: Any = None,
overlap: Any = None, language: Any = None, template: dict[str, Any] | None =
None, method_options: dict[str, Any] | None = None) -> dict[str, Any]`. The
constructor body is exactly:

```python
self._context = context
```

Preserve the exact current order:

1. type validation;
2. minimal empty return;
3. `_enforce_text_size(..., source="chunk_text_hierarchical_tree")`;
4. `dict(method_options or {})`;
5. `sanitize_output` truthiness inside the noncritical catch, then pop;
6. method/default size/overlap/language resolution with current truthiness;
7. `_resolve_method(...)` with the copied options;
8. one `_sanitize_input(..., suppress_security_log=True)` call;
9. `compute_paragraph_spans(original, template)`;
10. `build_hierarchy_tree(...)`;
11. wrap the returned root with the current schema-level method/language/size/overlap fields.

Do not add a top-level catch. Do not validate or coerce the frozen dataclasses.

- [ ] **Step 4: Replace the public tree body and remove the private title seam**

Keep the existing public signature and docstring, with this body:

```python
return HierarchyService(self).build_tree(
    text=text,
    method=method,
    max_size=max_size,
    overlap=overlap,
    language=language,
    template=template,
    method_options=method_options,
)
```

Delete `_extract_header_title(...)` and all now-stale tree imports/locals from `chunker.py`. Add assertions that `Chunker` has neither `_extract_header_title` nor `_compute_paragraph_spans`.

- [ ] **Step 5: Run builder, service, and end-to-end hierarchy tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchical_builder.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_rewrite_offsets.py \
  tldw_Server_API/tests/Chunking/test_offsets_additional.py \
  tldw_Server_API/tests/Chunking/test_template_classifier.py \
  tldw_Server_API/tests/Chunking/test_template_hierarchical_options.py \
  -q
```

Expected: all pass with public signature and option/call-trace tests unchanged.

- [ ] **Step 6: Commit the tree extraction**

```bash
git add Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md tldw_Server_API/app/core/Chunking/hierarchical/builder.py tldw_Server_API/app/core/Chunking/hierarchical/service.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_hierarchical_builder.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "refactor: extract hierarchical tree builder"
```

## Task 6: Extract Grouping and Activate It in Flattening

**Files:**

- Create `tldw_Server_API/app/core/Chunking/hierarchical/grouping.py`
- Create `tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`

- [ ] **Step 1: Add failing grouping tests**

Define these concrete helpers in the tests:

```python
def merge_texts(
    parts: list[tuple[str, dict[str, Any]]],
    *,
    method: Any,
    default_sep: str = " ",
    kind_hint: str | None = None,
) -> str: ...


def group_items_by_elements(
    items: list[dict[str, Any]],
    *,
    method: Any,
    max_elements: int | None,
    overlap: int,
) -> list[dict[str, Any]]: ...


def group_section_by_kind_weight(
    items: list[dict[str, Any]],
    *,
    method: Any,
    max_weight: int | None,
    overlap: int,
    weights: dict[str, Any],
) -> list[dict[str, Any]]: ...
```

Cover:

- empty and single-part joins;
- spaces, line separators, double-line header separators, and no-space languages;
- existing trailing newlines and header-like spacing;
- element windows with zero/negative overlap, clamped step, and suppressed overlap-only final window;
- nonpositive/`None` limits returning the original items list identity;
- min/max source offsets and `grouped_elements`;
- weighted grouping staying within a paragraph kind, default/nonpositive weights, overlap progress, and `group_kind`;
- invalid string weight raising `ValueError`;
- input list, item dictionaries, and nested metadata remaining unmodified.

Run the new file and expect a missing module/symbol failure.

- [ ] **Step 2: Move merge and grouping helpers exactly**

Implement the helpers in `grouping.py` by moving the three existing nested algorithms. Keep:

- `LANGUAGES_NO_SPACE = {"zh", "zh-cn", "zh-tw", "ja", "th"}` at module scope;
- current operator precedence in the `need_sep` condition;
- integer-conversion catches for offsets;
- `int(weights.get(...))` propagation for invalid weight strings;
- final-window suppression and overlap clamping;
- fresh grouped dictionaries while preserving current nested identities where no copy occurs.

The module may import only `Any` and the shared noncritical exception policy from outside the standard library. It must not import service, builder, flatten, leaves, `Chunker`, or `process_text`.

- [ ] **Step 3: Wire the active flatten implementation**

Import the three helpers into `chunker.py`, delete their nested copies, and pass the current `method` explicitly at each call. Keep section traversal and header buffering in `Chunker.flatten_hierarchical(...)` until Task 7.

- [ ] **Step 4: Run grouping and flatten compatibility tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py \
  tldw_Server_API/tests/Chunking/test_chunker_v2.py \
  -q
```

Expected: all pass. In particular, no-space joins, invalid weights, and malformed-tree outcomes remain unchanged.

- [ ] **Step 5: Commit the grouping extraction**

```bash
git add Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md tldw_Server_API/app/core/Chunking/hierarchical/grouping.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "refactor: extract hierarchical grouping"
```

## Task 7: Extract Flattening and Complete Public Delegation

**Files:**

- Create `tldw_Server_API/app/core/Chunking/hierarchical/flatten.py`
- Create `tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py`
- Modify `tldw_Server_API/app/core/Chunking/hierarchical/service.py`
- Modify `tldw_Server_API/app/core/Chunking/chunker.py`
- Modify `tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py`
- Modify `tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py`

- [ ] **Step 1: Add failing direct flatten tests**

Move the malformed matrix and identity assertions into parameterized direct tests for `flatten_tree(...)`, while retaining public/package integration assertions in `test_hierarchy_malformed_contracts.py`. Add direct tests for:

- ancestry and section paths;
- empty titles and nested section titles;
- normalizer call-time behavior and fallback from missing `chunk_type` to `paragraph_kind`;
- existing `chunk_index`/`total_chunks` preservation;
- header buffering into the next structure-aware item;
- header-only sections;
- structure-aware element and by-kind grouping;
- traversal of nested section children without duplicate content;
- caller-tree non-mutation, top-level metadata copy, nested identity, and shared ancestry-list identity.

Run the new file and expect the missing `flatten_tree` failure.

- [ ] **Step 2: Move traversal and output assembly into `flatten.py`**

Implement `flatten_tree(...)` with the required signature. Move `_append_with_titles`, `_gather_section_items`, header buffering, recursive walk, and final index normalization from `chunker.py`. Import only grouping helpers plus standard/shared low-level dependencies.

Preserve these deliberately permissive operations:

```python
root = tree.get("root") or {"children": tree.get("blocks", [])}
chunks = node.get("chunks") or []
children = node.get("children") or []
md = dict(ch.get("metadata") or {}) if isinstance(ch, dict) else {}
md["ancestry_titles"] = titles
md.setdefault("chunk_index", i + 1)
md.setdefault("total_chunks", len(out))
```

Do not add schema validation, deep copies, child normalization, or broad catches. Pass only the supplied normalizer callback, never the full hierarchy context.

- [ ] **Step 3: Add flatten coordination to the service**

Add:

```python
def flatten(self, tree: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(tree, dict):
        return []
    return flatten_tree(tree, self._context.normalize_chunk_type)
```

The non-dict check belongs here because `flatten_tree(...)` accepts the already public-validated dictionary and must preserve malformed dictionary behavior.

- [ ] **Step 4: Replace the public flatten body and preserve flat composition**

Keep the public flatten signature/docstring and replace its body with:

```python
return HierarchyService(self).flatten(tree)
```

Keep `chunk_text_hierarchical_flat(...)` exactly as public composition:

```python
tree = self.chunk_text_hierarchical_tree(
    text=text,
    method=method,
    max_size=max_size,
    overlap=overlap,
    language=language,
    template=template,
    method_options=method_options,
)
return self.flatten_hierarchical(tree)
```

Do not add a service-level `build_flat` method. Leave the package-level `Chunking.flatten_hierarchical(tree)` helper and its explicit exception tuple unchanged.

- [ ] **Step 5: Run all hierarchy and process integration tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_spans.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_builder.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py \
  tldw_Server_API/tests/Chunking/test_process_text_components.py \
  tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_rewrite_offsets.py \
  tldw_Server_API/tests/Chunking/test_offsets_additional.py \
  tldw_Server_API/tests/Chunking/test_template_classifier.py \
  tldw_Server_API/tests/Chunking/test_template_hierarchical_options.py \
  tldw_Server_API/tests/Chunking/test_thai_tables_spans.py \
  -q
```

Expected: all pass with the established skips only.

- [ ] **Step 6: Commit the flatten extraction**

```bash
git add Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md tldw_Server_API/app/core/Chunking/hierarchical/flatten.py tldw_Server_API/app/core/Chunking/hierarchical/service.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md"
git commit -m "refactor: extract hierarchical flattening"
```

## Task 8: Enforce Boundaries, Verify, and Prepare the PR Handoff

**Files:**

- Modify touched files only for verified cleanup.
- Update `TASK-13215` through Backlog MCP.
- Update this plan's checkboxes as tasks complete.

- [ ] **Step 1: Remove stale imports and verify ownership mechanically**

Run:

```bash
rg -n "_compute_paragraph_spans|_extract_header_title" tldw_Server_API/app/core/Chunking tldw_Server_API/tests/Chunking
rg -n "from .*Chunking\.chunker|import .*Chunking\.chunker|from .*process_text|import .*process_text" tldw_Server_API/app/core/Chunking/hierarchical
```

Expected:

- no production definition or call remains for either removed private helper;
- references in tests exist only in explicit absence assertions;
- no hierarchy module imports or names `Chunker` or `process_text`;
- `leaves.py` and `grouping.py` contain no service/builder/flatten imports.

Run the AST dependency tests after cleanup.

- [ ] **Step 2: Run focused hierarchy tests**

Run the exact Task 7 focused command again. Expected: all pass.

- [ ] **Step 3: Run the complete Chunking suite**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Chunking -q
```

Expected: all tests pass with only established skips. Investigate and record any unrelated failure before proceeding.

- [ ] **Step 4: Compile touched production and test modules**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m compileall -q \
  tldw_Server_API/app/core/Chunking/hierarchical \
  tldw_Server_API/app/core/Chunking/chunker.py \
  tldw_Server_API/app/core/Chunking/process_text/models.py \
  tldw_Server_API/app/core/Chunking/process_text/dispatch.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_spans.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_builder.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py
```

Expected: exit code `0` with no syntax errors.

- [ ] **Step 5: Run Ruff and scoped Black**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/Chunking/hierarchical \
  tldw_Server_API/app/core/Chunking/chunker.py \
  tldw_Server_API/app/core/Chunking/process_text/models.py \
  tldw_Server_API/app/core/Chunking/process_text/dispatch.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_spans.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_builder.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py

python -m black --check \
  tldw_Server_API/app/core/Chunking/hierarchical \
  tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_spans.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_builder.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_grouping.py \
  tldw_Server_API/tests/Chunking/test_hierarchical_flatten.py
```

Expected: both commands exit `0`. Do not run Black across legacy `chunker.py`, existing `process_text` modules, or existing tests.

- [ ] **Step 6: Run informational mypy on the new package**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m mypy tldw_Server_API/app/core/Chunking/hierarchical
```

Fix hierarchy-local type errors that can be corrected without changing runtime behavior. Record exact residual output in `TASK-13215`; do not add broad suppressions and do not treat the five known `process_text` baseline errors as part of this scope.

- [ ] **Step 7: Run Bandit on touched production code**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Chunking/hierarchical \
  tldw_Server_API/app/core/Chunking/chunker.py \
  tldw_Server_API/app/core/Chunking/process_text/models.py \
  tldw_Server_API/app/core/Chunking/process_text/dispatch.py \
  -f json -o /tmp/bandit_task_13215.json
```

Expected: no new findings in touched code. Inspect `/tmp/bandit_task_13215.json` and record the issue count and severity summary in `TASK-13215`.

- [ ] **Step 8: Review the final diff and repository checks**

```bash
git diff --check origin/dev...HEAD
git status --short
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- tldw_Server_API/app/core/Chunking tldw_Server_API/tests/Chunking
```

Confirm:

- only approved files changed;
- `Chunker` wrappers retain exact signatures;
- flat composition still uses both public methods;
- package helper exception handling is unchanged;
- no broad formatting churn exists;
- no dormant duplicate hierarchy implementation remains;
- no correction is mixed into a structural commit.

Use `superpowers:requesting-code-review` for a final code review. Validate every finding before editing, use `superpowers:receiving-code-review` for feedback, and rerun the affected focused tests after each accepted fix.

- [ ] **Step 9: Final verification commit**

Update `TASK-13215` with all commit hashes, touched files, focused/full test counts, compile/Ruff/Black/mypy/Bandit results, known skips, and the absence or evidence of gated corrections. Then commit any final verified cleanup and task evidence:

```bash
git add tldw_Server_API/app/core/Chunking tldw_Server_API/tests/Chunking "backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md" Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
git commit -m "test: verify hierarchical subsystem refactor"
```

If there is no code cleanup after the previous commit, make this a documentation-only evidence commit rather than an empty commit.

- [ ] **Step 10: Push and prepare the PR against `dev`**

Use `superpowers:finishing-a-development-branch`. Push the implementation branch, create the PR against `dev`, and include:

- factual change list by module;
- compatibility contracts covered;
- focused and full test counts;
- Ruff, Black, compileall, mypy, and Bandit evidence;
- any separately gated correction commit and rationale;
- `TASK-13215` and spec/plan links;
- an explicit merge blocker stating that the human requester must write the required `Change summary` in their own words, explaining both what changed and why these boundaries and compatibility choices were selected.

Do not mark `TASK-13215` Done or the PR merge-ready until that human-written summary exists and all acceptance criteria are checked.

## Behavior-Correction Stop Rule

No corrections are pre-approved. When a test exposes a possible pre-existing defect:

1. Reproduce it against the reconciled pre-refactor baseline in a clean state.
2. Cite the exact violated invariant or preserve it as compatibility behavior.
3. If it qualifies, add a focused failing regression test first.
4. Record evidence and rationale in `TASK-13215` before implementation.
5. Implement the smallest local correction.
6. Commit it separately with a `fix:` message.
7. Rerun all focused and complete Chunking gates.

If any step cannot be satisfied, document the finding and defer it without changing behavior.

## Completion Checklist

- [x] Reconciled current `origin/dev` baseline recorded before production edits.
- [ ] Frozen public/malformed/identity/call/log/span characterizations pass.
- [ ] Shared models and spans extracted; process protocol/private span seam removed.
- [ ] Leaves extracted with exact call multiplicity and offset fallback behavior.
- [ ] Builder/service extracted; private header-title seam removed.
- [ ] Grouping and flattening extracted with current malformed and aliasing behavior.
- [ ] Public signatures, public flat composition, and package helper remain compatible.
- [ ] AST dependency rules pass and no dormant duplicate hierarchy body remains.
- [ ] Full Chunking suite and all static/security gates recorded.
- [ ] Final review findings validated and addressed.
- [ ] PR targets `dev`; human-written `Change summary` merge gate remains explicit.
