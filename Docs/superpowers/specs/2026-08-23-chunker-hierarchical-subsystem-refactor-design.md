# Chunker Hierarchical Subsystem Refactor Design

Backlog: `TASK-13112`

## Purpose

`Chunker` still owns paragraph-span detection, hierarchical tree construction,
leaf offset mapping, structure-aware grouping, and tree flattening. These
responsibilities occupy roughly 960 lines in `chunker.py` and make hierarchy
changes difficult to review independently.

This design extracts the full hierarchy subsystem into focused internal modules.
It follows the merged `process_text` refactor and keeps the existing public
hierarchy methods and dictionary contracts. The implementation is compatibility
first, with narrowly reproduced corrections allowed under the gate defined below.

## Baseline

The design is based on refreshed `origin/dev` commit
`4958cfed65d3c6e9baa43ea47e2b155fed204e13`. The local `dev` branch is divergent
and dirty, so implementation must use an isolated worktree based on refreshed
`origin/dev` and must not modify or reset the local checkout.

Immediately before implementation begins, fetch `origin/dev` and compare it with
the recorded baseline. If it is unchanged, proceed from the pinned commit. If it
has advanced, do not silently mix the new code with these characterization
results: inspect the intervening commits for hierarchy or `process_text` changes,
rebase the isolated branch onto the new `origin/dev`, update the baseline hash in
this spec and the Backlog task, and rerun the focused characterization suite before
editing production code. Any resulting contract change returns to design review.

Focused baseline verification on the isolated worktree passed:

- 92 tests collected
- 91 passed
- 1 skipped
- 0 failures

The run covered hierarchy rewrite offsets, additional offsets, template
classification, Thai/table spans, hierarchical template options, `process_text`
components, and `process_text` output equivalence.

## Goals

- Move hierarchy-specific behavior out of `chunker.py` into small modules with
  explicit ownership.
- Keep `Chunker.chunk_text_hierarchical_tree(...)`,
  `Chunker.flatten_hierarchical(...)`, and
  `Chunker.chunk_text_hierarchical_flat(...)` public-compatible.
- Preserve schema-version-1 trees, legacy flatten inputs, flat chunk dictionaries,
  metadata, offsets, grouping semantics, logging, and fallback behavior.
- Share paragraph-span detection directly between hierarchy construction and
  multi-level `process_text` dispatch.
- Preserve public override and monkeypatch seams while deliberately removing two
  hierarchy-only private seams.
- Make hierarchy components independently testable without importing `Chunker`.
- Permit only reproduced, local correctness fixes that satisfy the correction
  gate.

## Non-Goals

- No public signature or output-schema redesign.
- No conversion of public trees or chunks to dataclasses or Pydantic models.
- No strategy implementation refactor.
- No streaming or file chunking changes.
- No template API redesign.
- No telemetry redesign.
- No speculative performance rewrite.
- No stricter validation or generalized malformed-tree normalization.
- No cleanup outside hierarchy ownership and the required `process_text` span
  dependency.

## Current Public Surface And Consumers

The public methods have direct consumers in ingestion persistence, media
endpoints, paper search, RAG retrieval, document and web-scraping services,
workflow adapters, templates, and `process_text` dispatch. The package-level
`Chunking.flatten_hierarchical(tree)` helper also constructs a `Chunker`, calls
the public flatten method, and converts a specific exception set to `[]`.

Those consumers continue to call the same public methods. This refactor does not
require caller migrations.

## Proposed Package

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

Ownership is intentionally narrow:

- `models.py`: `HierarchyContext`, `ResolvedHierarchyOptions`, and only similarly
  narrow internal state proven necessary during implementation.
- `spans.py`: safe template-boundary preparation and exact paragraph/block span
  classification.
- `leaves.py`: leaf strategy calls, rewrite-method handling, metadata mapping,
  source-slice selection, and bounded offset fallbacks.
- `builder.py`: section-stack, preface, header, bold-subsection, and tree
  construction.
- `grouping.py`: text merging and structure-aware element/weight grouping.
- `flatten.py`: traversal, ancestry, chunk-type normalization, and final indexes.
- `service.py`: call-time option resolution and coordination of tree construction
  or flattening against a supplied context.
- `__init__.py`: package marker with minimal exports. It does not create a new
  public API surface.

The split prevents the large method from merely becoming two large files.

## Dependency Direction

Dependencies remain one-way:

```text
chunker.py ----------------------> hierarchical.service
process_text.dispatch ----------> hierarchical.spans

hierarchical.service -----------> hierarchical.builder
hierarchical.service -----------> hierarchical.flatten
hierarchical.builder -----------> hierarchical.spans
hierarchical.builder -----------> hierarchical.leaves
hierarchical.flatten -----------> hierarchical.grouping

hierarchical.models <------------ shared internal types/protocols
```

No module under `hierarchical/` may import `chunker.py` or `process_text`. Leaf and
grouping modules must not import service, builder, or flatten. This keeps the
package free of circular ownership.

## Internal Contracts

### HierarchyContext

`HierarchyService` receives a context satisfying a narrow protocol. `Chunker`
passes `self`, but the hierarchy package is written against only these needs:

- current configuration defaults
- `_enforce_text_size(...)`
- `_normalize_method_argument(...)`
- `_resolve_method(...)`
- `_sanitize_input(...)`
- `chunk_text(...)`
- `chunk_text_with_metadata(...)`
- `normalize_chunk_type(...)`

The service is stateless beyond its context reference. It does not copy
configuration, cache strategies, or retain resolved options between calls.
Context methods are looked up at call time so runtime configuration and public
leaf-chunking overrides remain effective.

### ResolvedHierarchyOptions

Use a frozen internal dataclass to carry:

- resolved method
- max size
- overlap
- language
- shallow-copied method options
- `sanitize_output`

Method-option preparation preserves the current order and failure behavior:

1. Create `method_options` with `dict(method_options or {})`.
2. Default `sanitize_output` to `True`.
3. When the key exists, evaluate `bool(method_options.get("sanitize_output"))`
   inside the current noncritical-exception policy and fall back to `True` when
   that evaluation raises a covered exception.
4. Remove `sanitize_output` from the copied options before method resolution and
   before every leaf strategy call.
5. Preserve the identities of all other nested option values.

The model must not introduce validation or coercion absent from current behavior.
In particular, resolved option values retain current truthiness and runtime type
behavior. Internal annotations use `Any` where the existing API admits uncoerced
values; the frozen model does not add `__post_init__` validation.

Public hierarchy trees and chunks remain dictionaries.

## Shared Span API

`hierarchical.spans.compute_paragraph_spans(text, template=None)` becomes the one
supported internal detector. It preserves current handling for:

- blank lines and paragraphs
- ATX headers
- horizontal rules
- fenced code blocks, including unclosed fences
- ordered and unordered list lines
- Markdown table lines
- safe custom template boundary kinds
- template rule count and pattern-length caps
- regex-safety checks, flags, warnings, and fallback behavior

The first extraction preserves the current helper lookup behavior rather than
hardening it implicitly. Boundary preparation uses `check_pattern` and
`compile_flags`; failure under the current noncritical-exception policy disables
custom template patterns for that call. Accepted patterns use `safe_search` when
its call-time lookup succeeds. If that lookup raises a covered exception, matching
falls back to the compiled pattern's direct `search`, and per-pattern search
failures continue to skip that pattern. Changing or removing this fallback is a
security hardening change and must pass the behavior-correction gate separately.

Tree construction and multi-level `process_text` dispatch import this function
directly from `hierarchical.spans`.

`Chunker._compute_paragraph_spans(...)` is removed without a transitional alias.
This intentionally removes its private instance monkeypatch seam. Tests that need
to isolate multi-level span behavior patch the dependency in
`process_text.dispatch` instead.

The hierarchy-only `_extract_header_title(...)` helper moves into `builder.py`
and is also removed from `Chunker` without a transitional alias. Repository
inventory found no consumers outside the current hierarchy implementation.

## Public Entry Points

The tree and flatten methods become straight service delegates while retaining
their current signatures and doc-level behavior:

```python
def chunk_text_hierarchical_tree(...):
    return HierarchyService(self).build_tree(...)

def flatten_hierarchical(self, tree):
    return HierarchyService(self).flatten(tree)
```

The flat method must continue composing the two public methods:

```python
def chunk_text_hierarchical_flat(...):
    tree = self.chunk_text_hierarchical_tree(...)
    return self.flatten_hierarchical(tree)
```

It must not call a service-level `build_flat` shortcut. The current composition
allows subclasses and tests to override either public operation, and that public
seam remains supported.

The package-level `Chunking.flatten_hierarchical(tree)` helper remains unchanged.
It continues to catch its current explicit exception set and return `[]`; the
internal service does not absorb or broaden that compatibility policy.

## Tree Construction Data Flow

`HierarchyService.build_tree(...)` performs the existing sequence:

1. Validate that input is a string.
2. Return the current minimal empty hierarchy shape for empty text.
3. Enforce configured text size with the existing source label.
4. Shallow-copy method options.
5. Resolve `sanitize_output`, method, size, overlap, and language with current
   precedence and truthiness.
6. Sanitize once while retaining original-text coordinate space.
7. Compute spans from original text and the optional template.
8. Build the root, preface, nested header sections, bold subsections, and leaf
   blocks.
9. Delegate each leaf to `leaves.py` using the resolved options and source spans.
10. Close open section bounds and return the current dictionary schema.

The empty tree remains intentionally smaller than a non-empty tree. The refactor
does not fill in missing method, language, size, overlap, level, title, or offset
fields for empty input.

## Leaf Chunking And Offset Behavior

`leaves.py` preserves the current method-specific branches:

- rewrite methods emit rewritten text with `None` offsets and
  `offsets_valid=False`;
- words and sentences make one metadata-bearing strategy call on the successful
  path; returned results with missing or non-integer offsets are skipped rather
  than retried;
- tokens have the same per-result skip behavior for missing or non-integer
  metadata offsets;
- words, sentences, and tokens enter their local plain `chunk_text` fallback only
  when the metadata-bearing call raises an exception covered by the current local
  policy;
- structure-aware mode carries the exact block span;
- other methods use rolling, segment-bounded source search;
- the last-resort mapping remains bounded to the source block and makes monotonic
  progress under current rules.

Sanitized and raw source slices retain their current selection behavior. The
refactor does not reinterpret missing metadata, change exception breadth, or
normalize questionable offsets unless a separate correction satisfies the gate.

Leaf call multiplicity and order are compatibility behavior because rewrite
methods can invoke external or LLM-backed work. Per non-empty source block, a
successful rewrite or ordinary-method path calls `chunk_text` once; a successful
words, sentences, or tokens path calls `chunk_text_with_metadata` once and
`chunk_text` zero times; an exception fallback calls metadata first and plain
chunking second; and structure-aware mode calls neither leaf API. The outer
last-resort path reuses already computed chunks and calls `chunk_text` only when no
plain result has been obtained. Consequently, when an earlier plain `chunk_text`
attempt itself raises a covered exception, the current outer fallback makes one
additional plain attempt. The extraction preserves that bounded existing retry
but must not add preflight calls, retries, or duplicate invocations beyond these
characterized paths.

## Flattening And Grouping Data Flow

`HierarchyService.flatten(tree)` preserves current permissiveness rather than
adding a schema validator:

1. Non-dictionary top-level input returns `[]` through the public method.
2. Use `root` when truthy, otherwise construct the existing legacy `blocks`
   fallback.
3. Traverse sections and accumulate ancestry titles.
4. For structure-aware sections, gather child chunks and delegate element or
   weighted grouping to `grouping.py`.
5. Preserve header buffering, separators, language-sensitive joining, overlap,
   element weights, and source-offset aggregation.
6. Shallow-copy the top-level chunk metadata dictionary before adding ancestry,
   section path, normalized chunk type, chunk index, and total count.
7. Return fresh flat chunk dictionaries.

The implementation must characterize selected partial and malformed trees and
preserve their exact current result or exception. It must not assume all malformed
children are skipped, because current behavior is not uniformly tolerant.

Flattening creates a fresh flat chunk dictionary and a fresh top-level metadata
dictionary for each output row. Existing nested mutable metadata values retain
their identities; this is a shallow copy, not deep isolation. The traversal's
current `ancestry_titles` list is assigned directly and may therefore be shared by
multiple output rows in the same ancestry context. Flattening itself does not
mutate caller-owned nested values, but later caller mutation through these aliases
remains observable. Changing that aliasing requires a separately gated correction.
Existing `chunk_index` and `total_chunks` values retain `setdefault` semantics.

## Error And Logging Policy

- Non-string tree-construction input continues to raise `InvalidInputError`.
- Unsafe, invalid, or overlong template boundary rules retain current skipping
  and warning behavior.
- Expected metadata and offset failures retain their current local fallbacks.
- Unexpected exceptions that currently propagate continue to propagate.
- The service does not add a broad top-level catch.
- Existing compatibility catches continue using the shared
  `CHUNKER_NONCRITICAL_EXCEPTIONS` policy where current code does.
- Log levels and materially stable diagnostic text are preserved, especially for
  offset and metadata fallback paths.

## Behavior Correction Gate

Compatibility is the default. A correction may be included only when all of the
following hold:

1. A focused test fails against the pre-refactor `origin/dev` baseline, or an
   exact documented invariant is cited and demonstrably violated.
2. The correction is local to the extracted hierarchy subsystem.
3. A regression test proves the intended behavior.
4. The correction and rationale are recorded in the Backlog task and PR change
   summary inputs.
5. The correction does not redesign a public signature, hierarchy schema, or flat
   chunk shape.
6. The correction is committed separately from structural extraction.

Unreproduced findings are documented and deferred. Refactor-induced failures are
not classified as pre-existing corrections.

No behavior corrections are approved at design time. Removal of the two private
helpers is an explicit architectural decision, not a defect correction.

## Testing Strategy

Add frozen characterization outputs before moving production logic. Expected
trees and flat chunks must be explicit fixtures or dictionaries, not values
generated by the refactored implementation during the test.

Coverage includes:

- spans for paragraphs, blanks, headers, rules, lists, tables, code fences, and
  custom boundaries;
- safe template behavior for rejected, overlong, invalid, and accepted patterns;
  `safe_search` use and its current direct-search fallback; and warning level plus
  materially stable message assertions;
- tree construction for preface content, nested sections, bold subsections,
  header-only sections, duplicate text, and sanitization modes;
- leaf behavior for words, sentences, tokens, structure-aware mode, rewrite
  methods, metadata fallback, and bounded naive mapping;
- exact leaf API call order and multiplicity on successful, metadata-exception,
  missing-offset, rewrite, ordinary, structure-aware, and last-resort paths;
- exact `sanitize_output` default, truthiness, covered-exception fallback, removal
  before method resolution and strategy dispatch, and nested-option identity;
- ancestry, section paths, chunk-type normalization, and index defaults;
- legacy `blocks`, structure-aware element grouping, weighted grouping, overlap,
  header buffering, and no-space languages;
- input-tree and top-level metadata non-mutation, shallow nested-value identity,
  and current ancestry-list aliasing;
- exact current behavior for selected partial and malformed trees;
- public flat composition through overridable tree and flatten methods;
- unchanged `process_text` hierarchical outputs;
- unchanged multi-level `process_text` outputs after direct span import;
- the package-level flatten helper's explicit exception-to-empty-list behavior;
- exact public hierarchy signatures through `inspect.signature`;
- import boundaries enforced with AST-based import inspection, plus removal of
  private protocol/helper members;
- offset and metadata fallback log levels and materially stable diagnostic text.

Verification includes:

1. New hierarchy characterization and component tests after every stage.
2. Existing hierarchy, template, offset, `process_text`, and streaming-overlap
   tests.
3. The complete `tldw_Server_API/tests/Chunking` suite.
4. Compilation and unused-import checks for touched files.
5. Bandit over touched production Chunking paths.
6. `git diff --check`.

Any unrelated failure is investigated and recorded before proceeding rather than
silently excluded.

## Implementation Staging

1. Reconcile `origin/dev` against the recorded baseline and, if it advanced,
   update the baseline and rerun the focused suite before production edits.
2. Add and freeze baseline characterization tests, including option, aliasing,
   call-trace, regex, logging, signature, and import-boundary contracts.
3. Add models and span detection, wire both active callers, then remove
   `_compute_paragraph_spans` and its `ProcessTextContext` member.
4. Extract leaf processing and tree construction, wire the public tree method,
   then remove `_extract_header_title` from `Chunker`.
5. Extract grouping and flatten traversal, then wire the public flatten method.
6. Confirm the public flat method still composes the two public methods.
7. Remove stale imports, tighten protocols, and verify dependency direction.
8. Apply only qualifying corrections as separate red/green commits.
9. Run the complete verification gate and record results.

Every stage wires new code into the active production path immediately. No
dormant duplicate subsystem is added, and focused tests must pass before the next
stage begins.

## Review Risks

- Circular imports if hierarchy modules import `Chunker` or `process_text`.
- Behavior drift from option coercion, deep copying, or stricter tree validation.
- Public composition drift if the flat method bypasses overridable public methods.
- Offset drift across sanitized/raw text and repeated source content.
- Side-effect drift from duplicated leaf calls, implicit retries, or changed
  metadata-fallback triggers.
- Metadata drift from replacing `setdefault` with assignment.
- Mutation or identity drift if flattening writes into caller trees, deep-copies
  nested values, or changes current ancestry-list aliasing.
- Grouping drift in final-window overlap, by-kind weights, separators, or
  header-only sections.
- Error drift from broadening service-level catches.
- Security/logging drift from changing regex helper lookup, direct-search fallback,
  warning levels, or materially stable diagnostics during structural extraction.
- Accidental adoption of unmerged behavior from other historical worktrees rather
  than the refreshed `origin/dev` baseline.

## Acceptance Criteria For The Implementation PR

- The full hierarchy subsystem is extracted into the approved focused modules.
- Public hierarchy signatures, dictionary shapes, composition, outputs, and
  package-level fallback behavior remain compatible.
- The two approved private helper seams are removed and active callers are
  migrated.
- The hierarchy package does not import `Chunker` or `process_text`.
- Characterization and component tests cover spans, trees, leaves, grouping,
  flattening, shallow aliasing, exact call traces, logging, signatures, import
  boundaries, and integration behavior.
- The implementation baseline is explicitly reconciled with current `origin/dev`;
  an advanced baseline is recorded and re-characterized before production edits.
- Any included correction satisfies and records the correction gate.
- The complete focused verification gate passes, including Bandit.
