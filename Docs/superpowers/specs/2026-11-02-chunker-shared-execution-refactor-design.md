# Chunker Shared Execution Subsystem Refactor

Backlog design task: `TASK-15414`

Status: Approved by requester on 2026-11-02, together with ADR-086. Approval
authorizes implementation planning; execution still requires plan review and
an execution-method choice.

Planning task: `TASK-15415`.

Implementation plan: [Shared execution plan](../plans/2026-11-02-chunker-shared-execution-refactor.md).

Tracking reconciliation: the live scan found unrelated UserProfiles PR #3404
using `TASK-15413`. The official backlog-py CLI renumbered only this design
record to `TASK-15414`, preserving its history and directory. Historical
mentions of TASK-15413 remain provenance; unrelated records are untouched.

## Intent

Continue decomposing `Chunker` after the merged `process_text` and hierarchical
extractions. Move the execution bodies of `chunk_text`,
`chunk_text_with_metadata`, and `chunk_text_generator` into a focused internal
subsystem. Keep the existing public methods as thin delegates and make shared
preparation independently testable without erasing differences between paths.

The intended outcome is clearer ownership and less duplication, not a new
chunking API, changed output, or new behavior. The requester chose the full
internal-subsystem approach B over helper-only extraction A on 2026-11-02.

## Baseline And Isolation

Pinned `origin/dev`: `3ab1ce5d3a49d5a7fd130e8939d55d7639e5a373`.

Design branch: `codex/chunker-execution-refactor-design`, in the existing
isolated worktree. The dirty main checkout is untouched. The separate
bookkeeping branch and draft PR #3397 are not part of this design or its edits.
The implementation from PR #3095 is already merged as `6bf581da089a7aa6d90440a8801de07f8fdd238b`.

Chunking production files are unchanged between that merge and the pinned
baseline. The intervening Chunking test change updates a shared-provider
credential fixture to the authorized repository API and asserts its scope,
user and provider. Fresh full Chunking baseline: **796 passed, one existing
PyThaiNLP skip, 2000 warnings, 42.02s**
(`/tmp/chunker_execution_design_baseline.txt`).

Before implementation, fetch `origin/dev`, inspect scoped intervening changes,
pin the actual implementation baseline, and rerun characterization against
it. Do not silently apply stale expectations after relevant changes. Return
material contract differences to design review. Do not include the unrelated
bookkeeping commit merely because this worktree previously held that branch.

## Goals

- Retain public signatures, annotations, defaults, return shapes and hooks.
- Give each execution path a readable entry point outside `chunker.py`.
- Share equivalent option/strategy preparation without a universal execution
  loop or a collection of behavior-changing convenience flags.
- Preserve cache identity/copy policy, metadata mutation, logging, errors,
  generator timing and concurrency behavior.
- Keep `process_text`, hierarchy and file chunking calling the existing public
  methods, including subclass and instance overrides.
- Freeze subtle baseline behavior before extraction, with tests of actual
  output and failures in addition to interaction traces.

## Non-Goals And Anti-Goals

- No strategy implementation, factory registry or strategy-cache redesign.
- No movement of `LRUCache`, result-cache key construction or cache ownership.
- No sanitizer, security policy, method-resolution or LLM-scope redesign.
- No changes to file buffering, overlap retention or split-point algorithms.
- No telemetry framework migration or new metric names/labels.
- No change to `process_text` or hierarchy ownership/dependency direction.
- No new public configuration, dependency, async API or output model.
- No opportunistic behavior correction, broad formatting or unrelated test fix.
- Do not make the generator eager, retain a lock over a consumer pause, route
  metadata through string chunking, or add caching to a previously uncached path.

## Approaches And Decision

### A: Shared Setup Helpers Only

Extract common option resolution and strategy preparation, leaving all three
execution bodies in `Chunker`. This is smaller and reduces repeated setup, but
leaves caching, exception regions and output handling interleaved in the facade.

### B: Internal Execution Subsystem - Selected

Extract three path-specific execution entries and share only the preparation
that is equivalent. Leave thin public delegates in `Chunker`. This moves more
code and needs stronger characterization, but matches the successful existing
pipeline/service pattern and gives execution a clear owner.

The requester selected B, then approved the revised layout and contracts below
on 2026-11-02 after design review. The remaining builder choices are limited
to the discretion explicitly recorded later in this document.

## Ownership And Dependencies

Proposed internal package:

```text
tldw_Server_API/app/core/Chunking/execution/
    __init__.py
    models.py
    preparation.py
    strategy.py
    service.py
```

- `models.py`: narrow `ExecutionContext`, cache/hook protocols as needed, and
  a small frozen `PreparedExecution` dataclass. Public results stay unchanged.
- `preparation.py`: common option preparation, with explicit differences for
  text, metadata and generator calls. It does not sanitize or execute strategies.
- `strategy.py`: select and prepare a strategy using supplied resolved values
  and effective LLM hooks. It neither owns caches nor acquires/releases locks.
- `service.py`: `ChunkExecutionService` with `chunk_text`,
  `chunk_text_with_metadata`, and `iter_chunks` entries. It owns path-specific
  sequencing, lock scope, result-cache interactions, execution and finalization.
- `__init__.py`: minimal internal exports; no root-package public API expansion.

`chunker.py` imports the execution service. The service uses preparation and
strategy helpers; helpers depend on models and existing neutral modules such
as `base`, `option_utils`, `llm_context`, `exceptions` and `error_policy`.
None imports `Chunker`, `chunker.py`, `process_text` or `hierarchical`, even
through a local import. Do not add a circular dependency via package exports.

Keep these existing methods/objects on `Chunker`: sanitization and size checks;
method normalization/resolution, aliases and overlap override; strategy
factories, registry, per-call lookup and cache modes; LLM/language synchronization;
LLM signatures and cache keys; result cache and its inspection methods; file
streaming helpers. The extracted service calls these through the live context.
This phase changes execution ownership, not the ownership of those facilities.

The context protocol includes only attributes and methods actually used by
the service/helpers. In particular, it exposes the live config, cache mode,
lock and optional result cache, plus the retained methods named above. Use
structural protocols rather than importing `LRUCache` from `chunker.py`.
Do not expose the entire facade through `__getattr__`, clone its state, retain
an eager snapshot of its helpers, or build a second registry.

`PreparedExecution` carries effective method, size, overlap, language, raw/cache
options, strategy options, tokenizer override and path-specific alignment
state as needed. Its constructor does not validate/coerce values. A frozen
dataclass does not imply deep copying or freezing its dictionaries. Preserve
existing shallow copies, option values and observable hook call order.

### Keep User Options Separate From Internal Parameters

User strategy options are data, not internal calling parameters. Pass them to
preparation/strategy helpers as one explicit mapping argument; do not expand
them with `**options` beside internal arguments such as an execution kind,
context or hooks. Service construction receives only internal wiring, not
expanded user options. An execution entry may receive expanded options only
when its parameter-binding rules mirror the existing public method; otherwise
it also receives an explicit mapping. Do not create newly reserved option
names or use internal parameter names as implicit filters.

For example, `mode`, `context`, `hooks`, `source`, `path`, `request`, `strategy`
and `options` currently reach an injected strategy unchanged on all three
paths. An internal `mode="text"` parameter must neither consume a user-supplied
`mode` nor collide with it as a duplicate keyword. The only removals/coercions
are the baseline path-specific rules described below. Preserve their existing
mapping copies and value identities; "explicit mapping" does not authorize
deep copying, freezing values or passing unknown options to new validators.

## Public Facade And Live Hooks

Preserve the three declarations, annotations and defaults exactly; add no
public constructor state. Construct a lightweight service per call. Its
constructor only retains the live context and observability hooks; it does
not read config, resolve strategies, validate input or emit telemetry.

The two eager methods return their respective service result directly.
`chunk_text_generator` remains a generator function and uses `yield from`
inside its body. Service construction and all validation for that path occur
on first advancement, not when the caller creates the generator. The internal
`iter_chunks` entry is also lazy. Neither wrapper converts output to a list.

Keep inherited/subclass/instance overrides of the public methods effective for
existing callers. Retain the current private context methods used during
execution as callable seams, with their existing signatures. Resolve them on
the context when needed, including after the caller creates but before it
advances a generator. No private seam removal is proposed in this phase.

Keep `chunker.py`-level logging and `increment_counter` monkeypatch behavior
through narrow injected forwarding hooks. The forwarders resolve the current
module globals when an event is emitted, rather than caching those objects
in a long-lived service. Existing message text, levels, arguments, metric
names/labels, event order and suppression boundaries are preserved. Loguru's
source filename/function/line may naturally identify the extracted code;
do not promise byte-identical source-location metadata or new log redaction.
Do not import the facade module from the subsystem to perform forwarding.

## Path-Specific Compatibility

### Input And Option Preparation

Sanitize once through the existing context method. For text and metadata,
empty/whitespace-only sanitized input returns `[]` before size/default/strategy
work. The generator has no equivalent early return: preserve its size check
and subsequent strategy invocation, even for empty input. Keep size-check
`source` labels and sanitization/security behavior unchanged.

Use the current rules: size/overlap defaults apply only to `None`; language
falls back by truthiness; method normalization precedes resolution, aliases
and method-specific overlap override. Do not parse/coerce additional values
or reject values earlier. Preserve handling of bool-as-int, nonpositive sizes,
negative overlap, overlap at/above size, enums and method-like objects.

Preserve distinctions in `align_text_to_source` handling:

- Text removes the option before aliases/cache-key/strategy processing.
- Metadata removes it before resolution, coerces it with the existing helper,
  and forwards the coerced flag only when the original value is not `None`.
- Generator keeps it in the forwarded options as the baseline does.

All three remove `code_mode` and tokenizer-selector options before strategy
invocation. Preserve tokenizer alias precedence, string trimming, truthiness
and tokens-only per-call selection, including behavior for non-string values.
Keep raw/cache options separate from forwarded strategy options. Preserve the
plain-text overlap-check debug message versus the other paths' suppression.

### Plain Text Execution And Result Cache

Only `chunk_text` uses the result cache. Preserve size-based eligibility,
configuration fallbacks and reasons, key inputs, option copies, LLM signature,
hit/miss/skip/store metrics and result identity/copy semantics. Read current
context facilities at their existing points, not at service initialization.

Compute effective LLM hooks and their signature in the existing order, even
when result caching is disabled. A hit returns without constructing or
executing a strategy. Preserve skipped/empty-result storage behavior. Cache
lookup and signature/key construction remain outside the existing execution
catch region: do not newly suppress or wrap their failures. Cache writes
remain noncritical inside their existing suppression/metric handling.

For shared strategy mode, hold the existing lock through strategy acquisition,
LLM/tokenizer/language setup and `strategy.chunk`. Release it before result
storage and success logging. Other modes retain their current behavior.
Return the exact strategy/cache result rather than normalizing or copying it.

### Metadata Execution

Use `strategy.chunk_with_metadata` directly and do not route through
`chunk_text`, the result cache, a string generator or new output converters.
Preserve strategy result order and objects.

When alignment is enabled and the result is a list, retain the existing
per-item mutation, integer-only offset handling, clamping and suppression.
Use the sanitized source text, not the original unsanitized input. Do not
copy metadata, repair offsets, coerce additional values, or rebuild results.
Perform alignment and final success logging outside the strategy lock and
outside the existing strategy-execution catch region, as in the baseline.

### Generator Execution And Concurrency

Preserve `inspect.isgeneratorfunction`, first-advance validation, partial
iteration, no-iteration behavior, ordered yields, `close`, `throw`, cleanup
and exception timing. Do not prefetch, materialize or cache chunks.

In shared mode the generator deliberately creates an ephemeral strategy,
including the baseline behavior with explicitly injected strategies. Other
modes retain existing strategy selection; tokenizer overrides can still force
a per-call instance. Do not unify this rule with the eager paths' selection.

Acquire the lock only for setup and creation of the underlying iterator.
Look up/call a callable `chunk_generator` there; otherwise call `chunk` and
wrap its result in `iter` there. Actual `yield from` runs outside the lock,
including its first advancement and consumer-controlled pauses. Preserve
fallback logging and the distinction between iterator creation and iteration.
Use deterministic events/barriers for concurrency regressions, not sleeps.

### Errors, State And Observability

Import the existing shared noncritical exception tuple; do not duplicate,
broaden or narrow it. Text and metadata preserve their distinct catch regions,
unchanged `ChunkingError` passthrough, message text and exception chaining.
The generator gets no new enclosing translation/suppression handler.
Preparation failures, cache failures, alignment failures and iteration failures
must not be moved across those regions as a side effect of extracting helpers.

Obtain effective LLM hooks once at the current point per path. Pass them to
strategy preparation without a second lookup. Preserve synchronization order:
LLM hooks, tokenizer update/reset and then conditional language update. Keep
the existing shared/thread/call modes, custom factories and injected strategy
semantics; do not add deep copies, restore strategy state differently or
acquire new locks. The facade remains the owner of all mutable facilities.

## Characterization And Verification

Before product edits, add baseline characterization that passes against the
pinned implementation. New contracts must assert real returned text/metadata,
identity, exceptions or progress as applicable, not collector traces alone.
Freeze:

- Complete independent structural signatures and generator-function identity.
- Empty/non-string/oversized input, sanitized source and exact size labels.
- Default/alias/overlap handling and each path's exact forwarded options.
- User option names overlapping internal helper parameters, on all three
  paths. Include the example names above and the actual plumbing parameter
  names chosen during implementation; assert returned output and the exact
  strategy option values, not just that no `TypeError` was raised.
- Tokenizer overrides, reset flags and mutable LLM/language call ordering.
- Cache hit/miss/skip/store, key/signature inputs, copy modes, uncached paths
  and failures on both sides of the execution catch boundary.
- Metadata object/list identity, alignment mutation and malformed-item controls.
- All strategy-cache modes, injected strategies and shared-mode generator rules.
- First/partial/no iteration, callable/noncallable/missing generator fallback,
  failure timing, close/throw delegation and release of locks during iteration.
- Context override/monkeypatch resolution, including patching between generator
  creation and advancement and observability replacement during execution.
- Existing public callers in process_text, hierarchy and file-streaming paths.

Use deterministic fakes for factories, tokenizers and LLMs in component tests;
retain real public-output integration tests and existing assertions. Add
property-based cases where useful for generated option combinations, source
alignment and eager/generator equivalence only for strategies that already
promise equivalence. Do not assert universal equivalence or new validation.

Extraction-specific tests must demonstrate RED for absent/bypassed components
or wrong ordering, then GREEN for active public wiring. Fake-context component
tests need not import `Chunker` directly or construct it. The existing root
package eagerly exports the facade; redesigning that bootstrap is out of scope.
The boundary prohibits facade imports in the new subsystem itself, not the
root package's pre-existing initialization. Boundary tests must fail when
required modules are absent rather than skipping them.

### Cold-Import Gate

The future implementation must add fresh-interpreter smoke tests in addition
to static boundary tests and ordinary in-process integration tests. A warmed
test process can already contain the facade and conceal initialization-order
problems introduced by the root package's eager facade export.

Run each requested entry point in its own subprocess using the active Python
interpreter and the repository's isolated test configuration/import path:

- The root `tldw_Server_API.app.core.Chunking` package first, then the execution
  package and its modules.
- Each execution entry point first in a separate subprocess: `execution`,
  `execution.models`, `execution.preparation`, `execution.strategy` and
  `execution.service`, all under the same full Chunking package prefix.

After the requested import, import the root and execution service and assert
that their required `Chunker` and `ChunkExecutionService` exports resolve.
Requesting a submodule first still runs Python's real parent-package bootstrap;
do not fake away that behavior, preload the facade, stub modules in
`sys.modules`, or simulate a cold import by deleting modules in a warmed
interpreter. Do not require `Chunker` to be absent after bootstrap.

Capture exit status/stdout/stderr, bound subprocess waits, and fail on missing
modules/exports, exceptions or hangs. Do not skip or retry failed cold imports.
The smoke tests perform imports/export checks only: they do not call providers,
download tokenizers/models or initialize application services. Root bootstrap
redesign remains out of scope; an inherited failure still uses the separate
baseline-correction gate rather than silently changing initialization.

### Full Verification

Run full Chunking, targeted process_text/hierarchy/file regressions, static
typing on the new subsystem, scoped Ruff/Black/compileall, test-quality and
function-size ratchets, task-format checks and touched-scope Bandit. Record
baseline errors rather than claiming unrelated checks globally green. Add new
test files to the actual existing CI shard assignments and run their coverage
guard before publication. Do not weaken scanners, thresholds, assertions,
security checks or required CI; remove only demonstrably obsolete exceptions.

## Delivery And Correction Boundaries

This document is a design, not an implementation plan. After requester
approval, create the implementation plan and its separate task. The plan
should freeze contracts first, move path bodies into active delegation in
reviewable stages, then consolidate proven-equivalent preparation and finish
integration verification. Every moved component must be used by the public
path, not left as a dormant duplicate. Keep tests green between stages.

Refactor-induced regressions must be fixed before proceeding. A suspected
inherited defect requires reproduction against the pinned baseline and a
separate owner-approved behavior-correction scope before fixing it. Selection
of approach B is not approval for hardening changes, normalization of path
differences, or repairs to unrelated UX/AuthNZ tests.

Independent implementation review, fresh verification/Bandit and the current
live merge rules will gate its future PR. That PR needs its own requester-written
Change summary describing what changed and why; do not reuse the old hierarchy
paragraph or treat design approval as authority to merge without the gates.

## ADR Assessment And Builder Discretion

ADR required: **yes**. The proposed one-way execution-module dependency
boundary is a durable module-boundary rule, even though it is local to Chunker
and follows the existing pipeline/service pattern. Repository policy does not
require a decision to be repository-wide before an ADR is needed.

[ADR-086: Chunker Execution Live-Context Boundary](../../ADR/086-chunker-execution-live-context-boundary.md)
records this decision with status **Accepted** following requester approval
on 2026-11-02. Its scope is the
new execution subsystem; it does not retroactively change the accepted behavior
or ownership of process_text, hierarchy or other modules. Public, persistence,
security and concurrency behavior remains unchanged. This supersedes the
initial draft's ADR-required-no assessment; accepted ADRs are not rewritten.
ADR-002/059/084 still govern task tracking/tooling/IDs, and ADR-006 governs
portable Bandit output. Implementation planning is authorized; product/test
edits still wait for review of the resulting plan and an execution choice.

The builder may choose private helper names, precise internal annotations and
test-file organization within these responsibilities. The hooks container may
be a small dataclass/protocol if that removes wiring complexity. These are
engineering calls, not permission to change ownership, temporal semantics,
option handling, exception boundaries or public interfaces. Do not introduce
additional public modes, an executor hierarchy or speculative abstractions.

## Approval And Completion

- Approach B: selected by requester.
- Current production/test baseline: verified as recorded above.
- Original independent builder check: completed on 2026-11-02 and returned
  "nothing important left" for the initial draft. A subsequent design review
  identified option-namespace, ADR-recording and cold-import verification gaps;
  the requester authorized correcting all three.
- Bounded independent follow-up check: completed on 2026-11-02; the reviewer
  confirmed that all three revisions address their risks and returned
  "nothing important left". No material planning questions remain from that
  check. The review does not substitute for requester design/ADR approval.
- Requester approval of this revised document and ADR-086: received on
  2026-11-02 after the review found no further actionable issues.
- Implementation planning: authorized and tracked in TASK-15415.
- Product/test edits and execution: not started; plan review and an execution
  method are still required before implementation.
