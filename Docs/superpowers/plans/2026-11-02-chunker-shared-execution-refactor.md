# Chunker Shared Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the three chunk execution paths into an internal service while preserving their public and path-specific behavior.

**Architecture:** Keep `Chunker` as the public facade and owner of mutable facilities. Move each execution body into active service delegation first, then share proven-equivalent preparation and strategy setup. Use a live structural context and facade-supplied observability hooks, with no reverse imports or new result normalization.

**Tech Stack:** Existing Python >=3.12, protocols/frozen dataclasses, Loguru, pytest/Hypothesis, Ruff/Black/mypy, Bandit and repository CI guards; no new dependencies.

**Spec:** [Approved design](../specs/2026-11-02-chunker-shared-execution-refactor-design.md).

**ADR:** [Accepted ADR-086](../../ADR/086-chunker-execution-live-context-boundary.md).

**Tracking:** Implementation `TASK-15416`; completed planning `TASK-15415`; approved design `TASK-15414` (renumbered from colliding TASK-15413). Implementation links this exact plan, spec and ADR rather than executing under a completed planning task.

**Status:** Approved by requester on 2026-11-02 for Native execution followed by independent whole-branch review. Implementation tracked in TASK-15416.

## Global Constraints

- "No new public configuration, dependency, async API or output model."
- "No strategy implementation, factory registry or strategy-cache redesign."
- "No movement of `LRUCache`, result-cache key construction or cache ownership."
- "No sanitizer, security policy, method-resolution or LLM-scope redesign."
- "No changes to file buffering, overlap retention or split-point algorithms."
- "No telemetry framework migration or new metric names/labels."
- "No private seam removal is proposed in this phase."
- "Do not create newly reserved option names or use internal parameter names as implicit filters."
- Preserve all three public declarations, annotations, defaults and docstrings; preserve `inspect.isgeneratorfunction` for the generator.
- Preserve option copies/value identities, source alignment, returned objects, logging/metrics, exact catch regions and generator lifecycle; do not merge distinct execution policies.
- The new subsystem imports neither the facade nor process_text/hierarchy, including local imports. Existing parent-package eager exports remain unchanged.
- Use only the isolated worktree on `codex/chunker-execution-refactor-design`; main, the bookkeeping branch/PR and historical records are not edit targets.
- Track edits using backlog-py under ADR-059. Commit related task/plan updates with each stage. Never use Node Backlog/MCP or broad staging.
- Reproduced inherited defects need a separate requester-approved correction; refactor-induced failures must be repaired before the next stage.
- Plan approval/execution selection is not permission for an ungated merge. A future PR needs independent review, fresh verification, its own human Change summary and live required CI.

## Review Focus

1. A custom strategy receives `mode/context/hooks/source/path/request/strategy/options` and internal-plumbing names without collision or value replacement: Task 1 contracts, Task 5 helper tests.
2. Config/helpers/observability change after generator creation or during execution: Task 1 characterization, Tasks 2/4 wiring tests, Task 5 live-helper tests.
3. A shared-mode generator is paused, sent a value, thrown into, closed or never started: Task 1 characterization, Task 4 lifecycle/concurrency tests.
4. A cache returns a mutable hit or fails during lookup/key/signature/write: Task 1 identity/error characterization and Task 2 service tests.
5. Metadata has out-of-range, boolean, missing or raising offset attributes: Task 1 source-alignment characterization and Task 3 metadata tests.

## Baseline, Runtime And File Map

Approved design baseline: `3ab1ce5d3a49d5a7fd130e8939d55d7639e5a373`.
Planning-time refreshed dev: `5937f33065c4b5b5b2d0f731d1914c0651532eb5`.
The scoped diff in Chunking production/tests, AGENTS and ADRs is empty between
them. Fresh unchanged-code full Chunking: **796 passed, one existing PyThaiNLP
skip, 2000 warnings, 41.41s** (`/tmp/chunker_execution_plan_baseline.txt`).
These are baseline results, not proof of the future extraction.

Run commands from the isolated repository root in a shell where this setup has
been applied. Repeat activation in a fresh shell; do not assume a local `.venv`
exists inside the worktree:

```bash
source "${TLDW_VENV:-$(git rev-parse --path-format=absolute --git-common-dir)/../.venv}/bin/activate"
export PYTHONPATH="$PWD/tools/backlog-py/src${PYTHONPATH:+:$PYTHONPATH}"
```

At execution start, search for an existing implementation record first. If
none exists, use backlog-py to create `Implement Chunker shared execution
subsystem refactor` with a unique ID above HEAD/latest-dev/all paginated open-PR
task IDs. Record its exact ID/path as `IMPLEMENTATION_TASK_ID` and
`IMPLEMENTATION_TASK_FILE` in the shell/task notes. Each commit below stages
that record explicitly; never substitute the unrelated record now using 15413.

```bash
: "${IMPLEMENTATION_TASK_ID:?set from the official implementation task record}"
: "${IMPLEMENTATION_TASK_FILE:?set to the exact repository-relative task path}"
test -f "$IMPLEMENTATION_TASK_FILE"
```

Production files, introduced only when actively wired:

| Path | Owner / first stage |
| --- | --- |
| `tldw_Server_API/app/core/Chunking/execution/__init__.py` | Docstring-only internal package initializer; Task 2 |
| `tldw_Server_API/app/core/Chunking/execution/models.py` | Context/cache protocols and hooks; Task 2. Prepared request added in Task 5 |
| `tldw_Server_API/app/core/Chunking/execution/service.py` | Text entry in Task 2, metadata in Task 3, lazy iterator in Task 4 |
| `tldw_Server_API/app/core/Chunking/execution/preparation.py` | Shared option preparation; Task 5 |
| `tldw_Server_API/app/core/Chunking/execution/strategy.py` | Strategy selection/setup without owning locks; Task 5 |
| `tldw_Server_API/app/core/Chunking/chunker.py` | Three public delegates and private `_execution_hooks()` factory; Tasks 2-4 |

Test paths:

- `tldw_Server_API/tests/Chunking/_execution_test_support.py`: deterministic recording strategy/cache/lock and a minimal fake context, without constructing/importing the facade directly.
- `tldw_Server_API/tests/Chunking/test_chunker_execution_contracts.py`: frozen public baseline behavior, parametrized over the relevant paths.
- `tldw_Server_API/tests/Chunking/test_chunker_execution_components.py`: direct fake-context component tests.
- `tldw_Server_API/tests/Chunking/test_chunker_execution_boundaries.py`: active facade wiring, static dependencies and fresh-process imports.

The three test modules match the existing `test_chunker_*.py` assignment in all
six full-suite matrices (including the inherited release matrix). The support
file is not a collected test module. No workflow, ignore-list or baseline edit
is planned; prove coverage with the guard after the files exist.

## Fixed Internal Interfaces

These choices complete the spec's allowed private-interface discretion. Types
such as `Any` below deliberately retain dynamic baseline inputs/outputs; they
are not instructions to add validation or blanket mypy suppression.

`execution.models`:

```python
class ResultCache(Protocol):
    def get(self, key: str) -> Any: ...
    def put(self, key: str, value: Any) -> None: ...

@dataclass(frozen=True)
class ExecutionHooks:
    get_logger: Callable[[], Any]
    increment_counter: Callable[..., Any]
```

`ExecutionContext` has read-only protocol properties `config: ChunkerConfig`,
`_cache: ResultCache | None`, `_strategy_cache_mode: str`, `_strategy_lock: Any`.
Actual `Chunker` attributes still remain writable/live; the read-only protocol
avoids narrowing mutable protocol fields to a concrete `LRUCache` type.
Declare only these existing context methods, retaining their names/defaults:

```python
def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str: ...
def _enforce_text_size(self, text: str, *, source: str) -> None: ...
def _normalize_method_argument(self, method: Any) -> Any: ...
def _resolve_method(self, method: Any, language: Any, options: dict[str, Any] | None = None) -> str: ...
def _apply_option_aliases(self, method: str, options: dict[str, Any]) -> dict[str, Any]: ...
def _override_overlap_for_method(self, method: str, overlap: Any, options: dict[str, Any]) -> Any: ...
def _get_effective_llm_hooks(self) -> tuple[Any, Any]: ...
def _get_llm_signature(self, *, llm_call_func: Any = _LLM_UNSET, llm_config: Any = _LLM_UNSET) -> str: ...
def _get_cache_key(self, text: str, method: str, max_size: int, overlap: int, language: str, options: dict, *, llm_signature: str = "") -> str: ...
def _create_strategy_instance(self, method: str) -> Any: ...
def _get_strategy_for_call(self, method: str) -> Any: ...
def _sync_strategy_llm(self, strategy: Any, *, llm_call_func: Any = _LLM_UNSET, llm_config: Any = _LLM_UNSET) -> None: ...
def _apply_strategy_language(self, strategy: Any, language: str) -> None: ...
```

Import `_LLM_UNSET` from the existing neutral `llm_context` module. The service
constructor is `ChunkExecutionService(context: ExecutionContext, hooks:
ExecutionHooks) -> None`; it only stores those two references. Its entries
mirror the public argument names/defaults exactly, adding no internal keyword:

```python
def chunk_text(self, text: str, method: Optional[str] = None, max_size: Optional[int] = None, overlap: Optional[int] = None, language: Optional[str] = None, **options: Any) -> list[str]: ...
def chunk_text_with_metadata(self, text: str, method: Optional[str] = None, max_size: Optional[int] = None, overlap: Optional[int] = None, language: Optional[str] = None, **options: Any) -> list[ChunkResult]: ...
def iter_chunks(self, text: str, method: Optional[str] = None, max_size: Optional[int] = None, overlap: Optional[int] = None, language: Optional[str] = None, **options: Any) -> Generator[str, None, None]: ...
```

The explicit `Any` on internal variadic options does not change binding;
leave the existing public facade annotations verbatim.

`chunker.py._execution_hooks() -> ExecutionHooks` returns forwarding callbacks:
`get_logger()` resolves current `chunker.py.logger`, and `increment_counter`
forwards to the current module function when called. Do not capture either
global in a default argument or delegate to a newly imported metrics function.
Emit log calls through `hooks.get_logger().debug/info/warning/error` with the
original arguments and original suppression boundaries.

Task 5 adds frozen `PreparedExecution` with fields `method`, `max_size`,
`overlap`, `language`, `tokenizer_override` typed `Any`; `options_raw` and
`strategy_options` typed `dict[str, Any]`; and `use_per_call_strategy`,
`align_text_to_source` typed `bool`. No runtime coercion occurs in its constructor.
Private helper signatures are:

```python
ExecutionKind = Literal["text", "metadata", "generator"]
def prepare_execution(context: ExecutionContext, *, method: Any, max_size: Any, overlap: Any, language: Any, user_options: dict[str, Any], kind: ExecutionKind, hooks: ExecutionHooks) -> PreparedExecution: ...
def prepare_strategy(context: ExecutionContext, prepared: PreparedExecution, *, effective_llm: tuple[Any, Any], ephemeral: bool, hooks: ExecutionHooks) -> Any: ...
```

The service sanitizes/handles early return/enforces size before preparation.
Only its fixed internal keywords accompany `user_options=options`; never
unpack user options into these helper calls. The service controls the lock.
It computes the generator's `ephemeral` flag before entering the lock at the
baseline point, then passes it explicitly. The strategy helper never rereads
cache mode to reinterpret that decision or performs a second LLM-hook lookup.

## Task 1: Freeze Public Execution Contracts (Stage 1)

**Goal:** Establish an independent, passing compatibility oracle before extraction.
**Success Criteria:** New characterization passes on the reconciled unextracted implementation; no product edit or new skip.
**Status:** Complete

**Evidence:** Fresh original baseline 796 passed/1 established skip; initial characterization 81 passed and full suite 877 passed/1 skip. Final frozen characterization 103 passed, 280 warnings, 3.00s (`/tmp/chunker_execution_task1_frozen.txt`). Ruff/Black/compileall, task-ID/shard guards and test Bandit (B101 only excluded) pass; no production files changed. Independent literal docstrings use `inspect.cleandoc` for Python 3.12 versus 3.13+ compiler indentation portability; final source AST proof will preserve original docstrings exactly. Official implementation task is TASK-15416; all 39 open PR file lists paginated to allocate it.

**Files:** Create the support module and `test_chunker_execution_contracts.py` listed above; update only the active implementation task and this plan's stage evidence.

**Interfaces:** Consumes the current three public methods. Produces test helpers `RecordingLock` (`depth`, event log), `RecordingCache` (configurable get/put result/failure), `RecordingStrategy` (configurable text/metadata/iterator outputs), and `FakeExecutionContext` (neutral config plus the fixed protocol methods). Keep these private to tests; use existing `ChunkResult`/`ChunkMetadata`, not copies of their implementations. A `facade_case` fixture in the contracts module constructs a real `Chunker` with deterministic facilities; component tests later construct only `FakeExecutionContext`.

- [x] **Step 1: Allocate/find the execution record.** Follow the tracking procedure above and record the actual implementation task ID/path; this is required before the first product/test edit.
- [x] **Step 2: Reconcile the execution baseline.** Fetch dev and inspect scoped source/tests, AGENTS, ADR, `pyproject.toml`, `.github/workflows/ci.yml` and relevant guard changes. Follow current queue policy before rebasing the unpublished isolated branch if needed. Preserve all design commits and record the actual base SHA. Relevant behavior changes return to design review.
- [x] **Step 3: Verify the baseline.** Run `python -m pytest tldw_Server_API/tests/Chunking -q`; require zero failures and document the existing optional skip. Do not install dependencies or start services to hide a failure.
- [x] **Step 4: Write characterization tests with frozen expectations.** Capture complete signatures using independent expected `inspect.Signature` objects and resolved annotations; preserve literal baseline public docstrings. Add the named cases below. They assert existing behavior, so their initial expected result is GREEN, not fabricated RED.

| Test name | Required assertions |
| --- | --- |
| `test_public_execution_contracts` | All three signatures/defaults/docstrings equal the frozen baseline; generator is a generator function. |
| `test_path_input_and_option_contracts` | Text/metadata return `[]` before size/default work for sanitized whitespace; generator still invokes size/strategy on first advancement. Invalid text/size source and exceptions match each path. |
| `test_option_namespace_survives_all_paths` | Injected strategy emits the supplied options as actual output; the eight spec names, chosen helper names and nested object identity survive all three paths. Only documented path filters differ. |
| `test_path_alias_overlap_tokenizer_order` | Aliases, None-versus-zero, bool/non-integer controls, code routing, tokenizer precedence/reset and LLM/language order match baseline. Include `align_text_to_source` absent/None/false/true differences. |
| `test_text_cache_identity_and_error_regions` | Cache hit returns the controlled object, avoids factory work and emits exact labels; disabled/threshold/empty controls match. Get/key/signature failures propagate raw; put failures retain original suppression and metrics. |
| `test_metadata_alignment_identity_and_error_regions` | Same list/items/metadata are returned; sanitized source slicing/clamping, bool offsets, missing/raising attributes and per-item failure controls match. Strategy errors translate; alignment/success logging keep their original regions. |
| `test_generator_baseline_lifecycle_and_live_state` | No effects until first advance; later context patches are observed. Iterator setup is locked, next/send/throw/close is not; closing an unstarted generator does no work. Outer completion value remains `None`. |

For a namespace test case, set `user_options` to the three literal values in
the assertion below, and configure the controlled strategy to emit their JSON
as its actual result. Decode that result as `decoded_result`. A separate
identity case includes a mutable `payload` and inspects the exact mapping
received by the strategy as `strategy_options`; do not JSON-round-trip it:

```python
assert decoded_result == {"mode": "user-mode", "context": "user-context", "hooks": "user-hooks"}
assert strategy_options["payload"] is payload
```

Add three concrete Hypothesis contracts in this same stage, with a fresh case
constructed inside each example (`max_examples=50`, `deadline=None`, without
fixture-health-check suppression):

- `test_generated_user_options_reach_each_path`: draw JSON-compatible nested
  values under the eight spec names and `kind/user_options/effective_llm/ephemeral`;
  none is a public formal argument or recognized path option. The controlled
  strategy emits those values as JSON; assert decoded output equals the input
  mapping and received mutable values retain identity.
- `test_generated_alignment_preserves_objects_and_slices`: draw nonempty text
  prefixed with `a` and signed integer start/end offsets. Configure one existing
  `ChunkResult`/metadata object, align via the public metadata path, and assert
  those exact objects survive and `item.text == sanitized_source[start:end]`
  after the baseline clamp. Assert original metadata offsets are not repaired.
- `test_words_eager_generator_equivalence`: draw ASCII word lists, size 1-10
  and overlap 0 through size-1; compare actual public text output with fully
  consumed generator output for the real word strategy. Its eager and
  generator implementations both use `_prepare_chunk_records` on these
  inputs; first verify this invariant on the reconciled baseline. Do not
  generalize it to other strategies or malformed inputs.

Also characterize real LRU `copy_on_access=True/False`, all strategy-cache modes,
injected strategy/factory behavior, unchanged source offsets, cache metric
replacement during a strategy call, and a generator created in one thread and
first advanced under another thread's LLM override. Use events and bounded
joins, with cleanup in `finally`, not synchronization sleeps.

- [x] **Step 5: Run the new contracts against the unextracted baseline.** `python -m pytest tldw_Server_API/tests/Chunking/test_chunker_execution_contracts.py -q`; require all pass. Run `python Helper_Scripts/ci/check_shard_coverage.py`; require zero newly uncovered test files. Do not raise its baseline/ignore list.
- [x] **Step 6: Freeze the oracle and verify integration.** Run `python -m pytest tldw_Server_API/tests/Chunking -q`; require zero failures/no new skips. Run Ruff, Black check, compileall and test Bandit (`-s B101`) on only the two new test/support files, using the same command shapes as Final Verification. Existing assertions and frozen expectations are not editable to accommodate extraction. A correction needs baseline RED and separate requester approval.
- [x] **Step 7: Commit the independently reviewable characterization.**

```bash
git add -- tldw_Server_API/tests/Chunking/_execution_test_support.py tldw_Server_API/tests/Chunking/test_chunker_execution_contracts.py "$IMPLEMENTATION_TASK_FILE" Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md
git commit -m "test: characterize Chunker execution paths"
```

## Task 2: Move Plain Text Execution Behind The Facade (Stage 2)

**Goal:** Actively delegate the cached text path without changing its control flow.
**Success Criteria:** Direct fake-context tests and public wiring pass; all frozen paths still pass.
**Status:** Not Started

**Files:** Create `execution/__init__.py`, `models.py`, `service.py`, `test_chunker_execution_components.py` and `test_chunker_execution_boundaries.py`. Modify only the `chunk_text` body/imports and `_execution_hooks()` in `chunker.py`, plus tracking.

**Interfaces:** Consume Task 1's oracle/support. Produce `ResultCache`, `ExecutionContext`, `ExecutionHooks`, the storing-only service constructor, `ChunkExecutionService.chunk_text` and facade `_execution_hooks()` as defined above. Do not add unused prepared-request/strategy-helper modules yet.

- [ ] **Step 1: Add failing component and wiring tests.** `test_text_facade_delegates_without_extra_binding` replaces `chunker.py.ChunkExecutionService` with a sentinel service, asserts unchanged positional/keyword values and `result is sentinel_result`. `test_text_service_constructor_is_inert` uses a context that raises on attribute reads: construction emits no events. `test_text_service_cache_boundaries` asserts actual returned identity/output plus the cache/error/metric cases from Task 1 using a fake context. `test_hooks_resolve_replacement_during_execution` changes logger/counter after hook creation and verifies the replacement receives the original event arguments while output remains correct.
- [ ] **Step 2: Run only these new cases and capture RED.** `python -m pytest tldw_Server_API/tests/Chunking/test_chunker_execution_components.py tldw_Server_API/tests/Chunking/test_chunker_execution_boundaries.py -q`; expected failure is absent service/import or the old public body bypassing the sentinel, not an environment/network failure.
- [ ] **Step 3: Move the plain body mechanically.** Introduce the interfaces, store live context/hooks and replace only facade-owned accesses with context/hook lookups. Keep sanitization, early return, defaults, cache policy/signature/get and lock selection before the original try region; strategy execution/cache writes/success logging remain in their exact old regions. Preserve the exception tuple, identity, event ordering and exception chaining. Keep local setup duplication temporarily; do not consolidate here. The public method retains its declaration/docstring and returns the service result directly.
- [ ] **Step 4: Add active dependency and cold-import coverage.** At this stage require the real `execution`, `execution.models`, `execution.service` modules, with no missing-module skip. AST checks reject facade/sibling imports and `__getattr__` facade adapters. Fresh-process root-first and each introduced-module-first imports must resolve root `Chunker` and service `ChunkExecutionService` without preloading/stubbing modules.
- [ ] **Step 5: Verify GREEN.** Run the three execution test modules, then full Chunking; both must pass. Run scoped Ruff/Black/compileall, new-package mypy and production Bandit using the final verification commands restricted to introduced files. If a facade's pre-existing typing issue appears, prove its baseline rather than broadening this stage.
- [ ] **Step 6: Commit the active text migration and tracking.** No dormant helper or unrelated legacy formatting is included.

```bash
git add -- tldw_Server_API/app/core/Chunking/execution/__init__.py tldw_Server_API/app/core/Chunking/execution/models.py tldw_Server_API/app/core/Chunking/execution/service.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_chunker_execution_components.py tldw_Server_API/tests/Chunking/test_chunker_execution_boundaries.py "$IMPLEMENTATION_TASK_FILE" Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md
git commit -m "refactor: extract Chunker text execution"
```

## Task 3: Move Metadata Execution (Stage 3)

**Goal:** Delegate metadata execution while preserving direct strategy results and alignment.
**Success Criteria:** Same metadata/list identities, mutation, error regions and no result-cache calls.
**Status:** Not Started

**Files:** Modify `execution/service.py`, the `chunk_text_with_metadata` body in `chunker.py`, and the two component/boundary test modules; tracking only elsewhere.

**Interfaces:** Consume the Task 2 context/hooks/constructor. Produce `ChunkExecutionService.chunk_text_with_metadata` with the fixed mirrored signature. Do not route through `chunk_text` or introduce shared request preparation yet.

- [ ] **Step 1: Write `test_metadata_facade_returns_service_object` and `test_metadata_service_keeps_alignment_regions`.** For controlled results, assert the exact list/item/metadata identities and source-sliced `item.text`. Include out-of-range/boolean offsets, missing/raising metadata attributes, a non-list result, false/None alignment and an early empty-input control. For each case also assert no result-cache access and unchanged errors/events.

```python
assert result is strategy_result
assert result[0] is original_item
assert result[0].metadata is original_metadata
assert result[0].text == sanitized_source[clamped_start:clamped_end]
assert cache_events == []
```

- [ ] **Step 2: Capture wiring/component RED.** Run the execution component and boundary modules; require failure because the service metadata entry is absent or the public method bypasses it. Frozen baseline tests must not be changed.
- [ ] **Step 3: Move the metadata body, then wire its public delegate.** Keep raw alignment extraction/coercion before resolution, conditional strategy-option reinsertion and tokens-only ephemeral selection. Hold the lock only through direct `strategy.chunk_with_metadata`; leave per-item alignment and success logging outside the execution catch region. Add no copies, conversion, repairs or validation. Keep all retained context helpers on the facade.
- [ ] **Step 4: Run GREEN and regressions.** Run all execution modules, `test_process_text_components.py`, `test_process_text_refactor_equivalence.py`, `test_hierarchy_refactor_contracts.py` and `test_streaming_overlap.py` under `tldw_Server_API/tests/Chunking/`, then full Chunking. Require zero failures/new skips; run scoped static/type/Bandit checks.
- [ ] **Step 5: Commit the metadata migration and tracking.**

```bash
git add -- tldw_Server_API/app/core/Chunking/execution/service.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_chunker_execution_components.py tldw_Server_API/tests/Chunking/test_chunker_execution_boundaries.py "$IMPLEMENTATION_TASK_FILE" Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md
git commit -m "refactor: extract Chunker metadata execution"
```

## Task 4: Move Lazy Generator Execution (Stage 4)

**Goal:** Delegate generator execution without changing when work or failures happen.
**Success Criteria:** Generator protocol/timing is preserved and a paused consumer never retains the shared lock.
**Status:** Not Started

**Files:** Modify `execution/service.py`, `chunk_text_generator` in `chunker.py`, and execution component/boundary tests. Add new lifecycle cases to the execution contracts module without editing its frozen expectations. Tracking only elsewhere.

**Interfaces:** Consume the same context/hooks. Produce lazy `ChunkExecutionService.iter_chunks` with the fixed signature. Public `chunk_text_generator` remains a generator using a plain `yield from` statement; do not return an iterator from an eager wrapper or propagate a new completion value.

- [ ] **Step 1: Write failing active-wiring/lifecycle tests.** `test_generator_facade_defers_service_creation` asserts zero constructor calls until first advance, then actual yielded text. `test_generator_delegates_send_throw_close` controls each underlying protocol operation and asserts yielded values, cleanup and outer completion `None`. `test_generator_shared_selection_and_lock_regions` requires ephemeral factory selection even with an injected strategy, locked iterator creation and unlocked first/next/throw/close operations. Include callable/noncallable/missing `chunk_generator`, eager fallback, setup-versus-iteration failure and an unstarted close control.

```python
assert events == []  # after generator creation
assert next(generator) == "first"
assert generator.throw(SentinelError()) == "recovered"
generator.close()
assert events == [("factory", 1), ("iterator_created", 1), ("next", 0), ("throw", 0), ("close", 0)]
```

- [ ] **Step 2: Capture RED from the absent iterator entry or bypassed lazy delegate.** Run the relevant execution component/boundary tests; reject a RED caused only by a test-fixture mistake or optional provider initialization.
- [ ] **Step 3: Move and wire the generator body.** Construct the service inside the public generator body, not in facade initialization or a default argument. Preserve sanitize/size/default sequencing, lack of an eager empty return, LLM-hook lookup and `ephemeral` calculation before lock entry. Construct the iterator or eager fallback under the lock; delegate via `yield from` outside it. Do not add a generic exception handler, result caching, prefetch or a state-restoration scope around iteration.
- [ ] **Step 4: Add `test_paused_generator_allows_competing_call` and live-state controls.** Use events with bounded waits/joins and `finally` cleanup. Assert a competing text call returns actual expected chunks while the first generator remains paused. Check helpers/config/LLM override changes after generator creation, cross-thread first advancement, and no work when closing an unstarted generator. Keep the existing thread-safety test assertions unchanged.
- [ ] **Step 5: Run GREEN.** Run execution tests plus `test_thread_safety.py`, `test_streaming_overlap.py`, process_text and hierarchy contracts, then full Chunking and scoped static/type/Bandit checks. Require no lock leak, new skip or exception-translation change.
- [ ] **Step 6: Commit the lazy migration and tracking.**

```bash
git add -- tldw_Server_API/app/core/Chunking/execution/service.py tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_chunker_execution_contracts.py tldw_Server_API/tests/Chunking/test_chunker_execution_components.py tldw_Server_API/tests/Chunking/test_chunker_execution_boundaries.py "$IMPLEMENTATION_TASK_FILE" Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md
git commit -m "refactor: extract lazy Chunker execution"
```

## Task 5: Share Preparation And Complete Integration (Stage 5)

**Goal:** Remove proven setup duplication and verify the fully active subsystem.
**Success Criteria:** Every new helper is used by all applicable paths; all frozen contracts, real cold imports, guards and local verification pass before publication.
**Status:** Not Started

**Files:** Add `PreparedExecution`/`ExecutionKind` to `execution/models.py`; create `execution/preparation.py` and `execution/strategy.py`; update service calls and execution tests. Root package, existing strategies, LRU/registry/LLM helper bodies and workflow/check baselines are not edit targets. Tracking/plan updates accompany the work.

**Interfaces:** Consume the first four tasks. Produce exactly `PreparedExecution`, `prepare_execution` and `prepare_strategy` defined above. All execution entry signatures and context/hook interfaces remain unchanged.

- [ ] **Step 1: Write helper RED tests.** `test_preparation_preserves_path_specific_options` parametrizes the three kinds and raw alignment states; assert complete prepared fields, exact forwarded option values/identities and observed resolver order. `test_preparation_does_not_reserve_user_keys` includes every actual plumbing parameter name inside `user_options`, including `kind`, `user_options`, `effective_llm` and `ephemeral`. `test_strategy_preparation_uses_supplied_llm_and_selection` asserts a single supplied LLM pair, create-versus-lookup, tokenizer resets then conditional language update and no lock acquisition. Add active-helper spies asserting actual public outputs plus exactly one preparation/setup call on uncached paths and no setup call on a cache hit.
- [ ] **Step 2: Run RED.** Run the execution component/boundary modules; expect absent helpers or active service bypass. Do not edit previously frozen public expectations to make consolidation easier.
- [ ] **Step 3: Extract only equivalent preparation and strategy setup.** The preparation helper performs the original shallow copy, path-specific alignment handling, normalization/default/resolution/alias/overlap order, overlap handling/logging, tokenizer flag and forwarded-option copy. It performs no sanitization, size check, cache/LLM lookup or strategy work. The strategy helper uses supplied LLM hooks/ephemeral decision and the existing context hooks for create/lookup/sync/language; it changes no ownership, lock scope or error region. Pass user options as a mapping, never expanded with internal helper keywords.
- [ ] **Step 4: Expand required-module and cold-import coverage to the final module set.** Add `test_execution_cold_import` parametrized over the full root name and `execution`, `execution.models`, `execution.preparation`, `execution.strategy`, `execution.service` under that root. For each first-requested import, start a fresh interpreter with `sys.executable`, repository cwd/import path and isolated configuration. Use the fixed module allowlist, `shell=False`, `capture_output=True`, `text=True`, and a 30-second subprocess timeout. Have the child emit a final JSON line after real import/export checks, deriving names from the actual imported class objects rather than constants; keep diagnostics available. Assert the following. Do not preload/stub/delete modules, require the facade to remain absent, skip missing modules, call providers/download models or retry failures.

```python
assert completed.returncode == 0, completed.stdout + completed.stderr
assert json.loads(completed.stdout.splitlines()[-1]) == {
    "requested": requested_module,
    "chunker": "Chunker",
    "service": "ChunkExecutionService",
}
```
- [ ] **Step 5: Verify all matrix assignments and final gates.** Use the commands below. New test names must remain under `test_chunker_*.py`; require zero new uncovered files and prove coverage in each of the six parsed matrices, not just the guard's union. Do not edit YAML/ignore/baseline files if the existing patterns already cover them. Investigate any contrary result before changing scope.
- [ ] **Step 6: Review the complete diff and update evidence.** Prove only the three facade bodies/import/hook wiring and the new subsystem changed; retained facility bodies, declarations, public docstrings, existing assertions and sibling/file-streaming implementations remain intact. Record RED/GREEN and full test output, known skip/warnings, lint/type/security results and a spec-to-test map in the active implementation task. Remove no unrelated baseline exception or historical record.
- [ ] **Step 7: Commit the shared preparation and verified tracking.**

```bash
git add -- tldw_Server_API/app/core/Chunking/execution/models.py tldw_Server_API/app/core/Chunking/execution/preparation.py tldw_Server_API/app/core/Chunking/execution/strategy.py tldw_Server_API/app/core/Chunking/execution/service.py tldw_Server_API/tests/Chunking/test_chunker_execution_components.py tldw_Server_API/tests/Chunking/test_chunker_execution_boundaries.py "$IMPLEMENTATION_TASK_FILE" Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md
git commit -m "refactor: share Chunker execution preparation"
```
- [ ] **Step 8: Obtain independent whole-branch review.** Use `superpowers:requesting-code-review` on the immutable reconciled-base-to-final-head range; resolve confirmed in-scope findings with focused regressions and fresh verification. Disposition every declined-to-judge item explicitly. The requester-replaced Qodo provider/credit gate is not reinstated.
- [ ] **Step 9: Publish only after review and fresh verification.** Explicitly push the named execution branch, never a bare push to its possible `origin/dev` upstream. Prepare a draft PR against dev with rationale, compatibility/testing evidence, spec/ADR/plan links and a blank human-owned Change-summary gate. Do not reuse the old hierarchy summary. No merge is authorized by this plan alone; require separate authorization, live rules, exact-head CI and the new human summary. Preserve the isolated worktree.

## Final Verification Commands

After runtime activation, run these from the isolated root; log complete output
to unique `/tmp` paths and record exit status/counts, not just the last line.

```bash
python -m pytest tldw_Server_API/tests/Chunking -q
python Helper_Scripts/ci/check_shard_coverage.py
python Helper_Scripts/ci/check_backlog_task_ids.py
python -m pytest tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py tldw_Server_API/tests/lint/test_function_size_ratchet.py tldw_Server_API/tests/CI/test_test_quality_ratchet.py -q
python -m ruff check tldw_Server_API/app/core/Chunking/execution tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/_execution_test_support.py tldw_Server_API/tests/Chunking/test_chunker_execution_*.py
python -m black --check tldw_Server_API/app/core/Chunking/execution tldw_Server_API/tests/Chunking/_execution_test_support.py tldw_Server_API/tests/Chunking/test_chunker_execution_*.py
python -m compileall -q tldw_Server_API/app/core/Chunking/execution tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking
python -m mypy tldw_Server_API/app/core/Chunking/execution
python -m bandit -r tldw_Server_API/app/core/Chunking/execution tldw_Server_API/app/core/Chunking/chunker.py -f json -o /tmp/bandit_chunker_execution_production.json
python -m bandit tldw_Server_API/tests/Chunking/_execution_test_support.py tldw_Server_API/tests/Chunking/test_chunker_execution_*.py -s B101 -f json -o /tmp/bandit_chunker_execution_tests.json
git diff --check
```

Expected: full Chunking zero failures/no new skips; the existing PyThaiNLP
availability skip remains documented. New component/wiring/cold-import tests
all execute. Guard/ratchet/lint/format/compile/type commands exit zero without
raised baselines or new suppressions masking defects. Run Black separately on
only changed facade line ranges; never format the legacy file wholesale.
Production Bandit must have zero new findings/errors. In new pytest files,
exclude only B101 globally; any diagnostic on the fixed-argv subprocess smoke
test must be resolved or narrowly justified per line with its trusted inputs
and no-shell guarantee, not silenced through a new blanket skip.

Broader changed-file mypy/CI may expose inherited errors; reproduce them on the
immutable reconciled base and report them accurately. Fix extraction-local
typing without coercing runtime values. The future PR must pass all substantive
required checks on its exact head; missing/pending/failed/cancelled/skipped
required contexts block integration. Read current branch rules and MERGE_QUEUE
at that time; never dispatch required gates manually or bypass policy.

## Planning Self-Review And Handoff

Planner self-review completed on 2026-11-02, without a subagent dispatch:

| Check | Result |
| --- | --- |
| Spec coverage | Input/options/cache/error contracts map to Tasks 1/2/5; metadata to 1/3/5; generator/live hooks to 1/4/5; dependencies/cold imports to 2/5; CI/security/publication gates to 5. Task 1 pins concrete property-based option/alignment/word-equivalence invariants. No uncovered requirement remains. |
| Step scan | Baseline characterization is explicitly GREEN; extraction-only cases require meaningful RED. Removed an undefined example collector, separated operational steps and provided exact commit paths. Code bodies are moved rather than prescribed twice. |
| Type/interface consistency | Context/cache are structural protocols; entry names/defaults match facade bindings; prepared fields and helper signatures are defined once and reused unchanged in consumers. Internal Any annotations preserve dynamic inputs rather than adding coercion. |
| Review Focus | All five named failure classes have tests in the contract stage and the stage that owns their extraction; no silent outside-scope omission. |
| Proportion | Five independently testable stages, a single shared interface contract and one verification bundle; no implementation transcript or speculative subsystem. |

Record this review in TASK-15415. No product/test files change while writing or
reviewing this plan; documentation-only Bandit is not applicable. This is
planning verification, not proof that future extraction tests already pass.

Requester plan approval and the **Native** execution choice were received on
2026-11-02, followed by independent whole-branch review. Native means
one implementer owns these tightly coupled sequential migrations with an
independent whole-branch review at the end. **Subagent-driven** adds fresh
implementers and per-task review gates before the next task. Native is the
planner's recommendation for this five-stage plan because shared context,
exception timing and fixed interfaces span all paths; the requester chooses
the cost/review tradeoff before any task is executed.
