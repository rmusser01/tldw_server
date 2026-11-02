# PR 3187 Post-Merge CI Repair Plan

> **For agentic workers:** Use systematic debugging and test-driven development for each repair; an independent final reviewer must assess the combined branch.

**Goal:** Repair all four failing shards from completed PR 3187 CI without weakening runtime safeguards or test assertions.

**Architecture:** Preserve existing contracts. Keep independent repairs in disjoint files; change production behavior only when a failing regression proves a real defect. No new public interfaces or dependencies are planned.

**Tech Stack:** Python, pytest with explicit CI plugin loading, FastAPI, SQLite/PostgreSQL and existing lifecycle abstractions.

**Spec:** Requester instruction to check PR 3187 and address remaining CI failures; completed CI run `41511103471`, reviewed merge `b2b7d5d2b04256458fe6f33e02a36019e2c46795`.

## Global Constraints

- Track repairs through `TASK-2294.3.2.5` through `TASK-2294.3.2.8` using backlog-py.
- Preserve security checks, negative assertions, collection, plugin-isolation policy and CI baselines; no blanket skips or timeout increases.
- Use the project virtual environment and canonical AuthNZ database fixtures. Never log credentials or raw sensitive subprocess diagnostics.
- Keep the unrelated untracked audit ZIP untouched. Workers do not commit or run stashing hooks while other workers edit.
- PR 3187 is merged; publish repairs in a follow-up PR. Its human-written Change summary waiver applies only to PR 3187, not automatically to this follow-up.

## Review Focus

- Plugin autoload disabled: malformed handles must still fail before repository/governor effects.
- Cold and warm scoped storage reads: user, team and organization limits must resolve inside the owning transaction on both backends.
- Provider variants: the Parakeet variant must route to the intended mocked provider and return the expected transcript without real downloads.
- Reloaded/retired FastAPI objects: lifecycle tracking must not retain applications, and unrelated suite state must not produce a false leak assertion.
- Cross-suite execution: exact CI shard order and explicit plugin loading must not introduce failures hidden by isolated local tests.

## Stage 1: Diagnose And Reproduce
**Goal:** Bind each failure to its exact root cause before changing behavior.
**Success Criteria:** Exact failing tests reproduce under CI-compatible conditions or a deterministic equivalent isolates the cause.
**Tests:** Missing-mocker matrix; six generated-file scope/backend subprocess cases; Parakeet variant case; lifecycle weak-reference case plus surrounding order.
**Status:** Complete

- [x] Verify merged PR, all required gates green and all completed broad-matrix failures.
- [x] Create a current-dev branch in the existing isolated worktree and four repair tasks.
- [x] Record local RED evidence and root cause for each failure domain.

## Stage 2: Minimal Independent Repairs
**Goal:** Restore each existing contract with the smallest justified change.
**Success Criteria:** Preserved negative assertions and focused regressions pass; no unrelated behavior changes.
**Tests:** Exact failing cases, complete affected modules and equivalent CI plugin-isolation commands.
**Status:** Complete

- [x] `TASK-2294.3.2.5`: replace the undeclared pytest-mock dependency in `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_accounting.py` with standard-library mocks or equivalent explicit effect instrumentation, preserving all no-side-effect and redaction assertions.
- [x] `TASK-2294.3.2.6`: diagnose `tldw_Server_API/tests/AuthNZ/integration/test_vn_generated_file_idempotency.py` and its existing runtime helper; repair only the demonstrated transaction/fixture contract. Any production write-scope expansion must be justified before editing.
- [x] `TASK-2294.3.2.7`: diagnose `tldw_Server_API/tests/TTS_NEW/integration/test_transcription_auth.py` against the existing audio transcription provider selection and fix the demonstrated production or fixture defect.
- [x] `TASK-2294.3.2.8`: diagnose `tldw_Server_API/tests/Infrastructure/test_lifecycle_drain_isolation.py` with `tldw_Server_API/app/services/app_lifecycle.py`; fix a demonstrated reference leak or test-isolation defect without weakening collection semantics.

MCP checkpoint: CI-isolated RED reproduces 36 missing-fixture errors. Replacing
pytest-mock spies with standard-library async spies preserves their wrapped
behavior and all assertions; the full accounting module passes 63 tests. A
disposable in-process guard-bypass mutation makes all 36 negative cases fail,
proving the new instrumentation still detects the intended security defect.
Ruff/format pass; test Bandit has the same 49 assertion-only low findings as the
unchanged baseline and no new findings. Full in-app verification passes 4,339
tests with three existing optional parser skips under CI-equivalent plugin and
marker settings.

AuthNZ checkpoint: all six child failures came from fixture membership writes
missing the now-required write context, before quota resolution. Both writes
now receive `TrustedMembershipWriteContext` with reason `BOOTSTRAP`, following
existing fixture patterns. All six cases were RED before repair; the complete
affected module and related repository/boundary regressions pass 89 cases with
zero skips on canonical PostgreSQL/SQLite fixtures. No production change.

Audio checkpoint: earlier instance patch teardown left a bound Parakeet method
shadowing the later class stub. Patching the actual registry instance and adding
a cached-method parameter reproduces the order-dependent defect before repair.
The reproduced CI order and complete transcription-auth module pass all 24
cases with zero skips; both worker and parent runs completed successfully. No
provider routing, auth, quota, transcript assertion or model dependency changed.

Lifecycle checkpoint: an event-synchronized unrelated cyclic-GC finalizer
deterministically reproduces the false leak assertion: concurrent `gc.collect()`
does not start a second collection, leaving FastAPI router/closure cycles alive.
The original lifetime/reset assertions now execute in a fresh interpreter under
both idle and concurrent parent-GC conditions. Strong-registry, missing-reset
and removed-isolation mutations fail. Focused tests pass on Python 3.14 and
3.12; the full infrastructure/metrics/monitoring shard passes 490 cases with
zero skips. Production lifecycle code is unchanged. The exact uninstrumented
hosted initiator/finalizer is not recovered; this is a verified failure mode,
not proof of that specific initiating event. Parent full-shard rerun also passes
all 490 cases with zero skips. Independent combined review remains required.

## Stage 3: Integration And Independent Review
**Goal:** Verify the combined repair and close every review finding.
**Success Criteria:** Relevant full shard regressions, static checks and Bandit qualify the touched scope; independent review has no remaining findings.
**Tests:** Full MCP in-app, infrastructure/metrics/monitoring and relevant audio/AuthNZ groups with CI plugins; required ratchets, collection, normal hooks, Bandit on touched production scope and differential test diagnostics.
**Status:** In Progress

- [x] Inspect each worker delta and run combined regressions; record every failure/skip explicitly.
- [x] Run normal format/lint/security hooks and Bandit; fix new findings.
- [ ] Commit an immutable candidate and request a read-only independent review; repair all validated findings.

Integration evidence: MCP in-app 4,339 passed with three existing optional parser
skips; infrastructure/metrics/monitoring 490 passed without skips; canonical
SQLite/PostgreSQL generated-file regressions 89 passed without skips; reproduced
audio order plus transcription-auth module 24 passed without skips. Counts are
not a unique repository-wide total. Unfiltered collection finds 87,595 tests;
full repository execution is not claimed. Quality/RLS ratchets pass 40 tests;
tracking guards pass 10 tests; shard coverage has zero newly uncovered files.
Ruff and normal hooks pass. Combined Bandit reports the same 94 existing
assertion-only low findings across the four test files, no new findings or scan
errors; production scan scope is empty because production is unchanged. Existing
whole-file AuthNZ formatting drift is verified against HEAD and left untouched;
changed-line Black passes. No new ADR is required for test-only fixture repairs.

Independent review of source `957dc3605c` found one Important issue and no other
Critical/Important/Minor findings: the synthetic concurrent-GC fixture still
starts its own collector in the ambient pytest process. If another collector is
already active, that new collection returns before reaching its sentinel, so
fixture setup fails. Reviewer reproductions on Python 3.14 and 3.12 confirm the
race. Source approval is blocked until the synthetic orchestration also moves
into a fresh process and the shared probe driver is exercised with an already
active outer collector. Preserve all assertions, time bounds and cleanup; rerun
focused regressions, mutations and the complete infrastructure shard, then
request an immutable delta review.

The P2 reproduces independently before repair. The ambient pytest GC fixture is
removed: controlled synthetic collection now runs only in a fresh child, and
both pytest entry points and the child already owning GC share the same isolated
probe driver. Both public cases pass with an additional already-active outer
collector. Strong-registry, missing-reset and removed-isolation mutations fail
at their intended assertions. Project focused tests pass five cases; the full
infrastructure shard passes 490 with zero skips again. Direct probe calls also
pass on Python 3.12.12; two supplementary root-fixture command attempts failed
environment setup (duplicate plugin command, then missing diagnostic-venv httpx)
and are not claimed as a Python 3.12 pytest pass. No package installation, timeout
increase, retry or production change is used for the repair. Normal hooks,
Ruff/format and unchanged Bandit findings qualify the delta; independent
re-review remains pending.

## Stage 4: Publish And Report
**Goal:** Publish a follow-up PR with accurate evidence and no merge-policy bypass.
**Success Criteria:** Latest dev incorporated, review evidence and CI status documented, original PR linked to follow-up.
**Tests:** Git head/base equality, clean tracked worktree, remote PR/check readback.
**Status:** Not Started

- [ ] Update tasks and plan, commit with normal hooks, publish follow-up against `dev`.
- [ ] Read normal hosted CI and repair any new validated failure; do not hand-dispatch required gates.
- [ ] Report follow-up status and the human Change summary gate; do not extend the old per-PR waiver silently.
