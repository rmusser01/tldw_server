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
**Status:** Complete

- [x] Inspect each worker delta and run combined regressions; record every failure/skip explicitly.
- [x] Run normal format/lint/security hooks and Bandit; fix new findings.
- [x] Commit an immutable candidate and request a read-only independent review; repair all validated findings.

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

Final simplification after the next review: equal outer/inner subprocess
deadlines could orphan the grandchild. A bounded, self-cleaning external probe
independently reproduced that second P2. Rather than adding a cross-platform
process-tree supervisor, the candidate removes all branch-added synthetic GC
orchestration and nested subprocess paths. The original four lifecycle cases
remain; the original real FastAPI/reset/weakref assertions execute in one fixed
child, owned by `subprocess.run(timeout=30)`. The reviewer accepted this simpler
approach in principle, subject to immutable source review.

Current final evidence supersedes the earlier 490-case checkpoints: full
infrastructure shard 489 passed without skips; focused lifecycle tests four
passed; unfiltered collection 87,594 tests. Only the extra branch-added GC
parameter was removed; no pre-existing case or assertion was removed. The
external harness `/tmp/mcp3187_single_child_verification.py` holds an existing
collector active around the actual public test call (`ambient`, GREEN), removes
isolation to reproduce the weakref failure (`inline`, expected RED), and blocks
the sole child to prove the unchanged 30-second timeout kills and reaps it
(`hung`, GREEN). Run those modes with the project venv active and `PYTHONPATH`
set to the worktree root. The actual public ambient-GC probe also passes on
Python 3.12; strong-registry and missing-reset mutations still fail at their
intended unchanged assertions. Other three repairs are unchanged. Combined
Bandit retains the same 94 assertion-only low findings; Ruff/format pass.
Independent source approval remains pending.

Final independent re-review explicitly approves source
`bd7cbcaba746856a786a6ca4dcaf470d6abe2895` with zero Critical, Important or
Minor findings. Both lifecycle P2 findings are closed. The reviewer independently
exercised the actual public test under an already-active collector on Python
3.14 and 3.12, the intended mutations, and the unchanged 30-second sole-child
timeout/reaping proof. The other three previously approved repairs are unchanged.
All four repair child tasks are source-complete. Final committed-diff ratchets
pass 50 tests. Hosted CI and the human Change summary remain separate landing
gates, not waived by source approval.

## Stage 4: Publish And Report
**Goal:** Publish a follow-up PR with accurate evidence and no merge-policy bypass.
**Success Criteria:** Latest dev incorporated, review evidence and CI status documented, original PR linked to follow-up.
**Tests:** Git head/base equality, clean tracked worktree, remote PR/check readback.
**Status:** In Progress

- [x] Update tasks and plan, commit with normal hooks, publish follow-up against `dev`.
- [ ] Read normal hosted CI and repair any new validated failure; do not hand-dispatch required gates.
- [ ] Report follow-up status and the human Change summary gate; do not extend the old per-PR waiver silently.

Draft follow-up: [PR #3406](https://github.com/rmusser01/tldw_server/pull/3406),
based on current dev `b2b7d5d2b04256458fe6f33e02a36019e2c46795`. Source approval
applies to `bd7cbcaba746856a786a6ca4dcaf470d6abe2895`; the final evidence update
changes only this plan and task records, and source/test blobs must remain
identical before publication. Normal hosted qualification and the new PR's
human-written Change summary are pending. Auto-merge is disabled; the original
PR #3187 waiver does not transfer. Historical red checks on the merged commit
are not relabeled as successful.

## November 7 Continuation

PR #3406's completed hosted run has all required gates green. Its only failing
shard is media-audio (plus the aggregate): the provider-failure retry test checks
remaining delay after worker shutdown and observed 3.959 seconds against a
four-second floor. Current dev already repairs this in `98f9ead454`, asserting
the persisted five-second schedule against the UTC-second failure window rather
than subsequent shutdown duration. Do not duplicate that upstream repair.

The four follow-up commits are rebased onto dev
`dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c`. One Parakeet test conflict preserves
upstream's `monkeypatch.setitem(adapter.__dict__, ...)` for exact restoration of
absent instance overrides; its deterministic cached-method case remains.
The other three repair test blobs match the previously approved published head.
Backup `codex/mcp3406-pre-nov07-rebase` retains that head. ADR required: no;
this rebase and fixture verification change no durable architecture rule.

Independent semantic/rebase review approves source
`89de3acec971b5c73604fe6003bba37d2ac0d795` with no verified source findings.
Fresh focused audio passes 108 tests with one existing opt-in skip using a
run-specific AuthNZ database; canonical SQLite/PostgreSQL regressions pass 89
without skips. The exact infrastructure/metrics/monitoring shard passes 489
without skips. Quality/security/tracking guards pass 55; unfiltered collection
finds 88,179 tests. Ruff and normal hooks pass; Bandit retains the same 94
test-assertion-only low findings, no additions or scan errors. No production
Python, dependency, CI baseline or assertion changes are introduced.

An initial concurrent focused run had a startup SQLite lock before transcription
assertions (107 passed, one skip, one failure); the isolated database rerun
passes. An initial combined MCP/infrastructure run, unlike CI's grouping, had
three failures (model-runtime timeout, provider retry-window status and telemetry
restart alignment), 4,825 passed and three existing parser skips. All three
failing cases pass in isolated attribution without deadline/assertion changes;
the exact separate infrastructure shard also passes. Do not represent that
combined run as green. Exact MCP in-app and full media-audio runs are still in
progress at publication; their final results must be reported separately.

Source may be published to obtain normal current-head CI while qualification
continues, only with the PR still draft and auto-merge disabled. The explicit
publication lease is the verified remote head
`f06032b3e9b9c03ce60e9b2a4b58e66ed1f75948`. Do not claim current-head hosted
success from the prior run. The follow-up still needs a human-written Change
summary before merge; no Skills runner feature is enabled by this continuation.
