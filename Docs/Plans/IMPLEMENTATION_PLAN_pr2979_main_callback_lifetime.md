# Main callback lifetime — UAT572 / TASK-13260.278.18.83.45

UAT571 source-attested FastAPI 0.142.2 classification keys retain callbacks whose globals contain retired main apps. Test that a caller retaining a real registered callback can release its retired app after isolation. Move stateless control-plane callbacks into a stable endpoint module only if this causally breaks the retaining path. Keep main exports and registration metadata/security, live-module WeakSet isolation, request-app readiness, setup/error contracts and all budgets. No cache clearing, dependency patching, GC suppression or production cleanup change.

## Stage 1: Causal lifetime test
**Goal**: Reproduce retention with actual current source and FastAPI 0.142.2.
**Success Criteria**: Weakref assertion fails after real reload/registration/restoration; actual dependency imports and source hashes recorded.
**Tests**: One bounded app reload, real FastAPI classification and lifetime controls.
**Status**: Complete

## Stage 2: Shared callback ownership correction
**Goal**: Remove the callback-to-main-global app ownership path at its source.
**Success Criteria**: Same lifetime test passes; callbacks still work when deliberately retained by callers.
**Tests**: Lifetime, setup/root/readiness/metrics/security contracts and existing isolation tests.
**Status**: Complete

## Stage 3: Review and publication
**Goal**: Publish the smallest independently reviewed correction.
**Success Criteria**: Scoped lint/compile/Bandit, source review and reviewable PR; honest deferred native acceptance.
**Tests**: Required exact-head checks and uninstrumented whole Prompt shard natural exit.
**Status**: In Progress

## Current qualification — 2026-10-02

The identical final lifetime control fails twice on current-dev baseline main, then passes for ultra-minimal/full repeated reloads after extraction and three logging ownership fixes. The test holds a first-profile baseline app deliberately; a first-full-import lifetime failure is retained separately and is not accepted by this narrower control. Original live-module WeakSet isolation, real route registration/classification and caches remain unchanged.

Ten callbacks retain their original bodies/signatures, main aliases and registration/security/header/asset contracts. Existing logger unwrap helpers now save underlying add/configure callables immediately, and the removed-handler loop alias is released. Independent v1 review found the moved metrics decorator changed names; an actual counter failure preceded the explicit original decorator name and passing counter check.

Final scoped checks: 74 passes, zero skips/failures/errors, 72 inherited warnings, natural exit0. Baseline/candidate official canonical OpenAPI exports both exit0 and have identical SHA256 e384a65e765f4dac0c8f7608856c3bf559cdf6a99d4384be85dcaa989333dc0e, 2105 paths/3245 schemas. Shard coverage:837 patterns/4940 test files/4 ignored/44 baseline/zero new omissions. Actual local Python3.12.13/FastAPI0.142.2/Pydantic2.13.5/Starlette1.7.0; FastAPI models bytes match native UAT571.

Isolated branch normally fast-forwarded to dev dcae0cbd3f3ba6c7cd4287dc443628d93ee2cd64 after its 10 commits/11 UI-doc-task files were verified non-overlapping. Pre-dca and pre-correction diagnostic recoveries remain. No dependency/defaultCI/PG policy change or GC/cache/cleanup/exit/warning/timer bypass. Independent immutable v2 review is CLEAR with no actionable P1/P2; source publication is next. Nine inherited main Ruff findings remain, with zero new findings; Bandit exits1 for seven LOW test assertions and reports zero production findings/errors/HIGH/MEDIUM. Focused gc.collect() is identical baseline/candidate and proves collectability, not natural GC timing. Native uninstrumented whole Prompt natural exit and first-import ownership remain open; local results do not accept hosted/latest-head gates. Human-written Change summary is required by the repository merge policy for the new PR.
