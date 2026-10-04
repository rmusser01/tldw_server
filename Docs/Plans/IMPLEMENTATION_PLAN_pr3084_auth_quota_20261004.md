# PR3084 PostgreSQL quota test setup and latest-dev integration

**Task:** TASK-13260.278.18.83.45
**Goal:** Repair the actual auth/admin PostgreSQL CI failure and preserve the reviewed PR3084 implementation on latest dev.
**Architecture:** Configure the test user through the existing quota switch and UserProfileOverridesRepo. Keep production policy, DATE parameters, counters, login/profile paths and assertions intact.
**Stack:** Existing Python 3.12 virtual environment, official AuthNZ isolated PostgreSQL fixture and existing postgres:18 image; new upstream backlog-py CLI after rebase.
**ADR required:** No new decision. This is test setup and integration; Docs/ADR/059-backlog-py-task-editor-cutover.md governs upstream task tooling.

## Stage 1: Reproduce and inspect
**Goal**: Reproduce both unchanged profile cases using official isolated PostgreSQL fixtures and inspect incoming task-tooling changes.
**Success Criteria**: Both cases fail on remaining=None while HTTP/date/usage checks succeed; no new app-source or canonical-contract overlap.
**Tests**: The two profile parameter cases on the existing proper stack; static incoming-source review.
**Status**: Complete

## Stage 2: Integrate and correct setup
**Goal**: Rebase the five published commits onto immediately verified latest dev and configure the finite test quota explicitly.
**Success Criteria**: Owned patch remains exact, all original assertions remain, only the profile test gains master enablement and a real per-user 30-minute override; task notes preserved by the supported editor/normalizer.
**Tests**: Range-diff and binary-patch preservation; the full PostgreSQL DATE file plus quota switch/audio/resolver contracts.
**Status**: Complete

## Stage 3: Verify and review
**Goal**: Verify affected incoming task tooling/CI contracts and obtain independent actual-source review.
**Success Criteria**: Natural test success with no skipped PostgreSQL qualification; syntax/lint/format and baseline/current Bandit comparison have no new findings; review clear.
**Tests**: tools/backlog-py tests, affected CI contracts, owned task format ratchet, Sync activation/certification and Personalization activation suites on applicable SQLite/PostgreSQL stores, touched-test syntax/Ruff/Bandit.
**Status**: Complete

## Stage 4: Publish normally
**Goal**: Commit the reviewed patch and publish to the existing PR branch after exact remote/latest-dev checks.
**Success Criteria**: Normal hooks/commit and exact force-with-lease; human Change summary preserved verbatim; fresh new-head CI remains required. Remove only this plan after all stages are complete.
**Tests**: Diff check, clean checkout, exact reviewed commit patch, live remote/dev guards and PR body readback.
**Status**: In Progress

## Limits
Each declared test group has a 600-second outer cap and existing 300-second per-test timeout; no hidden retries, installation/environment reconstruction, new canonical export, CI bypass or stopped native controls. Use only the official fixture on task-owned PostgreSQL name tldw_pr3084_quota_20261004/port15492. No manual database or raw operator SQL, proof bundles, unrelated process/cache/dependency controls, frontend rebuild or browser action. The separate API replacement approval remains pending; native/first-import/full UAT/installer gates stay open.

## Current results
Unchanged profile cases:2failed/88warnings/56.72s pytest/64.130s outer/natural1, exact remainingNone failure. Latest dev73e adds40paths including four Sync/personal-context source files; independent source review and affected SQLite/PostgreSQL checks included. Rebase bf83774531 preserves all five published commits and complete owned binary patch exactly; old/new committed tree delta exactly40incoming paths. Canonical app routes/schemas remain unchanged by this new base. The new supported task editor normalized only the owned task without changing criteria/status. Numeric-only CLI id failed before writing; corrected to the actual TASK-prefixed id. Git GC warnings retained without cleanup.

Quota correction verification: full four-case PostgreSQL DATE file plus switch/audio/per-user/resolver contracts36passed/191warnings/51.01s pytest/57.294s outer/natural0, no skips reported. All original assertions and all other functions identical by AST;21 changed/owned Python files compile in memory. Ruff lint0 after normal formatting. Touched test baseline/current Bandit6 identical findings/0errors/no new findings; incoming CI helper0/0. Initial aggregate Bandit baseline selection included unchanged owned new control-plane file absent from old base and stopped there; only source scope is corrected to incoming diff for remaining scans, no source/environment change. Independent actual code/task review CLEAR; two Minor plan identifier/test-list corrections applied.

Remaining verification completed: incoming backlog-py/CI/task-format/license/path-classifier193passed/30warnings/14.31s pytest/17.413s outer/natural0. Sync activation/certification and Personalization activation89passed/29warnings/179.43s pytest/186.103s outer/natural0 across applicable SQLite/official PostgreSQL stores. No skips reported. Incoming eight production source files and CI helper baseline/current Bandit0findings/0errors; no new touched-test findings. Task normalize --check0, Ruff format--check0, diffcheck0. Complete incoming source and owned actual-code/task reviews clear for integration; two upstream task metadata defects remain outside this repair. Final exact three-path review and ordinary publication pending; native/live gates remain open.
