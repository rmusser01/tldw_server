# PR 2963 dev rebase and review plan

## Stage 1: Rebase the Persona stack
**Goal**: Replay the reviewed Persona commits onto current `origin/dev`.
**Success Criteria**: No unresolved conflicts; all prerequisite behavior and task records are present; new SQLite and PostgreSQL schema versions do not reuse existing versions.
**Tests**: Diff and migration registry inspection; `git diff --check`.
**Status**: Complete

## Stage 2: Verify integrated behavior
**Goal**: Validate startup, lifecycle, projection, import boundaries, and migrations against the new base.
**Success Criteria**: Focused tests pass on SQLite and live PostgreSQL, with existing skips and failures recorded honestly; no new Bandit findings.
**Tests**: Affected Persona/Workspace/Chat/Sync/Character suites and changed-scope Bandit.
**Status**: Complete. The 18-file core gate passed 711 tests; the Character/import gate passed 394 with three historical skips. Live PostgreSQL migration and dev migration gates passed 41 and 67 tests respectively. Production and integration-test Bandit scans found zero findings; integration-touched Python passed Ruff and compilation.

## Stage 3: Resolve PR review
**Goal**: Retarget PR #2963 to `dev` and process Qodo and other posted findings against the rebased code.
**Success Criteria**: Each actionable finding is fixed and tested, or answered with a verified reason; human-provided Change summary is published verbatim.
**Tests**: Targeted regressions and affected suites for review fixes.
**Status**: In Progress

## Stage 4: Merge and close tracking
**Goal**: Merge PR #2963 after checks, review, and policy gates, then update TASK-13245.5 and parent issue.
**Success Criteria**: PR is merged into `dev`; tracker records exact result and remaining Persona stages.
**Tests**: Verify PR merge commit and target branch through GitHub.
**Status**: Not Started
