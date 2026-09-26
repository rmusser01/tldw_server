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

**Final rebase verification (2026-09-26)**: On `a2826f103f02a67f57adb40ed048dbfa2ecfc6e5`, the 18-file core gate passed 713 tests with 26 warnings and no skips or failures. The official PostgreSQL fixture used isolated per-test databases on the existing standard test cluster with Docker auto-start disabled. Changed production Bandit found zero findings/errors across 14 files; all changed Python compiled. OpenAPI drift and shard coverage guards passed, and `git diff --check` was clean. No Persona runtime patch changed during this rebase; previous Character/import and full CI results remain historical evidence, not a substitute for the new head's CI.

## Stage 3: Resolve PR review
**Goal**: Retarget PR #2963 to `dev` and process Qodo and other posted findings against the rebased code.
**Success Criteria**: Each actionable finding is fixed and tested, or answered with a verified reason; human-provided Change summary is published verbatim.
**Tests**: Targeted regressions and affected suites for review fixes.
**Status**: Complete. Six Qodo findings were fixed and three were dispositioned with verified reasons in PR comment 5839111960. The human requester supplied the Change summary verbatim.

## Stage 4: Merge and close tracking
**Goal**: Merge PR #2963 after checks, review, and policy gates, then update TASK-13245.5 and parent issue.
**Success Criteria**: PR is merged into `dev`; tracker records exact result and remaining Persona stages.
**Tests**: Verify PR merge commit and target branch through GitHub.
**Status**: In Progress. All 53 checks passed on `f80a81d2e23bfbaacd22907d54107778bbc6c33c`, but GitHub refused merge because newer `dev` commits require an up-to-date branch. The stack was rebased without conflicts onto `a2826f103f02a67f57adb40ed048dbfa2ecfc6e5`; all 26 commits are identical in `git range-diff`. Fresh integration verification and CI are required before merge; no administrator bypass is permitted.

## ADR Check

**ADR required**: No for TASK-13245.5. The rebase, review and CI repairs preserve the approved startup contract and do not establish a new durable architecture rule. The existing choice/provenance design is `Docs/Design/2026-09-13-persona-workspace-choice-provenance-design.md`; the ADR index was searched. A historical decision backfill or later strict-startup design remains a separate assessment, not an automatic promotion of an inventory candidate.
