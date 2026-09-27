# Workspace Cascade Retry Transaction Ownership

Task: TASK-13245.7. Branch: `codex/persona-workspace-cascade-retry`.
Base: `35d6dd90d4` (latest dev at implementation start).

## Scope

Complete only reads owned by the staged Workspace cascade so a failed child
deletion can be retried on PostgreSQL. Reuse the existing opt-in read-only
transaction helper. Keep the outermost-transaction guard, caller-owned
transactions, committed admission closure and residual protected-chat check.
Own each soft-cascade message-page read with the existing transaction context;
finish that read before invoking independently owned child deletions.
No strict-startup activation, migrations, workflow changes or unrelated repairs.

ADR check: ADR required: no new ADR. Governing record:
`Docs/ADR/050-native-chat-fork-storage-lifecycle.md`. This repair restores the
existing committed admission fence and outermost-boundary rule without changing
storage, ownership or lifecycle policy.

## Stage 1: Reproduce and Cover Ownership
**Goal**: Reproduce hard-delete retry failure and cover both cascade paths.
**Success Criteria**: Regression tests expose leaked cascade reads without
intervening writes, and check managed/driver-owned caller transaction preservation.
**Tests**: Official isolated SQLite/PostgreSQL lifecycle fixtures, injected child
failure, immediate retry and rollback/commit of caller-owned transactions.
**Status**: Complete

## Stage 2: Minimal Ownership Repair
**Goal**: Opt side-effect-free cascade enumeration reads into existing ownership.
**Success Criteria**: Both paths settle their own reads, retries work, closure
remains committed, and caller-owned work is neither committed nor rolled back.
**Tests**: Focused RED/GREEN regressions plus existing lifecycle tests.
**Status**: Complete

## Stage 3: Verify and Review
**Goal**: Qualify the prerequisite repair independently from future Stage 2C.
**Success Criteria**: Affected SQLite/live-PostgreSQL suites pass, scoped lint,
compilation and Bandit introduce no new findings, review resolved, normal commit.
**Tests**: Native fork, Workspace lifecycle/creation and read-ownership regressions;
scoped Ruff, compilation, Bandit and whitespace check.
**Status**: In Progress

## Evidence

Prior causal reproduction and temporary diagnostic are documented in
`Docs/Design/2026-09-27-persona-workspace-strict-startup-refresh.md` and the task.
Latest dev did not change the relevant DB implementation or lifecycle fixtures.
Record current repair verification here before committing; historical evidence
does not certify the repair.

- Current exact baseline: the original PostgreSQL hard-delete retry test fails
  at the outermost guard (`persona-cascade-retry-baseline-20260927.log`).
- New RED: 3 failed, 7 passed, 4 SQLite-only driver-case skips, 4 warnings.
  All three failures are PostgreSQL cascade reads left INTRANS, including soft
  message-page reads. Caller-owned commit/rollback preservation cases pass.
  Log: `/private/tmp/persona-cascade-retry-red-20260927.log`.
- Initial full lifecycle GREEN: 79 passed, 4 SQLite-only driver-case skips,
  4 warnings in 210.83s. Log: `persona-cascade-retry-lifecycle-green-20260927.log`.
- Independent review: no production correctness findings; the P3 coverage gap
  for multi-page/image reads and committed partial progress was addressed with
  101 image-bearing messages, failure on the second page, independent observer
  verification of 100 durable deletions and closure, and immediate retry.
  New regression GREEN: 2 passed, 4 warnings. Its exact pre-repair soft cascade
  loaded in memory fails on PostgreSQL INTRANS (1 failed, 1 SQLite pass), proving
  the case detects the defect without changing the working tree or fixtures.
  Logs: `persona-cascade-retry-pages-{red,green}-20260927.log`.
- Scoped Ruff and compilation pass. Bandit reports no production findings or
  scan errors. The test scan contains only B101 assertions; with the standard
  test-only B101 exclusion it reports zero findings/errors.
- Final full lifecycle: 81 passed, 4 SQLite-only driver-case skips, 20 warnings
  in 78.74s. Expanded 14-file integration: 633 passed, no skips/failures,
  25 warnings in 313.33s. Together these cover 714 passing cases across 15 files
  on official isolated SQLite/live-PostgreSQL fixtures. Logs:
  `/private/tmp/persona-cascade-retry-lifecycle-final-20260927.log` and
  `/private/tmp/persona-cascade-retry-integration-20260927.log`.
- Test-run cleanup also reports pre-existing unreadable temporary-directory
  warnings; no cleanup of other agents' files or containers was attempted.
- Worktree module import verified; Ruff, compilation, production/test Bandit
  and whitespace checks pass. Independent review gap is resolved. Stage 3
  awaits the normal hooks-enabled commit and task closeout; no full repository
  suite, hosted CI, merge or Stage 2C implementation claim is made.
