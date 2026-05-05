---
id: TASK-62
title: Add sandbox cleanup and recovery contract tests
status: Done
assignee: []
created_date: '2026-05-05 03:58'
labels:
  - sandbox
  - runtime-reliability
  - phase-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a local-first Phase 4 sandbox reliability slice that defines and tests cleanup/recovery behavior for portable runtimes and session orchestration without requiring special host-gated VM infrastructure. Scope should validate failed-start cleanup, timeout/cancel cleanup, destroyed-session workspace cleanup, stale metadata behavior, and explicit no-warm-reuse claims for host-local runtimes while keeping repair APIs ownership-checked and not generalized beyond vz_linux.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local-first cleanup/recovery contract tests cover portable sandbox paths without requiring Apple Silicon VZ or other special host infrastructure.
- [x] #2 Tests verify failed-start and timeout/cancel cleanup semantics for at least one portable runtime path.
- [x] #3 Tests verify destroyed session workspace cleanup and stale metadata behavior through service/orchestrator seams.
- [x] #4 Tests preserve the runtime session contract distinction that host-local runtimes do not provide warm runtime reuse.
- [x] #5 Design and implementation plan reference sandbox architecture doctrine and current roadmap gaps.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Implemented local-first cleanup/recovery contract coverage in isolated worktree codex/sandbox-cleanup-recovery-contracts.

Added worktree timeout cleanup test asserting timed_out status, worktree destruction, run-dir removal, SIGTERM, and active tracking cleanup.

Added cross-service durable session destroy test asserting store-backed workspace root removal after service restart-style deletion.

Added host-local session contract test asserting seatbelt/worktree remain workspace_only with no live-health, recovery, or repair claims.

Verification: focused baseline before edits passed 64 tests; focused post-edit test set passed 67 tests; git diff --check passed.

Bandit: skipped because the branch only changes tests, docs, and Backlog.md task metadata; no production Python code changed.

PR #1294 review-fix pass: verified and fixed Qodo/CodeRabbit feedback that the worktree timeout test cleared active-run tracking before asserting cleanup. The test now asserts `_active_proc`, `_active_run_dir`, and `_cancelled_runs` before the test-isolation cleanup block runs.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a local-first sandbox cleanup/recovery contract-test slice. The branch adds design and implementation-plan docs, explicit worktree timeout cleanup coverage, cross-service durable session workspace cleanup coverage, and a host-local session contract assertion that seatbelt/worktree remain workspace-only with no warm runtime reuse or repair/recovery claims. PR review feedback was addressed by moving worktree active-tracking assertions before test-isolation cleanup. Verification passed with the focused sandbox suite: 67 tests passing; git diff --check clean. Bandit was not run because no production code changed.
<!-- SECTION:FINAL_SUMMARY:END -->
