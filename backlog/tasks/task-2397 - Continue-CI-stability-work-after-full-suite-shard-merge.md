---
id: TASK-2397
title: Continue CI stability work after full-suite shard merge
status: In Progress
labels:
- ci
- github-actions
- stability
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track follow-up work after PR #2258 merged the full-suite shard split into dev. Initial scope: monitor the merged workflow behavior on dev/main/release triggers, inspect any remaining queued or stale workflow-run behavior, and address residual CI runtime/stability issues without reintroducing early pytest maxfail behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Monitor the merged CI shard workflow behavior on `dev`, `main`, and release/manual triggers.
- [ ] Inspect any remaining queued, cancelled, or failing GitHub Actions jobs before pushing follow-up fixes.
- [ ] Address residual CI runtime or stability issues without reintroducing early pytest `--maxfail` behavior.
- [ ] Record local and GitHub verification evidence before merging follow-up work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR #2258 merged into `dev` at 2026-06-23T05:55:28Z with merge commit `c16aada8dd7e3061eb89c3c312ffcc1986fafa35`.
- Fresh PR #2258 checks before merge: 755 passed, 4 skipped, 0 pending, 0 failed, 0 cancelled.
- One unrelated stale queued CodeQL run for PR #1985 returned HTTP 500 when cancellation was attempted; it did not block #2258 completion.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
