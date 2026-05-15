---
id: TASK-304.2
title: Implement Research Studio degraded health pass-through
status: Done
assignee:
  - Codex
created_date: '2026-05-12 16:22'
updated_date: '2026-05-12 16:25'
labels:
  - implementation
  - research-studio
  - webui
  - health
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HTTP 200 and 206 health responses with body status ok healthy or degraded allow the app past ServerReadinessGate
- [x] #2 Malformed health responses network failures and explicit unhealthy statuses remain in retry/blocking path
- [x] #3 Focused ServerReadinessGate tests cover degraded enterable and unhealthy/malformed blocked states
- [x] #4 Implementation does not add capability-safety claims for chat or generation
- [x] #5 Verification and Bandit/frontend-only rationale are recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current ServerReadinessGate behavior and tests.
2. Add failing degraded-health tests before production code changes.
3. Implement the minimal readiness classification change.
4. Run focused readiness tests and diff hygiene.
5. Update this task with verification and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red step: after adding degraded-health tests, `bun run test:run components/networking/__tests__/ServerReadinessGate.test.tsx` failed with 2 degraded-response tests failing and 5 tests passing.
Green verification: `bun run test:run components/networking/__tests__/ServerReadinessGate.test.tsx` passed with 7 tests.
Diff hygiene: `git diff --check` passed.
Bandit not run: frontend TypeScript/test-only change with no backend Python touched.
Scope note: this only changes app entry readiness classification. It does not add capability-safety claims for chat or generation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented degraded health pass-through for ServerReadinessGate. Reachable health responses with `ok`, `healthy`, or `degraded` body statuses now enter the app, including HTTP 206 degraded responses, while malformed and explicitly unhealthy responses remain behind the readiness screen. Added focused regression tests for degraded, unhealthy, and malformed health envelopes.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
