---
id: TASK-2386
title: Implement sandbox operator status endpoint Slice 1
status: In Progress
labels:
- sandbox
- operator-ux
- vz_linux
- implementation
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 1 of sandbox operator/admin status consolidation from the approved design and plan: portable read-only status projection, service wrapper, schema, admin endpoint, tests, and docs. No evidence-file ingestion, generated_at, helper lifecycle mutation, launchd mutation, repair mutation, image-store cleanup mutation, or real VM execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Read-only operator status projection exists and validates through schema.
- [ ] #2 Admin-only GET /api/v1/sandbox/admin/operator-status endpoint returns structured payload.
- [ ] #3 Unconfigured VZ/evidence does not degrade otherwise usable installs.
- [ ] #4 Section failures are isolated and visible without preventing other sections from rendering.
- [ ] #5 Docs and RBAC coverage are updated.
- [ ] #6 Focused pytest and Bandit verification pass or any unrelated/pre-existing issues are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation follows Docs/superpowers/plans/2026-06-18-sandbox-operator-status-implementation-plan.md.
- [ ] #2 No mutation or host-gated execution paths are introduced.
- [ ] #3 Backlog task records verification commands and results.
- [ ] #4 Changes are committed on codex/sandbox-operator-status.
<!-- DOD:END -->
