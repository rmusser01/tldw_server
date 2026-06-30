---
id: TASK-598
title: Reduce frontend route metadata TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-03 01:11
labels:
- typescript
- frontend
- tsc-baseline
dependencies: []
modified_files:
- apps/packages/ui/src/routes/route-metadata.ts
- apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore route metadata heading-governance TypeScript compatibility and remove the missing route metadata import diagnostics from the frontend standalone tsc baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend standalone tsc no longer reports the route metadata/getRouteHeadingPolicy diagnostics targeted by this task.
- [x] #2 Shared UI package tsc remains clean after route metadata changes.
- [x] #3 Focused route heading metadata coverage passes or any blocker is documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification recorded for this tsc slice:
- RED: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend reported 12 diagnostics, including 3 route metadata/getRouteHeadingPolicy diagnostics.
- RED: bunx vitest run __tests__/smoke/route-heading-governance.metadata.test.ts failed because getRouteHeadingPolicy was missing at runtime.
- GREEN: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend now reports 9 remaining diagnostics, all in unrelated e2e-auth/chat-cockpit/agent-tasks/admin-llamacpp clusters.
- GREEN: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/packages/ui exits 0.
- Focused metadata test now reaches route coverage assertions and documents an existing broader gap: 54 active inventory routes lack metadata rows, mostly nested admin/settings/connector routes. That is outside this tsc slice.
- git diff --check exits 0.
- Bandit not applicable: touched files are TypeScript route metadata and Playwright spec imports only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the frontend standalone tsc baseline by removing the route metadata/getRouteHeadingPolicy cluster. The route metadata helper is restored in the shared UI package and the stage 2 route contract spec now imports getRouteMetadata. Remaining frontend tsc diagnostics are unrelated clusters in e2e auth, chat cockpit, agent tasks, and admin llama.cpp tests.
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
