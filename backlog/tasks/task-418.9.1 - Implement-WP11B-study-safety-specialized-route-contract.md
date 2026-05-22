---
id: TASK-418.9.1
title: Implement WP11B study safety specialized route contract
status: Done
labels:
- webui
- ux-audit
- wp11b
- routes
- study
- safety
priority: medium
parent_task_id: TASK-418.9
documentation:
- Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md
modified_files:
- apps/packages/ui/src/routes/study-safety-specialized-route-jobs.ts
- apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-jobs.test.ts
- apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
- apps/packages/ui/src/routes/option-evaluations.tsx
- apps/packages/ui/src/routes/option-kanban-playground.tsx
- apps/packages/ui/src/routes/route-metadata.ts
- backlog/tasks/task-418.9.1 - Implement-WP11B-study-safety-specialized-route-contract.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11B Task 1 from the WebUI study/safety/specialized implementation plan. Add a pure route-job metadata contract and focused route/sidepanel tests for evaluations, flashcards, quiz, moderation playground, content review, claims review, data tables, chunking playground, kanban, VN assets, and VN play. Keep this slice frontend metadata/test-focused and avoid backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Pure route-job contract covers every WP11B study, safety, review, and specialized route exactly once.
- [x] Contract tests verify route labels, user jobs, classifications, audit finding coverage, and alias ownership.
- [x] Route boundary tests verify shared route wrappers, moderation alias routing, claims-review alias routing, and Next-only VN page ownership.
- [x] Sidepanel flashcards test verifies flashcards is the only WP11B route registered for sidepanel.
- [x] Narrow route ownership fixes align evaluations error recovery, kanban route identity, and claims-review metadata with actual route behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified a red state for the initial route-job test before adding `study-safety-specialized-route-jobs.ts`.
- Cross-checked route ownership against `route-registry.tsx`, `deferred-options-route.tsx`, shared option route wrappers, and Next page files before locking the contract.
- Updated `/claims-review` metadata to match its current Next redirect to `/content-review`; `/moderation-playground` was already a shared redirect to `/moderation/rules`.
- Verification: `bunx vitest run src/routes/__tests__/study-safety-specialized-route-jobs.test.ts src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-registry.visibility.test.ts --maxWorkers=1 --no-file-parallelism` passed.
- Verification: `git diff --check` passed.
- Verification: `bunx tsc --noEmit --target ES2022 --module ESNext --moduleResolution bundler src/routes/study-safety-specialized-route-jobs.ts` passed.
- Typecheck note: package-wide `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json` still fails on inherited baseline TypeScript debt outside this slice; no reported errors referenced the new route-job contract or changed route wrappers.
- Bandit skipped because this slice touched TypeScript/Markdown only and no Python files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added a pure WP11B route-job contract for evaluations, flashcards, quiz, moderation/content review aliases, data tables, chunking, kanban, and VN routes.
- Added focused tests for route-job coverage, route boundary ownership, Next-only specialized pages, sidepanel flashcards isolation, and route metadata alias behavior.
- Fixed narrow ownership mismatches found by the tests: evaluations now has a route error boundary, `/kanban` reports the canonical route id/label, and `/claims-review` metadata now matches its redirect to `/content-review`.
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
