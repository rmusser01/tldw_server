---
id: TASK-431
title: Implement WebUI operations route job contract
status: Done
labels:
- webui
- extension
- ux-remediation
- routes
- wp10
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the WP10 operations route-job metadata contract for admin, MCP, sources, connector placeholders, integrations, scheduled tasks, watchlists, workflow editor, and skills routes. This is a frontend metadata/test slice only; it must not build missing backend systems or change route paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed operations route-job inventory covers the WP10 root route set from the implementation plan, including admin drill-down routes and connector placeholder child pages.
- [x] #2 Connector routes are explicitly marked as placeholders and scheduled tasks/integrations stay tied to existing probes, not invented backend capability maps.
- [x] #3 Admin root is represented as an operations overview with drill-down relations to existing admin modules.
- [x] #4 Route-job tests verify route coverage, backend-gate distinctions, implementation ownership, and metadata alignment.
- [x] #5 Focused Vitest verification and diff check are recorded; Bandit is not applicable because no Python code was touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `apps/packages/ui/src/routes/operations-route-jobs.ts` with a typed operations route-job inventory and lookup helpers.

Added `apps/packages/ui/src/routes/__tests__/operations-route-jobs.test.ts` using a red-green TDD cycle:
- Red: `bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts` failed because `../operations-route-jobs` did not exist.
- Green: `bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts` passed after adding the module.

Verification:
- `bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts`
- `bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-registry.visibility.test.ts`
- `git diff --check -- apps/packages/ui/src/routes/operations-route-jobs.ts apps/packages/ui/src/routes/__tests__/operations-route-jobs.test.ts`

Bandit was not run because this slice touched TypeScript and Backlog Markdown only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the WP10 operations route-job contract for admin, MCP, sources, connector placeholders, integrations, scheduled tasks, watchlists, workflow editor, and skills routes. The contract keeps connector pages honest as placeholders, ties scheduled tasks/integrations to existing probe behavior, and records admin root as an overview with drill-down module routes. No backend APIs or route paths were changed.
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
