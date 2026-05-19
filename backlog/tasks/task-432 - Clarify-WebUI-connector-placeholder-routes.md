---
id: TASK-432
title: Clarify WebUI connector placeholder routes
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
Replace the generic connector Coming Soon pages with honest placeholder route-state pages that reuse the operations route-job contract and point users to currently supported alternatives without implying connector catalog, jobs, or source workflows exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Connector placeholder routes identify themselves as connector placeholders and use route-job metadata for the current route.
- [x] #2 Each connector placeholder exposes exactly one primary action and points secondary alternatives to current supported routes such as Settings, Integrations, Sources, Scheduled Tasks, or Watchlists.
- [x] #3 Connector placeholder copy does not imply connector catalog, connector jobs, or connector source workflows are already implemented.
- [x] #4 Tests cover the shared connector placeholder component and page-level route wiring.
- [x] #5 Focused Vitest verification and diff check are recorded; Bandit is not applicable because no Python code was touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `apps/tldw-frontend/components/navigation/ConnectorRoutePlaceholder.tsx` and rewired:
- `apps/tldw-frontend/pages/connectors/index.tsx`
- `apps/tldw-frontend/pages/connectors/browse.tsx`
- `apps/tldw-frontend/pages/connectors/jobs.tsx`
- `apps/tldw-frontend/pages/connectors/sources.tsx`

The component reads labels/jobs from the operations route-job contract and renders connector-specific placeholder state, one primary action, and distinct supported alternatives.

Added tests:
- `apps/tldw-frontend/__tests__/navigation/connector-route-placeholder.test.tsx`
- `apps/tldw-frontend/__tests__/pages/connector-placeholder-routes.test.ts`

Updated focused Playwright expectations for the new connector headings:
- `apps/tldw-frontend/e2e/workflows/route-placeholder-settings.spec.ts`
- `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`

Note: `route-contract-stage2.spec.ts` already had pre-existing uncommitted changes from the route-contract slice; this task only changed the connector expected titles in that file.

Verification:
- Red: `bunx vitest run __tests__/navigation/connector-route-placeholder.test.tsx __tests__/pages/connector-placeholder-routes.test.ts` failed because `ConnectorRoutePlaceholder` did not exist and connector pages still used `RoutePlaceholder`.
- Green: `bunx vitest run __tests__/navigation/connector-route-placeholder.test.tsx __tests__/pages/connector-placeholder-routes.test.ts __tests__/navigation/route-placeholder-component.test.tsx`
- `bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts`
- `bunx playwright test e2e/workflows/route-placeholder-settings.spec.ts e2e/smoke/route-contract-stage2.spec.ts --reporter=line --workers=1`
- `git diff --check -- apps/tldw-frontend/components/navigation/ConnectorRoutePlaceholder.tsx apps/tldw-frontend/pages/connectors/index.tsx apps/tldw-frontend/pages/connectors/browse.tsx apps/tldw-frontend/pages/connectors/jobs.tsx apps/tldw-frontend/pages/connectors/sources.tsx apps/tldw-frontend/__tests__/navigation/connector-route-placeholder.test.tsx apps/tldw-frontend/__tests__/pages/connector-placeholder-routes.test.ts apps/tldw-frontend/e2e/workflows/route-placeholder-settings.spec.ts apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts "backlog/tasks/task-432 - Clarify-WebUI-connector-placeholder-routes.md"`

Bandit was not run because this slice touched TypeScript, TSX, Playwright test expectations, and Backlog Markdown only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Connector placeholder routes now render truthful connector-specific route-state pages instead of generic Coming Soon pages. The pages identify connector support as inactive in this build, avoid implying catalog/jobs/source workflows are implemented, and point users to supported alternatives such as Settings, Integrations, Sources, Scheduled Tasks, and Watchlists.
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
