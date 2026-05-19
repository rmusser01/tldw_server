---
id: TASK-439
title: Run WebUI operations integrations browser QA and final verification
status: Done
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- apps/tldw-frontend/components/networking/ServerReadinessGate.tsx
- apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx
- apps/tldw-frontend/e2e/utils/page-objects/SourcesPage.ts
- apps/tldw-frontend/e2e/workflows/tier-2-features/sources.spec.ts
- backlog/tasks/task-439 - Run-WebUI-operations-integrations-browser-QA-and-final-verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP10 Task 7 as a verify-first slice for the operations/integrations route work. Run the planned route/component/E2E verification, perform browser-observed QA for first-time and power-user paths across the scoped routes, and only make code changes if browser QA exposes a scoped defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Planned WP10 route contract tests are run and results recorded.
- [x] #2 Planned WP10 component state tests are run and results recorded.
- [x] #3 Parent-required and expanded WP10 E2E commands are run on an isolated dev-server port or any blocker is documented with evidence.
- [x] #4 Browser QA covers first-time and power-user paths for admin, MCP Hub, sources, connectors, integrations, scheduled tasks, watchlists, workflow editor, and skills routes.
- [x] #5 Any scoped browser QA defects are fixed with focused changes and verified; unrelated repo cleanup is avoided.
- [x] #6 Final task notes document verification evidence, Bandit applicability, and remaining risks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md#task-7-browser-qa-and-final-verification
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification found two scoped defects before closeout.

- ServerReadinessGate waited for backend health on placeholder/recovery routes even when E2E had seeded the existing offline/test bypass flags. It now honors `__tldw_allow_offline` and `__tldw_test_bypass` before issuing health checks, with component coverage.
- Sources E2E treated the real recovery heading "Cannot reach Sources" as a failure instead of a valid backend-unavailable state. The page object and lifecycle spec now recognize that state and wait for loading to settle before online-only assertions.
- Clean-branch E2E against current `dev` also exposed a backend-offline Sources shell that can keep showing loading under the heading. The Sources loading state now has a stable status/test hook, and the E2E only runs list/empty assertions after the page actually presents a finished data state.
- Browser QA found a mobile Watchlists orientation alert layout defect where Ant `Alert.action` squeezed the title into single-character wrapping. The action buttons now live in the alert description block, and the mobile screenshot was rechecked.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed WP10 Task 7 verification. Route contract tests passed: 8 files / 21 tests. Component state tests passed: 6 files / 48 tests, plus ServerReadinessGate passed 1 file / 4 tests. Focused placeholder E2E passed 9 tests. Parent-required E2E passed 13 tests with 8 expected skips. Expanded WP10 E2E passed 22 tests with 5 expected skips after the Watchlists mobile fix. Browser plugin was unavailable in this Codex session with `Browser is not available: iab`, so browser-observed QA used Playwright directly against the local Next dev server on port 18080. Desktop QA covered /admin, /admin/server, /mcp-hub, /sources, /connectors, /integrations, /scheduled-tasks, /watchlists, /workflow-editor, and /skills. Mobile QA covered /watchlists, /workflow-editor, and /mcp-hub. Inspected routes rendered expected content or recovery states with no readiness-gate or framework-overlay blockers. Console diagnostics had expected backend-offline fetch failures because the API server was intentionally not running, and no page errors. Static cleanup checks passed for touched files: git diff --check and trailing-whitespace scan. Clean-branch packaging on current `dev` preserved newer route/readiness behavior and reverified the final delta: ServerReadinessGate passed 8 tests, Watchlists orientation guidance passed 4 tests, SourcesWorkspacePage passed 8 tests, and focused Playwright Sources/placeholder routes passed 14 tests with 2 expected backend-dependent skips. Bandit was not applicable because this slice touched frontend TypeScript/E2E and Backlog metadata only; no Python code was changed. Remaining risk: this final QA validates offline/recovery rendering and route behavior, not live backend data paths.
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
