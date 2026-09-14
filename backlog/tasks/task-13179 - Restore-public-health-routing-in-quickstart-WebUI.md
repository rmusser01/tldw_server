---
id: TASK-13179
title: Restore public health routing in quickstart WebUI
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 15:38'
updated_date: '2026-09-05 15:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quickstart WebUI readiness probes same-origin /health, but the standalone Next proxy forwards only /api routes. Healthy backends therefore leave normal quickstart entry blocked with HTTP 404.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Quickstart forwards /health to the configured internal backend public /health endpoint.
- [x] #2 Advanced mode and quickstart without an internal backend keep existing routing behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing focused Next configuration regression for public health routing and mode isolation.
2. Add only the quickstart /health rewrite.
3. Run targeted configuration tests and static checks; hand off runtime restart and live retest.
ADR required: no
ADR path: N/A
Reason: Routine restoration of the existing quickstart public-health contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added only a quickstart /health rewrite to the configured internal backend public /health endpoint. Existing /api rewrites, advanced-mode routing, and fail-closed validation for a missing internal origin are unchanged.
Files: apps/tldw-frontend/next.config.mjs and __tests__/next-config-quickstart-health.test.ts.
ADR required: no; routine restoration of the documented quickstart public-health routing contract.
Red: new configuration regression showed 2 quickstart routing failures and 3 passing mode/configuration isolation cases. Green: bun x vitest run __tests__/next-config-quickstart-health.test.ts __tests__/next-config-dev-watch-guard.test.ts from apps/tldw-frontend passed 12/12 tests across 2 files. Logs: /private/tmp/task-13179-red.log and /private/tmp/task-13179-green.log.
Validation: targeted ESLint, new-test Prettier check, and git diff --check passed. Bandit inapplicable to JS/TS-only scope. Existing Node experimental localStorage warning remains.
Documentation: task captures this restoration; documented Next-owned same-origin quickstart proxy architecture already applies and requires no change.
Live runtime restart and fresh quickstart readiness retest remain with the coordinating agent; no runtime, browser, report, or commit changes were made here.

Live quickstart retest: same-origin /health returns200/ok and readiness state is ready. Actual cookie session mint also200. Persona then hits the separate legacy cookie-auth dependency gap TASK-13181; no quickstart asset or stream pass claimed.

Coordinated final validation: 265 focused frontend tests, 54 backend tests, production Bandit0 findings, scoped frontend ESLint0 errors (warnings documented), unchanged Python lint baseline, real browser evidence and limitations recorded in Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md. Repository-wide typechecking remains limited by80 diagnostics across6 unchanged unrelated files; no full suite run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quickstart public health routing repaired and verified in the real browser; remaining cookie-auth Persona failure is TASK-13181.
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
