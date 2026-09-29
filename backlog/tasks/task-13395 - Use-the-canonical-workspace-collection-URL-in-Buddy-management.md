---
id: TASK-13395
title: Use the canonical workspace collection URL in Buddy management
status: Done
assignee:
  - '@codex'
created_date: '2026-09-29 18:18'
updated_date: '2026-09-29 18:37'
labels: []
dependencies: []
references:
  - TASK-13227
documentation:
  - Docs/Reviews/2026-09-29-buddy-current-dev-qualification.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore workspace choices in Buddy and Persona Management when authenticated browser requests reject redirects. The collection caller omits the slash required by the server route, producing a 307 and a Failed to fetch error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared workspace listing reaches the canonical collection route without a redirect.
- [x] #2 Existing destination picker and workspace contract tests pass with the canonical route.
- [x] #3 Real disposable WebUI Buddy management loads workspace choices without the redirect error; request redirect protections remain intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: routine endpoint spelling repair within existing ownership and transport contracts. 1. Confirm GET /api/v1/workspaces returns307 while /api/v1/workspaces/ returns200. 2. Read all listWorkspaces callers and update existing canonical collection contract assertions; run them RED. 3. Correct the single shared collection path, retaining redirect:error. 4. Run focused domain and redirect security regressions, scoped lint, live Buddy management, and record source-bound evidence plus native/voice limits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected the shared listWorkspaces collection path to /api/v1/workspaces/ after real authenticated HTTP reproduced a307 and Buddy management failed under redirect:error. Updated two existing contract assertions, both RED before the one-line repair. Final scoped checks:37 workspace contracts,13 redirect security,37 Buddy management components and1 route lifecycle passed. Real quickstart WebUI selected Pixel Migu without Persona, attached the API-seeded disposable workspace, retained Static mode and visible Buddy on Watchlists, then persisted Dynamic. Independent review has no findings; diff check passes. Existing ESLint1 error/52 warnings and3-file Prettier debt unchanged against HEAD; Bandit N/A for TypeScript-only changes. No full suite. Existing ADR005 applies. Evidence and exact native/extension/voice/upgrade limits: Docs/Reviews/2026-09-29-buddy-current-dev-qualification.md and source-hashed artifact receipts. TASK13227 stays In Progress.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed workspace choices in Buddy management by using the existing canonical collection route without relaxing redirect security. Targeted regression and real disposable WebUI checks pass; broader native, extension and voice acceptance remains tracked separately.
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
