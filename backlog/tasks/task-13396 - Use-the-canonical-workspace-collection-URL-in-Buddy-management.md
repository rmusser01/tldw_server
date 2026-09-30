---
id: TASK-13396
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrected the shared listWorkspaces collection path to /api/v1/workspaces/ after real authenticated HTTP reproduced a307 and Buddy management failed under redirect:error. Updated two existing contract assertions, both RED before the one-line repair. Final scoped checks:37 workspace contracts,13 redirect security,37 Buddy management components and1 route lifecycle passed. Real quickstart WebUI selected Pixel Migu without Persona, attached the API-seeded disposable workspace, retained Static mode and visible Buddy on Watchlists, then persisted Dynamic. Independent review has no findings; diff check passes. Existing ESLint1 error/52 warnings and3-file Prettier debt unchanged against HEAD; Bandit N/A for TypeScript-only changes. No full suite. Existing ADR005 applies. Evidence and exact native/extension/voice/upgrade limits: Docs/Reviews/2026-09-29-buddy-current-dev-qualification.md and source-hashed artifact receipts. TASK13227 stays In Progress.
PR3056 Qodo follow-up: removed nested NOTES/final-summary markers from TASK13227, moved the qualification note into its existing canonical Implementation Notes section, and normalized this new task to the same marker spelling. Verified historical content preservation and parsed-section behavior with the repository Python Backlog tooling; temporary-copy append/summary edits retain exactly one section. Production Chrome build on5cbca66a passed117s and exported MV3 background/options/side-panel entrypoints validated; installed-extension interaction remains open. Build hashes recorded on PR3056; raw logs stay local.
Rebased PR3056 onto latest dev6110d2ae after PR3036 changed database handling. Range-diff preserves all3 reviewed commits exactly; shared apps sources are unchanged from the earlier base. Fresh focused checks:37 workspace contracts passed and25 independent-Buddy backend cases passed, with1 PostgreSQL case skipped. Original WebUI/build receipts retain their original source attribution; sanitized rebase-verification.json records this separate check. No full sweep or additional native/voice claim.
2026-09-30 dev rebase: dev607431154c introduced an unrelated RG task with provisional ID TASK-13395. The Python Backlog CLI has no renumber operation, so only this owned record filename/frontmatter ID was moved to unique TASK-13396 with every historical section retained. The merged RG task is untouched. Subsequent section edits continue through the Python CLI. All four existing PR commits are patch-identical after rebase; FastAPI0.141.1 qualification is in progress.
FastAPI rebase qualified at113653debd on dev607431154c: isolated FastAPI0.141.1 with existing Starlette1.2.1/Pydantic2.11.7 passed25 independent-Buddy cases and8 served-route guards in72.03s, with1 PostgreSQL fixture skip. Separate TestClient/real SQLite probe passed3.11s: canonical workspace200, noncanonical307, effective auth retained. Its first draft only had an envelope-shape assertion error, corrected without production changes. Both owned task files passed disposable Python parser/append/summary round trips. Separate sanitized fastapi-rebase-verification.json retains source/log hashes; prior WebUI/build evidence is unchanged. No full suite or extra native/voice claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
