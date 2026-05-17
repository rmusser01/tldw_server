---
id: TASK-406
title: Add Sources server path picker
status: Done
labels:
- webui
- extension
- sources
- filesystem
- implementation
documentation:
- Docs/superpowers/plans/2026-05-17-sources-server-path-picker-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an allowed-roots-only server directory picker for the Sources local-directory path field, including backend browse API, shared frontend client/hook, SourceForm picker UI, tests, Bandit, and browser QA.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend browse endpoint lists only configured ingestion source allowed roots and their immediate child directories.
- [x] #2 Browse endpoint rejects or normalizes paths outside configured allowed roots without leaking arbitrary filesystem contents.
- [x] #3 SourceForm local-directory path field has a Browse action that opens a server directory picker and writes the selected path back into the field.
- [x] #4 Picker handles empty roots, permission errors, loading states, and manual path entry fallback.
- [x] #5 Focused backend/frontend tests, Bandit, and browser QA are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-sources-server-path-picker-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started follow-up implementation in the existing `codex/sources-notes-ui-exposure` worktree. User approved limiting the picker to configured ingestion source allowed roots only.

Implemented `GET /api/v1/ingestion-sources/browse-directories`, shared client/hook support, and SourceForm Browse modal. The endpoint enforces the existing local-directory source entitlement and only exposes configured allowed roots plus immediate non-symlink child directories.

Rendered QA used Playwright because the in-app Browser could not reach the isolated localhost dev server. Desktop QA selected `/private/tmp/tldw-sources-picker-root/notes`; mobile QA confirmed the modal fit a `390x844` viewport. Replaced the initial Ant Design `List` usage with semantic list markup to avoid a new deprecation warning.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a safe server-folder picker for the Sources local-directory path field. The picker is backend-bounded to configured ingestion source allowed roots, supports root browsing and child directory selection, preserves manual path entry, and writes the selected folder back into the form.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_path_browser.py tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py -q` -> `35 passed, 5 warnings`
- `bunx vitest run src/services/__tests__/tldw-api-client.ingestion-sources.test.ts src/hooks/__tests__/use-ingestion-sources.test.tsx src/components/Option/Sources/__tests__/SourceForm.test.tsx` -> `3 passed (3), 27 tests passed`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py tldw_Server_API/app/api/v1/schemas/ingestion_sources.py -f json -o /tmp/bandit_sources_path_picker.json` -> exit 0, no findings
- `git diff --check` -> exit 0
- Playwright rendered QA desktop and mobile -> picker interaction passed with no console or page errors
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
