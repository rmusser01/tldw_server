---
id: TASK-12950
title: Preserve legacy single-user API key across media-page refresh
status: Done
labels:
- auth
- webui
- browser-extension
- regression
priority: high
references:
- TASK-12106
- TASK-12127
modified_files:
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
- apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts
- apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts
- apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts
documentation:
- Docs/superpowers/specs/2026-07-12-legacy-api-key-refresh-migration-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Latest dev scrubs pre-migration single-user tldwConfig records that contain a valid serverUrl and apiKey but lack authSource and device-persistence metadata. This makes both the WebUI and packaged browser extension lose authentication after a hard refresh of /media. Add a safe legacy migration to the shared TldwApiClient initialization path, with regression coverage for both surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A valid pre-migration single-user record with serverUrl and apiKey is upgraded to an origin-bound manual device credential during initialization.
- [x] #2 Refreshing /media in the WebUI preserves authentication for an upgraded legacy profile.
- [x] #3 Refreshing /media in the packaged browser extension preserves authentication for an upgraded legacy profile.
- [x] #4 Cookie, environment, and runtime auth replacements still supersede and scrub incompatible stored keys.
- [x] #5 Unit and browser regression tests cover the migration and fail-closed cases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation and verification completed on 2026-07-12 at HEAD 1fcb60243c73798d65c54fafdeb0d4d8107dde90.

Implementation:
- Tightened TldwApiClient.initialize() to migrate only exact eligible advanced/remote pre-metadata single-user credentials into complete manual/device/origin metadata.
- Preserved fail-closed handling for hosted/quickstart transports, malformed or placeholder credentials, invalid origins, and active cookie/environment/runtime replacement auth.
- Added WebUI and packaged Chrome MV3 /media hard-refresh regressions with exact offset-scoped authenticated GET /api/v1/media assertions.
- Packaged extension coverage seeds the released JSON-serialized chrome.storage.sync legacy record, verifies sync removal, and verifies exact migrated local state before and after reload.

Fresh verification:
- Shared auth Vitest matrix: 3 files, 50 tests passed.
- Ambient NEXT_PUBLIC_X_API_KEY isolation: 22/22 passed.
- Ambient VITE_TLDW_API_KEY isolation: 22/22 passed.
- Advanced WebUI persistence Playwright: 3/3 passed.
- Advanced Chrome production build: passed; .output/chrome-mv3 built, token sync OK.
- Packaged extension persistence Playwright: 3/3 passed.
- Extension compile: passed with no diagnostics.
- git diff --check for committed and working changes: passed.
- No generated artifacts are tracked or newly untracked.
- Bandit: not applicable because no Python files were touched.

Known unrelated/tooling baselines:
- Frontend typecheck remains nonzero only for the untouched pre-existing QuickIngestWizardModal.tsx:1813 overflowY TS2322 diagnostic; no touched-file diagnostics.
- Installed ESLint 9.39.2 cannot run from repo root because no root eslint.config file exists; no packages were installed or substituted.
- Existing extension build warnings (duplicate imports, circular chunks, chunk size, stale browser data) remain; build exits zero.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed legacy single-user API-key loss on /media refresh for both the WebUI and browser extension. Eligible pre-metadata advanced/remote credentials now migrate once to the existing origin-bound manual device format, while hosted/quickstart and higher-precedence authentication remain fail-closed. Browser regressions cover hard reload in the WebUI and the released extension chrome.storage.sync-to-local migration path. All task-specific unit, browser, build, and extension compile checks pass; unrelated repository typecheck/lint baselines are documented in Implementation Notes.
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
