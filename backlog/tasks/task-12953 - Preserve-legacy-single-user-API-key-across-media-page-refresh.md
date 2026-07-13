---
id: TASK-12953
title: Preserve legacy single-user API key across media-page refresh
status: In Progress
labels:
- auth
- webui
- browser-extension
- regression
priority: high
references:
- TASK-12106
- TASK-12127
- https://github.com/rmusser01/tldw_server/pull/2719
documentation:
- Docs/superpowers/specs/2026-07-12-legacy-api-key-refresh-migration-design.md
modified_files:
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
- apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts
- apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts
- apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts
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
Implementation complete on PR #2719. Review remediation and final verification are tracked in TASK-12952. Keep this task In Progress until the requester supplies the required human-written PR Change summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replacement for the auth-refresh task that was originally created as TASK-12950 before latest dev independently assigned TASK-12950 to Quick Ingest. The unrelated Quick Ingest task from dev must remain unchanged. This replacement remains In Progress until PR review remediation is verified and the required human-written Change summary is provided.

Implementation completed before review remediation:
- Tightened TldwApiClient.initialize() to migrate only exact eligible advanced/remote pre-metadata single-user credentials into complete manual/device/origin metadata.
- Preserved fail-closed handling for hosted/quickstart transports, malformed or placeholder credentials, invalid origins, and active cookie/environment/runtime replacement auth.
- Added WebUI and packaged Chrome MV3 /media hard-refresh regressions with exact offset-scoped authenticated GET /api/v1/media assertions.
- Packaged extension coverage seeds the released JSON-serialized chrome.storage.sync legacy record, verifies sync removal, and verifies exact migrated local state before and after reload.

Pre-review verification:
- Shared auth Vitest matrix: 3 files, 50 tests passed.
- Ambient NEXT_PUBLIC_X_API_KEY isolation: 22/22 passed.
- Ambient VITE_TLDW_API_KEY isolation: 22/22 passed.
- Advanced WebUI persistence Playwright: 3/3 passed.
- Advanced Chrome production build: passed; .output/chrome-mv3 built, token sync OK.
- Packaged extension persistence Playwright: 3/3 passed.
- Extension compile: passed with no diagnostics.
- git diff --check for committed and working changes: passed.
- No generated artifacts were tracked or newly untracked.
- Bandit: not applicable because no Python files were touched.

Known unrelated/tooling baselines:
- Frontend typecheck was nonzero only for the untouched pre-existing QuickIngestWizardModal.tsx:1813 overflowY TS2322 diagnostic; no touched-file diagnostics.
- Installed ESLint 9.39.2 could not run from repo root because no root eslint.config file existed; no packages were installed or substituted.
- Existing extension build warnings (duplicate imports, circular chunks, chunk size, stale browser data) remained while the build exited zero.

PR: https://github.com/rmusser01/tldw_server/pull/2719.
Review remediation is tracked in TASK-12952; append the fresh post-rebase verification here before final handoff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed legacy single-user API-key loss on /media refresh for both the WebUI and browser extension. Eligible pre-metadata advanced/remote credentials migrate once to the existing origin-bound manual device format, while hosted/quickstart and higher-precedence authentication remain fail-closed. Browser regressions cover hard reload in the WebUI and the released extension chrome.storage.sync-to-local migration path.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [ ] #7 Human-written PR Change summary provided
<!-- DOD:END -->
