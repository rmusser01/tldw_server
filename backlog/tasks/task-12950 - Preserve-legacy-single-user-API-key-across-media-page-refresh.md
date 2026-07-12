---
id: TASK-12950
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
- [ ] #1 A valid pre-migration single-user record with serverUrl and apiKey is upgraded to an origin-bound manual device credential during initialization.
- [ ] #2 Refreshing /media in the WebUI preserves authentication for an upgraded legacy profile.
- [ ] #3 Refreshing /media in the packaged browser extension preserves authentication for an upgraded legacy profile.
- [ ] #4 Cookie, environment, and runtime auth replacements still supersede and scrub incompatible stored keys.
- [ ] #5 Unit and browser regression tests cover the migration and fail-closed cases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-legacy-api-key-refresh-migration-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
