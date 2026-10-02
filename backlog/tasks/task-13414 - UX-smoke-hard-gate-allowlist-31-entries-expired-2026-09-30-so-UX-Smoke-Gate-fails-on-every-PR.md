---
id: TASK-13414
title: >-
  UX smoke hard-gate allowlist: 31 entries expired 2026-09-30, so UX Smoke Gate
  fails on every PR
status: Done
assignee: []
created_date: '2026-10-01 17:54'
updated_date: '2026-10-02 02:41'
labels:
  - bug
  - webui
  - testing
  - ci
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
apps/tldw-frontend/e2e/smoke/smoke.setup.ts SMOKE_HARD_GATE_ALLOWLIST has 31 entries (owners WebUI and Platform) with expiresOn 2026-09-30. Since 2026-10-01 UTC the spec 'hard-gate allowlist entries have current ownership metadata' fails first, so UX Smoke Gate (frontend-ux-gates.yml; not a required check) is red on every PR. First seen on #3065 run 36801831063. The expiry is a review deadline, so blanket-extending the dates defeats it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each expired entry is re-checked against a current all-pages smoke run: removed if its noise no longer occurs, otherwise renewed with a new expiresOn and a linked task for the underlying noise
- [x] #2 UX Smoke Gate passes the allowlist-metadata check
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Premise was stale by pickup time. The 31 expired entries had already been cut to 4 on dev (6b7cee876e for TASK-13377.9 / PR #3023 and 2f05d15913 for TASK-13260.278.18.83.51, merged in 6cd957e612), and the UX Smoke Gate was green again (run 36946184688: 102 passed, no allowlisted hits). Each of the original 31 entries was re-checked anyway against full all-pages runs on dev d81c13fddd. Setup: local copy of the CI job (MINIMAL_TEST_APP single_user backend via server_lifecycle.py on 8777, mock OpenAI on 18777, build:dev standalone bundle staged as in CI on 8788, Node 20.19.5, CI=true), plus a second run on next dev --webpack. An uncommitted recorder in classifySmokeIssues logged every classified issue and every old/current rule match; it was removed before commit. Results: none of the 27 already-removed rules matched anything. Moderation 404 (expiring 10-08): unmatched, because the browser 404 arrives after the gate classifies; removed. Drawer width deprecation: fixed at source (ArchivedItemsDrawer and two WritingPlaygroundShell drawers, width -> size) and removed; with the fix reverted the next dev Kanban recovery fixture fails on the warning. Forced route-boundary pair: still fires on all 16 fixture routes in next dev and cannot fire in production; kept at 2026-10-31, linked to TASK-13406. The deliberate /__wayfinding-missing-route__ document 404 was unallowlisted after the 27-rule cut, so the non-gated Wayfinding 404 test failed on all attempts; re-added as narrow entry m5-wayfinding-missing-route-document-404 (2026-10-31, TASK-13406). Verification: CI gate command 102 passed including the metadata test; Wayfinding 404 test plus smoke-allowlist.spec.ts 6 passed; next dev route boundaries 16/16 and Kanban 3/3 passed; dev's list fails metadata from 2026-10-09, this branch is valid through 2026-10-31; tsc 0 errors; ESLint on the e2e files 0 errors; vitest 12 passed. Bandit skipped: the change is TypeScript-only, with no Python touched. Docs: no allowlist doc lists entries, so none needed updating. Known non-allowlist failures in the full spec: Route Error Boundaries need next dev by design; Wayfinding settings alias/mobile selector filed as TASK-13419; intermittent next dev compile timeouts and ERR_EMPTY_RESPONSE. PR https://github.com/rmusser01/tldw_server/pull/3083
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Re-reviewed the smoke hard-gate allowlist against fresh full all-pages runs on both the CI-profile standalone bundle and next dev. The original 31 expired entries had already been cut to 4 on dev. This change removes the moderation 404 entry, which was due to expire 2026-10-08 and matched nothing. It fixes the kanban Drawer width deprecation at its source and drops that entry. The two deliberate route-boundary entries are kept and linked to TASK-13406. The deliberate wayfinding missing-route document 404, which had been unallowlisted since the cut, is restored as a narrow entry. All remaining entries expire 2026-10-31 under TASK-13406. The CI gate command passes locally (102 tests, including the metadata check). A separate full-spec failure, Wayfinding settings alias, is filed as TASK-13419. PR #3083.
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
