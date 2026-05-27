---
id: TASK-523
title: Implement PR 6 Notes capture destination pickers
status: Done
labels:
- notes
- ux
- extension
- webui
- pr6
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the PR 6 /notes remediation slice: replace raw destination IDs in directly connected capture flows where the app can discover valid destinations. First verify whether workspace and notes-folder list APIs exist; split to workspace picker only if notes-folder APIs are absent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace destination uses a picker/search when workspace list data is available.
- [x] #2 Folder destination uses a picker only after verifying or adding a notes-folder list/create API. Verified note folder attachments exist, but no public notes-folder list/create route was found in `tldw_Server_API/app/api/v1/endpoints/notes.py`; folder picker is split/deferred.
- [x] #3 Invalid destination is prevented before submit where possible.
- [x] #4 Raw ID fallback, if retained, is clearly secondary/advanced.
- [x] #5 Existing save-to-note-only flow remains fast.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR 6A as a workspace picker-only slice. Added `tldwClient.listWorkspaces()` for `GET /api/v1/workspaces/`. `WebClipperPanel` now lazy-loads workspaces only after Workspace/Both is selected, keeps note-only saves from calling the workspace list endpoint, uses a Workspace select when data is available, hides raw Workspace ID while loading, and exposes raw ID only as an advanced fallback when picker data exists or as fallback when loading fails or no options are available.

Folder picker is deferred because the backend exposes note folder data on notes, but no public note-folder list/create endpoint was discoverable in `tldw_Server_API/app/api/v1/endpoints/notes.py`. Existing folder ID validation remains in place.

Verification:
- RED: added failing client and Web Clipper tests for workspace list, picker submit, picker load failure fallback, note-only speed, and raw-ID hidden during loading.
- GREEN: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx` passed, 35 tests.
- `git diff --check` passed.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json --pretty false` reached the known unrelated baseline error in `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3)`.
- `bun run compile` in `apps/extension` passed.
- `bun run dev` in `apps/extension` built the WXT chrome-mv3 dev extension output. Direct browser rendering was not usable outside extension context: `http://localhost:3000/sidepanel.html` returned 404 and `file://.../sidepanel.html` was blocked by extension/dev-server CORS.
- Bandit skipped because this slice touched frontend TypeScript and Backlog task files only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workspace capture destinations now use a picker when workspace list data is available, with loading/error states and an advanced raw-ID fallback. Added regression coverage for workspace list client calls, picker submit payloads, failed-load fallback, hidden raw ID while loading, validation, and note-only save speed. Browser/WXT verification built the extension dev output, but direct browser rendering of sidepanel.html was not usable because the WXT dev URL returned 404 outside extension context and file loading was blocked by extension/dev-server CORS.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed or explicitly split with rationale recorded.
- [x] #2 Focused tests or verification recorded.
- [x] #3 Documentation updated when relevant.
- [x] #4 Bandit run for touched Python scope when applicable or frontend-only skip documented.
- [x] #5 Final summary added.
- [x] #6 Known skips or blockers documented.
<!-- DOD:END -->
