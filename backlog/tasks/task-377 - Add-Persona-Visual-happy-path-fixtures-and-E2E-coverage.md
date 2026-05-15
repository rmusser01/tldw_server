---
id: TASK-377
title: Add Persona Visual happy-path fixtures and E2E coverage
status: Done
assignee: []
created_date: '2026-05-15 06:09'
updated_date: '2026-05-15 19:21'
labels:
  - persona
  - webui
  - e2e
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1698'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/pull/1723'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bundled default sprite_frames fixture path is portable and deterministic.
- [x] #2 Uploaded .tldw-persona-vpack fixture path is portable and deterministic.
- [x] #3 E2E coverage proves default-pack setup through activation and BuddyShell rendering.
- [x] #4 E2E coverage proves uploaded-pack import commit through activation and BuddyShell rendering.
- [x] #5 Coverage remains scoped to sprite_frames runtime rendering; non-sprite formats stay preview/diagnostic only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented starter-pack service/UI copy flow, deterministic E2E sprite-frame fixtures, default/upload setup-path Playwright coverage, and BuddyShell refresh on visual-pack activation. Validation: focused Vitest passed (55 tests); setup-path Playwright passed (2 tests); git diff --check passed. Full persona-live file passed 4 visual-pack tests but the existing live-backend WebSocket proof timed out waiting for Disconnect.

Portable upload fixture determinism verified with a Bun import check: two generated buffers were byte-identical (uploaded-visual-pack.tldw-persona-vpack, 2055 bytes).

Addressed PR review findings: guarded starter pack loading against stale async responses, reset starter draft title on persona changes, localized starter labels/units, made uploaded fixture archive references consistent with its manifest, exercised upload import from true empty state, and aligned task status with completed DoD.

Review-fix validation: focused Vitest passed (56 tests); portable fixture determinism passed (1888 byte archive, stable across builds); setup-path Playwright passed (2 tests) from a fresh port with the upload path starting from empty state; git diff --check passed; apps/tldw-frontend lint passed with existing warnings only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Persona Visual starter-pack service/UI support, deterministic sprite_frames E2E fixtures including a portable .tldw-persona-vpack upload builder, default/upload happy-path Playwright coverage, and a BuddyShell activation refresh event so newly activated packs render immediately. Focused Vitest, fixture determinism, setup-path Playwright, and git diff checks passed; the broader persona-live file still has an unrelated live-backend WebSocket proof timeout.
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
