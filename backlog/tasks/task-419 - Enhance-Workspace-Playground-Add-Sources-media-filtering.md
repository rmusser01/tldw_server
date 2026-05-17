---
id: TASK-419
title: Enhance Workspace Playground Add Sources media filtering
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/1819
- https://github.com/rmusser01/tldw_server/pull/1819#discussion_r3255166540
- https://github.com/rmusser01/tldw_server/pull/1819#discussion_r3255166545
- https://github.com/rmusser01/tldw_server/pull/1819#discussion_r3255166549
- https://github.com/rmusser01/tldw_server/pull/1819#discussion_r3255166551
- https://github.com/rmusser01/tldw_server/pull/1819#discussion_r3255166554
documentation:
- Docs/superpowers/plans/2026-05-17-workspace-playground-media-filtering-pr1819.md
modified_files:
- Docs/superpowers/plans/2026-05-17-workspace-playground-media-filtering-pr1819.md
- apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx
- apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx
- apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up for PR #1819. Add My Media search/filter/sort controls in Workspace Playground Add Sources, using existing media search APIs for query, keyword, media type, and sort filtering. Also address current PR review threads around type safety, dropdown trigger handling, router navigation, and media fetch error logging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] My Media supports query, content type, keyword, and sort filtering.
- [x] Filtered My Media requests use the existing media search API and default empty-filter requests preserve listMedia behavior.
- [x] Clear filters resets controls and returns to the default media listing.
- [x] Current PR review threads are addressed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added My Media search/content-type/keyword/sort controls in Workspace Playground Add Sources. Active filters now call searchMedia with title/content fields, media_types, must_have keyword filters, sort_by, and pagination; empty filters continue to call listMedia with include_keywords. Media load failures are logged before the user-facing error. Addressed PR #1819 review threads by typing ChatModelSelectorDropdown menu items, removing the redundant model selector trigger click path, switching ChatPane navigation to useNavigate, and typing composerModels from fetchChatModels.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification: AddSourceModal.stage2.intake.test.tsx passed 15 tests; ChatPane.stage2.test.tsx passed 17 tests; focused Workspace Playground suite passed 7 files / 90 tests; git diff --check passed. Package-wide tsc still fails on unrelated baseline errors, but filtered output for touched files is clean. Bandit skipped because this change only touches TypeScript/React and docs/backlog metadata.
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
