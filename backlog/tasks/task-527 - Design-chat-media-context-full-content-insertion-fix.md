---
id: TASK-527
title: Design chat media context full-content insertion fix
status: Done
labels:
- chat
- media
- webui
- spec
priority: medium
documentation:
- Docs/superpowers/specs/2026-06-06-chat-media-context-full-content-design.md
- Docs/superpowers/plans/2026-06-06-chat-media-context-full-content-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-06-chat-media-context-full-content-design.md
- Docs/superpowers/plans/2026-06-06-chat-media-context-full-content-implementation-plan.md
- apps/packages/ui/src/utils/rag-format.ts
- apps/packages/ui/src/components/Knowledge/hooks/useKnowledgeSearch.ts
- apps/packages/ui/src/components/Knowledge/hooks/useFileSearch.ts
- apps/packages/ui/src/components/Knowledge/KnowledgePanel.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and specify a fix for /chat media context actions that can insert or pin title-only media search results instead of full media content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the frontend media-context fix. Media-library search results are origin-marked, pinned conversion preserves only contextOrigin, and full media content is fetched only for pinned results with mediaId plus contextOrigin === "media-library". Knowledge Search Insert/Ask/Pin/copy, File Search Attach/copy, and KnowledgePanel direct/confirmed/preview Ask now resolve full media text before formatting. Pin handling re-reads current pinned state after async resolution and guards pending media pins against Clear All races. Test scaffolding was also updated so the targeted component suites can run in a hydrated dependency environment.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed /chat media-library context actions so media items use full content instead of title-only fallback snippets when full media detail content is available. Verified submit/raw-preview paths already consume ragPinnedResults through formatPinnedResults when file retrieval is off, so no submit-path code change was needed. Verification: targeted Vitest suite passed after temporarily relinking the tracked antd symlink to the installed local package path, then restoring it before commit: Knowledge hooks, File hooks, KnowledgePanel QA preview, Playground pinned fallback, and raw preview MCP tools; 5 test files, 26 tests. The branch leaves the tracked antd symlink unchanged. Bandit is not applicable because the implementation touched TypeScript/TSX/docs only.
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
