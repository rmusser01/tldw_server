---
id: TASK-425
title: Address media read-along PR review feedback
status: In Progress
labels:
- pr-review
- media
- read-along
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/1835
modified_files:
- apps/packages/ui/src/components/Media/ContentViewer.tsx
- apps/packages/ui/src/components/Media/read-along/media-read-along-cache.ts
- apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache.test.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx
- apps/packages/ui/src/db/dexie/schema.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluate and address live GitHub PR #1835 review comments/check failures for the media viewer read-along TTS branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Address the four live PR #1835 review findings: avoid blob-heavy cache eviction reads, keep cache hits valid when lastUsedAt updates fail, use targeted active-segment DOM lookup, and register generated-audio terminal listeners as one-shot handlers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified live PR surface via gh pr view, gh pr checks, and GraphQL review threads. CodeRabbit skipped; actionable comments came from Gemini/Qodo. Local checks: targeted review-fix tests passed 45 tests; broader read-along suite passed 107 tests; route parity passed 6 tests; git diff --check passed. Bandit not applicable because only frontend TypeScript/backlog files changed.
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
