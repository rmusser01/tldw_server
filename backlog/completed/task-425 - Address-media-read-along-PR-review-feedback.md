---
id: TASK-425
title: Address media read-along PR review feedback
status: Done
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
- [x] #1 Live PR #1835 review comments are verified against GitHub and addressed.
- [x] #2 Regression coverage is added or updated for review-fix behavior.
- [x] #3 Review threads are resolved before merge.
- [x] #4 Local verification and any relevant skips are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Address the four live PR #1835 review findings: avoid blob-heavy cache eviction reads, keep cache hits valid when lastUsedAt updates fail, use targeted active-segment DOM lookup, and register generated-audio terminal listeners as one-shot handlers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Second live-thread pass found Qodo #1 and #2: resume null-audio crash and playback-only speed in cache signatures. Fixed resume to no-op when no generated audio is available; removed playbackSpeed fallback from generated-audio cache signatures while retaining synthesis-specific cacheSettings.speed. Regression checks: targeted session/cache tests passed 42 tests; broader read-along suite passed 109 tests; route parity passed 6 tests; git diff --check passed. Bandit not applicable because only frontend TypeScript/backlog files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved all actionable PR #1835 review feedback for media read-along TTS. Fixed cache eviction to avoid loading audio blobs, active segment lookup to avoid full DOM scans, generated-audio terminal listeners to use one-shot handlers, cache reads to tolerate best-effort lastUsedAt update failures, resume to no-op safely without generated audio, and cache signatures to avoid playback-only speed fragmentation. Verification recorded: targeted session/cache tests passed 42 tests, broader read-along suite passed 109 tests, route parity passed 6 tests, and git diff --check passed. Bandit was not applicable because only frontend TypeScript and Backlog metadata changed. GitHub review threads were resolved before merge; PR #1835 merged into dev at 2026-05-19T00:03:44Z with merge commit 57aa82909b3d09ddf0133947bbe32cb60a0fb0a5.
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
