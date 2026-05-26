---
id: TASK-505
title: Fix WatchlistsPlaygroundPage JSX syntax error
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-25 21:57'
labels:
  - watchlists
  - typescript
  - bugfix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`bunx tsc --noEmit --pretty false` fails because `WatchlistsPlaygroundPage.tsx` contains a stale duplicated Ant Design Alert JSX block before the canonical DesignSystemAlert block. Remove the malformed duplicate without changing Watchlists behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm TypeScript parse failure reproduces.
2. Remove only the stale duplicated Ant Design Alert orientation/teach-point block around the malformed JSX.
3. Re-run TypeScript check and focused syntax verification.
4. Record results in this Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: WatchlistsPlaygroundPage had a stale duplicated Ant Design Alert orientation/teach-point block before the canonical DesignSystemAlert block. The duplicate introduced an impossible JSX close at the former line 2355 and left a dangling fragment close.

Fix: removed the malformed duplicate and restored the watchlistViewsAvailable fragment/WatchlistsHealthBar wrapper around the remaining repeat-actions/orientation/teach-point block, matching the canonical structure in origin/dev while preserving current local content below it.

Verification:
- Red check: bunx tsc --noEmit --pretty false reproduced parser errors in WatchlistsPlaygroundPage.tsx before the fix.
- Green check: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false no longer reports WatchlistsPlaygroundPage.tsx syntax/parser errors. The command now reaches project-wide type checking and fails on unrelated existing type errors across tests/routes/services.
- git diff --check passes.

Bandit: not run; touched frontend TSX and Backlog task only, no Python/backend code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the WatchlistsPlaygroundPage JSX syntax break by removing the stale duplicated Ant Design Alert block and restoring the missing watchlistViewsAvailable fragment wrapper around the canonical DesignSystemAlert UI. The original Watchlists parser errors are gone; full TypeScript still fails on unrelated project-wide type errors outside this file.
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
