---
id: TASK-600
title: Reduce frontend chat cockpit snapshot TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:14'
labels:
  - typescript
  - frontend
  - tsc-baseline
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the frontend standalone tsc diagnostic in the chat cockpit real-server spec by making persisted playground session parsing type-safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend standalone tsc no longer reports the chat-cockpit.real-server.spec.ts session snapshot diagnostic targeted by this task.
- [x] #2 The change preserves support for both legacy plain snapshot and wrapped { state } localStorage shapes.
- [x] #3 Bandit applicability is documented for the TypeScript-only test change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification recorded for this tsc slice:
- RED: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend reported 7 diagnostics, including the chat-cockpit.real-server.spec.ts session snapshot union diagnostic.
- GREEN: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/tldw-frontend now reports 6 remaining diagnostics, all in unrelated agent-tasks/admin-llamacpp clusters.
- The parser now preserves both persisted shapes by normalizing either { state: snapshot } or the legacy plain snapshot from unknown JSON.
- No focused runtime test was run because this spec is a real-server Playwright workflow gated by TLDW_E2E_API_KEY/TLDW_API_KEY/SINGLE_USER_API_KEY; compiler coverage is the intended check for this type-only cleanup.
- git diff --check exits 0.
- Bandit not applicable: touched file is a TypeScript Playwright spec only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the frontend standalone tsc baseline by removing the chat cockpit persisted-session snapshot diagnostic. The real-server spec now parses localStorage JSON through a concrete unknown-to-snapshot normalizer that supports both wrapped and legacy plain snapshot shapes.
<!-- SECTION:FINAL_SUMMARY:END -->

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
