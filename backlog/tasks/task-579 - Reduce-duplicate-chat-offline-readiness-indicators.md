---
id: TASK-579
title: Reduce duplicate chat offline readiness indicators
status: Done
labels:
- frontend
- chat
- ux
priority: High
modified_files:
- apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.pinned-fallback.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundContextItems.role-play.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The chat cockpit currently surfaces offline/readiness state from multiple widgets at once, producing four or more visible offline indicators. Deduplicate the offline state so users see one primary recovery message and at most one compact status indicator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the chat page is offline, visible offline/readiness messaging is limited to one primary message plus at most one compact status indicator.
- [x] #2 Composer, empty state, and status strip do not repeat the same server-connection failure copy simultaneously.
- [x] #3 Behavior is covered by a regression test.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Root-cause investigation started from the authenticated full-shell verification screenshot where the empty state, composer placeholder/notice, status chips, and connection callouts all reported offline independently.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced duplicate chat offline/readiness messaging. The blank chat empty state now stays neutral while offline, the composer textarea keeps its normal drafting placeholder, the focus-triggered duplicate connect banner was removed, and the session status context chip is suppressed while the connection itself is offline but remains available for connected degraded sessions. Verification: targeted vitest suite passed, combined cockpit/rail/offline suite passed, git diff --check passed, and terminal Playwright confirmed exactly two visible offline indicators with no legacy duplicate connection strings. Bandit skipped because this task touched TypeScript/TSX and Backlog metadata only; the fresh worktree also has no project .venv.
Additional verification: `NODE_OPTIONS=--max-old-space-size=8192 ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json` passed from `apps/packages/ui`. The same command without the heap override hit Node's default heap limit before producing diagnostics.
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
