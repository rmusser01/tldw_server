---
id: TASK-578
title: Fix chat cockpit collapsed rail edge restore affordances
status: Done
labels:
- chat
- frontend
- ux
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the fresh-worktree /chat cockpit rail collapse fix: when either in-page cockpit rail is collapsed, show a same-side edge-mounted restore tab that clearly identifies Context or Runtime, keep the opposite rail expanded, and let the chat main area widen without vertical displacement. Scope is the actual Playground cockpit rails, not the global sidebar or sidepanel route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collapsing the left Context rail removes only that rail, keeps the Runtime rail visible, and shows a left edge-mounted restore tab identifying Context.
- [x] #2 Collapsing the right Runtime rail removes only that rail, keeps the Context rail visible, and shows a right edge-mounted restore tab identifying Runtime.
- [x] #3 If both rails are collapsed, both edge-mounted restore tabs remain visible and independently restore their rail.
- [x] #4 The cockpit main area occupies the freed grid column width without adding an in-flow collapsed banner or pushing the chat/composer downward.
- [x] #5 Focused tests cover the actual PlaygroundCockpitShell behavior and labels.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated the actual Playground cockpit shell rails, not the app sidebar or sidepanel route.
- Added full-width constraints to the Playground root and cockpit body so collapsed rail states do not shrink-wrap the chat surface.
- Replaced state-specific Tailwind grid-template classes with an explicit `--cockpit-grid-columns` style value for the four cockpit rail states.
- Converted collapsed rail restore controls into same-side vertical edge tabs labelled `Context rail` and `Runtime rail`.
- Fixed the tooltip button wrapper so absolute-positioned restore tabs are not also forced to `relative`.
- Added focused regression coverage for left-collapsed, right-collapsed, and both-collapsed restore behavior.
- Follow-up from screenshot review: both-collapsed mode still inherited the focus-mode max-width, leaving dead horizontal space despite both restore tabs being visible.
- Removed the inner chat/transcript/composer `max-w-[64rem]` constraint when any cockpit rail is collapsed, and force the composer form into its wide layout for those states.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the /chat cockpit rail collapse behavior in a fresh worktree. The remaining rail now stays expanded, the chat surface and inner composer/transcript stack widen into the freed grid column without vertical displacement, and collapsed rails are represented by same-side edge-mounted restore tabs. Rendered Playwright verification captured updated left-collapsed, right-collapsed, and both-collapsed screenshots under `/private/tmp/tldw-chat-rails-fresh-20260531`.
Additional verification after the offline-readiness cleanup: the combined cockpit/rail/offline Vitest suite passed, `git diff --check` passed, and `NODE_OPTIONS=--max-old-space-size=8192 ../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json` passed from `apps/packages/ui`. The same TypeScript command without the heap override hit Node's default heap limit before producing diagnostics.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented

Verification recorded:
- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx --reporter=verbose` passed: 1 file, 3 tests.
- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx --reporter=verbose` passed: 2 files, 34 tests.
- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx --reporter=verbose` passed: 3 files, 37 tests.
- `git diff --check` passed.
- Terminal-driven Playwright check against `http://127.0.0.1:18083/chat` passed for context-collapsed, runtime-collapsed, and both-collapsed geometry. It confirmed no collapsed in-flow summary, main `y=91` in each collapsed state, left restore tab at `x=32`, right restore tab at `x=1372`, and widened main widths of `1076px` and `1096px` for one-rail-collapsed states.
- Follow-up terminal-driven Playwright check also confirmed both-collapsed main width `1376px` on a `1376px` shell, no inner `max-w-[64rem]` constraints, and composer wide mode enabled.
- The earlier screenshots lacked the app header/sidebar because the route was loaded in an unauthenticated state and `_app.tsx` sets `hideShellNav` when auth is unresolved or missing. A follow-up authenticated Playwright run confirmed the full app chrome is present: app header `x=48 y=0 h=51`, collapsed app chat sidebar `x=0 w=48`, cockpit shell `x=48 y=51 w=1392`, and both rail restore tabs visible.

Known skips:
- Bandit not run because the fresh worktree has no `.venv` and the touched implementation is TypeScript/React only.
- Rendered console output still includes expected local setup noise for missing API key/CORS to the backend on `127.0.0.1:8000`; it is unrelated to cockpit rail layout.
<!-- DOD:END -->
