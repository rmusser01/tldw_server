---
id: TASK-520
title: Remove standalone chat character controls rail
status: Done
labels:
- chat
- frontend
- UX
references:
- Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the standalone far-right Character controls rail from the desktop /chat cockpit. Keep the restored core cockpit rails intact: Context on the left, Runtime on the right.

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The desktop /chat cockpit no longer renders the standalone far-right Character controls rail.
- [x] #2 Context and Runtime cockpit rails remain intact and covered by existing rail regression tests.
- [x] #3 Focused tests prove the character rail is not part of the main desktop cockpit layout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Removed the `CharacterControlRail` import, visibility subscription, and standalone desktop render block from `Playground.tsx`.
- Kept `PlaygroundContextRail` and `PlaygroundRuntimeInspector` mounted in the main cockpit shell.
- Updated `Playground.cockpit-regression.guard.test.ts` so it treats Context and Runtime as the core rails and asserts `CharacterControlRail` is not wired into `Playground.tsx`.
- Red test: `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts` failed before the implementation because `Playground.tsx` still contained `CharacterControlRail`.
- Green focused suite: `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.tts-clips-lazy-mount.test.ts` passed: 9 files, 105 tests.
- Browser verification: captured `/private/tmp/tldw-chat-no-character-rail.png` at `http://127.0.0.1:18014/chat`; cockpit shell and composer were visible, `Character controls` text was absent, and document/body width matched the 1440px viewport.
- `git diff --check` passed.
- Bandit skipped: touched files are frontend TypeScript/test Markdown only, with no Python execution path changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The desktop /chat cockpit now renders only the intended core rails: Context and Runtime. The accidental standalone Character controls rail has been removed from the main `Playground` layout and is guarded by a focused regression test.

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
