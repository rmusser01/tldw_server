---
id: TASK-422
title: Verify Buddy builder new-user flow with CDP screenshots
status: Done
labels:
- persona
- buddy
- frontend
- qa
- playwright
references:
- https://github.com/rmusser01/tldw_server/pull/1821
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run CDP-backed browser screenshots against the Persona Visuals guided Buddy builder flow for a new user, covering Basic bundled defaults, draft copy/review/state configuration, activation, and Codex/Petdex import preview. Use local API mocks only for server data so the real WebUI components render.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CDP screenshots are captured for desktop and mobile new-user catalog views.
- [x] #2 Copy-as-draft, review diagnostics, movement state configuration, and activation are exercised in the browser.
- [x] #3 Codex/Petdex zip import preview path is exercised in the browser.
- [x] #4 Console, page, and request failures are checked and documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-17: Fixed first-run Buddy builder visual QA issues found by the CDP flow. BuddySourcePicker source cards now render as native multi-line card buttons instead of AntD Button controls, preventing label/tag/description overlap on desktop and mobile. The default Buddy dock position now starts lower/right on desktop, and informational visual diagnostics are hidden while the dock is collapsed so the no-active-pack message does not cover builder controls; warning/error diagnostics still surface.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Captured CDP-backed screenshots for the new-user Buddy builder flow at /private/tmp/tldw-buddy-builder-cdp-new-user-flow. Verified desktop and mobile Basic catalog views, copy-as-draft review/readiness state, movement state configuration, activation, and Codex/Petdex zip import preview. Focused Vitest coverage passed: 3 files, 38 tests. CDP harness diagnostics were clean: no console errors, page errors, request failures, bad responses, or unhandled API mocks. git diff --check passed. Bandit is not applicable because this task touched only frontend TypeScript/TSX and Backlog markdown.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
