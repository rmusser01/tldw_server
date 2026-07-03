---
id: TASK-12115
title: Make chat focus mode truly fullscreen
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-03 01:43'
labels:
  - webui
  - chat
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update chat focus mode so it hides the app shell/header/sidebar and cockpit rails, leaving only the chat surface plus a clear escape-hatch control to exit focus mode. Add regression coverage before implementation and verify the WebUI rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focus mode hides the app header/sidebar shell chrome in WebUI and shared UI surfaces.
- [x] #2 Focus mode hides chat cockpit rails and the top shortcut/status strip, leaving chat transcript, composer, and one clear exit control.
- [x] #3 Exit focus returns the chat to cockpit mode without losing the current chat state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Plan: Docs/superpowers/plans/2026-07-03-chat-focus-fullscreen.md

Verification: targeted Vitest passed: bun run test:run ../packages/ui/src/components/Layouts/__tests__/Layout.shell-overrides.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx (22 tests passed).

Verification: browser check on http://127.0.0.1:18001/chat confirmed focus mode hides header/sidebar, chat sidebar, context/runtime rails, and top focus/shortcut controls while preserving transcript, composer, and Exit focus. Desktop screenshot: /tmp/tldw-chat-focus-fullscreen.png. Mobile screenshot: /tmp/tldw-chat-focus-fullscreen-mobile.png.

Known skip/failure: bun run typecheck still fails on existing unrelated files (AudioStudio TimelineEditor referrerPolicy, ScheduledTasks response typing, Skills Checkbox props, scheduled-tasks service params, mcp-hub path typing, voice-cloning ArrayBuffer, knowledge QA fixture assertions, flashcards E2E never callable). The touched Playground type error found during the first typecheck was fixed and absent in the rerun.

Bandit: skipped because this change only touches frontend TypeScript/TSX tests and Markdown; no Python touched.

Branch verification: after switching to codex/chat-focus-fullscreen from origin/dev, targeted Vitest was rerun and passed (22 tests). Broad typecheck still fails only on the same unrelated baseline files listed above.

PR: https://github.com/rmusser01/tldw_server/pull/2578
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a shared OptionLayout shell override hook and wired chat focus mode to hide app shell chrome, cockpit rails, top shortcut/status chrome, artifacts rails, and comparison breadcrumbs. Focus mode now leaves the chat transcript/composer plus a fixed Exit focus control, and the control returns the user to cockpit mode. Added regression coverage for the shared shell override hook and focus-mode UX.
<!-- SECTION:FINAL_SUMMARY:END -->

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
