---
id: TASK-13126
title: Make chat focus mode truly fullscreen
status: Done
assignee: []
created_date: ''
updated_date: 2026-08-26 15:51
labels:
- webui
- chat
- ux
dependencies: []
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2578
- TASK-13125
- TASK-12115
- legacy:TASK-12115 (chat focus mode)
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-03-chat-focus-fullscreen.md

Verification: targeted Vitest passed: bun run test:run ../packages/ui/src/components/Layouts/__tests__/Layout.shell-overrides.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx (23 tests passed after adding the shell cleanup regression).

Verification: browser check on http://127.0.0.1:18001/chat confirmed focus mode hides header/sidebar, chat sidebar, context/runtime rails, and top focus/shortcut controls while preserving transcript, composer, and Exit focus. Desktop screenshot: /tmp/tldw-chat-focus-fullscreen.png. Mobile screenshot: /tmp/tldw-chat-focus-fullscreen-mobile.png.

Verification: git diff --check passed.

Typecheck baseline: bun run typecheck was rerun and remains blocked by existing unrelated latest-dev errors outside this PR scope (AudioStudio TimelineEditor referrerPolicy, ScheduledTasks response typing, Skills Checkbox props, scheduled-tasks service params, mcp-hub path typing, voice-cloning ArrayBuffer, knowledge QA fixture assertions, flashcards E2E never callable). No touched files are present in the current typecheck error output.

Bandit: skipped because this change only touches frontend TypeScript/TSX tests and Markdown; no Python touched.

PR: https://github.com/rmusser01/tldw_server/pull/2578

Review follow-up: rebased on latest dev and addressed PR review comments for shell cleanup, focus accessibility/test coverage, and task marker structure. The Qodo broad-typecheck comment was evaluated against latest dev; the remaining typecheck failures are pre-existing, unrelated baseline errors and are not changed by this PR.
Identity normalization (TASK-13125): this completed record was originally created and merged as TASK-12115 in PR #2578. Its canonical ID is now TASK-13126. TASK-12115 now deterministically identifies the standalone HTML presentation rollout; no implementation or completion evidence was removed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a shared OptionLayout shell override hook and wired chat focus mode to hide app shell chrome, cockpit rails, top shortcut/status chrome, artifacts rails, and comparison breadcrumbs. Focus mode now leaves the chat transcript/composer plus a fixed Exit focus control, and the control returns the user to cockpit mode. Added regression coverage for the shared shell override hook and focus-mode UX. Review follow-up captures the exact shell setter used for cleanup, keeps focus accessibility state dynamic, verifies focus exit clears shell overrides, and fixes the Backlog final-summary marker structure.
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
