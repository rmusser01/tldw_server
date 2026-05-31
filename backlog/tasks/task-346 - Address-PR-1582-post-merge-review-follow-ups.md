---
id: TASK-346
title: Address PR 1582 post-merge review follow-ups
status: Done
assignee: []
created_date: '2026-05-14 20:02'
updated_date: '2026-05-15 01:12'
labels:
  - webui
  - chat
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582#discussion_r3243931786'
  - 'https://github.com/rmusser01/tldw_server/pull/1582#discussion_r3243931793'
  - 'https://github.com/rmusser01/tldw_server/pull/1582#discussion_r3243931800'
  - 'https://github.com/rmusser01/tldw_server/pull/1702'
  - 'https://github.com/rmusser01/tldw_server/pull/1702#discussion_r3245176513'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up slice for the merged chat cockpit PR #1582. Address only the three still-unresolved post-merge Qodo review findings: preserve diagnostic context for system-prompt lookup failures, preserve diagnostic context for real-server E2E JSON parse failures, and harden focus-return selector handling/event ingestion against malformed selectors. Base the work on current dev and open a new PR with focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt lookup failures set the UI unavailable state while retaining useful diagnostic logging/context.
- [x] #2 The real-server E2E apiPost helper reports JSON parse failures with response context instead of silently returning null.
- [x] #3 Focus-return helpers and CustomEvent returnFocusSelector ingestion tolerate malformed or empty selectors without throwing.
- [x] #4 Focused regression tests cover the new reliability behavior.
- [x] #5 A new PR is opened against dev from a clean follow-up branch.
- [x] #6 apiPost preserves successful empty 204 response behavior while retaining non-JSON diagnostics.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented focused PR #1582 review follow-ups: prompt lookup failures now log prompt id/error before unavailable UI state, apiPost non-JSON responses throw with status/body context, and focus-return selectors are normalized/guarded at utility plus PromptSelect/PlaygroundForm event ingestion.

Verification: focused Vitest for focus-return, Playground cockpit controls, PromptSelect modal; Playwright helper regression with TLDW_WEB_AUTOSTART=false; PlaygroundForm signals guard; git diff --check; frontend lint command completed with existing warnings only.

Opened follow-up PR #1702 against dev. Bandit not run: touched implementation/test files are TypeScript/TSX/Markdown only, with no Python touched scope.

PR #1702 review follow-up: Gemini identified a valid regression risk where parseApiJsonResponse would throw for successful 204 No Content responses. Reopening TASK-346 for a narrow test-first fix.

PR #1702 review follow-up implemented test-first: added a 204 No Content apiPost regression test, then returned null for status 204 before JSON parsing and consolidated response text trimming in diagnostics.

Verification: TLDW_WEB_AUTOSTART=false TLDW_E2E_API_KEY=dummy TLDW_E2E_SERVER_URL=http://127.0.0.1:1 bun run e2e:pw e2e/workflows/chat-cockpit.real-server.spec.ts -g 'API POST responses' --reporter=line; git diff --check; bun run lint e2e/workflows/chat-cockpit.real-server.spec.ts exited 0 with existing warnings only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all current PR #1702 review comments. The new Gemini 204 No Content finding was verified with a failing regression test, fixed by preserving null bodies for 204 responses before JSON parsing, and rechecked with the API POST helper tests. The earlier PR #1582 review follow-ups remain covered: prompt lookup diagnostics, non-JSON response diagnostics, and focus selector hardening.
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
