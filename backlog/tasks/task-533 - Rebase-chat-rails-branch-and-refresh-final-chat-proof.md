---
id: TASK-533
title: Rebase chat rails branch and refresh final chat proof
status: Done
labels:
- chat
- ux
- e2e
- rebase
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the chat rails UX branch onto current origin/dev and refresh the final /chat proof paths that remain after remediation: real-server chat green path where feasible, Web search/model-scope/assistant-clear contracts, and directly connected extension sidepanel smoke evidence. Keep scope to /chat and direct extension handoff only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Branch is rebased onto current `origin/dev` with `/chat` rail changes still present.
- [x] Real-server backend health, configured providers, cockpit rails, model selection, first send, and assistant clear/plain return are verified against the proper `/chat` page.
- [x] Direct extension handoff contracts remain covered for route-only `/chat` launch.
- [x] Rebaseline review document records current evidence and remaining non-goals.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Post-rebase proof refreshed. Branch rebased onto origin/dev and live backend proof used 127.0.0.1:18001 with mock OpenAI on 18088. Found and fixed a real assistant-clear race: clearing a tracked character/persona now waits for selected-assistant and selected-character storage clears before detaching the server chat, clears server assistant metadata, and drops serverChatId so the next persisted session is plain WebUI chat. Hardened the real-server persona selector proof to reuse the retrying selector helper after transient catalog load errors. Verification so far: focused cockpit-control Vitest passed 19/19; focused UI suite passed 92/92 plus moved-path selector/handoff tests passed 6/6; combined real-server Playwright proof passed 4/4 for running-server cockpit/focus, character clear/plain return, persona clear, and model provider confidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased `codex/chat-rails-ux-rebaseline` onto `origin/dev` and refreshed the final focused `/chat` proof against a live backend on `127.0.0.1:18001` with mock OpenAI on `18088`.

During live proof, found a real tracked-assistant clear race: visible assistant state could clear before the tracked server chat detached. Fixed `clearAssistantFromCockpit` so it awaits assistant/character storage clears, clears server assistant metadata, drops `serverChatId`, clears persisted tracked-session state, and returns to standard workflow before the next persisted plain WebUI chat.

Updated the cockpit unit guard and real-server E2E proof. The character clear test now captures the plain WebUI chat creation from the clear action, and the persona proof now uses the retrying selector helper after transient catalog-load errors.

Verification:
- `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx --reporter=verbose` passed: 1 file, 19 tests.
- Focused UI regression suite passed: 8 files, 92 tests.
- Moved-path selector/handoff tests passed: 2 files, 6 tests.
- Live Playwright proof passed: 4 tests for running-server cockpit/focus, tracked character clear/plain return, persona select/clear, and configured model provider first send.
- `git diff --check` passed.

Bandit was not run because this slice touched TypeScript/TSX, Playwright tests, Markdown docs, and Backlog task metadata only; no Python code was changed.
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
