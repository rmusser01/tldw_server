---
id: TASK-405
title: Implement main chat cockpit collapsible sidechannels
status: Done
labels:
- chat
- webui
- ux
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add discoverable collapsible sidechannel behavior to the main /chat cockpit Context and Runtime rails, scoped strictly to the WebUI main chat page. Reuse existing persisted rail visibility state, add focused tests, and preserve existing chat functionality.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Context and Runtime cockpit sidechannels expose rail-local collapse controls on desktop.
- [x] Collapsed sidechannels expose keyboard-accessible edge restore handles without hiding chat, composer, or status strip.
- [x] Existing header rail visibility controls and persisted visibility state continue to work.
- [x] The change is scoped to the main WebUI /chat cockpit, with no extension sidebar/sidepanel behavior changes.
- [x] Focused unit/accessibility tests and real-server /chat proof are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: add tests first around rail-local collapse and edge restore affordances, implement in PlaygroundCockpitShell using existing rail visibility props/callbacks, then verify with focused Vitest and real-server Playwright against the running server.

Verification:
- RED: focused cockpit tests failed for missing "Collapse context sidechannel" and restore handles before implementation.
- GREEN: `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx --config vitest.config.ts` passed, 19 tests.
- Real-server proof: `TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:18002 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line --grep "uses the running server"` passed, 1 test.
- `git diff --check` passed.
- `bunx prettier --check ...` passed on touched TS files after formatting.
- `bunx eslint ...` reported warning-only existing `any` usage in the real-server spec and ignored the UI package paths from the frontend base path; no errors.
- Bandit skipped: touched code is frontend TypeScript/tests plus Backlog/plan documents.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added rail-local collapse controls and collapsed edge restore handles for the main `/chat` Context and Runtime cockpit sidechannels. The implementation reuses existing rail visibility state, preserves the header hide/show controls, and keeps chat, composer, and status visible while rails are collapsed. Also created TASK-406 as the follow-on post-merge `/chat` cockpit live audit/enhancement tracker.
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
