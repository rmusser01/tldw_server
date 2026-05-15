---
id: TASK-369
title: Recertify issue 1646 main chat cockpit strict gaps
status: Done
assignee: []
created_date: '2026-05-15 03:23'
updated_date: '2026-05-15 06:52'
labels:
  - webui
  - chat
  - frontend
  - cockpit
  - recertification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1646'
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
documentation:
  - Docs/superpowers/plans/2026-05-15-chat-cockpit-1646-recertification-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recertification PR against dev scoped only to issue #1646 strict gaps for the main /chat cockpit page. This task tracks evidence-first closure of P0/P1/P2 items without mocked route data or sidepanel/sidebar drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 P0 strict gaps are covered with real-server proof or focused tests: real prompt, real persona, model setting persist/restore, MCP populated/unavailable distinction, and assistant transition matrix.
- [x] #2 Mobile and focus proof is expanded for the main /chat cockpit workflows.
- [x] #3 Visual QA screenshots are captured from the real WebUI flow and referenced in the PR/issue closeout.
- [x] #4 Issue #1646 is updated checkbox-by-checkbox based on evidence, and closed only after all P0/P1/P2 checklist items are proven.
- [x] #5 No browser extension sidepanel/sidebar or unrelated page changes are included.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the staged recertification plan in Docs/superpowers/plans/2026-05-15-chat-cockpit-1646-recertification-plan.md. Gate order: P0 proof gaps first, then mobile/focus proof, then visual QA and issue #1646 checkbox closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented recertification fixes for issue #1646 strict gaps: real-server prompt proof, real persona/character runtime rail proof, provider:model model setting scope handoff and persist/restore proof, MCP populated/unavailable state counts, assistant transition matrix coverage, mobile/focus screenshots, and no-route-stubbing guard.

Verification recorded:
- Focused Vitest suite passed: 68 tests.
- Real-server Playwright spec passed: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` 8/8 against `http://127.0.0.1:8000` using the configured `.env` key and no backend route mocking.
- Design-system product-state verifier passed from `apps/packages/ui`; output contains only existing allowed baseline exceptions.
- `git diff --check` passed.
- Bandit not applicable: no backend/Python files touched.
- Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1721
- Issue #1646 was updated checkbox-by-checkbox and closed with recertification evidence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recertified #1646 strict gaps for the main `/chat` cockpit page with focused frontend fixes and real-server proof. The branch adds scoped model-settings handoff through the cockpit model settings event, MCP state-count summaries, prompt persistence clear behavior, selected-model submit sync, assistant transition coverage, model metadata abort retry coverage, and full real-server visual/e2e proof for prompt/model/MCP/persona/character/mobile/focus states.
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
