---
id: TASK-391
title: Implement main /chat Context Stack and composition preview
status: Done
assignee: []
created_date: '2026-05-15 21:54'
updated_date: '2026-05-15 23:56'
labels:
  - webui
  - chat
  - ux
  - frontend
dependencies:
  - TASK-390
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
  - Docs/superpowers/plans/2026-05-15-chat-cockpit-composition-preview-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 1 of the post-merge main /chat cockpit maturity roadmap: add a Context Stack plus Prompt/Persona/Model Composition Preview for the main WebUI /chat page only. The work must preserve existing chat functionality, avoid sidepanel/sidebar scope, reuse existing /chat state and handlers, and keep provider:model settings identity intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main /chat cockpit shows an inspectable composition preview covering prompt, character/persona, provider:model route, settings scope, context stack, and MCP/tool policy.
- [x] #2 Context Stack clearly distinguishes active, disabled, degraded, unavailable, empty, and loading states without creating parallel state.
- [x] #3 Prompt/persona/character/model/context/tool updates from existing controls are reflected in the preview through shared /chat state.
- [x] #4 Focused Vitest coverage is added or updated for the summary contract, preview component, context rail wiring, cockpit maturity, and responsive/focus behavior.
- [x] #5 Real-server Playwright coverage is updated to prove prompt, persona/character, model settings scope, MCP state distinction, and screenshots without mocked server data or sidepanel/sidebar routes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation from Docs/superpowers/plans/2026-05-15-chat-cockpit-composition-preview-plan.md. Impeccable context loader found no PRODUCT.md or DESIGN.md in the worktree, so implementation will follow existing /chat components and WebUI design-system conventions as the source of truth.

Implemented the first-slice main /chat Context Stack and Composition Preview using the existing Playground coordinator state. Verification: focused Vitest suite passed 6 files / 40 tests; design-system product-state guard passed with baseline legacy exceptions only; full real-server Playwright spec `e2e/workflows/chat-cockpit.real-server.spec.ts` passed 8/8 against `http://127.0.0.1:8000` with the real API key and no route mocking; `git diff --check` passed. Bandit has no applicable Python touched scope for this TS/TSX/E2E slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a first-slice main /chat composition preview and context stack summary using existing prompt, assistant/persona/character, provider:model route, scoped settings, context source, and MCP/tool state. Wired it into the left cockpit rail without introducing parallel state, added focused unit/component/guard coverage, and expanded the real-server Playwright cockpit proof for prompt, model settings restore, MCP state, mobile, character, persona, and a working conversation screenshot.
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
