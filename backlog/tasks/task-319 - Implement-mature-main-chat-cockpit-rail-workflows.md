---
id: TASK-319
title: Implement mature main /chat cockpit rail workflows
status: In Progress
assignee:
  - codex
created_date: '2026-05-13 14:55'
updated_date: '2026-05-13 15:17'
labels:
  - webui
  - chat
  - frontend
  - cockpit
dependencies:
  - TASK-288
  - TASK-291
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-13-main-chat-cockpit-rail-completion-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved remaining main WebUI /chat cockpit rail work in PR #1582. Scope is strictly the main /chat Playground surface, not browser-extension sidepanel/sidebar. The cockpit must preserve existing composer chat functionality while making prompts, context, MCP, model/chat settings, character/persona, run controls, degraded warnings, accessibility, responsive behavior, and real-server verification work from the main chat window rails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Main /chat rails expose first-class prompt selection/status/clear controls using the existing prompt state without clearing unrelated context.
- [ ] #2 Runtime rail exposes direct MCP tool choice and settings workflows using existing chat MCP state, with clear unavailable/degraded states and no MCP Hub lifecycle/policy mutation.
- [ ] #3 Runtime rail presents Model & Chat settings with provider:model scope clarity and preserves scoped settings isolation.
- [ ] #4 Character / Persona is a first-class rail workflow distinct from Scene Director/ActorPopout and supports none/character/persona states without sidepanel/sidebar behavior.
- [ ] #5 Context rail includes prompt context and preserves isolated clear/remove behavior for context classes that affect the next reply.
- [ ] #6 Existing composer/focus-mode chat workflows remain present and working until rail equivalents are verified.
- [ ] #7 Focused component/integration tests cover the rail workflows, shared state paths, keyboard/focus behavior, and disabled/degraded states.
- [ ] #8 Real-server Playwright coverage uses the running server without mocked payloads or page.route for merge-critical proof, tolerates unrelated degraded subsystems with warnings, and restores any mutated setting or uses disposable data.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm current /chat cockpit seams in `Playground.tsx`, `PlaygroundForm.tsx`, `PlaygroundContextRail.tsx`, `PlaygroundRuntimeInspector.tsx`, and `playground-cockpit-actions.ts` against the approved 2026-05-13 spec.
2. TDD Stage A, shared action seams: add failing tests for direct prompt selection/clear, direct MCP settings, and first-class character/persona entry events; implement the smallest event/callback bridge in the main Playground path.
3. TDD Stage B, left rail prompts/context: add prompt rail tests for selected prompt, inline prompt, no prompt, clear isolation, and context inventory; implement prompt summary/actions in `PlaygroundContextRail` using existing prompt state.
4. TDD Stage C, runtime rail MCP and Model & Chat: add tests for direct MCP configure, tool choice, unavailable/degraded states, Model & Chat label/scope summaries; implement with existing MCP/model settings state and no MCP Hub lifecycle controls.
5. TDD Stage D, Character / Persona rail: add tests for none/character/persona/default-bootstrap-safe states and Scene Director separation; implement the first-class rail workflow without importing sidepanel/sidebar behavior.
6. Preserve composer and focus-mode workflows throughout, then run focused Vitest after each stage plus real-server Playwright against `http://127.0.0.1:8000` without mocked payloads or `page.route` before claiming completion.

User approval basis: the user approved the staged design, requested prompt management be added to the left rail, requested the plan/design review, and then said `continue` after the hardened plan was committed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the main /chat cockpit rail workflow slice: prompt selection/clearing in the left rail, shared prompt-select event handling, direct MCP settings opening, runtime rail separation for Model & Chat, Character / Persona, and MCP tool-choice controls. Removed duplicate prompt/MCP labels surfaced by tests so the rails scan cleanly.

Verification: focused Vitest cockpit suite passed: `bunx vitest run src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx` => 6 files / 41 tests passed. `git diff --check` passed.

Verification caveat: `bun run verify:design-system-state` still fails on existing shared-product-state baseline/stale entries outside this /chat slice, especially Chatbooks and other non-chat pages; no unrelated baseline cleanup performed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
