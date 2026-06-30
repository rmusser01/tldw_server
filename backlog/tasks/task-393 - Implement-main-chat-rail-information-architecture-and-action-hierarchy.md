---
id: TASK-393
title: Implement main /chat rail information architecture and action hierarchy
status: Done
assignee: []
created_date: '2026-05-16 00:30'
updated_date: '2026-05-16 01:35'
labels:
  - webui
  - chat
  - ux
  - frontend
dependencies:
  - TASK-391
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-cockpit-rail-ia-action-hierarchy-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 2 of the post-merge main /chat cockpit maturity roadmap: reorganize the main WebUI /chat cockpit rails into predictable work surfaces without changing sidepanel/sidebar behavior. Preserve existing controls and shared handlers while improving first-time comprehension and returning-user scan efficiency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main /chat left rail groups context stack, prompt management, search/RAG sources, files/media, and session persistence with clear headings and compact states.
- [x] #2 Main /chat right rail groups runtime state, model route/settings, assistant/persona, tools/MCP, and recovery controls with clear action hierarchy.
- [x] #3 Existing rail controls, shared handlers, keyboard-accessible names, focus behavior, and focus-mode behavior are preserved.
- [x] #4 First-time users can identify where to change prompt, persona/character, model, context, and tools without opening unrelated surfaces.
- [x] #5 Focused Vitest coverage and real-server /chat Playwright proof are updated for the reorganized rail IA without mocked backend routes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the PR 2 implementation plan for rail information architecture and action hierarchy. Scope is the main WebUI /chat cockpit only. The plan preserves existing shared state/handlers, keeps focus mode and mobile rail tabs intact, excludes sidepanel/sidebar and model selector redesign work, and requires TDD plus real-server Playwright proof.

Implemented PR 2 rail IA grouping for the main /chat cockpit only. Left rail now orders Composition, Context stack, Prompt, Search & sources, and Session. Right rail now orders Runtime, Model & Chat, Assistant, Tools, and Run controls. Preserved existing shared callbacks, accessible names, focus restoration, focus mode, and mobile rail tab behavior.

Verification: bunx vitest run src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts --reporter=verbose => 5 files, 54 tests passed.
Verification: TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=$KEY bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line => 8 passed against the real running server with no backend route mocks.
Verification: bun run verify:design-system-state => passed with existing allowed legacy product-state exceptions.
Verification: git diff --check => passed.
Bandit: skipped because no Python files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reorganized the main /chat cockpit rails into the PR 2 cockpit hierarchy while preserving the existing controls and shared handlers. Added focused left/right rail tests, updated integrated Playground/responsive coverage, and adjusted the real-server Playwright proof for the new Prompt/Tools rail labels and live-state variants.
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
