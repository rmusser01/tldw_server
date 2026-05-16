---
id: TASK-402
title: Polish main /chat cockpit visual hierarchy and copy
status: Done
labels:
- chat
- cockpit
- webui
- ux
- visual-polish
- copy
priority: HIGH
references:
- Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
modified_files:
- Docs/superpowers/plans/2026-05-16-chat-cockpit-visual-copy-polish.md
- apps/packages/ui/src/components/Option/Playground/playground-cockpit-rail-styles.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundCompositionPreview.tsx
- apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx
- apps/packages/ui/src/assets/locale/en/playground.json
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundCompositionPreview.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR7 slice from the main /chat cockpit maturity roadmap. Keep scope limited to main WebUI /chat cockpit rails/status/composer-adjacent copy and visual hierarchy. Do not touch extension/sidebar surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Main /chat cockpit rail sections no longer read as generic settings dumps; grouping and hierarchy remain dense but scannable.
- [ ] #2 Terminology for prompt, assistant/persona/character, context, provider route, scoped settings, and MCP tools is consistent across cockpit rails/status/composition surfaces.
- [ ] #3 Visual noise from repeated borders, duplicated labels, and low-value helper copy is reduced while preserving existing controls and keyboard names.
- [ ] #4 Design-system state labels/tokens remain the source of truth, and browser proof covers desktop and mobile cockpit states.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-chat-cockpit-visual-copy-polish.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed PR7 main /chat cockpit visual/copy polish. Shared rail styles now back the context/runtime/composition rails, right rail copy uses Model route, Provider:model settings, and MCP tools terminology, absent composition rows use the design-system Empty state label, and real-server desktop/mobile Playwright proof passes without mocked routes. Verification: focused/broader cockpit Vitest suite 102 tests passed; real-server chat-cockpit Playwright spec 9/9 passed against localhost backend/WebUI; targeted ESLint 0 errors with existing warnings; design-system product-state verifier passed with baseline exceptions; git diff --check passed; Bandit skipped because touched scope is frontend TS/JSON/docs only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
