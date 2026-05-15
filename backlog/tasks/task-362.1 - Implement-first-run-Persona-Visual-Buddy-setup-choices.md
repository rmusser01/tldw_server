---
id: TASK-362.1
title: Implement first-run Persona Visual Buddy setup choices
status: Done
assignee: []
created_date: '2026-05-15 03:44'
updated_date: '2026-05-15 19:40'
labels:
  - persona
  - buddy
  - visuals
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1695'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-persona-visual-buddy-setup-choices-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-persona-visual-buddy-setup-choices-implementation-plan.md
parent_task_id: TASK-362
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved first-run Visual Buddy setup choices for Persona Garden. The implementation should add a reusable setup-choice card, wire the Visuals tab to starter catalog copy/import/blank flows without automatic activation, and add a setup-wizard visual detour so users can open VisualPackEditor while assistant setup gating is active. Keep this slice frontend-focused; do not add backend starter-catalog routes, Live2D runtime support, MCP provider execution, VN/CYOA behavior, E2E fixture work, or import-polish beyond routing into existing controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reusable setup-choice card renders Use default, Import pack, and Start blank in Visuals, with compact optional wizard mode.
- [x] #2 VisualPackEditor shows setup choices only when the selected persona has no active visual pack and preserves the existing advanced reuse panel.
- [x] #3 Use default lists/copies bundled starter packs through existing frontend service patterns, creates an inactive draft, selects the returned draft, and never activates automatically.
- [x] #4 Import pack and Start blank route or focus existing editor controls without duplicating import or draft behavior in the setup card.
- [x] #5 AssistantSetupWizard can open Visuals through a route-level visual setup detour while setup is required, and returning clears the detour without changing setup completion state.
- [x] #6 Focused UI/service tests cover card rendering, no-active-pack behavior, starter copy selection, import/blank routing, compact wizard detour, and route-gating behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Follow `Docs/superpowers/plans/2026-05-15-persona-visual-buddy-setup-choices-implementation-plan.md`.
- Implement in the staged order: starter service/types, reusable setup card, VisualPackEditor integration, assistant setup visual detour, verification/task closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implementation plan created after user approval of the patched design spec.
- Plan explicitly calls out an implementation trap discovered during planning: current import/portability controls are guarded by `selectedPack`, but first-run Import pack must be reachable with zero packs, so the import preview/commit UI needs to be split from selected-pack-only upload/export/editing controls.

Task 3 VisualPackEditor setup flow integration completed and approved by code-quality reviewer. The editor now scopes setup state to the selected persona, loads starter packs defensively, copies defaults as inactive drafts, exposes first-run import and blank paths without requiring an existing pack, filters visible packs and candidates to the current persona, and guards async pack import and candidate mutation paths against stale persona or pack responses.

Task 3 verification passed: focused stale accepted candidate review test, focused stale-regression VisualPackEditor pattern run, service card editor Vitest suite with 60 tests, git diff --check, and bun run verify:design-system-state with existing baseline exceptions only.

Bandit not run for Task 3 because the slice changed only frontend TypeScript and TSX files.

Final wizard-detour closeout: AssistantSetupWizard now accepts optional compact visual setup content, the setup orchestrator tracks a route-level visual setup detour, sidepanel-persona can hide the wizard and show only the Visuals tab during the detour, and the detour return restores the setup overlay without marking setup complete. Focused validation passed: bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx ../packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx ../packages/ui/src/services/__tests__/persona-visuals.test.ts --testTimeout=30000 -> 5 files passed, 152 tests passed. git diff --check passed. Bandit remains not applicable because this task touched frontend TypeScript/TSX and Backlog Markdown only.

Quality review found that the first visual detour patch exposed all Persona Garden tabs while setup was still required. Resolved by filtering the detour view to the Visuals tab only and adding route regression assertions that Profiles, Commands, and Live Session are absent during the detour.

Post-rebase verification on latest origin/dev passed: bunx vitest run src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx src/services/__tests__/persona-visuals.test.ts --testTimeout=30000 -> 5 files passed, 153 tests passed. bun run verify:design-system-state passed with existing baseline exceptions only. git diff --check origin/dev...HEAD passed.

PR #1725 review-fix pass started. Verified live review surface: Qodo reported loadPacks invalidation/loading and refresh handler contract bugs; Gemini requested i18n cleanup for setup card copy, memoized visible pack derivations, broader stale mutation guards, and import-panel duplication review. Plan: fix verified low-risk bugs first, apply narrow memoization/i18n improvements where consistent with local patterns, evaluate broader refactors against scope, then run focused Vitest/design-system/diff verification before pushing.

PR #1725 review-fix pass completed. Fixed verified Qodo bugs by clearing pack loading/error state on inactive/persona invalidation and changing the refresh handler to call loadPacks with the expected options object. Addressed Gemini comments by memoizing visible/selected pack derivations, adding request IDs to draft create, asset upload, and manifest save mutation handlers with persona/pack invalidation, moving VisualBuddySetupChoiceCard copy through sidepanel i18n defaults plus English locale keys, and extracting the import preview/commit panel into a single shared JSX block for first-run and selected-pack flows. Also aliased the visual-library copy service import to avoid hook-rule false positives in direct package lint.

Review-fix verification: focused Vitest suite passed with 5 files and 154 tests; bun run verify:design-system-state passed with existing baseline exceptions only; git diff --check passed; direct ESLint on touched package files completed with zero errors and existing warnings only. Bandit remains not applicable because the review-fix touched frontend TypeScript/TSX, English locale JSON, and Backlog Markdown only.

Second PR #1725 review pass started after push. CodeRabbit added two still-valid sidepanel-persona detour findings: localize the visual-detour notice/button copy and derive one effective active tab so detour selection and rendered content cannot diverge. The repeated loadPacks string finding is already fixed in commit 7eb498dc2 and will be treated as stale after verification.

Second PR #1725 review pass completed. Fixed the still-valid sidepanel detour comments by routing visual/live detour notice and return-label copy through sidepanel locale keys and by deriving effectiveActiveTab so setup visual detour selection and lazy-rendered tab content stay aligned. Verified the repeated loadPacks(selectedPack?.id) thread is stale: current VisualPackEditor refresh uses loadPacks({ preferredPackId: selectedPack?.id }).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented first-run Persona Visual Buddy setup choices for issue #1695. The shared setup card supports full Visuals-tab actions and compact wizard routing. VisualPackEditor shows first-run setup choices only when no active visual exists, preserves advanced library/reuse affordances, copies bundled defaults into inactive drafts, focuses existing import controls, and routes blank setup to existing draft controls without auto-activation. Assistant setup now has an optional visual detour so users blocked by setup gating can open only the Visuals tab and return without changing setup completion state. Verification after rebasing onto latest dev: focused UI/service Vitest suite passed with 153 tests, design-system state verification passed with existing baseline exceptions only, git diff --check passed, and Bandit is not applicable for the frontend-only touched scope.

Review fixes for PR #1725 addressed the loadPacks invalidation/loading bug, corrected the refresh call contract, localized the setup card copy, memoized pack derivations, added request-id guards to the older mutation handlers, and shared the import preview panel between first-run and selected-pack paths. Validation passed with the focused 154-test Vitest suite, design-system state guard, git diff check, and touched-file ESLint with warnings only.

Second PR review pass: localized sidepanel setup detour copy, reused the return-to-setup locale key for live and visual detours, and made PersonaGardenTabs plus lazy tab rendering share effectiveActiveTab during setup visual detours. Validation: focused Vitest suite passed with 5 files and 154 tests; git diff --check passed; touched-file ESLint had zero errors and existing warnings only; design-system state guard passed with existing baseline exceptions only.
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
