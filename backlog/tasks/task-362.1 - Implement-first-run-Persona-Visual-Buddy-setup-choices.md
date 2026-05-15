---
id: TASK-362.1
title: Implement first-run Persona Visual Buddy setup choices
status: In Progress
assignee: []
created_date: '2026-05-15 03:44'
updated_date: '2026-05-15 07:19'
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
- [ ] #1 Reusable setup-choice card renders Use default, Import pack, and Start blank in Visuals, with compact optional wizard mode.
- [ ] #2 VisualPackEditor shows setup choices only when the selected persona has no active visual pack and preserves the existing advanced reuse panel.
- [ ] #3 Use default lists/copies bundled starter packs through existing frontend service patterns, creates an inactive draft, selects the returned draft, and never activates automatically.
- [ ] #4 Import pack and Start blank route or focus existing editor controls without duplicating import or draft behavior in the setup card.
- [ ] #5 AssistantSetupWizard can open Visuals through a route-level visual setup detour while setup is required, and returning clears the detour without changing setup completion state.
- [ ] #6 Focused UI/service tests cover card rendering, no-active-pack behavior, starter copy selection, import/blank routing, compact wizard detour, and route-gating behavior.
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
