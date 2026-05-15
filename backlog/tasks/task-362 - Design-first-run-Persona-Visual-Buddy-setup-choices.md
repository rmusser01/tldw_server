---
id: TASK-362
title: Design first-run Persona Visual Buddy setup choices
status: In Progress
assignee: []
created_date: '2026-05-15 03:22'
labels:
  - persona
  - buddy
  - visuals
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1695'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/specs/2026-05-14-persona-buddy-default-catalog-state-catalog-extension-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design/spec for GitHub issue #1695. The approved direction is a reusable setup-choice card rendered in both Persona Garden Visuals and the Assistant Setup Wizard, with the Visuals tab owning behavior. Setup is needed when the selected persona has no active visual pack. The default path should copy the recommended bundled starter into an inactive draft and select it without activation; Import and Start blank should route to existing controls. Keep import polish, E2E fixtures, Live2D runtime, external provider execution, and VN/CYOA behavior out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the reusable setup-choice component and its Visuals-tab and wizard placements.
- [x] #2 Spec preserves draft-first behavior and explicit activation after default copy or imported/blank drafts.
- [x] #3 Spec scopes Import pack to routing/focusing the existing import path and leaves import polish to #1696.
- [x] #4 Spec defines frontend service/type needs for starter catalog list/copy without adding backend routes.
- [x] #5 Spec defines focused test coverage for card rendering, no-active-pack behavior, default copy draft selection, import/blank routing, and wizard compact routing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Write the approved design spec for issue #1695.
- Run a spec-review pass and patch the spec if needed.
- Record docs-only verification and final task summary before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Design decisions captured: reusable setup-choice component in Visuals and wizard; setup-needed means no active visual pack; default path copies recommended starter first with optional picker; copied defaults remain inactive drafts and are selected for review; wizard card is optional and route-oriented; blank/import route to existing controls.
- Spec artifact: `Docs/superpowers/specs/2026-05-15-persona-visual-buddy-setup-choices-design.md`
- Spec review iteration 1 found four ambiguities: recommended starter field assumptions, wizard unknown-state behavior, compact component handler boundaries, and service/type test coverage. Patched the spec to use catalog order for V1 recommendation, route-only compact wizard behavior when state is unknown, optional mutation handlers only for editor use, and focused service tests.
- Spec review iteration 2 approved the patched design. Review confirmed catalog-order recommendation avoids backend field requirements, wizard known/unknown visual state behavior is clear, compact wizard mode remains route-only, and service/type test coverage is included.
- Verification: `git diff --check` passed in `.worktrees/persona-visual-setup-choices`. Bandit not applicable because this task changes Markdown/Backlog documentation only.
- Additional human-requested design critique before planning found three implementation risks and patched the spec: avoid relying on `loadPacks()` old-selection preference after starter copy, avoid adding a new route focus query for V1, and keep the wizard generic when active visual state is unknown instead of adding duplicate visual-pack loading.
- Second human-requested design critique found one blocking route-integration issue: while assistant setup is required, `sidepanel-persona.tsx` renders `AssistantSetupWizard` instead of `PersonaGardenTabs`, so a wizard action that only changes `tab=visuals` would not reveal `VisualPackEditor`. Patched the spec to require a route-level visual setup detour modeled on existing setup detours, plus tests for detour entry/return.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the design spec for first-run Persona Visual Buddy setup choices in `Docs/superpowers/specs/2026-05-15-persona-visual-buddy-setup-choices-design.md`. The spec defines a reusable setup-choice card for Visuals and Assistant Setup, keeps VisualPackEditor as the behavior owner, preserves inactive-draft and explicit-activation semantics, scopes Import pack to existing controls, defines starter catalog frontend service/type needs, and records focused test coverage. Independent spec review passed after one patch iteration. Later design critique patched implementation-risk notes around starter-copy selection, route focus scope, wizard unknown-state behavior, and the setup-required route gate that needs a visual setup detour before the wizard can open Visuals. Docs-only verification passed with `git diff --check`.
<!-- SECTION:FINAL_SUMMARY:END -->
