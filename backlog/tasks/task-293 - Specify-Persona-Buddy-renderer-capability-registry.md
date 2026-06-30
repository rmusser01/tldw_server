---
id: TASK-293
title: Specify Persona Buddy renderer capability registry
status: Done
assignee: []
created_date: '2026-05-12 04:49'
updated_date: '2026-05-12 04:53'
labels:
  - persona
  - buddy
  - visual-packs
  - design
dependencies: []
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for an end-to-end thin slice that adds a Persona/Buddy visual renderer capability registry. The spec should keep the first implementation target limited to registry-backed backend validation/API capability reporting plus frontend Buddy renderer registry/diagnostics, with sprite_frames as the only enabled renderer and Live2D/external providers explicitly deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec document is added under Docs/superpowers/specs with architecture, components, data flow, error handling, tests, and non-goals.
- [x] #2 Spec is scoped to Persona/Buddy display behavior and does not route work through Persona Chat, VN, or CYOA systems.
- [x] #3 Spec preserves fail-closed activation/rendering semantics for unsupported renderer types while keeping sprite_frames behavior unchanged.
- [x] #4 Spec review feedback is addressed or documented before moving to implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write the approved design to a dated spec under Docs/superpowers/specs.
2. Run a spec review loop and patch any valid issues.
3. Record verification and review notes in this task.
4. Commit the spec and task update on the isolated branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worktree: <local_worktree_path_redacted>

Branch: codex/persona-buddy-renderer-capability-spec

Spec review iteration 1 found three issues: endpoint/runtime ownership ambiguity, V1 future-renderer placeholders left too open, and sample asset roles blurring sprite-frame V1 with sprite-sheet follow-up work. Patched the spec to make the V1 endpoint list only enabled sprite_frames, keep Buddy runtime renderability local/deterministic, and limit sample asset roles to frame/preview.

Spec review iteration 2 approved the patched design with no remaining contradiction, fail-open behavior, or Persona Chat/VN/CYOA/Live2D implementation drift.

Human-requested design risk review found three improvements before implementation planning: keep draft manifest saves permissive until activation/import-preview validation, avoid new renderer-level asset-role enforcement, and explicitly name the API schema/frontend service additions so the capability endpoint is useful without making Buddy fetch it at runtime. Patched the spec accordingly.

Verification: git diff --check passed. ASCII scan found no non-ASCII characters. Bandit skipped because this task changed only markdown documentation and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Persona Buddy renderer capability registry design spec at Docs/superpowers/specs/2026-05-12-persona-buddy-renderer-capability-registry-design.md. The spec defines the end-to-end thin slice for backend capability registry/API reporting plus frontend Buddy renderer registry/diagnostics, keeps sprite_frames as the only enabled V1 renderer, defers disabled future renderer placeholders, preserves fail-closed unsupported-renderer behavior, keeps draft manifest saves permissive, and avoids new asset-role enforcement. Spec review passed after one patch iteration and the follow-up design risk review was incorporated.
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
