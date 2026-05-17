---
id: TASK-420
title: Plan Buddy default selection and Codex import UX
status: In Progress
labels:
- persona
- buddy
- visual-packs
- frontend
- design
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/issues/1787
- https://github.com/rmusser01/tldw_server/issues/1803
- https://github.com/rmusser01/tldw_server/pull/1818
documentation:
- Docs/superpowers/specs/2026-05-17-buddy-guided-builder-ux-design.md
- Docs/superpowers/plans/2026-05-17-buddy-guided-builder-ux-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan the WebUI/extension Persona Buddy selection and configuration UX so the six basic Codex Buddy defaults are presented as the basic tier, Codex/Petdex pet import is a first-class reuse path, and current bundled 96x96 runtime assets are clearly distinguished from the Codex-compatible atlas interchange target. Start with repo-grounded inspection of existing Persona Garden Visuals, Assistant Setup, shared UI service/types, and extension-side surfaces before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current Persona Visual/Buddy setup and selection surfaces are inspected and summarized from code/docs.
- [x] #2 Spec proposes how users select bundled Buddy defaults, import Codex/Petdex pets, and understand draft/review/activation status without inventing a parallel avatar system.
- [x] #3 Spec keeps intermediate/intricate asset production out of this fork and focuses on selection/configuration UX.
- [x] #4 Spec identifies focused implementation slices and tests for the next PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the guided Buddy builder UX spec in
`Docs/superpowers/specs/2026-05-17-buddy-guided-builder-ux-design.md`.

Repo-grounded findings:

- `VisualPackEditor` already owns the Persona Visual pack lifecycle:
  starter-copy draft creation, import preview/commit, library reuse,
  duplicate-to-persona, generated-candidate review, manifest editing,
  validation, and activation.
- `VisualBuddySetupChoiceCard` is currently only a three-choice entry card; it
  should become an entry point into a full Visuals-tab builder, not the final
  setup UX.
- `AssistantSetupWizard` already has a visual setup detour that can show the
  normal Visuals tab while setup is still in progress.
- The server starter catalog now uses `search-lens-basic` as the default ID and
  exposes six art-ready basic defaults before six higher-tier scaffolds.
- The Codex/Petdex backend adapter accepts `.zip` packages with `pet.json` or
  `petjson.json` and a 1536x1872 8x9 spritesheet, then maps rows into normal
  Persona Visual `sprite_frames`, including `moving_right` and `moving_left`.
- The current frontend file gate still only accepts `.tldw-persona-vpack`, so
  the first implementation slice must allow Codex/Petdex `.zip` archives to
  reach backend import preview.

The approved design direction is the full guided Buddy builder:
source selection, draft creation/import/reuse, review diagnostics,
state/trigger configuration, and explicit activation.

Design review pass before implementation:

- Added responsive guidance so the builder does not assume a wide WebUI rail;
  sidepanel/narrow layouts need a compact stepper or accordion.
- Added explicit builder state-machine invariants so persona/source/file
  changes clear stale preview, copied-draft, review, and activation state.
- Tightened Codex/native import UX so frontend checks only admit candidate
  archives and backend preview remains the adapter/source-type authority.
- Clarified that higher-tier scaffold catalog entries remain visible but must
  not look like reviewed Basic defaults.
- Added the missing runtime follow-through risk for `moving_right` and
  `moving_left`: the builder can configure them, but Buddy drag needs its own
  short-lived runtime override slice before those states are actually used.
- Added test expectations for `search-lens-basic` default fixtures, intentional
  `research-buddy-starter` legacy coverage, i18n/accessibility, and narrow
  layout behavior.

Implementation plan:

- Created
  `Docs/superpowers/plans/2026-05-17-buddy-guided-builder-ux-implementation-plan.md`.
- The plan keeps implementation in `apps/packages/ui`, preserves
  `VisualPackEditor` as the lifecycle/mutation owner, and extracts focused
  builder components instead of expanding the editor inline.
- Planned stages cover archive admission, builder source/draft shell, review
  diagnostics, state/trigger configuration, movement runtime follow-through,
  browser QA, and closeout.
- Plan review subagent was not dispatched in this pass because this session did
  not have explicit approval to spawn a reviewer for the new implementation
  plan.

Verification:

- `git diff --cached --check` passed for the spec and Backlog task draft.
- Post-review `git diff --check` passed for the critique refinement.
- Bandit is not applicable yet because this slice is docs/tracker only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
