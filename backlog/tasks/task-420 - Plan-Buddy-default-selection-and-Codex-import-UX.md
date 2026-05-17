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
modified_files:
- apps/packages/ui/src/components/PersonaGarden/buddyBuilderArchive.ts
- apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts
- apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts
- apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts
- apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx
- apps/packages/ui/src/components/PersonaGarden/BuddySourcePicker.tsx
- apps/packages/ui/src/components/PersonaGarden/BuddyStarterCatalogPicker.tsx
- apps/packages/ui/src/components/PersonaGarden/BuddyImportFormatPanel.tsx
- apps/packages/ui/src/components/PersonaGarden/BuddyDraftReviewPanel.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx
- apps/packages/ui/src/components/PersonaGarden/BuddyStateConfigurationPanel.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx
- apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx
- apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
- apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx
- apps/packages/ui/src/store/persona-visual-runtime.ts
- apps/packages/ui/src/store/__tests__/persona-visual-runtime.test.ts
- apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx
- apps/packages/ui/src/services/__tests__/persona-visuals.test.ts
- apps/packages/ui/src/assets/locale/en/sidepanel.json
- apps/packages/ui/src/public/_locales/en/sidepanel.json
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

Plan review pass before implementation:

- Moved archive-admission i18n key work into Task 1 so new import error copy
  cannot land without locale coverage.
- Tightened builder scope so `VisualPackEditor` renders the guided builder for
  first-run, draft, review, and active-pack states instead of treating it as
  only a replacement for the first-run setup card.
- Grounded Codex/Petdex review diagnostics in the current
  `PersonaVisualImportPreviewResponse` fields (`schema_version` and
  `bundle_summary.assets`) instead of assuming a future `source_format` field.
- Removed loose `as any` guidance from the sample state and movement code and
  pointed implementers to the existing Persona Visual typed helpers.
- Added movement-runtime implementation notes for pointer capture and
  stale-closure risks in `BuddyShellHost`.

Verification:

- `git diff --cached --check` passed for the spec and Backlog task draft.
- Post-review `git diff --check` passed for the critique refinement.
- Post-plan-review `git diff --check` passed for the implementation-plan
  refinement.
- Bandit is not applicable yet because this slice is docs/tracker only.
Task 1 implementation completed: added shared Buddy import archive admission helpers, wired VisualPackEditor import preview gating to admit native Persona Visual archives and Codex/Petdex .zip archives, refreshed default starter fixtures to search-lens-basic, and added English builder import error copy. Focused verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/services/__tests__/persona-visuals.test.ts src/routes/__tests__/sidepanel-persona-locale-keys.test.ts --testTimeout=30000` (4 files, 76 tests). Bandit is not applicable to this frontend-only TypeScript/JSON slice.
Task 2 implementation completed: added the Buddy builder state helpers, source picker, tiered starter catalog, import-format panel, and top-level guided builder shell; wired the shell into VisualPackEditor as the primary Visuals surface while preserving existing editor-owned mutations and draft/import controls; updated compact Assistant Setup copy to open the Buddy builder. Focused verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx --testTimeout=30000` (5 files, 150 tests). Locale guard passed: `bunx vitest run src/routes/__tests__/sidepanel-persona-locale-keys.test.ts --testTimeout=30000` (1 file, 3 tests). `jq empty` on English sidepanel locale JSON and `git diff --check` both passed. Bandit is not applicable to this frontend-only TypeScript/JSON slice.
Task 3 implementation completed: added pure Buddy draft readiness summarizers for source labels, atlas metadata, required-state blockers, movement states, custom states, warnings, and activation readiness; added BuddyDraftReviewPanel with backend-preview source semantics and SpriteFrameRenderer-backed draft previews; integrated the review panel into BuddyGuidedBuilder. Focused verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testTimeout=30000` (4 files, 74 tests). Locale guard passed: `bunx vitest run src/routes/__tests__/sidepanel-persona-locale-keys.test.ts --testTimeout=30000` (1 file, 3 tests). `jq empty` on English sidepanel locale JSON and `git diff --check` both passed. Bandit is not applicable to this frontend-only TypeScript/JSON slice.
Task 4 implementation completed: added BuddyStateConfigurationPanel with core state ordering, separate movement states, custom state metadata/fallbacks, grouped tool-name/tool-category/runtime triggers, accessible read-only state controls, and a Save visual state configuration action wired through the existing VisualPackEditor manifest save callback. Focused verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testTimeout=30000` (3 files, 69 tests). Shared helper and locale verification passed: `bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts src/routes/__tests__/sidepanel-persona-locale-keys.test.ts --testTimeout=30000` (2 files, 9 tests). `jq empty` on English sidepanel locale JSON and `git diff --check` passed. Bandit is not applicable to this frontend-only TypeScript/JSON/docs slice.
Task 5 implementation completed: added a Persona Visual runtime `clearOverride()` helper and wired BuddyShellHost drag handling to set short-lived `moving_right` / `moving_left` runtime overrides only when the active pack declares those movement states in `states` or `state_catalog`. Pointer release/cancel clears only Buddy-drag overrides, and runtime custom-state allowance now uses the union of declared manifest states and state catalog entries. Focused verification passed: `bunx vitest run src/store/__tests__/persona-visual-runtime.test.ts src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx --testTimeout=30000` (2 files, 32 tests). Bandit is not applicable to this frontend-only TypeScript/docs slice.
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
