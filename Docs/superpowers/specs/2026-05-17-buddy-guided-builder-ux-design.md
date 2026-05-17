# Buddy Guided Builder UX Design

Date: 2026-05-17
Status: Draft for TASK-420 user review
Backlog: TASK-420
Parent epic: https://github.com/rmusser01/tldw_server/issues/1510
Related trackers:
- https://github.com/rmusser01/tldw_server/issues/1787
- https://github.com/rmusser01/tldw_server/issues/1803

## Summary

Replace the current first-run Persona Buddy setup surface with a full guided
Buddy builder inside Persona Garden's Visuals tab. The builder should guide a
user through choosing a source, creating or importing a draft, reviewing
runtime readiness, configuring states and triggers, and explicitly activating a
pack.

The builder does not create a new avatar system. It is a product-grade shell on
top of the existing Persona Visual pack contract:

1. bundled starter packs,
2. portable native visual-pack import,
3. Codex/Petdex pet import,
4. personal visual library reuse,
5. duplicate-to-persona,
6. blank draft creation,
7. manifest validation,
8. generated-candidate review,
9. explicit activation.

The immediate implementation target is selection and configuration UX for the
basic Buddy/default and Codex import path. Intermediate and intricate asset
production remains out of this fork.

## Current Foundations

The current repo already contains the contracts the guided builder should use:

- `VisualPackEditor` already owns pack list loading, active-pack selection,
  starter copy, import preview/commit, library reuse, duplicate-to-persona,
  generated candidate review, manifest editing, validation, and activation.
- `VisualBuddySetupChoiceCard` currently exposes three setup choices, but it is
  only a compact card, not a guided builder.
- `VisualPackReusePanel` exposes reuse affordances after the main pack controls.
- `AssistantSetupWizard` already has a visual setup detour that can render the
  normal Visuals tab while setup is still in progress.
- `BuddyShellHost`, `SpriteFrameRenderer`, and `personaVisualState` already
  render active `sprite_frames` packs and resolve built-in/custom states.
- `visual_starter_fixtures.py` currently defines twelve starter IDs in stable
  order. The first six are the art-ready basic tier:
  `search-lens-basic`, `index-card-basic`, `archive-cube-basic`,
  `paperclip-basic`, `terminal-tile-basic`, and `migu-marker-basic`.
- The Codex pet import adapter accepts `.zip` packages containing
  `pet.json` or `petjson.json` and a 1536x1872 8x9 PNG/WebP spritesheet. It maps
  Codex rows into normal Persona Visual `sprite_frames`, including
  `moving_right` and `moving_left` custom states.

The current frontend has one important mismatch: import file validation in
`VisualPackEditor` only accepts `.tldw-persona-vpack`, so Codex/Petdex `.zip`
archives are rejected before the backend adapter can preview them.

## Design Review Findings

The direction is sound, but implementation planning should account for these
issues before UI work starts:

1. The builder must work in the sidepanel, not just the wide WebUI. A desktop
   rail is acceptable, but narrow surfaces need a compact stepper or accordion
   with a stable action footer.
2. Codex/Petdex archives and native Persona Visual archives are both ZIP-like.
   The frontend should only do extension/media-type admission; backend import
   preview must decide which adapter applies.
3. Changing persona, source type, selected file, or selected starter must clear
   downstream preview, commit, review, and activation state so stale results do
   not apply to the wrong draft.
4. Higher-tier scaffold packs should stay visible for tracker continuity, but
   the UI must not present them as equal to reviewed Basic defaults.
5. `moving_right` and `moving_left` are already importable/custom states, but
   the current Buddy drag handler only moves the dock. The implementation plan
   needs a runtime follow-through slice or explicit defer note for temporary
   drag movement overrides.
6. Existing frontend tests still contain many `research-buddy-starter`
   fixtures. The builder work should update stale test defaults to
   `search-lens-basic` without removing intentional legacy compatibility tests.
7. New user-facing labels need sidepanel i18n keys and accessible step/button
   semantics from the start, otherwise review fixes will be predictable.

## Goals

1. Make the Visuals tab feel like a Buddy builder, not a raw manifest editor.
2. Present the six art-ready basic defaults as the default bundled tier.
3. Make Codex/Petdex pet import a first-class path alongside native pack
   import.
4. Keep every path draft-first and review-before-activation.
5. Let users configure state mappings, custom states, movement states, and
   authored triggers without editing raw manifest JSON first.
6. Preserve the current runtime/rendering/backend contracts.
7. Keep WebUI and extension behavior shared through `apps/packages/ui`.

## Non-Goals

1. No automatic activation.
2. No new renderer runtime.
3. No replacement for Persona Visual packs, assets, library, or import jobs.
4. No intermediate or intricate asset production in this fork.
5. No Codex pet export generation in this slice unless a later implementation
   plan explicitly scopes it.
6. No VN/CYOA work.
7. No raw image generation workflow in the builder until generation readiness
   and review surfaces are intentionally extended.

## Builder Shape

The Visuals tab should render a guided builder as the primary content. Existing
raw controls remain available, but they should be grouped under builder steps
and advanced sections.

Recommended structure:

1. Source
2. Draft
3. Review
4. Configure
5. Activate

Use a persistent in-page stepper or rail rather than a modal. This flow has
state, diagnostics, file uploads, draft selection, and advanced editing. A modal
would make long review and configuration work cramped, and it would duplicate
the Visuals tab navigation model.

Responsive behavior matters. Wide WebUI layouts can use a left rail. Sidepanel
and narrow layouts should collapse to a top stepper or accordion-style flow,
with the primary action and current blocker kept visible near the bottom of the
builder.

The current `VisualBuddySetupChoiceCard` should become the first-run entry
affordance into this builder, not the complete UX. In Assistant Setup, the
compact card continues to open the visual setup detour and lands the user in
the builder.

The builder should maintain an explicit local state machine:

- selected persona,
- selected source,
- selected starter/library/duplicate source,
- selected import file,
- import preview job/result,
- committed draft pack,
- active review/configuration step.

Any upstream change clears downstream state. For example, changing the selected
persona clears selected starter, selected file, import preview, committed draft,
and activation readiness.

## Step 1: Source

The first step asks what the user wants to start from.

Source choices:

1. Bundled Buddy
2. Import Codex/Petdex pet
3. Import native Persona Visual pack
4. Personal library
5. Duplicate from another persona
6. Start blank

The primary/default choice should be Bundled Buddy. Its catalog should be tier
aware:

- Basic: show the six art-ready defaults as selectable, usable packs.
- Intermediate: show scaffold entries as future/production-packet targets, not
  as polished defaults.
- Intricate: same as intermediate.

Do not hide intermediate or intricate rows if the API returns them. They are
useful tracker visibility. The UI should make their status clear so users do
not confuse scaffold catalog entries with completed default packs.

The six Basic packs are the recommended selectable defaults. Intermediate and
Intricate scaffold entries should either disable the primary "use as default"
action or route through an explicit scaffold-copy affordance that says the pack
is a production packet, not finished Buddy art.

## Step 2: Draft

Every source path creates, imports, selects, or reuses a draft.

Behavior by source:

- Bundled Buddy: copy selected starter with
  `POST /api/v1/persona/visual-starter-packs/{starter_pack_id}/copy`, select
  the returned draft, and stay in the builder.
- Codex/Petdex pet: accept `.zip`, run import preview, then commit as a draft
  after review.
- Native visual pack: accept `.tldw-persona-vpack`, run import preview, then
  commit as a draft after review.
- Personal library: use existing library item flow to create a draft for the
  selected persona.
- Duplicate from another persona: use the existing duplicate flow to create a
  draft for the selected persona.
- Start blank: create an empty draft or focus the existing draft title control,
  depending on implementation slice.

The builder must never make a copied/imported draft active automatically.
Draft creation, import commit, and duplicate/library reuse should only select
the resulting draft for review.

## Step 3: Review

The review step should answer one question: can this draft be trusted enough to
activate?

Show:

- pack status and selected persona,
- source type: bundled starter, Codex pet, native visual pack, library, duplicate,
  blank, or generated candidate,
- renderer type and manifest version,
- production status and complexity tier when from the starter catalog,
- source format and atlas details when from Codex/Petdex import,
- state coverage for built-in states,
- custom states from `state_catalog`,
- movement states `moving_right` and `moving_left` when present,
- fallbacks,
- authored triggers,
- asset role/group summary,
- import-preview blockers, warnings, conflicts, and renderer diagnostics,
- pack-health diagnostics,
- activation blockers.

Codex/Petdex imports should explicitly show that the `.zip` was adapted into a
normal Persona Visual draft. The UI should not imply a separate Codex-pet store.

When asset bytes are available, previews should use the existing
`SpriteFrameRenderer` path against the selected draft. If preview data is not
available yet, show diagnostics and state coverage instead of a handcrafted
HTML/image mockup.

## Step 4: Configure

Configuration should expose common Buddy behavior before raw manifest details.

Configuration groups:

1. Core states
   - `idle`
   - `listening`
   - `thinking`
   - `speaking`
   - `tool_running`
   - `approval_needed`
   - `wake_armed`
   - `error`
   - `offline`

2. Movement states
   - `moving_right`
   - `moving_left`
   - These are Buddy drag/screen movement states, not generic task-running
     states.
   - The configuration UI can expose and validate these states independently
     from runtime support, but implementation planning should include a runtime
     test proving drag movement can request the corresponding temporary visual
     state when that behavior is wired.

3. Custom states
   - Display `state_catalog` entries with label, kind, description, tags, and
     fallback.
   - Keep ID creation bounded by the existing custom state ID rules.

4. Tool and runtime triggers
   - Prefer exact `tool_name` triggers for per-tool animation variants.
   - Use `tool_category` only for broader fallback behavior.
   - Use `mcp_runtime` for bounded transient runtime reasons.

5. Advanced manifest editing
   - Keep the existing state mapping, animation frames, fallbacks, and authored
     trigger controls available.
   - Move them behind an advanced/edit-details section when possible.

The builder should not hide validation errors. It should route them to the
specific step that can fix them.

## Step 5: Activate

Activation remains an explicit final step.

The final step should show:

- selected persona,
- selected draft,
- active pack currently in use,
- what will change on activation,
- unresolved blockers,
- warnings that are allowed but should be acknowledged,
- a single Activate action once required states resolve.

If the user already has an active pack, activation should make it clear that
the active pack will be replaced, while archived/draft packs remain available.

## Import Format UX

The import UI should present two supported archive formats:

1. Native Persona Visual pack: `.tldw-persona-vpack`
2. Codex/Petdex pet: `.zip`

Frontend file validation should accept both before calling import preview.
Backend preview remains the source of truth for actual archive validation.
Do not infer the import adapter from MIME type alone. `.tldw-persona-vpack` may
arrive with a generic ZIP media type, and Codex/Petdex `.zip` may arrive as
`application/octet-stream`.

Accepted media types should include normal ZIP types for both formats:

- `application/zip`
- `application/x-zip-compressed`
- `application/octet-stream`

Copy should distinguish early file-picker help from backend validation:

- file picker: "Choose a `.tldw-persona-vpack` or Codex/Petdex `.zip` archive."
- preview result: "This Codex pet will be imported as a Persona Visual draft."

## Assistant Setup And Extension Placement

Assistant Setup remains optional for visuals. The compact visual card should
open the existing visual detour and land in the guided builder.

In sidepanel/extension surfaces, the same shared builder components should be
used through `apps/packages/ui`. The extension should not implement separate
Buddy import or selection logic. It should route to Persona Garden Visuals or
render the shared Visuals surface where already supported.

## Component Boundaries

Keep behavior in existing services and explicit builder components:

- `BuddyGuidedBuilder`
  - orchestrates source, draft, review, configure, activate steps.
- `BuddySourcePicker`
  - bundled/default/import/library/duplicate/blank choices.
- `BuddyStarterCatalogPicker`
  - tier-aware starter display and copy action.
- `BuddyImportFormatPanel`
  - native versus Codex import help and file picker.
- `BuddyDraftReviewPanel`
  - normalized review/readiness summary.
- `BuddyStateConfigurationPanel`
  - common state/custom state/trigger editing shell.
- Existing `VisualPackEditor`
  - remains the integration owner for data loading, service calls, and final
    mutation handlers until there is a clean extraction point.

Avoid expanding one file indefinitely. `VisualPackEditor` is already broad, so
the builder should be extracted into focused presentational components as soon
as practical.

## Implementation Slices

### Slice 1: Format Gate And Copy Corrections

Goal: make the existing import path accept Codex/Petdex `.zip`, update copy so
the current UI no longer blocks backend-supported imports, and refresh stale
test fixtures away from `research-buddy-starter`.

This slice is preparatory. If it ships as a standalone PR, it should not be
described as the full guided builder. Its success condition is that existing
paths stop contradicting the backend-supported import and starter catalog.

Tests:

- `persona-visuals` service tests for starter catalog normalization using
  `search-lens-basic`.
- `VisualPackEditor` import-file validation accepts `.zip` with
  `application/zip`, `application/x-zip-compressed`, and
  `application/octet-stream`.
- `VisualPackEditor` import-file validation still accepts
  `.tldw-persona-vpack` with normal ZIP media types.
- `VisualPackEditor` still rejects unsupported extensions.
- stale test fixtures use `search-lens-basic`, while explicit legacy alias
  coverage still names `research-buddy-starter`.
- existing import preview tests remain green.

### Slice 2: Guided Source/Draft Shell

Goal: introduce the builder shell and source picker while wiring actions to
existing handlers. This is the first visible builder slice.

Tests:

- no-active-pack state renders builder source step.
- basic tier shows six art-ready default IDs.
- choosing a bundled default copies it as a draft and selects it.
- changing persona/source clears stale selected file, import preview, copied
  draft, and activation readiness.
- narrow layout renders a compact stepper/accordion instead of a cramped rail.
- Assistant Setup compact card opens the visual detour and renders the builder.

### Slice 3: Review And Import Diagnostics

Goal: make review status readable before activation.

Tests:

- Codex/Petdex import preview shows source format, atlas dimensions, and draft
  semantics.
- import blockers disable commit/activation path.
- native archive and Codex archive copy are distinct.
- preview source type comes from backend preview data, not from ZIP MIME type.
- selected draft preview uses existing sprite-frame rendering when renderable.
- activation remains unavailable when required states do not resolve.

### Slice 4: Configure States And Triggers

Goal: expose common state mapping, movement states, custom states, fallbacks,
and authored triggers through builder sections.

Tests:

- custom states from `state_catalog` render with kind/label/fallback.
- `moving_right` and `moving_left` render as movement states.
- exact `tool_name` trigger editing preserves structured match fields.
- manifest save still uses the existing update endpoint.
- user-facing labels use sidepanel i18n keys and controls have accessible names.

### Slice 4B: Movement Runtime Follow-Through

Goal: connect configured `moving_right` and `moving_left` states to Buddy drag
movement when those states exist for the active pack.

Tests:

- dragging right sets a short-lived runtime override to `moving_right` when the
  active pack declares it.
- dragging left sets a short-lived runtime override to `moving_left` when the
  active pack declares it.
- releasing the drag clears the movement override back to normal state
  resolution.
- packs without movement states keep the current drag behavior.

### Slice 5: Browser QA

Goal: verify the builder in the rendered WebUI/extension surface before calling
the UI work complete.

Checks:

- Persona Garden Visuals no-active-pack state.
- bundled Basic selection flow.
- Codex `.zip` file selection flow with mocked backend response where needed.
- draft review state.
- compact Assistant Setup detour.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| The builder duplicates backend concepts into a parallel system. | Keep all mutations on existing Persona Visual services and endpoints. |
| Users confuse scaffold higher-tier entries with ready defaults. | Tier the catalog and label production status prominently. |
| Codex import appears unsupported because frontend rejects `.zip`. | Accept `.zip` at the file gate and let backend preview validate. |
| Codex/native import routing is guessed incorrectly from file MIME type. | Treat frontend checks as admission only; render backend preview source type. |
| A stale import preview or copied draft applies after persona/source changes. | Model builder state explicitly and clear downstream state on upstream changes. |
| Visuals become required in Assistant Setup. | Keep the wizard card optional and detour-based. |
| `VisualPackEditor` becomes harder to maintain. | Extract builder panels instead of adding another large inline block. |
| Movement states are mistaken for `tool_running`. | Label `moving_right`/`moving_left` as drag/screen movement states. |
| Movement states are configurable but never used at runtime. | Add the movement runtime follow-through slice or document it as deferred. |
| Sidepanel layout becomes cramped. | Use a compact stepper/accordion on narrow surfaces and browser-QA it. |
| New copy bypasses localization/accessibility conventions. | Add sidepanel i18n keys and accessible labels in the first UI slice. |
| Activation happens too early. | Preserve draft-first semantics and final explicit activation. |

## Open Questions For Implementation Planning

1. Whether Slice 1 and Slice 2 should be one PR or separate PRs.
2. Whether the first builder shell should replace `VisualBuddySetupChoiceCard`
   immediately or wrap it for one compatibility slice.
3. Whether blank-start should create a draft immediately or keep focusing the
   existing draft title input in the first implementation.
4. Whether Codex import review should show a generated atlas row table in Slice
   3 or defer that to a later visual QA pass.
5. Whether movement runtime follow-through should be included with Configure or
   handled as the next PR after builder configuration lands.
