# Persona Visual Buddy Setup Choices Design

Date: 2026-05-15
Status: Approved for implementation planning for TASK-362 / issue #1695
Owner: Codex brainstorming pass
Parent: #1510

## Summary

Add a reusable first-run setup choice surface for Persona Visual Buddy setup:
`Use default`, `Import pack`, and `Start blank`.

The component should render in both Persona Garden Visuals and the Assistant
Setup Wizard, but `VisualPackEditor` remains the behavioral source of truth for
draft creation, starter copy, import routing, pack reloads, selection, errors,
and explicit activation.

Setup is considered needed when the selected persona has no active visual pack.
Drafts may already exist; in that case the setup surface should shift its copy
toward reviewing and activating a draft while still offering default, import, or
blank alternatives.

This design uses the bundled starter catalog added by #1701 and does not add
new backend routes.

## Current Foundations

Relevant current behavior on `origin/dev`:

- Bundled starter packs can be listed and copied into user-owned inactive draft
  packs through:
  - `GET /api/v1/persona/visual-starter-packs`
  - `GET /api/v1/persona/visual-starter-packs/{starter_pack_id}`
  - `POST /api/v1/persona/visual-starter-packs/{starter_pack_id}/copy`
- Starter copies are normal user-owned drafts and do not activate
  automatically.
- `VisualPackEditor` already owns:
  - visual pack list loading and selected-pack state.
  - blank draft creation controls.
  - import archive picker and import-preview/commit controls.
  - activate/deactivate controls.
  - personal-library and duplicate-to-persona affordances.
- `VisualPackReusePanel` exposes advanced reuse actions, but it is not a
  first-run "set up your visual buddy" flow.
- `AssistantSetupWizard` has a fixed setup sequence and should not make visuals
  a required step for text-first assistants.
- #1696 covers archive upload/import handoff polish separately. This slice
  should only route users into the existing import path.
- #1698 covers deterministic fixtures and E2E coverage separately.

## Goals

1. Make the first visual-buddy setup path discoverable from Persona Garden.
2. Reuse one setup-choice component in both the Visuals tab and the Assistant
   Setup Wizard.
3. Keep `VisualPackEditor` as the owner of visual-pack behavior and mutations.
4. Copy the recommended bundled starter into an inactive draft and select it
   without activating it.
5. Route `Import pack` and `Start blank` into existing editor controls.
6. Preserve review-before-activation semantics for default, imported, and blank
   drafts.
7. Keep the implementation scoped to frontend service/types/components unless a
   missing response field blocks draft selection.

## Non-Goals

1. No automatic activation.
2. No Live2D runtime adapter or new renderer.
3. No external MCP provider execution.
4. No VN/CYOA behavior.
5. No import-polish work beyond routing/focusing existing controls.
6. No E2E fixture work.
7. No backend starter-catalog route changes.
8. No replacement of the advanced `VisualPackReusePanel`.

## User-Confirmed Decisions

- Use one reusable setup-choice component in both surfaces.
- The first concrete behavior target is the Visuals tab.
- Setup is needed when there is no active visual pack.
- `Use default` should be hybrid:
  - primary action copies the recommended bundled starter.
  - secondary action can open a starter picker when choices are available.
- After copying a default, select the new draft and stay in the Visuals tab.
- The setup wizard gets a compact optional card, not a new required setup step.
- `Start blank` focuses existing draft creation controls; it does not create a
  draft automatically.
- `Import pack` routes to existing import controls; import polish remains #1696.

## Component Model

Add a reusable `VisualBuddySetupChoiceCard` in
`apps/packages/ui/src/components/PersonaGarden/`.

The component should be presentational and compact. It should not load data or
call services directly. It receives state and handlers from its parent.

Suggested props:

```ts
type VisualBuddySetupChoiceCardProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  hasActiveVisual: boolean
  packCount: number
  recommendedStarter?: PersonaVisualStarterPackSummary | null
  starterCount?: number
  starterCatalogLoading?: boolean
  starterCatalogError?: string | null
  copyingDefault?: boolean
  compact?: boolean
  onUseDefault?: () => void
  onChooseDefault?: () => void
  onImportPack?: () => void
  onStartBlank?: () => void
  onOpenVisuals?: () => void
}
```

The exact prop names can follow local style during implementation, but the
boundary should stay the same: component renders choices; parent owns behavior.
Only `VisualPackEditor` should pass mutation/focus handlers for default copy,
starter picker, import picker, or blank draft focus. In compact wizard mode,
those actions should either be hidden or mapped to `onOpenVisuals`. The wizard
must not call starter-copy or import-preview services directly in this slice.

### Visuals Tab Placement

`VisualPackEditor` should render the setup card near the top of the Visuals tab,
before lower-level library/import/generation controls.

Rendering rule:

- Hide the setup card when `active_pack` exists.
- Show the setup card when `active_pack` is absent.
- If `packs.length === 0`, copy should frame this as first setup.
- If `packs.length > 0`, copy should frame this as unfinished setup with drafts
  available for review and activation.

The existing `VisualPackReusePanel` should remain available as the advanced
reuse/library path. It should not be the only first-run entry point.

### Assistant Setup Wizard Placement

The wizard should show a compact optional visual setup card without adding a new
required setup step.

Preferred behavior:

- If the route already knows the selected persona has no active visual pack,
  render a compact card that states visual setup is still available.
- If the active-visual state is unknown, render only a generic optional
  "set up visual buddy" route affordance. It should not claim that setup is
  required or missing.
- The card opens the Visuals tab with the selected persona already selected,
  using local tab state when the route is already mounted and route navigation
  only when needed from outside the Persona Garden route.
- Do not make assistant setup completion depend on visual setup.

If the route does not have enough visual-pack state to know whether an active
visual exists, do not duplicate full visual-pack loading in the wizard for this
slice. Use the generic route action to the Visuals tab instead.

For the first implementation, the wizard should assume active-visual state is
unknown unless that state is already available from existing route/profile data.
Do not add a second visual-pack fetch path just to make the wizard card smarter.
Do not add a separate focus query parameter for V1; local tab transition or the
existing `/persona?persona_id=...&tab=visuals` route is enough once the route can
actually render the Visuals tab.

Important setup-gating constraint: `sidepanel-persona.tsx` currently renders
`AssistantSetupWizard` instead of `PersonaGardenTabs` while assistant setup is
required, unless an existing setup detour is active. A compact wizard card that
only changes `tab=visuals` would leave the user inside the wizard and would not
reveal `VisualPackEditor`. The wizard integration therefore needs a route-level
visual setup detour, modeled on the existing command/live detour pattern:

- clicking the compact visual setup card starts the detour, selects the persona,
  and sets the active tab to `visuals`.
- while the detour is active, the route renders the normal tabs/Visuals tab
  instead of the setup wizard.
- a small route-level "return to setup" affordance clears the detour and restores
  the wizard.

This detour does not make visuals required, does not add visual setup to the
setup progress model, and does not move visual-pack mutations into the wizard.
If the implementation chooses not to add the detour in the first slice, the
wizard card should remain informational and must not claim that it can open the
editor during required setup.

## Frontend Services And Types

Extend `apps/packages/ui/src/types/persona-visuals.ts` with starter catalog
types matching the current backend response shapes:

- `PersonaVisualStarterPackSummary`
- `PersonaVisualStarterPackDetail`
- `PersonaVisualStarterPackAssetSummary`
- `PersonaVisualStarterPackCopyRequest`

Extend `apps/packages/ui/src/services/persona-visuals.ts` with:

- `listPersonaVisualStarterPacks()`
- `getPersonaVisualStarterPack(starterPackId)`
- `copyPersonaVisualStarterPack(starterPackId, payload)`

The implementation should preserve the existing service style:

- use `fetchPersonaVisualJson`.
- encode path ids.
- normalize list responses defensively if the API returns either list or
  wrapper shape in tests.

No backend route changes are expected.

## Data Flow

In `VisualPackEditor`:

1. Continue loading visual packs as today.
2. Load the starter catalog independently.
3. Derive:
   - `activePack = response.active_pack`.
   - `hasActiveVisual = Boolean(activePack)`.
   - `showSetupChoices = !hasActiveVisual`.
   - `recommendedStarter = first returned starter`.
4. On `Use default`:
   - call `copyPersonaVisualStarterPack(recommendedStarter.id, { target_persona_id })`.
   - reload or merge the persona visual pack list without letting the old
     selected pack override the newly copied draft.
   - select the returned draft pack by id after the refresh.
   - show status copy that activation is still explicit.
5. On `Choose another default`:
   - open a simple picker using loaded starter summaries.
   - copy the selected starter through the same path.
6. On `Import pack`:
   - call the existing import archive picker/focus handler.
7. On `Start blank`:
   - call the existing draft-title/create focus handler.

In `sidepanel-persona.tsx` / setup orchestration:

1. Keep the compact wizard card route-only/generic unless active visual state is
   already known.
2. When the user opens visual setup from the wizard, start a visual setup detour
   rather than only changing the URL or active tab.
3. During the visual detour, render `PersonaGardenTabs` with `activeTab =
   "visuals"` so `VisualPackEditor` owns all actual setup actions.
4. Provide a route-level return affordance that clears the detour and restores
   the wizard without changing setup completion state.

If the returned copied pack cannot be found after reload, the editor should
select the returned pack directly if it has enough pack fields. Otherwise it
should leave the current selection unchanged and show a non-fatal status/error
message.

Implementation should avoid relying on the current `loadPacks()` preference
order for starter-copy selection. That helper currently prefers active pack or
existing selection. Add a narrow preferred-pack mechanism, merge the copied pack
before selection, or set the selected id after refresh so the newly created
draft remains visible even when older drafts already exist.

The V1 frontend should not require a new backend `recommended` field. If a
future response already contains a recommendation marker, the frontend can use
it, but this slice should treat catalog order as the recommendation source.

## Error Handling

- Starter catalog load failure must not break the Visuals tab.
- When the starter catalog is unavailable, keep `Import pack` and `Start blank`
  enabled.
- Disable `Use default` when no recommended starter is available.
- Starter copy failure should use the existing editor error surface.
- Import and blank flows inherit existing error handling.
- The UI should never imply that copying a default activates it.

## Starter Picker

The first implementation can keep the picker simple:

- Open from `Choose another default`.
- Show starter title, description, renderer type, tags, and license label when
  available.
- Copy selected starter as an inactive draft.
- Do not preview advanced renderer behavior beyond current metadata.

If only one starter exists, the secondary picker control can be hidden or
disabled with clear copy.

## Testing Plan

Focused component tests:

1. `VisualBuddySetupChoiceCard` renders `Use default`, `Import pack`, and
   `Start blank`.
2. The card explains first setup when no packs exist.
3. The card explains draft review/activation when drafts exist but no active
   pack exists.
4. The card disables `Use default` when starter catalog loading failed or no
   starter is available, while leaving import and blank actions usable.
5. Compact mode renders suitable wizard copy and open-visuals action.

Focused `VisualPackEditor` tests:

1. Shows setup choices when no active visual pack exists.
2. Hides setup choices when an active pack exists.
3. Clicking `Use default` calls starter copy, reloads packs, selects the
   returned draft, and does not call activation.
   - Include a regression where another draft already exists, so selection does
     not fall back to the old draft after reload.
4. Clicking `Import pack` routes to the existing archive picker.
5. Clicking `Start blank` focuses existing draft creation controls.
6. Catalog load failure leaves import and blank flows usable.
7. Optional starter picker copies a selected non-recommended starter.

Focused service/type tests:

1. `listPersonaVisualStarterPacks` accepts both direct-list and wrapper-list
   shapes if tests mock either form.
2. Starter ids are URL-encoded in detail and copy paths.
3. Copy requests send `target_persona_id` and optional title without extra
   activation fields.
4. Copy responses are consumed as existing `PersonaVisualPack` drafts.

Focused setup wizard/route tests:

1. Wizard compact visual setup affordance starts a visual setup detour when
   assistant setup is required.
2. During that detour, the normal Visuals tab renders `VisualPackEditor` instead
   of leaving the user inside `AssistantSetupWizard`.
3. Return-to-setup clears the detour and restores the wizard.
4. A plain route/tab transition still opens the Visuals tab when setup is not
   gating the tabs.
5. The setup progress model does not add a required `visuals` step.
6. Assistant setup can still finish without an active visual pack.
7. Unknown active-visual state renders generic optional setup copy rather than
   claiming that a visual is missing.

Verification commands should be scoped to changed UI tests, for example:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx \
  src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx \
  src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx \
  src/routes/__tests__/sidepanel-persona.test.tsx
```

Run `git diff --check`. Bandit is not applicable if the implementation remains
frontend-only.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Wizard duplicates editor behavior | Wizard uses compact route/focus behavior; Visuals tab owns mutations |
| Users think default copy activates immediately | Copy and status text explicitly say inactive draft and activation required |
| Setup card competes with reuse panel | Setup card handles first-run choices; reuse panel remains advanced/library path |
| Starter catalog failure blocks setup | Import and Start blank remain available |
| Import-polish PR overlaps #1695 | This slice only routes to existing import controls; #1696 owns upload/status/failure polish |
| Future multi-starter catalog needs rework | Hybrid primary recommended starter plus optional picker supports growth |
| Wizard route action does not reveal Visuals because setup gating hides tabs | Add a route-level visual setup detour that temporarily renders normal tabs with Visuals active |

## Implementation Notes

This design intentionally keeps behavior in the existing editor rather than
creating a parallel onboarding state machine. The one exception is a thin
route-level visual detour, because the current setup-required route gate hides
normal tabs. If implementation finds that the wizard already receives reliable
active-pack state, it can render a smarter compact card. Otherwise the wizard
should keep the visual setup copy generic and use the detour only to hand the
user to `VisualPackEditor`.

The design assumes starter-copy responses include enough pack identity for
selection. If the response is insufficient in current code, prefer a frontend
reload-and-match strategy before adding backend fields.
