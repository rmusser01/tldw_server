# Main Chat Role-play Preset Remediation Design

**Date:** 2026-05-17
**Surface:** Web `/chat` main chat Playground
**Status:** Approved in-session for spec review
**Backlog:** TASK-402

---

## Goal

Make the main `/chat` role-play preset workflow predictable, recoverable, and understandable without drifting into a general WebUI redesign.

The end state is not "more role-play features." The end state is control:

- users know which character, prompt/template, scene, and generation style are active;
- users can preview what a preset changes before applying it;
- users can undo or clear a bad preset choice;
- users can tell whether the next request will include character context;
- desktop and mobile users can reach the same role-play controls.

## Problem

The current `/chat` role-play experience has useful pieces but weak mapping:

- `Chat as a character` opens character selection.
- `Character mode` can lead to scene/actor settings.
- `Templates` applies system prompt snippets.
- parameter `Presets` change generation behavior.
- startup templates can save bundles of model, prompt, character, context, and generation style.

Those controls are scattered and use overlapping language. Applying a role-play prompt template can leave the UI saying only `Custom prompt`, which hides the role-play state. Some recovery paths are missing when the prompt library is empty. Mobile users cannot reach the same template and generation-style controls. Browser testing also found a hard crash when selecting `Default Assistant` from the character picker.

## Implementation Anchors

The plan should stay grounded in the current `/chat` implementation:

- `/chat` route chain: `apps/tldw-frontend/pages/chat/index.tsx` -> `apps/packages/ui/src/routes/option-chat.tsx` -> Playground.
- First-run starter: `PlaygroundEmpty.tsx` dispatches the `Chat as a character` starter and `PlaygroundForm.tsx` handles it.
- Character/persona selection: `AssistantSelect.tsx`.
- Behavior templates: `SystemPromptTemplates.tsx`.
- Prompt recovery/editing: `PromptSelect.tsx`.
- Generation style: `ParameterPresets.tsx`.
- Startup bundle persistence candidate: `startup-template-bundles.ts` and `usePromptTemplates.ts`.
- Request inclusion and character-flow eligibility: `usePlaygroundRawPreview.ts`.
- User-facing labels: existing i18n locale files under `apps/packages/ui/src/public/_locales/` and `apps/packages/ui/src/assets/locale/`.

## Non-Goals

- No general redesign of `/chat`.
- No broad chat cockpit refactor.
- No character library, Persona Garden, or Buddy runtime redesign.
- No RAG UX redesign except compatibility notices that directly affect character context.
- No backend/API changes unless existing frontend/API contracts cannot truthfully represent role-play state or request inclusion.
- No new persistence model unless existing startup template bundles cannot support saved role-play setups.
- No new route for role-play chat.
- No deliberate extension sidepanel parity work in this design. Shared component changes must not break the extension, but extension-specific UX follow-up belongs in a separate task.

## Coordination Constraints

This work must coordinate with existing chat cockpit/sidebar planning:

- If cockpit rails or a runtime inspector land first, the Role-play setup surface should live inside that existing structure.
- If the current composer remains the active shell, Stage 4 may use a right-side drawer on desktop and a full-height mobile sheet.
- Do not add a second permanent role-play panel that competes with cockpit rails, runtime inspector, or mobile overflow.
- Role-play controls may move, but chat transcript, composer, and send behavior must remain on the existing Playground pipeline.

## Product Direction

Use a hybrid remediation strategy:

1. Stabilize existing role-play paths first.
2. Make current state visible and reversible.
3. Restore mobile and accessibility parity.
4. Add a dedicated Role-play setup surface that reuses existing controls.
5. Present saved startup bundles as saved role-play setups where appropriate.
6. Harden compatibility and request-shaping guardrails.

This deliberately avoids building a new drawer on top of broken state. The first stages make the existing state model safer; the later stages consolidate the workflow.

## Terminology

User-facing terms should follow the existing character-chat taxonomy:

| Term | Meaning |
| --- | --- |
| Character | A saved speaking identity or card. |
| Persona | A persistent profile concept distinct from a character card. |
| Scene | Optional context layered onto character chat. |
| System prompt | Behavior instructions applied to the model. |
| Behavior template | A reusable system-prompt template. |
| Generation style | Parameter preset such as Creative, Balanced, Precise, or Custom. |
| Role-play setup | A bundle of character/persona, behavior template, scene, generation style, and optional context policy. |

Avoid using `Actor` as the primary runtime label. It may remain internal or secondary if required by existing implementation, but user-facing flow should say `Scene` or `Role-play setup`.

## Stage 1: Crash, Recovery, And Accessibility Fixes

**Goal:** Make the current role-play entry path safe before adding new product surface.

**Scope:**

- Reproduce the `Chat as a character` starter path and picker-selection crash in the current branch before changing behavior. If the crash no longer reproduces, keep the regression test and record the current behavior.
- Fix the `Chat as a character` starter path crash observed when selecting `Default Assistant` or the current equivalent default entry.
- Add regression coverage for opening the character/persona picker from the empty-state starter and selecting an entry.
- Ensure the current custom system prompt can always be edited or cleared, even when the prompt library is empty.
- Add useful accessible names to compact parameter preset controls.
- Preserve current behavior where possible; do not introduce the Role-play setup drawer in this stage.

**Success Criteria:**

- The empty-state character starter does not crash.
- A user can recover from a bad system prompt/template choice without needing saved prompts.
- Parameter presets expose meaningful labels to assistive technology and visible UI where space allows.
- Focus remains in a predictable place after opening and closing picker/modal flows.

**Likely touched areas:**

- `PlaygroundEmpty.tsx`
- `PlaygroundForm.tsx`
- `AssistantSelect.tsx`
- `PromptSelect.tsx`
- `ParameterPresets.tsx`
- focused Playground/Prompt/Assistant tests

## Stage 2: Visible State And Terminology Cleanup

**Goal:** Show what role-play state is active and make each label map to one user decision.

**Scope:**

- Preserve behavior template identity after applying a system prompt template.
- Stop collapsing applied role-play behavior to only `Custom prompt`.
- Add active-context chips for:
  - selected character/persona;
  - behavior template or custom system prompt;
  - scene context;
  - generation style;
  - role-play-relevant pinned/context state when present.
- Add clear/remove actions to chips where safe.
- Rename or clarify misleading labels:
  - `Templates` should become `System prompts` or `Behavior templates`.
  - `Character mode` should not open scene settings without making the scene mapping explicit.
  - `Preset` should not be used ambiguously for both generation parameters and saved setups.
- Update locale keys or fallbacks for user-facing terminology changes instead of hard-coding English labels.

**Success Criteria:**

- A user can look at the composer/context area and understand the active role-play state.
- Applying `Character Actor` or another role-play behavior template shows the template name until edited.
- Clearing a character, behavior template, or generation style is obvious.
- Existing character/persona selection continues to work.

**Likely touched areas:**

- active context item derivation
- system prompt template apply path
- composer toolbar labels
- mode launcher labels
- chip/removal tests

## Stage 3: Mobile Parity

**Goal:** Mobile users can find, apply, inspect, and clear role-play settings.

**Scope:**

- Add role-play behavior templates and generation style to mobile composer overflow or a compact mobile role-play entry.
- Prefer stable entry points that Stage 4 can reuse. Avoid mobile-only controls that will be deleted immediately when the Role-play setup surface lands.
- Ensure active role-play chips wrap cleanly on narrow screens.
- Preserve first-message usability; do not bury the send box under a large setup panel.
- Validate that picker, template modal, and recovery controls are reachable by keyboard/touch on mobile widths.

**Success Criteria:**

- Mobile users can apply a behavior template.
- Mobile users can choose a generation style.
- Mobile users can clear character/prompt/style state.
- No desktop-only role-play preset function remains essential to the main workflow.

**Likely touched areas:**

- `ComposerToolbar.tsx`
- `ComposerToolbarOverflow.tsx`
- responsive composer tests
- browser/mobile smoke tests

## Stage 4: Dedicated Role-play Setup Surface

**Goal:** Consolidate scattered controls into one understandable role-play workflow.

**Scope:**

Add a dedicated `Role-play setup` drawer or panel that reuses existing underlying controls:

1. Character/persona
   - choose a saved character or persona;
   - show selected name and basic identity;
   - expose clear/change.

2. Behavior
   - choose a behavior template;
   - preview the system prompt;
   - allow custom prompt editing;
   - show whether the behavior template has been modified.

3. Scene
   - optional scene fields;
   - no requirement to fill scene before role-play can start;
   - use `Scene`, not `Actor`, as the primary label.

4. Generation style
   - choose Creative, Balanced, Precise, or Custom;
   - show the parameter changes that will be applied.

5. Preview and apply
   - show before/after changes;
   - apply changes atomically to existing Playground state;
   - offer clear/revert.

**Success Criteria:**

- First-time users can start role-play from one flow without learning every underlying toolbar control.
- Returning users can open the setup surface, see current state, change one piece, and close.
- Existing advanced controls remain available but no longer carry the primary role-play path.
- The setup surface is a thin orchestration layer, not a second state system.
- Stage 3 mobile access remains valid after this surface ships, either by opening the setup surface or by keeping equivalent overflow entries.

**Likely touched areas:**

- new Role-play setup component(s)
- `AssistantSelect`
- `SystemPromptTemplates`
- scene/actor settings component path
- parameter preset component path
- Playground state integration tests

## Stage 5: Saved Role-play Setups

**Goal:** Make reusable role-play presets understandable without inventing parallel persistence.

**Scope:**

- Reuse startup template bundle fields when they can represent role-play setups:
  - model;
  - system prompt/template identity;
  - generation preset;
  - selected character/persona;
  - pinned/context state where relevant.
- Present role-play-relevant bundles as `Saved role-play setups`.
- Add exact preview before apply:
  - what character changes;
  - what behavior prompt changes;
  - what generation values change;
  - what context/pinned sources change.
- Support apply, update current setup, duplicate/save as, rename, and delete where existing storage supports it.
- Avoid auto-migrating unrelated startup templates into role-play setups unless they contain role-play-relevant fields.

A startup template bundle is role-play-relevant when it has at least one of:

- a selected character/persona;
- a behavior template categorized as role-play;
- scene settings;
- a saved custom system prompt explicitly marked or named as role-play by the user.

Generation style alone is not enough to make a startup template a role-play setup.

**Success Criteria:**

- Power users can save and reuse a role-play setup without guessing what it contains.
- Applying a saved setup previews changes before mutation.
- Deleting or renaming a saved role-play setup does not affect unrelated startup templates unexpectedly.
- No second persistence model exists unless a specific limitation is documented.

**Likely touched areas:**

- startup template bundle helpers
- startup template modal
- new saved setup list inside Role-play setup surface
- persistence tests

## Stage 6: Compatibility And Guardrails

**Goal:** Make the UI truthful about whether character context is included, blended, or excluded.

**Scope:**

- Derive compatibility status for combinations of:
  - selected character/persona;
  - custom or templated system prompt;
  - scene context;
  - RAG/pinned sources;
  - uploaded/context files;
  - compare mode;
  - docs/search modes.
- Show notices that are specific enough to be actionable:
  - `Character context included`;
  - `Character context blended with sources`;
  - `Character context excluded in this mode`;
  - `Custom prompt may override character behavior`.
- Align UI notices with the actual request-shaping/send path.
- Add tests proving the UI status matches request behavior.

**Success Criteria:**

- Users know before sending whether character context will be used.
- The UI does not claim character role-play is active when the request path excludes it.
- Compare/RAG/context conflicts have specific explanations and resolution actions.
- Request-shaping tests guard against drift.

**Likely touched areas:**

- request preview/send hooks
- role-play state adapter
- compatibility notice components
- request-shaping tests
- browser/e2e coverage

## Role-play State Adapter

Introduce a small derived-state adapter in the Playground layer.

The adapter should derive a readable `rolePlayState` from existing state:

- selected character/persona;
- system prompt content and template identity;
- scene settings;
- generation preset and parameter values;
- pinned/RAG/docs/context/compare state;
- request-path eligibility for character context.

The adapter should be derived-only at first:

- no persistence;
- no API calls;
- no direct mutation;
- no independent state copy.

Consumers:

- active-context chips;
- Role-play setup preview;
- compatibility notices;
- saved setup preview;
- tests.

This avoids duplicating Playground state while giving role-play UX one shared source of readable truth.

## Data Flow

1. User changes role-play inputs through existing controls or the Role-play setup surface.
2. Existing Playground state updates.
3. The role-play adapter derives readable state and compatibility status.
4. Active-context chips and notices render from the derived state.
5. Preview surfaces show before/after changes before apply.
6. Apply actions update existing Playground state.
7. Request preview/send logic reports whether character context is included, blended, or excluded.

## Error Handling And Reversibility

Every role-play mutation should have a visible recovery path:

- clear selected character/persona;
- clear behavior template/custom system prompt;
- reset generation style;
- clear scene context;
- revert a preview before apply;
- undo or clear a saved setup application where feasible.

Errors should be state-specific. Avoid vague notices such as "verify intended behavior" when the UI can say which behavior is at risk.

Use precise recovery language:

- `Cancel` means no state has been applied yet.
- `Revert` means restore the before-preview state after a previewed apply.
- `Clear` means remove one active role-play layer, such as character, scene, prompt, or generation style.
- `Reset` means return generation style or scene fields to their default values.

## Accessibility Requirements

- Character, behavior, scene, and generation controls need stable accessible names.
- Icon-only controls require `aria-label` and tooltip text that name the action and current state.
- Modals/drawers must trap focus, restore focus on close, and support Escape.
- Active-context chips need keyboard-accessible clear actions.
- Mobile overflow entries must not be the only accessible path for a control if they are unreachable by keyboard.

## Testing Strategy

Each stage should add tests proportional to risk:

| Stage | Required Tests |
| --- | --- |
| Stage 1 | Character starter regression, picker selection, prompt recovery, parameter preset accessibility. |
| Stage 2 | Template identity preservation, active chips, clear/remove actions, terminology-sensitive assertions where stable. |
| Stage 3 | Mobile overflow access, narrow-width chip wrapping, mobile apply/clear flows. |
| Stage 4 | Role-play setup preview/apply/clear, one-field updates, focus behavior. |
| Stage 5 | Save/apply/update/delete saved role-play setup, exact preview fields, startup template compatibility. |
| Stage 6 | Request-shaping compatibility matrix, browser/e2e smoke for included/blended/excluded states. |

Browser verification is required before declaring the workflow fixed because the original audit found a browser-observed crash.

## Backlog And PR Split

Implementation should use one parent task plus six child tasks:

1. Crash/recovery/accessibility fixes.
2. Visible state and terminology cleanup.
3. Mobile parity.
4. Role-play setup consolidation.
5. Saved role-play presets.
6. Compatibility/guardrail tests.

Each child task should be a reviewable PR-sized unit with:

- scoped acceptance criteria;
- touched files;
- focused verification commands;
- browser verification notes where relevant;
- explicit skips/blockers.

## Rollout

- Stages 1-3 can ship independently and improve the current UI.
- Stage 4 should introduce `Role-play setup` as the primary role-play entry while keeping existing controls available.
- Stage 5 should expose saved role-play setups only for bundles with role-play-relevant fields.
- Stage 6 is the reliability gate before calling the workflow stable.

## Risks

- The Role-play setup surface could become a second state system. Keep it as orchestration over existing state.
- Terminology cleanup can break tests or screenshots that assume old labels.
- Saved startup templates may contain mixed-purpose bundles; do not force all of them into role-play naming.
- Compatibility notices must match actual request behavior or they become worse than no notice.
- Mobile additions can overcrowd the composer if they are added as permanent visible controls instead of overflow/setup entries.

## Acceptance Criteria

- The staged design covers all audit findings and identified improvements.
- The plan remains scoped to the main `/chat` role-play preset workflow.
- The implementation split is reviewable as 4-6 PR-sized stages.
- The design uses existing Playground state and components where possible.
- The dedicated Role-play setup surface is introduced after stabilization, not before.
- Testing includes browser verification for the original crash path and request-state guardrails.
