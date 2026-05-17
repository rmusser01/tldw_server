# Main Chat Role-play Preset Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the main `/chat` role-play preset workflow predictable, recoverable, mobile-reachable, and truthful about character/persona context.

**Architecture:** Keep `/chat` on the existing Playground route and reuse existing character selection, system prompt templates, parameter presets, scene settings, startup template bundles, and send/preview hooks. Add a small derived role-play state layer that powers active chips, setup previews, saved setup previews, and compatibility notices without becoming a second state system. Introduce the dedicated Role-play setup surface only after the current role-play paths are stable and visibly recoverable.

**Tech Stack:** React, TypeScript, Next.js route wrappers, shared `@/components` UI package, Zustand model settings store, Plasmo storage hooks, Ant Design controls, `react-i18next`, Vitest, Testing Library, in-app browser or Playwright for final browser verification.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-17-main-chat-role-play-preset-remediation-design.md`
- Planning Backlog task: `TASK-406`
- Design Backlog task: `TASK-402`

## Scope Constraints

- Main `/chat` role-play preset workflow only.
- No new route for role-play chat.
- No deliberate extension sidepanel parity work, but shared-component changes must not break extension consumers.
- No broad chat cockpit implementation. If cockpit rails or runtime inspector already exist when Stage 4 starts, put Role-play setup inside that structure. Otherwise use a drawer/sheet.
- No backend/API change unless the frontend cannot truthfully represent role-play state or request inclusion from current contracts.
- No new saved-setup persistence model unless startup template bundles cannot support the required UX.

## File Map

### Existing Files To Modify

- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
  - Keep `Chat as a character` starter dispatch.
  - Add or preserve test identifiers/labels needed for regression tests.

- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - Wire role-play state adapter.
  - Preserve template identity after template apply.
  - Open Role-play setup surface in Stage 4.
  - Pass mobile/desktop role-play entry props through composer controls.

- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
  - Fix current starter selection crash if reproduced here.
  - Preserve character/persona tab semantics and focus behavior.

- `apps/packages/ui/src/components/Common/PromptSelect.tsx`
  - Always expose current system prompt edit/clear recovery when a custom prompt exists, even with an empty prompt library.

- `apps/packages/ui/src/components/Option/Playground/ParameterPresets.tsx`
  - Add accessible names and compact-mode text/fallbacks.
  - Preserve `detectCurrentPreset` and `getPresetByKey` contracts.

- `apps/packages/ui/src/components/Option/Playground/SystemPromptTemplates.tsx`
  - Export template metadata or a pure lookup helper.
  - Preserve selected template identity after apply.
  - Rename/fallback labels through i18n, not hard-coded English.

- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts`
  - Render role-play active chips from the derived role-play state.
  - Keep pinned/context as summary and compatibility state only.

- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
  - Add desktop/mobile entry points for behavior templates, generation style, and Role-play setup.

- `apps/packages/ui/src/components/Option/Playground/ComposerToolbarOverflow.tsx`
  - Add mobile-reachable role-play entries that Stage 4 can reuse.

- `apps/packages/ui/src/components/Option/Playground/PlaygroundModeLauncher.tsx`
  - Clarify `Character mode`/scene terminology without creating a new workflow.

- `apps/packages/ui/src/components/Option/Playground/PlaygroundComposerNotices.tsx`
  - Keep startup template controls working.
  - Add saved role-play setup affordances only when Stage 5 starts.

- `apps/packages/ui/src/components/Option/Playground/PlaygroundStartupTemplateModal.tsx`
  - Extend preview fields for saved role-play setup apply.

- `apps/packages/ui/src/components/Option/Playground/startup-template-bundles.ts`
  - Reuse bundle shape for role-play setup eligibility and save/apply previews.

- `apps/packages/ui/src/components/Option/Playground/hooks/usePromptTemplates.ts`
  - Reuse startup template persistence for saved role-play setup operations.

- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts`
  - Use shared compatibility/request-inclusion helper in Stage 6.

- Actor/scene sources:
  - `apps/packages/ui/src/types/actor.ts`
  - `apps/packages/ui/src/utils/actor.ts`
  - `apps/packages/ui/src/store/actor.tsx`
  - `apps/packages/ui/src/services/actor-settings.ts`
  - `apps/packages/ui/src/components/Common/Settings/ActorPopout.tsx`
  - `apps/packages/ui/src/components/Common/Settings/ActorEditor.tsx`

- Locale files:
  - `apps/packages/ui/src/public/_locales/en/playground.json`
  - `apps/packages/ui/src/assets/locale/en/playground.json`
  - Other locale files only if the repo pattern requires mirrored keys.

### New Files To Create

- `apps/packages/ui/src/components/Option/Playground/role-play-state.ts`
  - Pure derived role-play state helpers.
  - No React, no storage, no API calls.

- `apps/packages/ui/src/components/Option/Playground/role-play-compatibility.ts`
  - Pure compatibility/request-inclusion helpers.
  - Stage 6 may fold this into `role-play-state.ts` if it stays small.

- `apps/packages/ui/src/components/Option/Playground/role-play-scene.ts`
  - Pure adapters for summarizing Actor scene settings, creating before/after scene previews, and identifying clear/reset changes.

- `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
  - Stage 4 setup surface.
  - Reuses existing controls and derived state.

- `apps/packages/ui/src/components/Option/Playground/RolePlaySetupPreview.tsx`
  - Compact before/after preview shared by setup and saved setup flows.

- `apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx`
  - Stage 5 saved setup list and actions.

### New Or Modified Tests

- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx`
- `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`
- `apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts`
- `apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts`
- `apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts`
- `apps/packages/ui/src/components/Option/Playground/__tests__/role-play-compatibility.test.ts`
- Existing startup template tests:
  - `apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.integration.test.ts`
  - `apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.prompt-mapping.test.ts`

## Task 0: Execution Setup And Current-State Verification

**Files:**
- Backlog tasks only unless the implementation branch/worktree setup requires a plan note.

- [ ] **Step 1: Verify branch and decide whether to isolate work**

Run:
```bash
git branch --show-current
git status --short
```

Expected:
- You know whether you are on `codex/chat-sidebar-tools-first`, `dev`, or a dedicated role-play branch.
- If the branch contains unrelated sidebar/cockpit work, create or switch to a dedicated role-play branch before code changes.

- [ ] **Step 2: Create implementation Backlog tasks**

Use the Backlog MCP or CLI to create:
- parent task: `Implement main chat role-play preset remediation`
- child task 1: `Fix chat role-play starter crash, prompt recovery, and preset accessibility`
- child task 2: `Implement chat role-play visible state and terminology cleanup`
- child task 3: `Restore chat role-play mobile preset parity`
- child task 4: `Add main chat Role-play setup surface`
- child task 5: `Add saved role-play setup UX`
- child task 6: `Add role-play compatibility and request-inclusion guardrails`

Expected:
- Each code-edit stage has an associated Backlog task before files are changed.

- [ ] **Step 3: Install or verify frontend dependencies**

Run from repo root or `apps/tldw-frontend` as appropriate:
```bash
bun install
```

Expected:
- Dependencies are available.
- If network/sandbox blocks install, request approval and document the result in the active Backlog task.

- [ ] **Step 4: Reproduce or retire the observed crash**

Start the frontend:
```bash
cd apps/tldw-frontend
bun run dev -- -p 3000
```

Browser path:
1. Open `http://127.0.0.1:3000/chat`.
2. Click `Chat as a character`.
3. Select `Default Assistant` or the current default/equivalent entry.
4. Record whether the route crashes.

Expected:
- If it crashes, capture the console error and stack.
- If it does not crash, keep the regression test and record "not currently reproducible" in the child task.

- [ ] **Step 5: Commit only task setup if task files changed**

Replace the path placeholder with the exact Backlog task files created in Step 2. Do not stage unrelated existing Backlog changes.

```bash
git status --short backlog/tasks
git add <exact Backlog task files created in Step 2>
git commit -m "chore: track chat role-play preset remediation tasks"
```

## Task 1: Stage 1 Crash, Recovery, And Accessibility Fixes

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify as needed: `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptSelect.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/ParameterPresets.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts`

- [x] **Step 1: Write the failing starter regression test**

Add `PlaygroundForm.role-play-starter.integration.test.tsx`.

Test cases:
- Dispatching or clicking the `Chat as a character` starter opens `AssistantSelect` on the character tab.
- Selecting the default/equivalent entry closes the picker without throwing.
- Focus returns to a stable composer or trigger control after close.

Example assertion shape:
```ts
expect(screen.getByRole("button", { name: /select character/i })).toBeInTheDocument()
await user.click(screen.getByRole("button", { name: /default assistant|helpful ai assistant/i }))
expect(screen.queryByText(/something went wrong/i)).not.toBeInTheDocument()
```

- [x] **Step 2: Run the starter test and verify failure or current pass**

Run:
```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx --reporter=verbose
```

Expected:
- Fails if the crash or missing test harness is active.
- If it passes before implementation, record that the browser crash is not currently reproduced by unit coverage and keep the test.

- [x] **Step 3: Fix the starter/picker crash minimally**

Only change the path that causes the update loop.

Likely fix areas:
- `AssistantSelect.tsx` selection state effects.
- `PlaygroundForm.tsx` starter event handler and picker open/close state.
- Header/drawer interactions if the crash stack still points there.

Do not add Role-play setup or state adapter in this task.

- [x] **Step 4: Write prompt recovery tests**

Extend `PromptSelect.system-prompt-modal.test.tsx`.

Test cases:
- With empty prompt library and non-empty `systemPrompt`, menu still exposes `Edit current system prompt`.
- Editing and saving calls `setSystemPrompt` with the draft.
- Clearing a custom prompt calls `setSystemPrompt("")`.
- Empty prompt library can still render `No saved prompts` without hiding recovery actions.

- [x] **Step 5: Implement prompt recovery**

In `PromptSelect.tsx`:
- avoid returning early before appending current-prompt recovery actions;
- add a current prompt item when `systemPrompt.trim().length > 0`;
- keep saved prompt behavior unchanged when prompt data exists.

Implementation shape:
```ts
const hasCurrentSystemPrompt = String(systemPrompt || "").trim().length > 0
const recoveryItems = hasCurrentSystemPrompt ? [editCurrentPromptItem, clearCurrentPromptItem] : []
const menuItems = [...promptItems, ...recoveryItems]
```

- [x] **Step 6: Write parameter preset accessibility tests**

Extend `ParameterPresets.guard.test.ts`.

Test cases:
- Compact preset control has an accessible label such as `Generation style`.
- Each preset option has an accessible name: Creative, Balanced, Precise, Custom.
- Tooltip/detail rows still include temperature/top-p/top-k values.

- [x] **Step 7: Implement parameter preset labels**

In `ParameterPresets.tsx`:
- add `aria-label={t("playground:presets.ariaLabel", "Generation style")}` to the segmented control or wrapper;
- include visually hidden preset text in compact labels if Ant Design strips accessible names;
- keep visible labels in non-compact mode.

- [x] **Step 8: Run Stage 1 focused tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts
```

Expected: pass.

- [x] **Step 9: Browser verify Stage 1**

Verify on `/chat`:
- `Chat as a character` opens the selector.
- selecting the default/equivalent entry does not crash.
- an applied custom prompt can be edited/cleared with an empty prompt library.
- compact generation presets are understandable by keyboard/screen-reader inspection.

- [x] **Step 10: Commit Stage 1**

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx \
  apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx \
  apps/packages/ui/src/components/Common/AssistantSelect.tsx \
  apps/packages/ui/src/components/Common/PromptSelect.tsx \
  apps/packages/ui/src/components/Option/Playground/ParameterPresets.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts
git commit -m "fix: stabilize chat role-play starter controls"
```

## Task 2: Stage 2 Visible State, Template Identity, And Terminology

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/role-play-state.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/SystemPromptTemplates.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundModeLauncher.tsx`
- Modify: `apps/packages/ui/src/public/_locales/en/playground.json`
- Modify: `apps/packages/ui/src/assets/locale/en/playground.json`

- [x] **Step 1: Write role-play state tests**

Create `role-play-state.test.ts`.

Test cases:
- no character, prompt, scene, or preset yields inactive role-play state;
- selected character yields a character layer;
- selected persona yields a persona layer distinct from character;
- applied role-play template yields behavior template layer with template id/title;
- edited prompt after template apply marks behavior as `modified`;
- non-custom generation preset yields generation style layer;
- clearing generation style resets to `Balanced`;
- pinned/context state is summarized but does not expose source management actions.

- [x] **Step 2: Implement derived state helper**

Create `role-play-state.ts`.

Start with pure types:
```ts
export type RolePlayIdentityKind = "character" | "persona" | "assistant"
export type RolePlayPromptSource = "none" | "template" | "custom" | "modified-template"

export type RolePlayState = {
  active: boolean
  identity: { kind: RolePlayIdentityKind; id?: string; name?: string } | null
  behavior: { source: RolePlayPromptSource; templateId?: string; title?: string; modified: boolean } | null
  scene: { active: boolean; summary?: string } | null
  generationStyle: { key: string; label: string } | null
  context: { pinnedCount: number; hasExternalContext: boolean }
}
```

Rules:
- helper is derived-only;
- no React imports;
- no storage;
- no API calls.

- [x] **Step 3: Export template lookup metadata**

In `SystemPromptTemplates.tsx`:
- export `PROMPT_TEMPLATES` or add `getPromptTemplateById(id)`;
- preserve icons as UI-only where practical;
- expose category/title/content for role-play state and saved setup detection.

- [x] **Step 4: Preserve behavior template identity after apply**

In `PlaygroundForm.tsx`:
- when `SystemPromptTemplatesModal` applies a template, store template id/title/category alongside `systemPrompt`;
- if the prompt text changes after apply, derive `modified-template`;
- if the prompt clears, clear template identity.

Do not convert this into a backend object.

- [x] **Step 5: Write active chip tests**

Extend or create tests around `usePlaygroundContextItems`.

Test cases:
- applied `Character Actor` shows `Behavior: Character Actor`, not just `Prompt: Custom prompt`;
- custom prompt shows `System prompt: Custom`;
- modified template shows `Character Actor modified`;
- character chip remove calls the existing selected-character clear path;
- generation style chip clear/reset uses `Balanced`.

- [x] **Step 6: Render role-play chips from adapter**

In `usePlaygroundContextItems.ts`:
- accept `rolePlayState` or input fields needed to derive it;
- render separate chips for identity, behavior, scene, and generation style;
- keep pinned source chip as summary only;
- add safe clear/remove actions where handlers already exist.
- when clearing generation style, apply `getPresetByKey("balanced")` settings.

- [x] **Step 7: Clarify labels and update i18n**

Update user-facing copy through locale keys:
- `Templates` -> `System prompts` or `Behavior templates`;
- `Parameter Presets` -> `Generation style` where exposed to role-play users;
- `Character mode` -> `Character / Scene` or equivalent that does not imply scene equals character selection.

Update:
```bash
apps/packages/ui/src/public/_locales/en/playground.json
apps/packages/ui/src/assets/locale/en/playground.json
```

- [x] **Step 8: Run Stage 2 focused tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.composer-options.guard.test.ts
```

Expected: pass.

- [x] **Step 9: Browser verify Stage 2**

On `/chat`:
- apply `Character Actor`;
- active context names it as a behavior template;
- edit prompt and verify modified state appears;
- clear behavior without clearing character;
- verify changed labels are not hard-coded English in components.

Result:
- Focused tests, locale parsing, and diff whitespace checks were completed.
- Browser verification was attempted, but the in-app browser policy blocked `http://127.0.0.1:3001/chat` and explicitly prohibited raw CDP or alternate-browser workarounds for that target.

- [x] **Step 10: Commit Stage 2**

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/role-play-state.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts \
  apps/packages/ui/src/components/Option/Playground/SystemPromptTemplates.tsx \
  apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx \
  apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts \
  apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx \
  apps/packages/ui/src/components/Option/Playground/PlaygroundModeLauncher.tsx \
  apps/packages/ui/src/public/_locales/en/playground.json \
  apps/packages/ui/src/assets/locale/en/playground.json
git commit -m "feat: show active chat role-play state"
```

## Task 3: Stage 3 Mobile Parity For Current Controls

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/ComposerToolbarOverflow.tsx`
- Modify as needed: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts`

- [x] **Step 1: Write mobile parity tests**

Create `ComposerToolbar.role-play-mobile.test.tsx`.

Test cases:
- at mobile/casual configuration, overflow includes `System prompts` or `Behavior templates`;
- overflow includes `Generation style`;
- active role-play chips expose clear/change actions without requiring desktop-only controls;
- entries use callbacks that Stage 4 can reuse to open Role-play setup.

- [x] **Step 2: Implement reusable mobile entry callbacks**

In `ComposerToolbar.tsx`:
- pass `onOpenSystemPrompts`, `onOpenGenerationStyle`, and future `onOpenRolePlaySetup` callbacks through one shared prop shape;
- avoid mobile-only state that will be deleted in Stage 4.

- [x] **Step 3: Add role-play entries to overflow**

In `ComposerToolbarOverflow.tsx`:
- add `System prompts`/`Behavior templates`;
- add `Generation style`;
- make both keyboard reachable and labelled;
- keep composer send box reachable.

- [x] **Step 4: Run mobile tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts
```

Expected: pass.

- [x] **Step 5: Browser verify mobile**

At 390px wide viewport on `/chat`:
- find and open system prompts/behavior templates;
- choose a role-play template;
- open generation style;
- clear active role-play state;
- verify composer remains visible.

Result:
- Focused Stage 3 tests, existing responsive/layout guards, locale parsing, and diff whitespace checks were completed.
- Browser verification was blocked by the same in-app browser target policy for `http://127.0.0.1:3001/chat`; raw CDP and alternate browser surfaces were not used because the policy explicitly prohibited routing around the blocked target.

- [x] **Step 6: Commit Stage 3**

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx \
  apps/packages/ui/src/components/Option/Playground/ComposerToolbarOverflow.tsx \
  apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/mobile-composer-layout.test.ts
git commit -m "feat: expose chat role-play presets on mobile"
```

## Task 4: Stage 4 Dedicated Role-play Setup Surface

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- Create: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupPreview.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/ComposerToolbarOverflow.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
- Modify as needed: `apps/packages/ui/src/components/Option/Playground/role-play-state.ts`
- Create: `apps/packages/ui/src/components/Option/Playground/role-play-scene.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/role-play-scene.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx`

- [x] **Step 1: Write setup drawer tests**

Create `RolePlaySetupDrawer.test.tsx`.

Test cases:
- renders current character/persona state;
- renders behavior template/custom prompt state;
- renders scene as optional;
- loads scene state from current Actor settings through `useActorStore` and `actor-settings` service helpers;
- renders generation style and exact changed parameter values;
- previews scene summary from `buildActorPrompt` and `estimateActorTokens`;
- preview apply calls one `onApply` with selected changes;
- cancel closes without calling `onApply`;
- clear actions remove one layer only;
- scene clear disables Actor and clears notes/aspects for the previewed setup only;
- scene reset restores `createDefaultActorSettings()`;
- Escape closes and focus returns to trigger.

- [x] **Step 2: Write scene adapter tests**

Create `role-play-scene.test.ts`.

Test cases:
- `createDefaultActorSettings()` summarizes as inactive/empty scene;
- enabled settings with notes/aspects summarize as active scene;
- GM-only notes are not included in the prompt preview;
- clear scene returns disabled settings with empty notes/aspects values;
- reset scene returns `createDefaultActorSettings()`;
- preview text uses `buildActorPrompt(settings)` and token count uses `estimateActorTokens`.

- [x] **Step 3: Implement scene adapter**

Create `role-play-scene.ts`.

Use existing types/helpers:
```ts
import type { ActorSettings } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"
import { buildActorPrompt, estimateActorTokens } from "@/utils/actor"
```

Export pure helpers:
```ts
export type RolePlayScenePreview = {
  active: boolean
  summary: string
  prompt: string
  tokenCount: number
}

export function summarizeRolePlayScene(settings: ActorSettings | null): RolePlayScenePreview
export function clearRolePlayScene(settings: ActorSettings | null): ActorSettings
export function resetRolePlayScene(): ActorSettings
```

- [x] **Step 4: Implement preview component**

Create `RolePlaySetupPreview.tsx`.

Props:
```ts
type RolePlaySetupPreviewProps = {
  before: RolePlayState
  after: RolePlayState
  onRevert?: () => void
}
```

Keep it presentational. No store access.

- [x] **Step 5: Implement setup drawer**

Create `RolePlaySetupDrawer.tsx`.

Rules:
- use existing `AssistantSelect` or selector trigger for character/persona;
- use existing `SystemPromptTemplatesModal` or extracted template selector for behavior;
- read current scene from `useActorStore().settings` and load persisted per-chat settings through `actor-settings` service helpers when the drawer opens;
- keep a local scene draft in the drawer until the user applies;
- use `ActorEditor` or a compact scene summary/editor, labelled `Scene`, backed by `ActorSettings`;
- preview scene using `summarizeRolePlayScene(sceneDraft)`;
- apply scene by calling the same save path used by `ActorPopout` for the active chat key;
- clear scene by applying `clearRolePlayScene(sceneDraft)`;
- reset scene by applying `resetRolePlayScene()`;
- use `ParameterPresets` or preset metadata for generation style;
- show preview before apply;
- do not duplicate source-of-truth state.

- [x] **Step 6: Wire setup drawer into PlaygroundForm**

In `PlaygroundForm.tsx`:
- add `rolePlaySetupOpen` local state;
- pass current derived state to drawer;
- pass current `historyId`/`serverChatId` or whatever key `actor-settings` needs to load/save current chat scene settings;
- apply confirmed changes through existing setters;
- do not change send behavior.

- [x] **Step 7: Reuse Stage 3 mobile entry**

In `ComposerToolbar.tsx` and `ComposerToolbarOverflow.tsx`:
- `Role-play setup` entry opens the drawer/sheet;
- Stage 3 mobile template/generation entries either remain equivalent shortcuts or route into the setup surface.

- [x] **Step 8: Run setup tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-scene.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx
```

Expected: pass.

Recorded verification:
- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/role-play-scene.test.ts ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx --reporter=verbose` passed: 4 files, 37 tests.
- Follow-up CDP regression pass found and fixed a null `documentContext` crash, missing desktop setup trigger, mobile overflow first-click failure, and chat header horizontal overflow. `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/role-play-scene.test.ts ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx --reporter=verbose` passed: 6 files, 53 tests.
- `node -e "JSON.parse(require('fs').readFileSync('../packages/ui/src/assets/locale/en/playground.json','utf8')); JSON.parse(require('fs').readFileSync('../packages/ui/src/public/_locales/en/playground.json','utf8')); console.log('json ok')"` passed.
- `git diff --check` passed.
- `bunx tsc --noEmit --pretty false` still fails only on the existing unrelated baseline errors in `EmbeddingsModelSelectionConfig.tsx`, `persona-visuals.ts`, and `lib/api/vnPlay.ts`; no new Stage 4 type errors were reported.

- [x] **Step 9: Browser verify setup flow**

Recorded status: verified through CDP after explicit user override. CDP connected to `Chrome/145.0.7632.6` with `webSocketDebuggerUrl` present and verified `/chat` on `http://127.0.0.1:3001/chat` with seeded single-user API config. Computer Use was not used.

Desktop:
- direct Role-play setup trigger is present;
- Role-play setup opens the drawer;
- choose character/persona;
- choose behavior template;
- add scene context;
- choose generation style;
- preview and apply.

Mobile:
- More options opens on the first click while the composer textarea is focused;
- Role-play setup is present in overflow;
- setup drawer exposes preview, Character, Behavior, Scene, Generation style, Apply, and Cancel;
- `/chat` at 390px has `scrollWidth` equal to `clientWidth` and no horizontal overflow offenders.

- [x] **Step 10: Commit Stage 4**

Committed with message `feat: add chat role-play setup surface`.

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx \
  apps/packages/ui/src/components/Option/Playground/RolePlaySetupPreview.tsx \
  apps/packages/ui/src/components/Option/Playground/role-play-scene.ts \
  apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx \
  apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx \
  apps/packages/ui/src/components/Option/Playground/ComposerToolbarOverflow.tsx \
  apps/packages/ui/src/components/Option/Playground/role-play-state.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-scene.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx
git commit -m "feat: add chat role-play setup surface"
```

## Task 5: Stage 5 Saved Role-play Setups

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/startup-template-bundles.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePromptTemplates.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundStartupTemplateModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupPreview.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.integration.test.ts`

- [x] **Step 1: Write saved setup helper tests**

Create `saved-role-play-setups.test.ts`.

Test cases:
- bundle saved from Role-play setup is role-play relevant;
- selected character/persona makes a bundle relevant;
- role-play behavior template makes a bundle relevant;
- generation style alone does not make a bundle relevant;
- template name substring alone does not make a bundle relevant;
- preview includes character, behavior, generation values, and context counts.

- [x] **Step 2: Extend startup bundle metadata minimally**

In `startup-template-bundles.ts`:
- add optional source marker such as `source: "startup-template" | "role-play-setup"` if compatible with existing storage;
- add structured role-play metadata, for example:
  ```ts
  rolePlay?: {
    source: "role-play-setup"
    identity: { kind: "character" | "persona"; id: string | number; name: string } | null
    behavior: {
      source: "template" | "custom" | "modified-template"
      templateId: string | null
      templateTitle: string | null
      templateCategory: string | null
      systemPrompt: string
      modified: boolean
    } | null
    scene: ActorSettings | null
    generation: {
      presetKey: PresetKey
      settings: Partial<ChatModelSettings>
    }
    context: {
      ragPinnedCount: number
      ragPinnedResultIds: string[]
    }
  }
  ```
- add pure `isRolePlayRelevantBundle(bundle, promptLibrary?)`;
- add pure `describeRolePlaySetupPreview(bundle, promptLibrary?)`.
- normalize this metadata defensively in `normalizeStartupTemplateBundle`.
- keep existing top-level `character`, `systemPrompt`, `presetKey`, and `ragPinnedResults` for backward compatibility and generic startup template behavior.

Do not break existing startup template tests.

- [x] **Step 3: Add saved setup panel**

Create `SavedRolePlaySetupsPanel.tsx`.

MVP actions:
- save current role-play state;
- preview/apply;
- rename;
- delete.

Do not add update-current or duplicate unless existing storage already makes it cheap.

- [x] **Step 4: Wire panel into Role-play setup drawer**

In `RolePlaySetupDrawer.tsx`:
- show saved setups when role-play-relevant bundles exist;
- keep ordinary startup templates out of this list;
- use `RolePlaySetupPreview` before apply.

- [x] **Step 5: Update startup template modal preview**

In `PlaygroundStartupTemplateModal.tsx`:
- show exact role-play fields when previewing a role-play-relevant bundle;
- keep generic startup template preview behavior for other bundles.

- [x] **Step 6: Run saved setup tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.integration.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.prompt-mapping.test.ts
```

Expected: pass.

Recorded verification:
- Red helper test run failed before implementation on missing role-play bundle helpers and metadata, then passed after implementation.
- `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts ../packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.integration.test.ts ../packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.prompt-mapping.test.ts ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx` passed: 4 files, 23 tests.
- `bunx tsc --noEmit --pretty false` still fails only on existing unrelated baseline errors in `EmbeddingsModelSelectionConfig.tsx`, `persona-visuals.ts`, and `lib/api/vnPlay.ts`; no new Stage 5 type errors were reported.

- [x] **Step 7: Browser verify saved setups**

On `/chat`:
- create a role-play setup with character, behavior, scene, generation style;
- save it;
- rename it;
- preview/apply it;
- delete it;
- verify unrelated startup templates do not appear as role-play setups.

Recorded status: verified through CDP. CDP connected to Chrome on `127.0.0.1:9222` and verified `/chat` at `http://127.0.0.1:3001/chat` with seeded single-user API config. Computer Use was not used.

CDP verification covered:
- generic startup template remains stored but hidden from the saved role-play setup list;
- role-play setup saves with `source: "role-play-setup"`;
- saved setup captures Character Actor behavior, Precise generation, and enabled scene state;
- preview shows role-play fields;
- applying from the preview modal persists the saved scene;
- rename, apply, and delete complete.

- [x] **Step 8: Commit Stage 5**

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx \
  apps/packages/ui/src/components/Option/Playground/startup-template-bundles.ts \
  apps/packages/ui/src/components/Option/Playground/hooks/usePromptTemplates.ts \
  apps/packages/ui/src/components/Option/Playground/PlaygroundStartupTemplateModal.tsx \
  apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx \
  apps/packages/ui/src/components/Option/Playground/RolePlaySetupPreview.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.integration.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.prompt-mapping.test.ts
git commit -m "feat: save chat role-play setups"
```

Committed with message `feat: save chat role-play setups`.

## Task 6: Stage 6 Compatibility And Request-Inclusion Guardrails

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/role-play-compatibility.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/role-play-state.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/role-play-compatibility.test.ts`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx`
- Test as needed: `apps/packages/ui/src/components/Option/Playground/__tests__/compare-interoperability.test.ts`

- [ ] **Step 1: Write compatibility tests**

Create `role-play-compatibility.test.ts`.

Matrix:
- character selected, no RAG/docs/compare/context files -> `included`;
- persona selected, no RAG/docs/compare/context files -> explicit persona status;
- character plus custom prompt -> `blended` or `override-risk`;
- character plus pinned RAG -> `blended`;
- character plus selected knowledge -> exact current behavior;
- character plus file-retrieval RAG with scoped `ragMediaIds` -> exact current behavior;
- character plus `contextFiles` -> `excluded`;
- character plus selected documents or document context -> `excluded`;
- character plus docs/search modes -> exact current behavior;
- character plus image command -> `excluded`;
- character plus compare mode -> exact current behavior, either `excluded` or `included-in-compare`;
- character plus uploaded docs/context files -> exact current behavior;
- no character/persona -> `none`.

- [ ] **Step 2: Implement pure compatibility helper**

Create `role-play-compatibility.ts`.

Type shape:
```ts
export type RolePlayContextStatus = "none" | "included" | "blended" | "excluded" | "override-risk"

export type RolePlayCompatibility = {
  status: RolePlayContextStatus
  reasonCode:
    | "no_identity"
    | "character_flow"
    | "persona_flow"
    | "custom_prompt"
    | "rag_sources"
    | "compare_mode"
    | "context_files"
  messageKey: string
}
```

Input should mirror the current `usePlaygroundRawPreview` eligibility checks:
```ts
type RolePlayCompatibilityInput = {
  hasCharacter: boolean
  hasPersona: boolean
  compareModeActive: boolean
  isImageCommand: boolean
  hasContextFiles: boolean
  hasSelectedDocuments: boolean
  hasDocumentContext: boolean
  hasSelectedKnowledge: boolean
  fileRetrievalEnabled: boolean
  hasScopedRagMediaIds: boolean
  ragPinnedResultsLength: number
  hasCustomPrompt: boolean
}
```

- [ ] **Step 3: Replace duplicate eligibility logic**

In `usePlaygroundRawPreview.ts`:
- keep request behavior unchanged unless tests prove current behavior is wrong;
- use the helper to derive status that the UI can display;
- avoid changing endpoint selection as part of this task unless the UI currently lies.

- [ ] **Step 4: Show actionable notices**

In setup drawer and/or active context chips:
- show `Character context included`;
- show `Character context blended with sources`;
- show `Character context excluded in this mode`;
- show `Custom prompt may override character behavior`.

Each notice needs a resolution action when possible:
- clear custom prompt;
- clear pinned sources;
- turn off compare mode;
- open Role-play setup.

- [ ] **Step 5: Run compatibility tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-compatibility.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/compare-interoperability.test.ts
```

Expected: pass.

- [ ] **Step 6: Browser verify compatibility states**

On `/chat`, verify at least:
- character-only says included;
- character plus custom prompt says override/blended risk;
- character plus pinned sources says blended;
- character plus compare mode says the actual request behavior;
- persona path has its own status and does not inherit character-only copy.

- [ ] **Step 7: Commit Stage 6**

```bash
git add \
  apps/packages/ui/src/components/Option/Playground/role-play-compatibility.ts \
  apps/packages/ui/src/components/Option/Playground/role-play-state.ts \
  apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts \
  apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx \
  apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundContextItems.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-compatibility.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/compare-interoperability.test.ts
git commit -m "feat: clarify chat role-play context inclusion"
```

## Task 7: Final Verification And Closeout

**Files:**
- Backlog task updates.
- No code files unless verification reveals defects.

- [ ] **Step 1: Run focused role-play suite**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx \
  apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-state.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/role-play-compatibility.test.ts
```

Expected: pass.

- [ ] **Step 2: Run broader affected Playground tests**

Run:
```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.integration.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/startup-template-bundles.prompt-mapping.test.ts \
  apps/packages/ui/src/components/Option/Playground/__tests__/compare-interoperability.test.ts
```

Expected: pass.

- [ ] **Step 3: Run frontend lint**

Run:
```bash
cd apps/tldw-frontend
bun run lint
```

Expected: pass, or document pre-existing lint failures separately from role-play changes.

- [ ] **Step 4: Run frontend build or compile if feasible**

Run one of:
```bash
cd apps/tldw-frontend
bun run build
```

or, if Turbopack build is too expensive in the local environment:
```bash
cd apps/tldw-frontend
bun run compile
```

Expected: pass, or document environment/pre-existing failure.

- [ ] **Step 5: Run browser verification**

Start frontend:
```bash
cd apps/tldw-frontend
bun run dev -- -p 3000
```

Desktop verification:
- first-time user can discover role-play from empty state;
- returning user can switch character/template/generation style;
- power user can edit, clear, save, apply, rename, and delete setup;
- bad setup choice can be reverted or cleared;
- active chips make state obvious.

Mobile verification:
- role-play setup is reachable;
- templates and generation style are reachable;
- composer remains usable;
- chips wrap without overlap.

Compatibility verification:
- included, blended, excluded, and override-risk states match actual request behavior.

- [ ] **Step 6: Security validation**

This plan is frontend TypeScript/React only. Bandit does not apply unless a Python file is touched unexpectedly.

If Python files are touched:
```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_chat_role_play_preset.json
```

If no Python files are touched, record Bandit skip in Backlog final summaries.

- [ ] **Step 7: Update Backlog tasks**

For each implementation child task:
- mark status;
- add touched files;
- add focused test commands and browser verification notes;
- add skips/blockers.

- [ ] **Step 8: Final commit or PR handoff**

If all stages are already committed, create or update the PR. The PR body must preserve the human-owned `Change summary` placeholder required by repo policy.

Do not fabricate the human `Change summary`.

## Execution Order

Recommended order:

1. Task 0
2. Task 1
3. Task 2
4. Task 3
5. Task 4
6. Task 5
7. Task 6
8. Task 7

Do not start Task 4 before Tasks 1-3 pass in browser. The setup surface should consolidate stable behavior, not hide broken behavior.

## Known Planning Risks

- The active branch may contain chat sidebar/cockpit work. Verify branch isolation before code edits.
- The original browser crash may no longer reproduce. Keep the regression test anyway.
- Ant Design `Segmented` may not expose option accessible names from custom React labels. Test with DOM/accessibility queries, not visual assumptions.
- Template identity can become false after prompt edits. Use `modified-template` state.
- Saved startup templates may be mixed-purpose. Only role-play-relevant bundles belong in saved role-play setup UI.
- Request-inclusion status must match actual send path. A pretty notice that lies is worse than no notice.
