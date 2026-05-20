# Character Chat Phase 3 Setup Safety And Accessibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining Phase 3 PRD gaps in the existing Role-play setup surface: safe saved-setup deletion and valid generation-style radio semantics.

**Architecture:** Reuse the shipped `RolePlaySetupDrawer` and `SavedRolePlaySetupsPanel`; do not introduce a new role-play state system or persistence model. Keep deletion safety local to the saved setup panel and keep generation style staging inside the existing drawer payload path.

**Tech Stack:** React, TypeScript, Ant Design buttons/inputs, Testing Library, Vitest, existing Playground role-play setup helpers.

---

## Source Context

- PRD: `Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md`
- Backlog: `TASK-447`
- Current implementation:
  - `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
  - `apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx`
  - `apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx`

## Scope

- In scope:
  - Saved role-play setup delete confirmation or undo.
  - Accessible radio semantics for the Role-play setup generation style selector.
  - Focused component tests.
  - Backlog closeout notes.
- Out of scope:
  - New Character Chat session rail.
  - Extension sidepanel parity.
  - Command palette shortcuts.
  - Backend or persistence schema changes.
  - Redesigning the existing Role-play setup drawer.

## Task 1: Saved Setup Delete Safety

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx`

- [x] **Step 1: Add a failing test for delete confirmation**

Add a test that renders one saved role-play setup, clicks `Delete`, verifies the setup is not deleted immediately, then clicks a confirmation button and verifies `onDeleteSavedSetup` is called.

Expected assertion shape:

```ts
fireEvent.click(screen.getByRole("button", { name: "Delete Mira detective scene" }))
expect(onDeleteSavedSetup).not.toHaveBeenCalled()
expect(screen.getByRole("status")).toHaveTextContent("Delete Mira detective scene?")
fireEvent.click(screen.getByRole("button", { name: "Confirm delete Mira detective scene" }))
expect(onDeleteSavedSetup).toHaveBeenCalledWith("saved-role-play")
```

- [x] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx --reporter=verbose
```

Expected: the new test fails because delete currently calls `onDeleteSetup` immediately.

- [x] **Step 3: Implement local pending-delete confirmation**

In `SavedRolePlaySetupsPanel.tsx`:

- add `pendingDeleteId` state;
- first click sets `pendingDeleteId`;
- render inline confirmation copy and `Confirm delete` / `Cancel` buttons for that setup only;
- confirmation calls `onDeleteSetup(setup.id)` and clears pending state;
- cancel clears pending state;
- changing the setup list should clear stale pending ids.

Keep copy routed through `t(...)` fallback calls.

- [x] **Step 4: Re-run the focused test**

Expected: the delete confirmation test passes.

## Task 2: Generation Style Radio Semantics

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx`

- [x] **Step 1: Add a failing accessibility test**

Add a test that locates the `Generation style` radiogroup in the Role-play setup drawer and verifies each preset is exposed as a `radio`, with the active preset checked.

Expected assertion shape:

```ts
const group = screen.getByRole("radiogroup", { name: "Generation style" })
expect(within(group).getByRole("radio", { name: "Creative" })).toBeChecked()
expect(within(group).getByRole("radio", { name: "Precise" })).not.toBeChecked()
```

- [x] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx --reporter=verbose
```

Expected: the new test fails because the current children are buttons with `aria-pressed`.

- [x] **Step 3: Implement radio semantics without changing staged behavior**

In `RolePlaySetupDrawer.tsx`, change each generation style option from a pressed button to a radio input wrapped in a label or a button with `role="radio"` and `aria-checked`.

Preferred implementation:

- render a visually-hidden `input type="radio"` per preset;
- keep the existing visual card styling on the label;
- set `name="role-play-generation-style"`;
- use `checked={selected}`;
- call `selectGenerationPreset(preset.key)` on change.

- [x] **Step 4: Re-run focused tests**

Expected: Role-play setup tests pass.

## Task 3: Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-447 - Implement-Character-Chat-Phase-3-role-play-setup-safety-and-accessibility.md`

- [x] **Step 1: Run focused role-play setup tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/saved-role-play-setups.test.ts --reporter=verbose
```

Expected: pass.

- [x] **Step 2: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: pass.

- [x] **Step 3: Record TypeScript and Bandit applicability**

Run `bunx tsc --noEmit --pretty false` if time permits. If the known repo-wide baseline still fails, verify no touched files are listed. Bandit is not applicable unless Python files are touched.

- [x] **Step 4: Update `TASK-447`**

Record touched files, verification commands, TypeScript/Bandit notes, and final summary.

- [x] **Step 5: Commit**

```bash
git add \
  Docs/superpowers/plans/2026-05-20-character-chat-phase3-setup-safety-accessibility-plan.md \
  "backlog/tasks/task-447 - Implement-Character-Chat-Phase-3-role-play-setup-safety-and-accessibility.md" \
  apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx \
  apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx \
  apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx
git commit -m "fix: harden character role-play setup controls"
```
