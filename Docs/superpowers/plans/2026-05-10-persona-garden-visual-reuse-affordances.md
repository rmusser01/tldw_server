# Persona Garden Visual Reuse Affordances Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Persona Garden visual-pack reuse decision surface that makes existing draft, library, duplicate, and import workflows discoverable without adding a parallel management model.

**Architecture:** Keep backend/API contracts unchanged. Add one focused React presentational component for the reuse affordance surface and wire it into `VisualPackEditor` with callbacks to the existing create, library, duplicate, and import controls. Preserve user-owned, persona-attached, draft/review-before-activation language.

**Tech Stack:** React, TypeScript, Ant Design buttons/tags, lucide-react icons, Vitest, Testing Library.

---

### Task 1: Add Reuse Decision Surface Tests

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/VisualPackReusePanel.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackReusePanel.test.tsx`

- [x] **Step 1: Write failing tests for available actions**

Test that the panel renders the four reuse paths:
- Create a new draft pack.
- Use a personal library pack.
- Import a portable archive for preview.
- Duplicate the selected pack to another persona as a draft.

Expected assertions:
- Copy includes `draft`, `review`, and `activate` semantics.
- Copy does not include `marketplace`, `shared with other users`, `VN`, or `CYOA`.
- Clicking enabled actions invokes their callback props.

- [x] **Step 2: Write failing tests for disabled and empty states**

Test that duplicate is disabled when no selected pack or no other persona target exists, and that the empty library state still routes to the library area instead of hiding the workflow.

- [x] **Step 3: Run focused panel tests to verify RED**

Run:
`cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackReusePanel.test.tsx`

Expected: FAIL because `VisualPackReusePanel` does not exist yet.

### Task 2: Implement Panel Component

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/VisualPackReusePanel.tsx`

- [x] **Step 1: Implement typed props and four actions**

Props:
- `selectedPersonaName`
- `hasSelectedPack`
- `libraryItemCount`
- `hasDuplicateTargets`
- `duplicateTargetsLoading`
- `onCreateDraft`
- `onOpenLibrary`
- `onOpenImport`
- `onOpenDuplicate`

Render compact action buttons with icons and state notes. Disable duplicate when there is no selected pack, no target persona, or targets are loading.

- [x] **Step 2: Run panel tests to verify GREEN**

Run:
`cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackReusePanel.test.tsx`

Expected: PASS.

### Task 3: Wire Panel Into VisualPackEditor

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Add failing integration tests**

Add tests that:
- The panel appears in the Persona Garden Visuals tab.
- The create action focuses the draft title input.
- The library action focuses or scrolls to the personal library panel.
- The import action opens the existing import file input.
- The duplicate action focuses the existing duplicate target select and remains disabled until a pack and target persona exist.

- [x] **Step 2: Wire callbacks through existing refs**

Use existing handlers and DOM controls:
- Add refs for draft title, library panel, duplicate target select.
- Reuse existing `importPreviewInputRef`.
- Implement callbacks that focus/scroll or click existing controls only; do not add new API calls.

- [x] **Step 3: Run editor tests to verify GREEN**

Run:
`cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: PASS.

### Task 4: Documentation, Backlog, and Verification

**Files:**
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `backlog/tasks/task-214 - Implement-Persona-Garden-reusable-visual-pack-affordances.md`

- [x] **Step 1: Update the PRD implementation snapshot**

Add a concise entry that Persona Garden now exposes the reusable visual-pack decision surface backed by existing draft, library, duplicate, and import flows.

- [x] **Step 2: Run focused verification**

Run:
`cd apps/packages/ui && bunx vitest run src/components/PersonaGarden/__tests__/VisualPackReusePanel.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Expected: PASS.

- [x] **Step 3: Record Bandit applicability**

This slice touches TypeScript/Markdown only. Record that Python Bandit is not applicable unless Python files are added later.

- [x] **Step 4: Update Backlog task**

Check acceptance criteria and add verification notes.
