# Persona Visual Empty State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Persona Garden Visuals no-pack state so first-time users understand how to start the existing Persona Buddy visual-pack workflow.

**Architecture:** Keep this as a focused frontend hardening slice in `VisualPackEditor`. The empty state stays inside the existing Visuals tab and points to the already-present draft creation flow without adding backend APIs or duplicate editor controls.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, existing Persona Garden UI primitives.

---

### Task 1: Add First-Run Empty-State Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Write the failing test**

Add assertions to the existing no-pack test for copy that says the selected persona's Buddy has no visual pack yet, that users should create a draft pack first, and that upload/import/generation/activation happen after a draft exists.

- [x] **Step 2: Run the focused test**

Run: `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
Expected: FAIL because the current empty state only renders `No visual packs yet.`

### Task 2: Implement Empty-State Copy

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`

- [x] **Step 1: Replace passive empty copy**

Update the `persona-visual-pack-empty` block to include concise Persona Buddy / Persona Live framing, the draft-first action, and follow-on workflow summary.

- [x] **Step 2: Keep controls scoped**

Do not expose upload, import, generation, activation, or manifest controls when `selectedPack` is absent; those remain behind the existing selected-pack branch.

- [x] **Step 3: Run focused tests**

Run: `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
Expected: PASS.

### Task 3: Verify Related Frontend Behavior

**Files:**
- Verify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- Verify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
- Verify: `apps/packages/ui/src/utils/__tests__/persona-garden-route.test.ts`

- [x] **Step 1: Run related tests**

Run: `bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/utils/__tests__/persona-garden-route.test.ts`
Expected: PASS.

- [x] **Step 2: Run diff hygiene**

Run: `git diff --check`
Expected: no output and exit 0.

### Task 4: Closeout

**Files:**
- Modify: `backlog/tasks/task-164 - Harden-Persona-Visuals-first-run-empty-state.md`

- [x] **Step 1: Record verification in Backlog**

Check acceptance criteria, record red/green verification, and note Bandit is not applicable for this frontend-only slice.

- [x] **Step 2: Commit**

Run:
```bash
git add apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx Docs/superpowers/plans/2026-05-09-persona-visual-empty-state-plan.md "backlog/tasks/task-164 - Harden-Persona-Visuals-first-run-empty-state.md"
git commit -m "fix: improve persona visuals empty state"
```
