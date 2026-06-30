# Persona Buddy Visual Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Persona/Buddy visual-pack failures visible and actionable while preserving the existing text fallback.

**Architecture:** Add one small UI diagnostics helper that classifies visual-pack health from load state, pack metadata, assets, manifest, and renderer errors. Use that model in the Buddy runtime for compact diagnostics and in Persona Visuals for a fuller health summary.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, existing Persona/Buddy UI services.

---

### Task 1: Shared Diagnostics Helper

**Files:**
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts`
- Test: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`

- [ ] Write failing tests for no active pack, load failure, unsupported renderer, missing manifest, missing assets, missing animation, and missing asset render failures.
- [ ] Run the diagnostics test and confirm it fails because the helper does not exist.
- [ ] Implement the minimal helper with stable reason codes, severity, title, message, and optional action label.
- [ ] Re-run the diagnostics test and confirm it passes.

### Task 2: Buddy Runtime Diagnostics

**Files:**
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellPopover.tsx`
- Test: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
- Test: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx`

- [ ] Write failing Buddy runtime tests for active-pack load failure and renderer missing-asset diagnostics.
- [ ] Run the Buddy tests and confirm the new assertions fail.
- [ ] Track load status/errors in `BuddyShellHost` and pass diagnostics to `BuddyShellDock`.
- [ ] Use `SpriteFrameRenderer.onRenderError` to report render failures without disabling fallback rendering.
- [ ] Show compact diagnostics in the dock/popover with an existing Visuals workflow link when a persona id is available.
- [ ] Re-run Buddy tests and confirm they pass.

### Task 3: Persona Visuals Editor Health Summary

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [ ] Write failing editor tests for selected-pack health summary output.
- [ ] Run the editor test and confirm the new assertions fail.
- [ ] Render the shared diagnostics summary near the selected pack/validation area.
- [ ] Re-run the editor test and confirm it passes.

### Task 4: Verification And Closeout

**Files:**
- Update: Backlog task `TASK-175`

- [ ] Run targeted Vitest commands for the diagnostics, Buddy, renderer, and VisualPackEditor tests.
- [ ] Run any practical TypeScript/lint check for the touched UI package.
- [ ] Document Bandit as not applicable if no Python files are touched.
- [ ] Update Backlog acceptance criteria and final notes.
