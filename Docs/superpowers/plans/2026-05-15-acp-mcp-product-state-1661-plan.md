# ACP/MCP Product-State Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the first ACP/MCP product-state guard slice from AntD state widgets to shared design-system primitives for issue #1661.

**Architecture:** Replace direct AntD product-state rendering with `Alert` from `@/components/ui/primitives` while leaving form, list, and layout AntD usage intact. Update focused component tests to assert the shared primitive contract, then remove only the baseline entries proven stale by the guard.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, shared `apps/packages/ui` design-system primitives.

---

### Stage 1: Establish Current Guard Baseline

**Files:**
- Read: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
- Read: `apps/packages/ui/scripts/verify-design-system-product-state.mjs`
- Read: `apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx`
- Read: `apps/packages/ui/src/components/Option/MCPHub/AcpProfilesTab.tsx`

- [x] Run `bun run verify:design-system-state` from `apps/packages/ui`.
- [x] Confirm the guard state before edits.
- [x] Record unrelated baseline noise in the Backlog task if present.

Baseline note: the full guard was red on `dev` in this worktree because unrelated Admin/Llamacpp product-state findings no longer matched their stored baseline IDs. The review-fix pass refreshed those unrelated Admin/Llamacpp baseline IDs so `bun run verify:design-system-state` passes again without migrating additional product-state surfaces.

### Stage 2: ACP Modal Alert Migration

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts`

- [x] Add a failing static test asserting `ACPSessionCreateModal.tsx` imports the shared `Alert` primitive and no longer imports AntD `Alert`.
- [x] Run `bunx vitest run src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts --maxWorkers=1 --no-file-parallelism` from `apps/packages/ui` and confirm the new assertion fails.
- [x] Replace the AntD creation-error `<Alert>` with the shared `Alert` primitive, preserving the error title and suggestion list.
- [x] Re-run the focused ACP modal test and confirm it passes.

### Stage 3: MCP ACP Profiles Alert Migration

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/AcpProfilesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/AcpProfilesTab.test.tsx`

- [x] Add a failing render test asserting load failures render a shared `Alert` primitive.
- [x] Run `bunx vitest run src/components/Option/MCPHub/__tests__/AcpProfilesTab.test.tsx --maxWorkers=1 --no-file-parallelism` from `apps/packages/ui` and confirm the new assertion fails.
- [x] Replace the AntD profile error `<Alert>` with the shared `Alert` primitive, preserving the existing error message.
- [x] Re-run the focused MCP ACP profiles test and confirm it passes.

### Stage 4: Baseline Cleanup and Verification

**Files:**
- Modify: `apps/packages/ui/scripts/design-system-product-state-baseline.json`
- Modify: `backlog/tasks/task-392 - Migrate-design-system-product-state-for-MCP-and-ACP-issue-1661.md`

- [x] Remove the stale baseline entries for `ACPSessionCreateModal.tsx:Alert` and `AcpProfilesTab.tsx:Alert`.
- [x] Run a focused product-state guard over the two touched source files and confirm no new blocked findings.
- [x] Refresh the unrelated Admin/Llamacpp baseline drift and run `bun run verify:design-system-state` from `apps/packages/ui`.
- [x] Run both focused Vitest commands from stages 2 and 3.
- [x] Run `git diff --check`.
- [x] Skip Bandit with an explicit UI-only rationale in the Backlog task.
- [x] Update the Backlog task with changed files, verification results, and remaining issue #1661 baseline debt.
