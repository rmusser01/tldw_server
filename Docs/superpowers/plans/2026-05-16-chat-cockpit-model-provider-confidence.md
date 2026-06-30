# Chat Cockpit Model Provider Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the main `/chat` cockpit model/provider confidence slice so configured models are the default, all-model search remains explicit, and provider:model settings work without leaking internal `tldw:` identifiers into user-facing or validation paths.

**Architecture:** Reuse the existing Playground-only `useModelSelector` and model-scope utilities. Keep the older shared `ModelSelect` out of scope because the main `/chat` cockpit uses `PlaygroundForm` and `useModelSelector`. Normalize internal server model ids at the selector/availability boundary so stored choices, runtime rail labels, settings scopes, and submit validation agree on `provider:model`.

**Tech Stack:** React, TypeScript, Zustand, Plasmo storage, Vitest, Playwright real-server workflow.

---

### Task 1: Lock Provider-Qualified Model Id Normalization

**Files:**
- Modify: `apps/packages/ui/src/hooks/playground/modelSelectorUtils.ts`
- Modify: `apps/packages/ui/src/hooks/playground/__tests__/modelSelectorUtils.test.ts`
- Modify: `apps/packages/ui/src/services/tldw-server.ts`

- [x] **Step 1: Write failing selector utility coverage**

Add coverage that a tldw server model descriptor such as `{ id: "gpt-4o", model: "tldw:gpt-4o", provider: "openai" }` resolves to `getModelId(...) === "gpt-4o"` and `getCanonicalModelKey(...) === "openai:gpt-4o"`.

- [x] **Step 2: Run the focused utility test**

Run: `bunx vitest run apps/packages/ui/src/hooks/playground/__tests__/modelSelectorUtils.test.ts --config apps/packages/ui/vitest.config.ts`

Expected: FAIL before implementation because the selector currently keeps the internal `tldw:` prefix in canonical keys.

- [x] **Step 3: Normalize internal ids**

Update `getModelId` to strip the internal `tldw:` prefix and prefer a real `id` when the `model` field is just the prefixed transport value. Add the raw backend `id` to `mapTldwModelToUi` output so the selector can keep backend ids and display labels separate.

- [x] **Step 4: Re-run focused utility test**

Expected: PASS, with duplicate model ids still distinct through provider-qualified keys.

### Task 2: Make Availability Checks Accept Provider-Qualified Selections

**Files:**
- Modify: `apps/packages/ui/src/utils/chat-model-availability.ts`
- Modify: `apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts`

- [x] **Step 1: Write failing availability coverage**

Add coverage that available descriptors with provider metadata include both raw and provider-qualified ids, and that `findUnavailableChatModel(["openai:gpt-4o"], ids)` passes when OpenAI exposes `gpt-4o`.

- [x] **Step 2: Run the focused availability test**

Run: `bunx vitest run apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts --config apps/packages/ui/vitest.config.ts`

Expected: FAIL before implementation for provider-qualified selected models.

- [x] **Step 3: Add provider-aware availability ids**

Extend `buildAvailableChatModelIds` to include normalized raw ids and provider-qualified ids when provider metadata is present. Keep empty-catalog behavior unchanged.

- [x] **Step 4: Re-run focused availability test**

Expected: PASS.

### Task 3: Certify Existing Configured/Catalog/Recent UX Contract

**Files:**
- Modify: `apps/packages/ui/src/hooks/playground/__tests__/useModelSelector.capabilities.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx`

- [x] **Step 1: Add hook-level coverage**

Assert the default `modelListScope` is `configured`, catalog-only models remain absent until the explicit catalog scope is selected, and recent/frequent configured choices are promoted without dropping provider grouping.

- [x] **Step 2: Add component-level copy/state coverage**

Keep the `Search all models` toggle action/hint under test and assert the catalog state exposes `All known models` while configured state exposes `Usable configured models`.

- [x] **Step 3: Run focused Playground selector tests**

Run: `bunx vitest run apps/packages/ui/src/hooks/playground/__tests__/useModelSelector.capabilities.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx --config apps/packages/ui/vitest.config.ts`

Expected: PASS.

### Task 4: Real-Server Proof for Main `/chat`

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`

- [x] **Step 1: Add real-server model selector proof**

Use the running server only. Open `/chat`, open the model selector, prove configured scope is default, toggle to catalog scope, pick a real configured model from the server selector, verify composition preview/runtime rail show the provider route, then send a short real conversation.

- [x] **Step 2: Run the focused real-server workflow**

Run with the existing local server and `.env` API key:
`TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=<from-env> bunx playwright test apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts --config apps/tldw-frontend/playwright.config.ts --project=chromium --grep "model provider confidence"`

Expected: PASS and screenshot output for the working conversation.

### Task 5: Verification and Task Closeout

**Files:**
- Modify: `backlog/tasks/task-399 - Implement-main-chat-model-provider-confidence.md`
- Update if needed: `Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md`

- [x] **Step 1: Run focused Vitest**

Run the selector, availability, settings-scope, and cockpit tests touched by this slice.

- [x] **Step 2: Run real-server Playwright proof**

Use the real running server at `127.0.0.1:8000`; do not use mocked backend routes.

- [x] **Step 3: Run static checks**

Run: `git diff --check`

Run design-system verification if UI classes changed: `bun run verify:design-system-state`.

- [x] **Step 4: Bandit decision**

No Python files should be touched. Record Bandit as skipped for no Python changes unless the slice unexpectedly edits backend Python.

- [x] **Step 5: Update TASK-399**

Check completed acceptance criteria, record verification commands/results, add final summary.

- [x] **Step 6: Commit**

Commit the PR3 slice separately with a message like `Implement chat cockpit model provider confidence`.
