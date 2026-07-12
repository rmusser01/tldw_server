# Quick Ingest Preset Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Standard and Deep Quick Ingest honor saved preset providers and route missing-provider runs to an accessible Configure control in both the WebUI and browser extension.

**Architecture:** The always-mounted Quick Ingest event hook owns preset-storage hydration and captures one immutable preset snapshot for each open or new draft. The reducer resolves and edits against that snapshot by merging changes into the session's full current config. The Configure step provides a session-scoped provider combobox; existing validation redirects missing-provider runs before any request.

**Tech Stack:** React, TypeScript, Zustand, Plasmo storage, Ant Design, Vitest/Testing Library, Playwright.

**Spec:** `Docs/superpowers/specs/2026-07-12-quick-ingest-preset-provider-design.md`

---

## Stages

### Stage 1: Snapshot-aware reducer
**Goal:** Preserve full session configuration and resolve named presets against a supplied snapshot.
**Success Criteria:** First-source chunking survives every edit; configured presets match correctly.
**Tests:** Focused `IngestWizardContext.test.tsx` cases.
**Status:** Not Started

### Stage 2: Hydrated open lifecycle
**Goal:** Capture saved presets before creating, rebasing, or rendering eligible drafts.
**Success Criteria:** No fallback persistence during hydration; lifecycle exclusions and Ingest More boundaries hold.
**Tests:** `QuickIngestButton.resume.test.tsx` and modal session cases.
**Status:** Not Started

### Stage 3: Provider recovery UI
**Goal:** Add the editable provider control and recoverable redirect.
**Success Criteria:** Missing providers focus Configure with localized accessible feedback and no request.
**Tests:** Configure/modal integration and session cases.
**Status:** Not Started

### Stage 4: Runtime verification
**Goal:** Prove shared behavior under WebUI and extension adapters.
**Success Criteria:** Focused suites, typechecks, adapter smoke, lint, and targeted browser harnesses pass.
**Tests:** Commands in Task 4.
**Status:** Not Started

### Stage 5: Review and delivery
**Goal:** Review, finalize Backlog, commit, push, and open the PR to `dev`.
**Success Criteria:** No blockers and fresh verification is recorded.
**Tests:** Fresh focused rerun and `git diff --check`.
**Status:** Not Started

---

### Task 1: Make reducer updates snapshot-aware and lossless

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx`

- [ ] **Step 1: Write failing tests**

Pass a saved `PresetMap` whose Standard preset contains `advancedValues.api_name = "openai"`. Assert initial/default state, named preset switches, matching, and reset use it. Seed `FIRST_SOURCE_QUICK_PRESET_CONFIG`, edit OCR/provider, and assert chunking remains true. Clear `api_name`, edit another option, serialize/rehydrate, and assert it stays absent. Assert `setPreset("custom")` preserves the full current config.

- [ ] **Step 2: Prove red**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the provider accepts no map and edits reconstruct from hard-coded defaults.

- [ ] **Step 3: Implement minimally**

Add `presetMap?: PresetMap` to `IngestWizardProvider` and pass the resolved snapshot into initial-state/reducer helpers. Use it for named selection, matching, and reset. In `SET_CUSTOM_OPTIONS`, merge nested input into `state.presetConfig`, remove incoming advanced keys whose value is `undefined`, then call `detectPreset(nextConfig, presetMap)`. Preserve the merged config for Custom; store the captured named config on an exact match. `SET_PRESET("custom")` changes only the label/base metadata.

- [ ] **Step 4: Prove green and commit**

Rerun the command, then:

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx
git commit -m "fix: preserve quick ingest preset configuration"
```

### Task 2: Gate opens on hydration and own snapshot boundaries

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Test: `apps/packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`

- [ ] **Step 1: Write failing lifecycle tests**

Mock `useStorage("quickIngestPresetConfigs", ...)` with controllable `isLoading`. Cover pending opens, a preloaded visible draft, reopen after Settings changes, and auto-process waiting for combined readiness. Assert named idle drafts rebase; Custom, first-source, processing, interrupted, cancelled, `partial_failure`, and completed sessions do not. Change Settings while results remain open, invoke Ingest More, and assert the new draft uses the new map without the prior session provider.

- [ ] **Step 2: Prove red**

```bash
bunx vitest run ../packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 3: Implement event-hook ownership**

In `useQuickIngestEvents`, load `quickIngestPresetConfigs`, compute `quickIngestReady = sessionHydrated && !presetMeta.isLoading`, and gate pending consumption plus modal rendering in both `QuickIngestButton` and `QuickIngestModalHost`. Capture a resolved map and incrementing revision on open and new-draft boundaries. Before showing an eligible draft, rebase its selected named preset from the snapshot; seed ordinary new drafts from it. Preserve all excluded lifecycle/custom/first-source records. Return `presetMap`, `openRevision`, and `createNewDraft`.

- [ ] **Step 4: Wire modal snapshot inputs**

Pass the snapshot into `IngestWizardProvider`. Remount eligible idle drafts by session id plus revision while keeping processing/terminal identity stable. Route Ingest More through `createNewDraft` instead of the store action.

- [ ] **Step 5: Prove green and commit**

```bash
git add apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx apps/packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
git commit -m "fix: hydrate quick ingest preset snapshots"
```

### Task 3: Add provider recovery UI

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx`
- Modify: `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/option.json`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx`
- Test: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`

- [ ] **Step 1: Write failing provider and redirect tests**

Mock configured, unconfigured, duplicate, blank, and local provider entries. Assert an associated editable combobox, “For this ingest” help, trimmed/deduplicated configured suggestions, keyboard selection, arbitrary free text, clear behavior, and catalog-failure tolerance. For fresh Standard and Deep, assert Quick Process reaches step 2, stays idle, fires no start/submit request, focuses the combobox, links its localized alert through `aria-describedby`, and announces with `role="alert"`. Repeat with `autoProcessQueued=true` and assert it never dispatches a running state or start/submit request. Cover the late Review guard and a configured success path.

- [ ] **Step 2: Prove red**

```bash
bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 3: Implement the combobox**

Use Ant Design `AutoComplete` bound to `presetConfig.advancedValues.api_name`. Fetch `getProvidersStatus()` only while Configure is visible and analysis is enabled; ignore stale responses. Suggest trimmed/deduplicated `configured === true` names, retain arbitrary typed values, and write clear as `undefined`. Add stable label/help/error IDs and English locale keys for label, “For this ingest” help, placeholder/status, and required warning. Use the service helper only as the validation predicate.

- [ ] **Step 4: Implement navigation/focus**

Share provider warning/focus state between modal steps and route both Quick Process and the `autoProcessQueued` effect through one provider preflight. On step 1, use `goNext()` so the forward-navigation guard cannot block recovery. When auto-process is invalid, do not dispatch `SKIP_TO_PROCESSING`; redirect through the same step-1 path and stay idle. On late validation, set idle and use `goToStep(2)`. Never call start/submit while invalid; focus after Configure mounts and clear the warning after a valid provider is entered.

- [ ] **Step 5: Prove green and commit**

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx apps/packages/ui/src/assets/locale/en/option.json apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
git commit -m "fix: recover missing quick ingest providers"
```

### Task 4: Verify both runtime boundaries

**Files:**
- Modify if needed: `apps/tldw-frontend/__tests__/extension/plasmo-storage-watch.test.tsx`
- Modify if needed: `apps/tldw-frontend/e2e/quick-ingest-render-loop.spec.ts`
- Modify if needed: `apps/extension/tests/e2e/quick-ingest-workflows.spec.ts`

- [ ] **Step 1: Add only missing adapter smoke coverage**

Exercise `quickIngestPresetConfigs` through the WebUI storage shim and extension Vitest config; assert readiness and watched map updates reach the event hook. Do not duplicate shared reducer/UI cases.

- [ ] **Step 2: Run focused shared and adapter tests**

```bash
bunx vitest run ../packages/ui/src/components/Common/QuickIngest/__tests__/IngestWizardContext.test.tsx ../packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx ../packages/ui/src/services/__tests__/quick-ingest-batch.test.ts __tests__/extension/plasmo-storage-watch.test.tsx --maxWorkers=1 --no-file-parallelism
bunx vitest run -c vitest.extension.config.ts ../packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 3: Run lint and both typechecks**

```bash
bunx eslint ../packages/ui/src/components/Layouts/QuickIngestButton.tsx ../packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx ../packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx ../packages/ui/src/components/Common/QuickIngestWizardModal.tsx
bun run typecheck
cd ../extension && bun run compile
```

- [ ] **Step 4: Run targeted browser smoke tests**

Build both current artifacts before launching either harness:

```bash
bun run build
bunx playwright test e2e/quick-ingest-render-loop.spec.ts --reporter=line --workers=1
cd ../extension && bun run build:chrome
bunx playwright test tests/e2e/quick-ingest-workflows.spec.ts --reporter=line --workers=1
```

Record an exact environmental blocker rather than claiming a skipped run passed.

- [ ] **Step 5: Run repository gates**

Run `git diff --check`. Bandit is not applicable if touched files remain TypeScript/JSON/docs; record that in TASK-12950. If Python becomes touched, activate `.venv` and run Bandit recursively on the touched Python path.

### Task 5: Review and deliver

**Files:**
- Update through Backlog MCP: `TASK-12950`
- Remove after all stages complete: `Docs/superpowers/plans/2026-07-12-quick-ingest-preset-provider-implementation.md`

- [ ] **Step 1: Request a dedicated correctness review**

Review snapshot ownership, render loops, persistence, lifecycle exclusions, async cleanup, accessibility, locale usage, and unrelated changes. Resolve blockers test-first.

- [ ] **Step 2: Rerun fresh verification**

Rerun focused Vitest, lint, both typechecks, `git diff --check`, and browser smoke affected by review changes.

- [ ] **Step 3: Finalize Backlog and commits**

Record touched files, implementation choices, exact verification, browser blockers, Bandit disposition, commits, and final summary through the Backlog MCP. Remove this completed temporary plan, stage all intended files, and commit without bypassing hooks.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin codex/fix-quick-ingest-preset-provider
gh pr create --base dev --head codex/fix-quick-ingest-preset-provider
```

The PR body must explain the fix, list exact verification, and state that a human requester must add their own Change summary explaining what changed and why before merge.
