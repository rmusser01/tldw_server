# Knowledge QA Stage 3 Extension Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make browser-extension Knowledge QA setup, backend reachability, auth, allowlist, search, and sync failures visible and recoverable.

**Architecture:** Keep the shared Knowledge QA route but add extension-specific diagnostic state where the extension has extra failure modes. Treat the WXT runtime E2E build stall as release-risk work: either fix the harness enough to launch `options.html#/knowledge`, or record it as a release blocker with an owner.

**Tech Stack:** React, WXT, Chrome extension APIs, Playwright, Vitest.

**Backlog Task:** TASK-2279.5

---

## Boundaries

- Do not change WebUI readiness flows except where shared diagnostics require it.
- Do not add flashcard behavior to `/knowledge`.
- Do not hide extension workflow failures in console-only logs.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`
- Modify: `apps/extension/tests/e2e/knowledge-qa-setup-diagnostics.spec.ts`
- Modify: `apps/extension/tests/e2e/knowledge-qa-states.spec.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-common.ts`
- Add: `apps/extension/tests/e2e/utils/extension-launch-health.spec.ts`

## Task 1: Define Extension Failure States

- [x] **Step 1: Write failing unit tests**

Update `KnowledgeQA.connection.test.tsx`:

```ts
it.each([
  ["setup_missing", "Finish setup"],
  ["setup_invalid", "Fix setup"],
  ["backend_unreachable", "Retry connection"],
  ["backend_auth_failed", "Update credentials"],
  ["api_allowlist_blocked", "Request host access"],
  ["search_succeeded_sync_failed", "Retry sync"],
  ["search_failed", "Retry search"],
])("shows recovery for %s", async (state, action) => {
  renderKnowledgeQaWithConnectionState({ extensionFailureState: state })
  expect(screen.getByRole("button", { name: action })).toBeInTheDocument()
})
```

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx
```

Expected: fail until states are modeled and rendered.

- [x] **Step 2: Add typed failure state**

Extend `types.ts` with:

```ts
export type ExtensionKnowledgeFailureState =
  | "setup_missing"
  | "setup_invalid"
  | "backend_unreachable"
  | "backend_auth_failed"
  | "api_allowlist_blocked"
  | "search_succeeded_sync_failed"
  | "search_failed"
```

## Task 2: Surface Sync Failure

- [x] **Step 1: Write failing provider persistence test**

In `KnowledgeQAProvider.persistence.test.tsx`, simulate search success followed by thread/message persistence failure. Assert:

- answer remains visible
- `isLocalOnlyThread` is true
- trust state is `unsynced_local_result`
- visible UI includes retry sync action

- [x] **Step 2: Implement local-result sync state**

Update `KnowledgeQAProvider.tsx` so thread creation or message persistence failures set an unsynced state instead of only logging or silently proceeding.

- [x] **Step 3: Render retry action**

Update shared UI to retry sync without rerunning the search when possible.

## Task 3: Repair Or Gate Extension Runtime Harness

- [x] **Step 1: Add explicit harness health test**

Added `apps/extension/tests/e2e/utils/extension-launch-health.spec.ts` to assert the built extension launches and options route `#/knowledge` is reachable. The probe is marked expected-failure with a `TASK-2279.5` release-blocker reason so the known packaged MV3 launch issue remains visible in CI/local runs.

Run:

```bash
cd apps/extension
bunx playwright test tests/e2e/utils/extension-launch-health.spec.ts --project=chromium-extension --reporter=line
```

Expected before fix: reproduce WXT build stall or route launch blocker.

- [x] **Step 2: Investigate build stall without broad refactors**

Inspect:

- `apps/extension/tests/e2e/setup/build-extension.ts`
- `apps/extension/tests/e2e/utils/extension-build.ts`
- `apps/extension/wxt.config.ts`
- `apps/extension/playwright.config.ts`

Limited to three headed/sandbox launch attempts, then moved to headless default. Current blocker is documented in `TASK-2279.5`: sandboxed Chromium aborts before route code; unsandboxed headless Chromium launches but exposes no extension targets for `resolveExtensionId`; headed launch times out locally.

- [x] **Step 3: Run Knowledge QA extension E2E**

```bash
cd apps/extension
bunx playwright test tests/e2e/knowledge-qa-setup-diagnostics.spec.ts tests/e2e/knowledge-qa-states.spec.ts --project=chromium-extension --reporter=line
```

Result: blocked before page assertions. The latest run failed 6/6 with `Could not determine extension id from [no extension targets]`.

## Task 4: Verify

- [x] **Step 1: Run focused unit and extension tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx

cd apps/extension
bun run compile
bunx playwright test tests/e2e/utils/extension-launch-health.spec.ts --project=chromium-extension --reporter=line
bunx playwright test tests/e2e/knowledge-qa-setup-diagnostics.spec.ts tests/e2e/knowledge-qa-states.spec.ts --project=chromium-extension --reporter=line
```

- [x] **Step 2: Run diff hygiene**

```bash
git diff --check -- apps/packages/ui/src/components/Option/KnowledgeQA apps/extension/tests/e2e backlog/tasks Docs/superpowers/plans/2026-06-07-knowledge-qa-stage3-extension-reliability-plan.md
```

- [x] **Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Option/KnowledgeQA apps/extension/tests/e2e apps/extension/playwright.config.ts "backlog/tasks/task-2279.5 - Harden-Knowledge-QA-extension-runtime-and-sync-reliability.md"
git commit -m "fix: harden knowledge qa extension reliability"
```
