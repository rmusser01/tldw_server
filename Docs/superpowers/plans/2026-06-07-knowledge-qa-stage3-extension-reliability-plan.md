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
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx`
- Modify: `apps/extension/tests/e2e/knowledge-qa-setup-diagnostics.spec.ts`
- Modify: `apps/extension/tests/e2e/knowledge-qa-states.spec.ts`
- Modify: `apps/extension/tests/e2e/utils/extension-build.ts`
- Modify: `apps/extension/tests/e2e/utils/extension.launch.test.ts`
- Modify: `apps/extension/playwright.config.ts` if harness timeout or build profile needs adjustment

## Task 1: Define Extension Failure States

- [ ] **Step 1: Write failing unit tests**

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

- [ ] **Step 2: Add typed failure state**

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

- [ ] **Step 1: Write failing provider persistence test**

In `KnowledgeQAProvider.persistence.test.tsx`, simulate search success followed by thread/message persistence failure. Assert:

- answer remains visible
- `isLocalOnlyThread` is true
- trust state is `unsynced_local_result`
- visible UI includes retry sync action

- [ ] **Step 2: Implement local-result sync state**

Update `KnowledgeQAProvider.tsx` so thread creation or message persistence failures set an unsynced state instead of only logging or silently proceeding.

- [ ] **Step 3: Render retry action**

Update shared UI to retry sync without rerunning the search when possible.

## Task 3: Repair Or Gate Extension Runtime Harness

- [ ] **Step 1: Add explicit harness health test**

Update `apps/extension/tests/e2e/utils/extension.launch.test.ts` to assert the built extension launches and options route `#/knowledge` is reachable.

Run:

```bash
cd apps/extension
bunx playwright test tests/e2e/utils/extension.launch.test.ts --project=chromium-extension --reporter=line
```

Expected before fix: reproduce WXT build stall or route launch blocker.

- [ ] **Step 2: Investigate build stall without broad refactors**

Inspect:

- `apps/extension/tests/e2e/setup/build-extension.ts`
- `apps/extension/tests/e2e/utils/extension-build.ts`
- `apps/extension/wxt.config.ts`
- `apps/extension/playwright.config.ts`

Limit attempts to three. If unresolved, document exact command, stall point, process tree, and owner in `TASK-2279.5`.

- [ ] **Step 3: Run Knowledge QA extension E2E**

```bash
cd apps/extension
bunx playwright test tests/e2e/knowledge-qa-setup-diagnostics.spec.ts tests/e2e/knowledge-qa-states.spec.ts --project=chromium-extension --reporter=line
```

Expected: tests execute in browser. If blocked, mark release-blocking with evidence.

## Task 4: Verify

- [ ] **Step 1: Run focused unit and extension tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.persistence.test.tsx

cd apps/extension
bun run compile
bunx playwright test tests/e2e/utils/extension.launch.test.ts tests/e2e/knowledge-qa-setup-diagnostics.spec.ts tests/e2e/knowledge-qa-states.spec.ts --project=chromium-extension --reporter=line
```

- [ ] **Step 2: Run diff hygiene**

```bash
git diff --check -- apps/packages/ui/src/components/Option/KnowledgeQA apps/extension/tests/e2e apps/extension/playwright.config.ts
```

- [ ] **Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Option/KnowledgeQA apps/extension/tests/e2e apps/extension/playwright.config.ts "backlog/tasks/task-2279.5 - Harden-Knowledge-QA-extension-runtime-and-sync-reliability.md"
git commit -m "fix: harden knowledge qa extension reliability"
```
