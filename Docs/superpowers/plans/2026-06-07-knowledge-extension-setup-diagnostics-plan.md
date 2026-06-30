# Knowledge Extension Setup Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the extension `/knowledge` setup-required state diagnostic enough for users to recover without opening DevTools.

**Architecture:** Keep the shared Knowledge QA route, but add extension-aware setup diagnostics that can report missing server URL, missing API key, host permission or allowlist problems, and backend reachability failures. The extension setup state should not compete with a feature tour while setup is blocking.

**Tech Stack:** React, WXT/browser extension options route, shared `@tldw/ui`, Vitest, Playwright.

**Backlog Task:** TASK-528.3

---

## Boundaries

- This phase is extension setup and diagnostics only.
- WebUI readiness recovery is handled by TASK-528.2.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx`
- Create: `apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/knowledgeQaStateFixtures.ts`
- Create or modify: `apps/extension/tests/e2e/knowledge-qa-setup-diagnostics.spec.ts`

## Task 1: Write Failing Setup Diagnostic Tests

- [x] Add unit tests for missing server URL, missing API key, blocked absolute URL or allowlist issue, and unreachable backend.
- [x] Assert the setup state shows specific checks and next actions, not only "Setup Required."
- [x] Assert a setup-blocked state suppresses the ready search workspace while recovery is blocking.
- [x] Run:

```bash
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx
```

Result: failed before implementation on missing diagnostic content, then passed after implementation.

## Task 2: Add Extension-Aware Diagnostics Model

- [x] Identify the source of extension server URL and API key configuration.
- [x] Add a typed diagnostic summary that can report:
  - server URL missing
  - API key missing
  - backend unreachable
  - request blocked by extension permission or allowlist
  - server configured but auth failed
- [x] Keep diagnostics generic enough to be reused by other extension setup states if local patterns support it.

Notes: The shared connection store already exposes the needed route-level facts through `useConnectionState` and `useConnectionUxState`: `serverUrl`, `configStep`, `errorKind`, `lastError`, `lastStatusCode`, and `isChecking`. No new connection store or request-core state was needed.

## Task 3: Update Setup Required UI

- [x] Replace the single vague setup message with a compact checklist.
- [x] Provide primary action to finish setup.
- [x] Provide secondary action to retry or open diagnostics when useful.
- [x] Show the configured server origin when available, redacting paths/query strings.
- [x] Defer the page tour until setup passes by returning the blocking diagnostics surface before the ready workspace renders.

Notes: WebUI and extension use the same diagnostics panel. The extension additionally shows `Request host access` when `chrome.permissions.request` is available; WebUI users receive origin/CORS guidance in the browser access row.

## Task 4: Verify Extension Route

- [x] Add Playwright tests for extension options `#/knowledge` in missing config, missing auth, unreachable/backend-blocked, and configured healthy states.
- [x] Use the extension test config and static/mock runtime already used by the repository.
- [ ] Run:

```bash
npx playwright test --config apps/extension/playwright.config.ts apps/extension/tests/e2e/knowledge-qa-setup-diagnostics.spec.ts
```

Result: blocked before Playwright launched because the extension WXT production build hung twice after the duplicated-import warnings. Both hung processes were terminated.

## Task 5: Close Verification

- [x] Run related unit tests and extension compile.
- [x] Record WebUI versus extension setup differences in TASK-528.3 notes.
- [x] Record Bandit as not applicable because no Python backend files were touched.

Verification:

```bash
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeQA.connection.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeQALayout.behavior.test.tsx
bun run compile
```

Results: Vitest passed 20 tests. Extension TypeScript compile passed. Extension runtime E2E is authored but not executed due WXT build hang; see TASK-306 for the existing build pre-render hang track.
