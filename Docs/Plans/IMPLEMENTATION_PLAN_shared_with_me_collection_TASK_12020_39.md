# Shared With Me Collection Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/shared` render the canonical `{items, total}` recipient response and preserve reliable populated, empty, loading, error, and Research Workspace handoff behavior.

**Architecture:** Keep `useSharedWithMe()` and `SharedWithMeResponse` as the single typed API contract. Update only the active component's collection selection and strengthen its existing tests so array-shaped mocks can no longer mask response drift. Do not add a compatibility normalizer for a response shape the client does not support.

**Tech Stack:** React 18, TypeScript, TanStack Query, Ant Design, Vitest, Testing Library, FastAPI, Playwright over CDP.

---

## Stage 1: Prove The Contract Failure

**Files:**
- Create: `apps/packages/ui/tsconfig.shared-with-me.json`
- Create: `apps/packages/ui/src/components/Option/__tests__/SharedWithMe.states.test.tsx`
- Delete: `apps/packages/ui/src/components/Option/SharedWithMe.test.tsx` (outside the configured Vitest discovery pattern)
- Modify: `apps/packages/ui/src/components/Option/__tests__/SharedWithMe.research-workspace-route.test.tsx`

- [x] Add a narrow TypeScript project that extends the package config and includes `vitest.setup.ts`, required ambient declarations, the active component, and both task-owned tests; imported production dependencies remain typechecked transitively.
- [x] Replace array-shaped hook data with typed `SharedWithMeResponse` fixtures.
- [x] Define every canonical response fixture with `satisfies SharedWithMeResponse` so array-shaped or incomplete mocks fail the package typecheck.
- [x] Keep the existing unknown-access-level fallback case as a separate, explicitly documented runtime-drift fixture; confine its intentional cast to that one test so it cannot weaken canonical contract coverage.
- [x] Add populated, empty, loading, and error state assertions.
- [x] Preserve the exact `/research-workspace?shared=<share_id>` navigation assertion.
- [x] Run the focused tests and confirm the populated and route cases fail because the active component discards the envelope.

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/__tests__/SharedWithMe.states.test.tsx \
  src/components/Option/__tests__/SharedWithMe.research-workspace-route.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

**Status:** Complete

## Stage 2: Consume The Canonical Envelope

**Files:**
- Modify: `apps/packages/ui/src/components/Option/SharedWithMe.tsx`
- Modify: `apps/packages/ui/src/types/sharing.ts`

- [x] Read `data?.items ?? []` from the typed hook result.
- [x] Add no legacy array branch and no new abstraction.
- [x] Re-run the focused tests and confirm all state and navigation cases pass.
- [x] Run the pinned scoped TypeScript project so the active component, both test fixtures, and their imported dependencies are checked rather than merely transpiled by Vitest.
- [x] Record the unrelated package-wide TypeScript baseline separately; do not treat existing diagnostics outside this scoped project as task failures or silently claim the package-wide project is clean.

Prepare the worktree dependencies once if `apps/packages/ui/node_modules` is absent:

```bash
cd apps
bun install --frozen-lockfile
```

Run the scoped typecheck with the repository-pinned TypeScript 5.9.3 binary:

```bash
cd apps/packages/ui
NODE_OPTIONS=--max-old-space-size=8192 \
  ./node_modules/.bin/tsc -p tsconfig.shared-with-me.json --noEmit --pretty false
```

**Status:** Complete

## Stage 3: Live Recipient Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-12020.39 - Normalize-Shared-With-Me-collection-responses-in-the-WebUI.md`

- [x] Start a task-owned multi-user backend and WebUI with isolated data/config paths.
- [x] Seed an owner, a distinct recipient identity, recipient membership, an identifying owner-workspace title marker, and a canonical share through supported backend/API paths.
- [x] Launch a clean browser profile, authenticate it as the recipient, and control it only through Playwright's CDP connection.
- [x] Capture the recipient's live `GET /api/v1/sharing/shared-with-me` response and assert an object envelope with `items`, `total`, and the seeded share rather than accepting a mocked or array-shaped response.
- [x] Verify `/shared` lists the identifying owner workspace and opens the exact `/research-workspace?shared=<share_id>` route.
- [x] After navigation, assert the live share-metadata request returns the seeded `share.workspace_id` and the page renders the recipient's shared-access banner/access level. Do not claim owner-source or shared-chat hydration: TASK-12020.40 explicitly owns replacing the unrelated local workspace data plane with canonical shared sources and chat.
- [x] Assert zero unexpected browser-console errors and zero unexpected failed HTTP requests across list, route open, and share-metadata initialization; retain known recovered local-context diagnostics in the task evidence under TASK-12020.40.
- [x] Replace the task-owned deprecated Ant Design `List` with a semantic row list while preserving the existing populated layout and controls; verify the live deprecation diagnostic is gone.
- [x] Run final focused tests, lint, design-system state verification for the touched surface, and `git diff --check`.
- [x] Record Bandit as not applicable in the Backlog closeout because the task-owned implementation diff is TypeScript/TSX and task metadata only; run Bandit if the scope expands into Python.
- [x] Record evidence, complete TASK-12020.39, request code review, and commit only task-owned files.

Verification evidence:

- Focused Vitest: 2 files, 10 tests passed, including nullable-name fallback and unique multi-row action labels.
- Scoped TypeScript 5.9.3 project: passed with no diagnostics.
- Scoped ESLint: passed with no file findings; the Next.js pages-directory notice is unrelated to the shared package files.
- Touched-surface design-system analysis: zero findings. The repository-wide verifier remains non-zero because of pre-existing baseline drift outside this task.
- Live CDP: canonical envelope, seeded recipient row, exact Research Workspace handoff, matching metadata, read-only banner, and no unexpected HTTP errors all passed. The recovered local-context exact-path miss and unbound shared source/chat data plane remain assigned to TASK-12020.40.
- Independent review findings were addressed by aligning nullable API fields, strongly typing hook mocks, using workspace-specific action labels, and adding the missing regression coverage.
- Bandit: not applicable because this task changes TypeScript/TSX, tests, and task documentation only.

**Status:** Complete
