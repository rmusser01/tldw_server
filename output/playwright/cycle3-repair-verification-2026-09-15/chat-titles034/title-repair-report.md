# TASK-13260.34 — Chat browser and header titles

## Frozen scope

Production frozen at **2026-09-15T21:42:49Z**, unchanged since; six exact hashes are in `/private/tmp/uat034-production-freeze.json`. No runtime, browser, commit, global tracker, or shared-plan changes by this agent. TASK-13260.34 stays In Progress pending independent review and root's native acceptance.

### Root causes and repair

- `pages/chat/index.tsx` mounted no declarative Next Head title. Loader/selection code wrote `document.title` imperatively, while Next's real head manager computes an empty title when no mounted title element exists. The Chat page now owns its fallback and active title declaratively.
- Header read/wrote the legacy `@/db` chrome-storage catalog, despite restored mirrors using Dexie and the loader's canonical `serverChatTitle`. Both surfaces now use `useActiveChatTitle`: canonical metadata for saved chats and the current reactive Dexie adapter for local chats.
- Local rename remains a local transaction with no scope/network acquisition. Saved rename resolves the existing verified scope lease, checks captured account/conversation after awaited work, calls versioned `updateChat`, then updates canonical metadata and the mirror. The existing client/domain API gains optional scope/signal forwarding; only exact PUT `/api/v1/chats/{id}` joins the scoped route policy. Unscoped callers and conflict retry behavior remain supported.
- Per-selection cancellation rejects late reads/writes and A→B→A completions. Canonical same-principal rotation remains benign; unknown authority masks the title. A render-time, owner-bound edit draft prevents a one-commit glimpse of the previous conversation's draft.
- Failed server rename keeps the last canonical title, reports a small inline error, and offers **Retry rename**, which reopens the attempted text. Error/retry state is masked outside its captured owner.

### Owned production files

1. `apps/tldw-frontend/pages/chat/index.tsx`
2. `apps/packages/ui/src/components/Layouts/Header.tsx`
3. `apps/packages/ui/src/hooks/useActiveChatTitle.ts` (new)
4. `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
5. `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
6. `apps/packages/ui/src/services/tldw/service-prompt-scope-error.ts`

### Owned test files

- `apps/tldw-frontend/__tests__/pages/chat-title.integration.test.tsx` (new)
- `apps/tldw-frontend/__tests__/pages/chat-title.ssr.test.tsx` (new)
- `apps/packages/ui/src/services/tldw/__tests__/chat-title-scope-policy.test.ts` (new)
- `apps/packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts` (one added case)

Backlog TASK-13260.34 updated through official CLI; no .33-owned files changed.

## Verification

- Stable initial RED: 4 title failures (`uat034-title-red.log`); expanded RED: 9 failures / 21 passes (`uat034-title-red-expanded.log`). These reproduced missing Head, wrong catalog, stale reads, and missing updateChat scope forwarding.
- Additional behavior RED: Profiler captured `['Private draft title', '']` during selection switch (`uat034-title-edit-boundary-red.log`); Retry rename action absent (`uat034-title-retry-red.log`). Both are permanent controls now.
- **88/88 passed in 10 files**, including the Node SSR test, final `/private/tmp/uat034-title-final.log`.
- **1/1 actual Node SSR test passed**, `/private/tmp/uat034-title-ssr.log`. It imports the real Chat page/shared hook with `window` absent, renders through the actual Next Head context, and observes `Chat | tldw`. It does not substitute for a Next build or browser check.
- Tests use actual Header + ChatHeader, actual Next Head/head manager, and controlled account/store/transport fixtures. The committed local persistence test uses an explicit reactive adapter mock; it does not claim a real browser IndexedDB verification. A private exploratory installed-only fake-indexeddb run passed 14 tests before the final retry UI addition (`uat034-real-dexie-exploratory.test.tsx` / `.log`), but that undeclared dependency is not imported by committed tests and no package/lock changes were made.
- Covered: metadata hydration; Head rerender/remount; local rename/reload and two mounted title consumers; Strict Mode; saved scope/version/mirror rename; delayed local read/history change; canonical account change before IDs clear; A→B→A delayed save; deferred scope resolving under another account; same-owner token rotation and benign event; initial unresolved authority; failure/retry; first-render edit-draft masking; scoped route siblings/malformed IDs/method negatives.
- ESLint from repository root: **0 errors / 0 new warnings**. Existing client **532** and domain **235** warnings match exact HEAD messages; every other owned file has zero warnings. `/private/tmp/uat034-title-lint-final.json` and `uat034-lint-baseline-comparison.json`.
- Full TypeScript: final rerun matches all **90/90** merged-baseline diagnostic signatures with no additions/removals; comparison uses multiplicities and ignores line/column movement. `uat034-title-typecheck-final.log` and `uat034-title-typecheck-comparison.json`.
- Scoped `git diff --check`: clean. Bandit N/A (no Python files touched).

### Existing unrelated failure, explicitly preserved

Running the existing whole `services/__tests__/tldw-api-client.chat-mutations.test.ts` also gives **4 passed / 1 failed**: the Chat completion payload sanitization case retains synthetic `trace=/Users/private/stack.txt`. The identical failure was reproduced by loading the exact HEAD client and domain source, without repository edits, via `/private/tmp/uat034-api-baseline.config.mts`; baseline and log are retained. The existing **updateChat version-conflict retry passes** in both runs. This failure is not counted as passing and was not repaired under title scope.

### Independent overlap review correction

The first freeze had a P2 lost-edit race: a second rename submitted while the first was pending closed its editor and was discarded by the save lock. Original reviewer probe `uat034-independent-overlap.config.ts` was rerun unchanged: **1 failed / 14 passed** (`uat034-overlap-reproduced-red.log`). A permanent test also failed before repair (`uat034-overlap-permanent-red.log`).

The corrected Header keeps the newer edit open on busy Enter, displays **Saving conversation title**, and lets the user submit the retained text after the previous save completes. The hook publishes pending/completed state; no extra ChatHeader/API changes were needed. The permanent scenario confirms the second request uses the first response's updated version. Original reviewer probe plus permanent tests now **16/16 passed** (`uat034-overlap-independent-green.log`). The second freeze above supersedes the first; independent re-review is clear in `/private/tmp/uat034-independent-review.md` (original overlap, focused Header/Head+SSR, supplemental first-save-failure and principal-invalidation controls).

## Exact verification commands

From `apps/tldw-frontend`:

```sh
npm exec -- vitest run __tests__/pages/chat-title.integration.test.tsx __tests__/pages/chat-title.ssr.test.tsx ../packages/ui/src/components/Layouts/__tests__/Header.character-mode.test.tsx ../packages/ui/src/components/Layouts/__tests__/Header.share-links.integration.test.tsx ../packages/ui/src/components/Layouts/__tests__/Header.tts-clips-lazy-mount.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx ../packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts ../packages/ui/src/services/tldw/__tests__/chat-title-scope-policy.test.ts ../packages/ui/src/hooks/__tests__/useCanonicalConnectionConfig.test.tsx
npm exec -- vitest run __tests__/pages/chat-title.ssr.test.tsx
npm exec -- vitest run ../packages/ui/src/services/__tests__/tldw-api-client.chat-mutations.test.ts
npm exec -- vitest run --config /private/tmp/uat034-api-baseline.config.mts ../packages/ui/src/services/__tests__/tldw-api-client.chat-mutations.test.ts
npm exec -- tsc --noEmit
```

ESLint from repository root, explicit frontend config, on the ten owned source/test files:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/hooks/useActiveChatTitle.ts apps/packages/ui/src/components/Layouts/Header.tsx apps/tldw-frontend/pages/chat/index.tsx apps/tldw-frontend/__tests__/pages/chat-title.integration.test.tsx apps/tldw-frontend/__tests__/pages/chat-title.ssr.test.tsx apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/domains/chat-rag.ts apps/packages/ui/src/services/tldw/service-prompt-scope-error.ts apps/packages/ui/src/services/tldw/__tests__/chat-title-scope-policy.test.ts apps/packages/ui/src/services/tldw/__tests__/TldwApiClient.request-scope.test.ts -f json
```

## Native limitation

Root encountered a Next module-resolution overlay for the new hook while edits/HMR were active. No browser acceptance is inferred from those intercepted controls. Root owns the isolated frontend restart and native Robot/Cedar/Aster title, rename, local-mirror and reload checks against the frozen production hashes.
