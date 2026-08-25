# Task 8 Implementer Report

## Implementation Summary

- Added strict TypeScript contracts for the canonical recipient bootstrap, sources, preview, history, chat request, and chat response envelopes.
- Added `sharedWorkspacesApi` using `getTldwServerURL`, `fetchWithTldwAuth`, `buildTldwApiError`, canonical `/api/v1/sharing/shared-with-me/{share_id}` paths, `URLSearchParams`, source-ID encoding, and direct `AbortSignal` propagation.
- Added bounded runtime response parsing. Invalid server envelopes become a typed retryable `shared_workspace_unavailable` client error, while malformed generation defaults fail closed and deny grounded questions.
- Extended structured API errors with normalized `code`, `recovery_action`, and non-negative `retry_after_ms` while preserving existing unrelated detail fields.
- Added a pure shared-workspace reducer whose initial actions are denied and whose state owns bootstrap, source query/page/selection, messages/history cursor, exact draft, provider/model, immutable pending submission, preview, rate limit deadline, and in-pane errors.
- Added `useSharedResearchWorkspace` with one abort controller per bootstrap/source/history/preview/submission operation, cleanup of every active operation, a monotonically increasing share generation, reset-before-bootstrap behavior, and stale response fencing.
- Added immutable submission receipts: normalized query and sorted unique source IDs, one UUID allocation, exact object reuse only after ambiguous transport failure, edit invalidation, source-conflict refresh/new UUID, bounded rate countdown, and draft clearing only after a stored success.
- Replaced Task 1's static shell with stable loading, neutral not-found, unavailable, and loaded placeholders driven only by the shared controller.

## TDD Evidence

Exact RED command from the brief:

```text
cd apps/packages/ui && bunx vitest run src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx --maxWorkers=1 --no-file-parallelism
```

RED result:

```text
2 failed files, 1 passed file
Client and reducer/controller suites failed import because the required modules were absent.
The unchanged route suite passed all 14 existing assertions.
```

Final GREEN result for the same exact command:

```text
3 test files passed
44 tests passed
```

Covered behavior includes authenticated fetch and exact canonical URLs/query encoding; bounded response/error normalization; neutral 404 behavior; fail-closed generation defaults/actions; synchronous route reset; all-operation abort cleanup; source reconciliation; older-history deduplication; immutable same-UUID transport retry; draft/source/provider/model invalidation; sorted unique source scope; stale-response and stale-conflict suppression; source conflict refresh/new UUID; rate-limit deadlines; context-budget draft preservation; and all four route placeholders.

Additional self-review RED/GREEN cycle:

```text
bunx vitest run src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts --maxWorkers=1 --no-file-parallelism
RED: 3 failed, 24 passed
GREEN: 27 passed
```

The three failures proved that the first implementation rejected the backend's canonical preview shape, did not sort/deduplicate source IDs before freezing, and could refresh sources after an aborted stale source-conflict response.

## Focused Regressions

```text
cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__/auth-fetch.test.ts src/services/tldw/domains/__tests__/workspace-api.skills.test.ts src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts --maxWorkers=1 --no-file-parallelism
4 test files passed
42 tests passed
```

These regressions preserve authenticated-fetch behavior, existing domain-client exports, and local Research Workspace route parsing.

## Static And Security Gates

```text
../../tldw-frontend/node_modules/.bin/tsc -p tsconfig.json --noEmit --pretty false
exit 134: Node exhausted its 4 GB heap before emitting diagnostics
```

A temporary focused config including the Task 8 files reached diagnostics. It reported the route test's existing jest-dom matcher type gap and imported package baseline errors for `chrome`, `browser`, `import.meta.env`, and one optional OCR module; no Task 8 production file had a diagnostic. The temporary config was removed.

```text
../../tldw-frontend/node_modules/.bin/eslint --config ../../tldw-frontend/eslint.config.mjs <all touched TS/TSX files>
exit 0
```

ESLint emitted only the shared Next.js config's missing-pages notice because `apps/packages/ui` is a library package. Bandit is not applicable because Task 8 changes no Python production code.

```text
git diff --check
passed
```

## Exact Files Changed

- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/progress.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-8-implementer-report.md`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/shared-research-workspace-reducer.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/useSharedResearchWorkspace.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts`
- `apps/packages/ui/src/services/tldw/api-error.ts`
- `apps/packages/ui/src/services/tldw/domains/index.ts`
- `apps/packages/ui/src/services/tldw/domains/shared-workspaces.ts`
- `apps/packages/ui/src/services/tldw/domains/__tests__/shared-workspaces.test.ts`
- `apps/packages/ui/src/types/shared-workspace.ts`
- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`

## Scope And Self-Review

- Confirmed the recipient client exposes only bootstrap, list sources, preview source, list messages, and ask. It contains no owner mutation methods and does not import `useSharing.ts`.
- Confirmed every URL uses the canonical recipient prefix. No alias, redirect, legacy local workspace endpoint, or mutation path was added.
- Confirmed the shared component tree imports no local Research Workspace store. The existing local lazy route and `/research-workspace` behavior are unchanged.
- Confirmed route work is placeholder-only: no banner/trust UI, no complete source/chat visual surface, and no visual redesign from Task 9.
- Corrected runtime fail-closed handling for impossible TypeScript generation-default states and canonical chat-turn citation placement during the green cycle.
- Self-review corrected canonical preview parsing, normalized frozen source scopes, and prevented stale aborted source conflicts from launching refresh work.
- Confirmed the two unrelated untracked watchlist templates remain untouched and unstaged.

## Concerns And Residual Risks

- Package-wide TypeScript could not complete within Node's 4 GB heap; the focused check remains non-green on imported package/test baseline diagnostics, with no Task 8 production diagnostics.
- Task 8 intentionally provides placeholders rather than the complete recipient interface. Task 9 must render the typed state/actions without introducing local store dependencies.
- Live owner/member/non-member browser and real backend/provider execution are deferred to Task 11 UAT.

## Fix Round 1/5 - Controller Lifecycle And Scope Corrections

### Implementation Summary

- Replaced first-page-only chat selection semantics with explicit `sourceScopeMode: "all" | "include"`. Bootstrap starts in `all`; immutable all-scope requests carry no source IDs. Explicit subsets switch to `include` and retain a deduplicated cross-page ID set.
- Source refresh now receives its exact query. Filtered or paginated pages remove only returned IDs that are no longer retrieval-ready and preserve off-page IDs; a complete unfiltered page may also reconcile authoritative removals. Unfiltered source summaries remain the all-scope count authority.
- Blocked all-scope chat when the authoritative queryable count exceeds 500 until an explicit include subset of 1-500 IDs is selected. Exposed select-all and clear transitions for Task 9 without adding Task 9 visuals.
- Added current-controller identity checks to bootstrap, sources, history, preview, and submission success/error paths. Chat responses must match the submitted `request_id`; mismatches fail closed without appending messages, clearing drafts, or retaining retry UUIDs.
- Added an operative bounded `rateLimitRemainingMs` countdown and submit/retry deadline gates. Successful/new submissions clear expired rate state.
- Pending submissions now retain the exact raw draft and draft revision. A stored success clears only that unchanged exact revision; edits made in flight survive.
- Reclassified retry receipts so only ambiguous fetch/transport `TypeError` failures retain the immutable UUID. Any received HTTP error, including an untyped detail object, discards it.

### RED/GREEN Evidence

Covering test file: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts`.

```text
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts --maxWorkers=1 --no-file-parallelism
RED: 1 failed file; 11 failed, 15 passed.
Failures covered missing all/include state, filtered/paginated persistence, all payload shape, untyped HTTP retry classification, same-share source/preview/history/submission reordering, response-ID mismatch, over-500 all-scope gating, and countdown state.
GREEN: 1 passed file; 27 passed.
```

Exact Task 8 command from the brief:

```text
cd apps/packages/ui && bunx vitest run src/services/tldw/domains/__tests__/shared-workspaces.test.ts src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspaceRouteGate.test.tsx --maxWorkers=1 --no-file-parallelism
3 test files passed; 54 tests passed.
```

Focused existing regressions:

```text
cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__/auth-fetch.test.ts src/services/tldw/domains/__tests__/workspace-api.skills.test.ts src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts --maxWorkers=1 --no-file-parallelism
4 test files passed; 42 tests passed.
```

### Files Changed In Fix Round 1

- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/shared-research-workspace-reducer.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SharedResearchWorkspace/useSharedResearchWorkspace.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/shared-research-workspace-reducer.test.ts`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/progress.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-8-implementer-report.md`
- `backlog/tasks/task-12020 - Research-Workspace-2026-06-25-UAT-follow-up-remediation.md`
- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`

### Self-Review

- Confirmed partial pages never drop absent include IDs, while returned nonqueryable IDs are removed and a bounded complete unfiltered page can authoritatively reconcile removals.
- Confirmed all mode always serializes `source_ids: []`; include mode sorts/deduplicates IDs before freezing, and scope/provider/model/draft edits still invalidate failed replay receipts.
- Confirmed current-controller fencing applies symmetrically to success and error paths and stale source-conflict responses cannot launch refreshes.
- Confirmed response-ID mismatch preserves the current draft, appends no turn, clears the pending receipt, and cannot reuse the UUID.
- Confirmed rate gating uses the absolute deadline even if timer delivery is delayed, while the exposed countdown is clamped to 0-1,800,000 ms.
- Confirmed no client path, shared shell visual, local Research Workspace store, browser-extension contract, owner mutation, or watchlist template changed.

### Typecheck And Environment State

```text
../../tldw-frontend/node_modules/.bin/eslint --config ../../tldw-frontend/eslint.config.mjs <three touched TS/test files>
exit 0; only the shared Next.js missing-pages notice for this library package.

../../tldw-frontend/node_modules/.bin/tsc -p tsconfig.task8-fix.json --noEmit --pretty false
exit 2; 14 existing imported-package `ImportMeta.env` diagnostics, zero diagnostics in either changed production file. Temporary config removed.

git diff --check
passed.
```

Bandit is not applicable because this fix round changes frontend TypeScript only. No server, schema, PostgreSQL, provider, or browser process was started.

### Concerns

- Package typecheck remains non-green on the existing `ImportMeta.env` declaration baseline; the focused run emitted no changed-production-file diagnostic.
- Task 9 must derive visible checkbox state from explicit `sourceScopeMode` plus include IDs and retain the 500-source subset behavior. Task 11 still owns live owner/member/non-member and provider-backed UAT.
