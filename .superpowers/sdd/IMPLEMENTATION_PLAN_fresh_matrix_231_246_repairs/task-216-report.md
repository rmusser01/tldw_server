# TASK13260.216 / UAT275 — World Book parent summary refresh

## Outcome

Successful entry writes now invalidate the manager's existing per-book entry query and the parent `['tldw:listWorldBooks']` query. The parent query supplies both the table rows and `selectedWorldBookRecord`, so this refreshes the visible table count and selected-detail header without new local state.

The change stays inside `WorldBookEntryManager.tsx`. It does not change transports, API contracts, global query defaults, or error behavior.

## Causal coverage

A real `QueryClient` regression renders a parent summary query together with the actual entry manager. Before the fix, a successful add updates the backing entries state but leaves the visible parent count at `0`; the corrected RED is retained in [worldbook275-causal-red.log](../../../.tmp/uat-repairs-231-246/worldbook275-causal-red.log). After the fix, the visible count updates `0 → 1 → 0` after add and delete without reload. A failed add leaves the count at `0` and does not refetch the parent query.

A second real-QueryClient control covers a partial move: it copies an entry to the destination, makes source deletion reject, and verifies the visible destination parent count still updates to `1`. This failed before the post-copy parent invalidation and is retained in [worldbook275-move-causal-red.log](../../../.tmp/uat-repairs-231-246/worldbook275-move-causal-red.log).

The first test harness attempt failed while the initial parent query was still loading; it is explicitly separated and is not product evidence: [worldbook275-initial-readiness-attempt.md](../../../.tmp/uat-repairs-231-246/worldbook275-initial-readiness-attempt.md).

## Implementation

- Existing `onSuccess` paths for add, update, delete, bulk operations, and partial-success bulk add call a shared helper that invalidates the entry query and parent list query.
- Bulk move independently invalidates the parent list after at least one destination copy succeeds, before its source-delete request. This covers a successful copy followed by a rejected source delete.
- Partial bulk add is source-reviewed: its existing `result.succeeded > 0` branch now calls the shared invalidator even when failures coexist. The targeted suite includes existing bulk action controls, but this task does not add a second full UI lifecycle test for that branch.
- Existing `onError` paths remain responsible for errors and do not receive a new invalidation.

## Validation

- Focused suite: [worldbook275-focused-final.log](../../../.tmp/uat-repairs-231-246/worldbook275-focused-final.log) — **17 passed, 1 skipped**, exit 0. The skipped test is the existing `WorldBooksManager.entryStage1.test.tsx` responsive-drawer test, intentionally skipped because entries moved to the detail panel; this task did not change it.
- Causal green: [worldbook275-causal-green.log](../../../.tmp/uat-repairs-231-246/worldbook275-causal-green.log) — 3 passed, exit 0.
- Scoped ESLint: [worldbook275-eslint-final.log](../../../.tmp/uat-repairs-231-246/worldbook275-eslint-final.log) — exit 0, 42 warnings and 0 errors. Warnings are present elsewhere in the existing manager; none are in the new test.
- `git diff --check` for the manager and new regression — exit 0.
- Bandit: [worldbook275-bandit-final.json](../../../.tmp/uat-repairs-231-246/worldbook275-bandit-final.json) — 0 findings and 1 TypeScript parse error. Bandit does not provide TypeScript security assurance.

No clean project typecheck is claimed. The direct UI compiler was not repeated because it emits a repository-wide diagnostic set unrelated to this narrow change; the changed source is exercised by Vitest transforms and scoped ESLint.

Exact hashes are retained in [worldbook275-hashes.sha256](../../../.tmp/uat-repairs-231-246/worldbook275-hashes.sha256) and [worldbook275-verification.json](../../../.tmp/uat-repairs-231-246/worldbook275-verification.json).
