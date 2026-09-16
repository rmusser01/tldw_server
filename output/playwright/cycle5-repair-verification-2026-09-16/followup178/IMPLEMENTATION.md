# UAT178 / TASK-13260.115 — failed Cram queue recovery

## Change and boundary

A failed queue request left `data` absent, which ReviewTab treated as an empty, completed queue. The existing auto-end effect already required a successful query; it remains unchanged.

ReviewTab now displays a visible queue error and Retry, including when cached cards remain. Retry invokes only the existing scoped query's `refetch()`. Initial loading has explicit feedback. Successful completion and its recovery actions require a successful, settled Cram query. Cached active cards, practiced identities, deck/tag/scheduling choices and rating behavior remain intact.

Owned production scope is only `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx` (31 added lines). No query hook, service, backend, browser or runtime changes. Two existing test files are owned: `ReviewTab.cram-mode.test.tsx` gains real QueryClient/useCramQueueQuery boundary cases; `ReviewTab.create-cta.test.tsx` adds truthful success/error flags to one existing successful-empty fixture.

## Tests and causal evidence

The new cases mount ReviewTab with the real Cram query hook and QueryClient, substituting `listFlashcards` only at the service boundary. They cover initial failure and Retry success, repeated failure, initial loading, cached cards and practiced identities through failed refresh/retry, failed refresh after the last cached card is rated, and successful empty completion. They assert no rating or session-end mutation from loading/failure; the scheduled cached-last-card case ends exactly once only after the query recovers. Existing identity, rerating, scope, due-mode, orientation and session controls remain in the regression run.

- Original RED: **5 failed / 15 passed**, `red.log`.
- Final tests against the frozen original component via a nonmutating Vite loader: **5 failed / 15 passed**, `final-baseline-red.log`. The same final tests run against current code pass.
- Final regression: **65 passed / 5 files**, `final-tests.log`.
- Intermediate corrections are retained: `green.log` exposed one test expecting a disabled Retry during a no-data refetch. TanStack Query correctly clears that error while pending; the test now checks visible loading and absent Retry. `regression-tests.log` exposed the old empty-tag fixture's missing success flag; it was corrected to reflect the actual query contract.

From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx
bunx vitest run --config ../../.tmp/uat178-repair-20260916/baseline.config.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx
bunx tsc --noEmit --incremental false
```

## Static checks

- Full TypeScript: **90 existing errors before and after**, zero diagnostics added/removed after normalizing line positions. Exit 2 both times; this is not a clean project compiler. See `typecheck-{baseline,final}.log` and `typecheck-comparison.json`.
- Scoped ESLint: **0 errors / 118 existing warnings before and after**, exact file/rule/message multiset unchanged. See `eslint-{baseline,final,comparison}.json`. The comparison uses the ESLint API and the actual file paths/config for original snapshots and current files.
- `git diff --check -- <three owned paths>`: exit 0.
- Required Bandit run after activating `.venv`: 0 findings, **all three TSX files failed Python AST parsing**. Bandit gives no security assurance for this TypeScript scope; see `bandit-final.json` and log.

```sh
source .venv/bin/activate
python -m bandit apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx -f json -o .tmp/uat178-repair-20260916/bandit-final.json
```

## Handoff limits

This repairs truthful frontend status/recovery, not the backend queue 500 under separate DB work. Native HTTP-failure-to-recovery acceptance and independent review remain parent-owned. No source commit, task/tracker/shared-plan edit, browser action or service action was performed. `owned-manifest.json`, `owned.patch`, and `review-snapshot/` define the frozen source/test unit; original copies are validated against the manifest's baseline commit.
