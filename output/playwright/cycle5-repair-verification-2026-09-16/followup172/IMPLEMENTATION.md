# UAT172 — generated-card save recovery

Task TASK-13260.109. Frozen source and tests are listed in `owned-manifest.json`; baseline source copies are byte-identical to HEAD `1d790537974aadcce71c78b2c5b2329c4cf6007f`. Design was recorded before production edits in `DESIGN.md`.

## Cause and change
A failed account-scoped deck query remained errored after backend recovery. Global query focus/reconnect refetch is disabled, and Save/Retry only checked readiness; it never retried the read. The live DOM established that Save was not stuck busy: the earlier loading icon was stale snapshot output, so no spinner change was made.

GeneratePanel now explicitly refetches only an errored deck list through the existing scoped observer before resolving the save target. It uses the successful returned catalogue, rather than stale render data, and checks the original account again after the await. Pending queries still receive the existing wait error; ready queries keep their existing behavior. A still-failed read follows the existing visible error/retained-draft path. Deleted selected decks are rejected. No query-client, backend, service, or account protocol changes.

## Owned files
- `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`: one resolver change and stable refetch/error destructuring for callback dependencies.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx`: mounted actual panel, scoped deck query and mutation hooks; only request service boundaries mocked.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx`: the existing errored-query fixture supplies a failing refetch and checks that pending reads are not restarted.

## Evidence and results
- Initial before-production RED contains the direct causal failure: visible Retry makes no second deck read. The full final-source-independent replay `final-baseline-red-reset.log` is authoritative: **7 expected failures / 4 passing controls**, using only a private Vite loader for the baseline GeneratePanel. Production files were never reverted for replay.
- Final current source: **65 tests / 4 files PASS**, including all **11 real-query recovery controls**. Checks include edited draft and source/account attribution; blocked repeat click while recovery is pending; still-failing fetch with retry enabled afterward; initially pending query; account abort/change/unresolved during read; abort/unresolved before read; deleted selected deck; recovered empty catalogue creating exactly one deck/card; ready catalogue without a new read before save.
- Scoped ESLint: **0 errors / 16 warnings**, exact diagnostic multiset equal to baseline. No new warning. The root-cwd Next pages-directory advisory is retained in command output; source was linted with the actual frontend config.
- Fresh compiler runs: **90 baseline / 90 final diagnostics**, position-normalized comparison has **0 added / 0 removed**. Both exit2; no whole-project compile pass is claimed.
- Bandit: **0 findings, 3 TSX parse errors**. Bandit cannot analyze these TypeScript files; this gives no TSX security assurance.
- Diff checks emit no whitespace diagnostics. The untracked test's `git diff --no-index --check` returns1 because the file is added; the retained log is empty.

### Intermediate results retained honestly
The first GREEN attempt was 8/9 because a test clicked before TanStack's batched failed-query notification reached React; the fixture now flushes that task before Save. The first expanded baseline replay carried an unused deferred mock into later cases after an expected failure; the service mocks now reset implementations between every test, producing the final 7/4 result. An initial callback dependency warning was fixed by destructuring the stable query fields. An incorrect root bunx/compiler invocation and incorrect ESLint executable path were retried with the installed frontend tools; final logs/results above are the valid executions.

## Reproduce
From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.generate.test.tsx
bunx vitest run --config ../../.tmp/uat172-repair-20260916/baseline.config.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx
bunx tsc --noEmit --incremental false
```

From repo root:

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx -f json
source .venv/bin/activate
python -m bandit apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx -f json
```

## Boundaries / next acceptance
Source and tests are frozen for independent review. Parent owns the preserved real native draft, source provenance, native Retry/save request receipts, canonical reload acceptance, task/tracker closure and commits. No agent browser, runtime, tracker, task or git mutation was performed. The native saved-draft result is not yet claimed.
