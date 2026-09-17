# UAT205 — Study Pack pending status and duplicate-submission guard

TASK13260.143. Source and tests are frozen for independent review; native queued→terminal acceptance remains parent-owned. The worker/configuration investigation is separate.

## Change

The accepted POST202 job ID now keeps the Create study pack button disabled and loading throughout first-response waiting, queued/running polling gaps and transient status-fetch errors. A visible `role=status`, polite, atomic message explains acceptance, queued work, generation, or temporarily unavailable progress. Five English keys use the existing translation/fallback mechanism.

Previously `canSubmit` ignored the job ID and loading depended on the short-lived `isFetching` flag. Every settled poll could re-enable another job POST. Failed/cancelled jobs still clear the job ID, report the existing error and preserve title/source inputs for a deliberate retry; completed with a deck still calls the existing callback, closes and navigates once. Independent review also proved that the backend can return a completed job without a usable deck, and polling then stops. The first pending guard incorrectly retained that terminal job forever. The approved correction reports “Study pack completed, but its review deck is unavailable.”, clears the accepted job ID and preserves inputs for deliberate retry without claiming success or navigating.

Owned paths only:

1. `apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx`
2. `apps/packages/ui/src/assets/locale/en/option.json` — five additions; every existing value unchanged.
3. `apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.pending-job.test.tsx`

`owned-manifest.json`, `owned.patch` and `review-snapshot/` bind the final bytes. The baseline drawer/English resource were copied before production edits. No worker, job hook, API client, polling interval, existing test, backend or account policy was changed.

## Causal tests and controls

The new suite mounts the real AntD drawer and real TanStack create/status hooks with the production English resource and real i18next. It stubs only remote Study Pack service calls, notifications/navigation and feature availability. Query responses use normal state transitions; settled-poll controls verify `fetchStatus=idle` before attempting another actual button click. The tests do not assert source text or CSS classes.

- Original source: **5 failures / 5 passing controls**, `causal-red.log`.
- Independent reviewer actual-hook completed/null probe: **1 failure**, retained in `reviewer-completed-missing-result-red.log`; the first reviewed release is preserved in `first-reviewed-release/`. Permanent missing-pack/missing-deck controls on the first guard implementation: **2 failures / 10 deselected**, `completed-missing-permanent-red.log`.
- Final twelve-case test replay against only the frozen original drawer via a nonmutating Vite loader: **7 failures / 5 passing controls**, `review-corrected-baseline-red.log`, 4.50s. Current hooks/resource remain loaded; this is a drawer-only baseline replay.
- Final corrected source: **34 passed / 4 files / zero skips**, `review-corrected-green.log`, 4.48s. Earlier 32-case receipts describe the initial implementation and remain retained as history.

The original five RED cases are accepted-first-response status, queued and running idle-gap duplicate guards, queued→active fetch→running lifetime, and a temporary polling error. Positive controls cover failed/cancelled retry with exact original source/title payload; successful deck callback/navigation once; rejected creation; and an old POST acceptance resolving after a freshly keyed drawer is mounted under the same QueryClient. Existing four drawer tests, five hook tests and thirteen ImportExport decomposition tests pass unchanged.

The two added terminal controls prove both missing pack and missing deck release the guard, show a truthful error, retain exact inputs, make no success callback/navigation, and accept a deliberate new job.

The keyed-remount control exercises the drawer boundary used by `FlashcardsManager`'s keyed ImportExportTab; it is not a native account-switch or full AuthNZ test. No account scope policy was added or bypassed.

## Exact verification commands

From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.pending-job.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useStudyPackQueries.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx
bunx vitest run --config ../../.tmp/uat205-repair-20260917/baseline.config.ts ../packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.pending-job.test.tsx
bunx tsc --noEmit --incremental false
```

From the repository root:

```sh
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.pending-job.test.tsx -f json
source .venv/bin/activate && python -m bandit apps/packages/ui/src/components/Flashcards/components/StudyPackCreateDrawer.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/StudyPackCreateDrawer.pending-job.test.tsx -f json -o .tmp/uat205-repair-20260917/bandit-review-corrected.json
```

ESLint: **zero errors / zero warnings**, matching the clean original drawer. New test was formatted with installed Prettier before final RED/GREEN. Full compiler: **90 baseline / 90 current**, output byte-identical, zero owned diagnostics. This is not a clean whole-project compiler claim. Compiler output is retained under `tsc-baseline.log` and `tsc-review-corrected.log`.

Bandit: **zero findings, two TSX parse errors**. Bandit cannot analyze these TypeScript files; this provides no TSX security assurance. Manual scope review finds only local pending derivation and translated feedback, no new network request, credential access, privilege path or raw content rendering. JSON parse and exact five-key locale delta pass; the owned patch adds no trailing whitespace. `verification.json` summarizes these receipts.

## Boundaries and handoff

The accepted job remains pending during this open drawer session until terminal handling acts, including the explicit completed-without-usable-deck failure path. Existing close→reopen or new-intent resets and keyed parent remount semantics are unchanged; this unit does not introduce durable cross-close/reload tracking or cancellation of remote jobs. Inputs remain editable and are retained on failure just as before. Poll failures do not imply a terminal job failure and cannot enable another POST.

Parent was notified of final production hashes after the last edit so subsequent native receipts can distinguish HMR/source boundaries. No browser, native runtime, native data, model/provider, git, tracker or task mutation occurred here. Parent owns independent review, native queued→terminal acceptance and integration. Root-reported native queued job evidence is in `.tmp/uat198-181-native-20260917/pack-queued-events.txt` and `pack-queued-no-status.png`; this author did not operate or reclassify the worker.
