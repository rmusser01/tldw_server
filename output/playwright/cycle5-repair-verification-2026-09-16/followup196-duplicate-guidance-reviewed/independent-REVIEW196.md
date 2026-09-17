# Independent UAT196 source review

TASK13260.134. **Clear for parent-owned native acceptance.** Reviewed the two frozen files and the original component snapshot. No source, browser, server, database, task, tracker, or git changes made.

The new catch belongs only to `createDeckMutation.mutateAsync`: HTTP409 becomes actionable duplicate-name guidance through the existing status extractor and translation fallback. The original error remains its cause; the mutation cache and existing hook diagnostic still receive the original request error. Card409 and other failures retain their paths. Existing account checks and draft/source preservation remain intact. This introduces no query, auth, or retry-policy change.

## Fresh verification

Run from `apps/tldw-frontend`:

```sh
node node_modules/vitest/vitest.mjs run __tests__/flashcards-generated-save-errors.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/GeneratePanel.deck-recovery.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.deck-creation.test.tsx
```

**29 passed / 3 files / 0 skipped**, 6.75s (`green.log`). The real service/request/TanStack boundary and installed Next Pages observer use synthetic HTTP responses; this is not native or real-provider acceptance. Assertions cover inline/toast/summary guidance, retained original mutation diagnostic, edited-name retry, preserved card draft and source, and no unhandled Next overlay.

Independent nonmutating Vite loader replay replaces only GeneratePanel with the author's saved original bytes:

```sh
node node_modules/vitest/vitest.mjs run --config ../../.tmp/uat196-independent-20260917/baseline.config.ts __tests__/flashcards-generated-save-errors.test.tsx -t 'keeps a (deck|card) HTTP'
```

**1 expected failure / 4 passed / 8 deselected**, 4.32s (`baseline-red.log`). The failed assertion shows the original raw `Entity: decks` and HTTP path instead of guidance. All other selected HTTP controls pass. A preliminary loader-config write used the frontend working directory with a root-relative path and failed before creating anything; it was corrected before this replay.

Fresh scoped ESLint uses the same two paths: zero errors, one existing `no-explicit-any` warning. Author compiler comparison records 90 baseline/current diagnostics, none owned; not independently rerun here. Bandit cannot parse TSX (two author parse errors), so it provides no TS security assurance.

Both owned hashes and saved snapshots match before and after tests. Native acceptance and actual translated-language presentation remain parent-owned; the mounted test uses the existing translation fallback stub, not a claim of full ICU/locale coverage.
