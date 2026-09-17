# Independent review — UAT235 / UAT242

**Verdict: clear; no actionable findings. Native acceptance remains pending.**

Reviewed the exact two production and three test changes against HEAD; hashes in `hashes.json`. The successful-response merge at `ReviewTab.tsx:626` replaces stale scheduling fields/interval previews in the retained re-rate card while preserving front/back and other content absent from the response. The actual API response schema contains the reviewed UUID and scheduling fields. The merge remains after the real review-run authority guard and outside the practice-only branch. Failed saves do not reach it. Existing scope reset, queue membership, undo counters, and timeout behavior are unchanged.

Inspected the real review-run lifecycle and canonical response schema as integration context. Its stale/invalidated result returns null before the snapshot merge; owner rotation, invalidation and pending-operation controls are included in the independently rerun adjacent suites. Cram tests assert remaining-queue behavior after reordered and unchanged refetch; Due tests assert successive returned interval updates plus preserved card content. Failure control verifies the existing safe error and unchanged preview, rather than expecting raw exception text.

English and default Due completion now use the established ICU plural mechanism. The tests mount the actual ReviewTab with the project's ICU adapter and actual English resource, then independently omit resources to exercise fallback. Both one and two completed-card counts render correctly. Other locales are unchanged and were not claimed repaired or exhaustively validated.

## Verification

Independent installed-local Vitest: **100 passed / six files / zero skips**, including the 40 focused cases (not 140 distinct cases). Receipt: `tests.log`.

Command, from `apps/tldw-frontend`:

```sh
node node_modules/vitest/vitest.mjs run ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-completion.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.scope-change.guard.test.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardReviewRun.test.tsx --reporter=verbose
```

Inspected author RED: three stale interval-preview failures plus two real ICU singular failures; sixth failure was a corrected test expectation for raw versus safe error wording. The author retained a corrected negative control on original production. Independent review did not swap production to recreate RED.

Limits: no browser/native generation, backend/DB actions, compiler rerun, source/test/task edits or commits. Parent owns final compiler/lint baseline comparison; those author results are not counted as independent checks here. Two initial review command path errors did not launch tests; the command above is the actual successful run. No source changes were needed for the test run.
