# UAT189 / UAT190 frozen author handoff

TASK13260.127 / TASK13260.128. Parent approved the design and reviewed causal RED before releasing production. [Design](DESIGN.md), [owned diff](owned.patch), [exact file hashes](owned-manifest.json), [scope checks](scope-check.json).

## Final change

- UAT190: the existing filtered no-match branch now also requires `!hasCramPracticeCards`. A successful queue still containing practiced matching cards displays the existing Cram completion message after exhaustion. Initial empty tagged results keep their no-match guidance. No queue, scheduler, session, query or mutation logic changed.
- UAT189: the component fallback and new English `reviewedThisCramSession` key use the existing ICU one/other pattern. One displays “1 card practiced in this cram session”; zero/multiple use “cards.” Counts and the existing positive-count UI guard stay unchanged.
- A new actual-component suite mounts real AntD, React i18next with ICU/English resources or empty-resource fallback, real Cram query and useFlashcardReviewRun. External data/mutations and unrelated query boundaries are fixtures. Existing Cram tests switch their fallback formatter to real ICU and update two formerly plural-only singular assertions. Their scheduling, identity, recovery and session assertions remain intact.

## Test-first evidence

`red-final.log`: **9 expected failures / 4 passing controls / 0 skipped**. Five fail resource/singular text; four fail completed nonempty tagged queues (scheduled, practice-only, recovered query and reordered queue). Initial empty and tag/deck/account reset controls pass. The exact new test bytes are retained in `red-final-test.tsx` and match the frozen final test.

The initial test run had two asynchronous harness mistakes (query-notification timing and a detached DOM element after deck change); these were corrected to await observable state and retained separately in `red-initial-harness.log`. They are not counted as product failures. Newly copied test mocks were then typed; the final RED remains9/4 and the new file has0 lint warnings.

`adjacent-baseline.log`: original Cram suite **20 passed** before production edits. `green-initial-with-legacy-test.log`: new13 all pass; old suite has3 failures caused by the handcrafted interpolation stub not supporting ICU and its two “1 cards” expectations. Only the formatter and those two expected strings were adjusted after preserving that evidence.

`green-final.log`: **64 passed / 6 files / 0 skipped** (5.28s). Exact command from `apps/tldw-frontend`:

```sh
bunx vitest run \
  ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-completion.test.tsx \
  ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx \
  ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx \
  ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx \
  ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.scope-change.guard.test.ts \
  ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.cram-queue.test.tsx
```

## Static verification

- `static-comparison.json`: scoped ESLint **0 errors,21 warnings before and after**, exact diagnostic messages unchanged. ReviewTab has2 existing warnings, existing Cram test19, new test0. Run from repository root with `apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs` and the three owned TSX paths; this checks the actual files. Initial discovery attempts used a nonexistent binary and then omitted the explicit config; neither is counted as validation. The correct run's Next pages-directory advisory is retained in stderr.
- `bunx tsc --noEmit --pretty false` from frontend: **90 baseline /90 current diagnostic messages, no changes and no owned-file diagnostics**. This is not a globally passing compiler claim.
- Bandit was run through the project venv on all three TSX paths. It reports0 findings and3 TSX syntax parse errors because it is a Python scanner; no JavaScript security coverage is claimed. Manual security review: only a display condition, plain localized strings and test fixtures change; no data, permission, network or error-reporting boundary changes.
- Exact source reconstruction confirms only the two intended ReviewTab replacements and one English resource addition. JSON parses; added lines have no trailing whitespace. All four source/test snapshots are frozen and hash-listed.

Native evidence establishing the defects is retained by the parent under `.tmp/uat-study-native-20260917/disposable-rating-success-evidence.txt` and `disposable-completed-session.png`: actual completed session2 count1 while the filtered list still contains the matching saved card. I viewed the screenshot and inspected the receipt. This author run adds no native acceptance claim. Independent review and parent final native completion acceptance remain pending. No browser/runtime/provider/task/tracker/git actions were performed.
