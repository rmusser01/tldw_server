# UAT211 — display the saved review gap accurately

TASK13260.149. Author implementation is frozen for independent review. **96 focused tests and 112 adjacent tests pass.** Native acceptance remains parent-owned; this agent did not access, rate, reset, restore or delete either native card.

## Result and scope

The saved-review toast, both Manage density views, editor metadata and reset confirmation now share `formatFlashcardReviewGap`. A saved ten-minute learning/relearning step displays `10 minutes` or `10 min`, including when the scheduler retains a former positive day interval. The formatter describes the saved schedule and does not use the current clock.

Five production files:

- `apps/packages/ui/src/components/Flashcards/utils/date-display.ts`: small pure formatter reusing `parseFlashcardTimestamp`.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`: saved toast only.
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`: compact and expanded gap labels only.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`: metadata, reset summary and related explanatory tooltip only.
- `apps/packages/ui/src/assets/locale/en/option.json`: twelve related messages; all original resource values remain identical.

Three existing mounted test fixtures were extended and one pure formatter test was added. Exact nine-path hashes, original hashes, patch and copies are in `owned-manifest.json`, `baseline-manifest.json`, `owned.patch`, `baseline/` and `review-snapshot/`. Owned manifest SHA256: `880d5fc5a3d26d1b89c1a9d6f2072ff50dcb74fc9364c30e1f1e74f709118039`.

The approved design is `../uat211-diagnosis-20260917/DESIGN211.md`; implementation stages are in `PLAN211.md`. Existing date formatting, shared `TFunction` formatters, numeric ICU plurals and scheduler preview units were inspected before implementation. Scheduler source establishes that intermediate learning/relearning steps can retain an old positive `interval_days`, so a zero-only fix would still be inaccurate. This unit changes no scheduling algorithm, request, rating value, version, query ownership, persistence or production source link.

## Display contract

| Saved fields | Result |
| --- | --- |
| Learning/relearning with a valid positive due-minus-last-reviewed gap | Use that gap, including when the old day interval is positive. |
| Ordinary review or legacy response with a positive finite day interval | Preserve the existing day interval, including absent timestamps. |
| Zero-day review or legacy response with a valid positive timestamp gap | Display the measured gap. Explicit `queue_state=review` does not hide a measurable subday schedule. |
| New/suspended, or missing/invalid/nonpositive measured gap | Compact `—`; full `not available`. Never invent zero days or reuse an invalid learning card's previous day interval. |

Measured gaps round fractional minutes upward, with a minimum displayed minute. Exact whole hours/days use those units; other gaps retain minutes (90 minutes stays 90 minutes). This preserves the existing integer-minute preview convention without calling the scheduler. Full words use numeric ICU cardinal plural messages; compact units use the existing interpolation adapter. Ordinary positive day-scale review intervals remain unchanged.

## Causal RED and controls

The permanent mounted cases were written before production changes. `permanent-red.log` recorded **13 failures / 23 passes**: eleven subday failures, one related day-plural assertion, and one stale preexisting source-label assertion. `same-instance-red.log` separately recorded **2 failures** showing the same mounted Manage/editor view retained the wrong display after new saved fields arrived.

The initial private diagnosis's source-label mismatch was not caused by replacing its translator. This earlier attribution is superseded by `preexisting-source-label-baseline.log`: a nonmutating Vite loader replayed the **exact original reset fixture plus all five original production files** and reproduced its expected `Message #m-12` mismatch. Current accepted `source-reference.ts` already says `Conversation for message #m-12`. Parent explicitly approved that one test-only assertion alignment; production source links are untouched. The first green run (`green.log`, 95 passes/1 failure) retained this failure before alignment.

After the alignment, the final unchanged-original-production replay (`final-baseline-replay.log`) produces **15 failures / 23 passes**, with all 38 mounted tests running. Thirteen failures are subday or same-instance display mismatches; two are the related reset summary's old `day(s)` wording. There is no stale source-link mismatch in that final replay. The loader validates each original snapshot SHA before loading it. The new pure test is deliberately excluded from baseline replay because the old source has no formatter export; no fake implementation is added to make old-source unit tests run.

The private AntD/JSDOM visibility limitation is preserved in the diagnosis packet. Permanent reset-modal controls assert the actual opened title and rendered text content, without claiming JSDOM animation/layout visibility. Manage label checks use visible text. No tests submit a native reset.

## Final verification

| Check | Result | Receipt |
| --- | --- | --- |
| Focused formatter + three mounted suites | **96 passed / 4 files**, 7.69s | `final-green.log` |
| Adjacent date, Cram practice, rerating, queue, editor metadata/save and review-run suites | **112 passed / 7 files**, 4.41s | `adjacent-green.log` |
| Exact original production replay, current mounted tests | **15 expected failures / 23 passes**, 8.11s | `final-baseline-replay.log` |
| Full frontend compiler | **90 baseline / 90 final**, byte-identical output; no added error | `tsc-baseline.log`, `tsc-final.log` |
| Scoped ESLint | **0 errors / 28 existing warnings**, identical path/rule/severity/message multiset; zero added | `eslint-complete-baseline.json`, `eslint-final.json` |
| English resource comparison | Exactly twelve additions; every old value unchanged | `static-comparison.json` |
| Diff whitespace | PASS | `diff-check.txt` |
| Bandit | Zero findings, **eight TS/TSX parse errors** | `bandit-final.json` |

Bandit does not support these TypeScript files; its zero findings are not TypeScript security assurance. A first final ESLint invocation from the frontend directory ignored the sibling package as outside its base path; `eslint-outside-base-not-validation.json` is retained and is **not** counted as validation. The corrected final invocation uses the repository root and explicit existing frontend config and matches the actual logical-path baseline.

The 58 pure cases use real i18next and the project's ICU adapter with both production English resources and fallback messages. Controls include singular/plural minutes/hours/days, 90-minute precision, retained relearning days, explicit legacy review/zero, missing/invalid/inverted gaps, new/suspended cards, epoch/timezone parsing, same translator count changes, a frozen input object and a throwing `Date.now` spy. The mounted tests use real ICU fallback translation and controlled remote mutation/query seams. They preserve rating UUID/value assertions, reset callback semantics, ordinary days, practice-only behavior, save behavior and same-instance updates. These are frontend behavioral tests, not new native/API/DB persistence claims.

## Exact commands

From `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-previews.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx ../packages/ui/src/components/Flashcards/utils/__tests__/review-gap.test.ts

bunx vitest run ../packages/ui/src/components/Flashcards/utils/__tests__/date-display.test.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.scheduling-metadata.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.save.test.tsx ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardReviewRun.test.tsx

bunx tsc --noEmit --pretty false
```

From repository root:

```sh
node apps/tldw-frontend/node_modules/vitest/vitest.mjs run --config .tmp/uat211-repair-20260917/baseline.config.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-previews.test.tsx ../packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx

UAT211_ORIGINAL_FIXTURE=1 node apps/tldw-frontend/node_modules/vitest/vitest.mjs run --config .tmp/uat211-repair-20260917/baseline.config.ts ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx -t 'confirms and invokes reset scheduling callback'

node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Flashcards/utils/date-display.ts apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-previews.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx apps/packages/ui/src/components/Flashcards/utils/__tests__/review-gap.test.ts -f json

source .venv/bin/activate
python -m bandit apps/packages/ui/src/components/Flashcards/utils/date-display.ts apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-previews.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.reset-scheduling.test.tsx apps/packages/ui/src/components/Flashcards/utils/__tests__/review-gap.test.ts -f json
```

## Handoff and limits

Source and tests are frozen. The parent was notified when production edits stopped. Independent review and the parent's real UI acceptance remain pending. The native 600-second source receipt is parent-provided evidence; this author did not claim new native observations. No runtime/browser/provider/DB/task/tracker/git mutation was performed by this unit. The existing 90 compiler errors and 28 scoped warnings are baseline, not a whole-project clean claim. This package is ready for review, not final native closure.
