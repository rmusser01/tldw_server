# TASK13260.40 / UAT099 independent final review

## Disposition

**CLEAR for the frozen seven-file scope; no actionable finding confirmed.** Reviewed the five production files and two tests frozen at 2026-09-15T23:56:55.774Z. All seven SHA-256 values matched before and after independent checks. No repository source, tests, browser, runtime, global documents or commits were changed by this review. Temporary review controls and this report are under `/private/tmp` only.

## Reviewed behavior

- The original three layout changes use the existing exact `/media` viewport policy, allow the Media root and ContentViewer to shrink, and retain the toolbar outside the existing scrolling source body. The independently scrolling sidebar and scrollable empty state remain intact. No competing document scroll listener or new layout mechanism is added.
- `ViewMediaPage.tsx:458` gates a seek on explicit current-view chapter intent. Passive remembered/default selection no longer increments the nonce; actual chapter actions still do. The nonce is reset on selected media/kind changes, and the existing deferred selection cancellation guard remains.
- Existing detail readiness is forwarded to the reading-progress hook. Restoration and listening begin when content is ready. The bounded frame retry policy is unchanged.
- `useMediaReadingProgress.ts:181` retains response lifetime/token guards and additionally checks the captured scroll revision both after GET and before a queued restore frame. A newer scroll cannot be replaced by that older restore operation.
- `useMediaReadingProgress.ts:244` records the progress payload at the scroll event. Cleanup flushes a real pending payload, using the original media/save closure even when the current ref has detached. No-input cleanup no longer manufactures a zero-progress write. Percentage zoom remains100; original debounce/deduplication and explicit save behavior are retained.

## Fresh independent evidence

1. **75 tests /15 suites pass**, exact command below. Log: `/private/tmp/uat099-independent-current-tests.log`. Includes the permanent original RED controls for actual page-owner passive/default and remembered selection, explicit chapter navigation, delayed content readiness, no-input cleanup, late GET versus newer scroll, pending geometry resize/ref detachment, and existing old/new media flush.
2. **Two additional private lifetime controls pass**, with the nine existing hook controls (11/1 total). They hold real queued animation-frame callbacks, verify newer user scroll wins, and execute an old-media callback after the new-media callback to verify cancellation and absence of no-input writes. Files: `/private/tmp/uat099-independent-lifetime.config.ts`, `/private/tmp/uat099-independent-lifetime-cases.txt`, `/private/tmp/uat099-independent-lifetime.log`. These are two additional cases, not11 additional unique tests. The initial temporary fixture accidentally created a fresh ref object every render, producing an extra effect/frame; the fixture was corrected to use the real stable-ref contract. Its retained `uat099-independent-lifetime-fixture-red.log` is a **harness failure, not a product RED**.
3. Fresh ESLint from repository root covers all seven actual paths: **0 errors,38 unchanged warnings,0 added/removed signatures** against the implementer's retained exact HEAD baseline. `/private/tmp/uat099-independent-eslint.json` and `uat099-independent-eslint-comparison.json`. The existing missing-root-pages advisory remains. No ignored-file result is counted as coverage.
4. Scoped `git diff --check` is clean.
5. `/private/tmp/uat099-independent-pre-hashes.json` and `uat099-independent-post-hashes.json`:7/7 frozen hashes match independently before/after.

## Commands

Working directory: `apps/packages/ui`.

```sh
./node_modules/.bin/vitest run src/hooks/__tests__/useMediaReadingProgress.test.tsx src/components/Review/__tests__/useMediaSelection.reading-progress.test.tsx src/components/Media/__tests__/ContentViewer.stage1.test.tsx src/components/Media/__tests__/ContentViewer.stage2.test.tsx src/components/Media/__tests__/ContentViewer.stage3.test.tsx src/components/Media/__tests__/ContentViewer.stage4.accessibility.test.tsx src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx src/components/Media/__tests__/ContentViewer.analysis-markdown.test.tsx src/components/Media/__tests__/ContentViewer.stage14.reprocess.test.tsx src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Review/__tests__/ViewMediaPage.stage13.error-handling.test.tsx src/components/Review/__tests__/ViewMediaPage.stage14.bulk-actions.test.tsx src/routes/__tests__/route-paths.viewport.test.ts src/routes/__tests__/option-media-multi.connection-state.test.tsx --maxWorkers=1 --no-file-parallelism
./node_modules/.bin/vitest run --config /private/tmp/uat099-independent-lifetime.config.ts --maxWorkers=1 --no-file-parallelism
```

Working directory: repository root.

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/routes/route-paths.ts apps/packages/ui/src/components/Review/ViewMediaPage.tsx apps/packages/ui/src/components/Media/ContentViewer.tsx apps/packages/ui/src/components/Media/hooks/useReadingProgress.tsx apps/packages/ui/src/hooks/useMediaReadingProgress.ts apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx apps/packages/ui/src/hooks/__tests__/useMediaReadingProgress.test.tsx --format json
```

## Limits

The actual owner regression runs real page/navigation/reading hooks with controlled external storage and scroll geometry through a ContentViewer seam. These tests do not prove CSS overflow or native restored position. Parent owns rebuilt-source native wheel, progress save/GET/reload, mobile geometry and source-preservation acceptance, plus the combined compiler. No full-fresh UAT or clean-TypeScript claim is made here. Bandit is inapplicable to this TypeScript-only review.

The earlier optional WebLayout chat-scroll-contract fixture has a documented unchanged missing-auth-mock-export failure and mocks the route policy to `/chat`; it is not meaningful Media geometry coverage. This review did not rerun that unrelated failing baseline or count it as green.
