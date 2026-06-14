# Media-Multi Page UAT Remediation Plan

**Created:** 2026-06-14
**Source review:** `Docs/Design/2026-06-14-media-multi-page-uat-review.md`
**Scope:** `/media-multi` (`MediaReviewPage`).

`/media-multi` is well-built and functional — selection, batch actions, Compare/Focus/Stack views, Escape-to-close, and de-duplicated list fetches all work (the latter two validate the merged #2351 and #2353 fixes). The two findings are fixed in this PR.

---

## Stage 1: Testability — stable selectors + e2e coverage — **Done (this PR)**

**Finding:** #1. The result rows and batch-action controls lacked `data-testid`s, leaving the core multi-select → batch flow without stable selectors or dedicated e2e coverage.
**Fix:** added `data-testid="media-review-result-row"` to result rows and `media-multi-batch-*` testids to the batch controls (keywords/add-tags/export-format/export/reprocess/trash). Added `e2e/media-multi-bulk-select.spec.ts` covering selection → batch toolbar + actions, and Escape closing the selected-items drawer (regression guard for the merged Escape fix on this page).
**Tests:** `e2e/media-multi-bulk-select.spec.ts` (2 tests, green); existing `MediaReviewPage.stage5.batch-toolbar` / `…export-trash-handoff` unit tests still pass.
**Status:** Completed.

## Stage 2: `@tanstack/react-virtual` flushSync warning — **Done (this PR)**

**Finding:** #2. A dev-only "flushSync was called from inside a lifecycle method" warning fires on load, originating from `@tanstack/react-virtual` (`^3.13.18`) — `listVirtualizer.measureElement(el)` in the row ref triggers the library's internal `flushSync`.
**Fix:** defer row measurement with `requestAnimationFrame` and skip disconnected nodes before calling `listVirtualizer.measureElement(el)`, moving the virtualizer call out of the synchronous React ref/commit path.
**Status:** Completed.
