# Media-Multi Page UAT Remediation Plan

**Created:** 2026-06-14
**Source review:** `Docs/Design/2026-06-14-media-multi-page-uat-review.md`
**Scope:** `/media-multi` (`MediaReviewPage`).

`/media-multi` is well-built and functional — selection, batch actions, Compare/Focus/Stack views, Escape-to-close, and de-duplicated list fetches all work (the latter two validate the merged #2351 and #2353 fixes). Only two findings; one is fixed here, one is a library follow-up.

---

## Stage 1: Testability — stable selectors + e2e coverage — **Done (this PR)**
**Finding:** #1. The result rows and batch-action controls lacked `data-testid`s, leaving the core multi-select → batch flow without stable selectors or dedicated e2e coverage.
**Fix:** added `data-testid="media-review-result-row"` to result rows and `media-multi-batch-*` testids to the batch controls (keywords/add-tags/export-format/export/reprocess/trash). Added `e2e/media-multi-bulk-select.spec.ts` covering selection → batch toolbar + actions, and Escape closing the selected-items drawer (regression guard for the merged Escape fix on this page).
**Tests:** `e2e/media-multi-bulk-select.spec.ts` (2 tests, green); existing `MediaReviewPage.stage5.batch-toolbar` / `…export-trash-handoff` unit tests still pass.
**Status:** Completed.

## Stage 2: `@tanstack/react-virtual` flushSync warning — **Open (library)**
**Finding:** #2. A dev-only "flushSync was called from inside a lifecycle method" warning fires on load, originating from `@tanstack/react-virtual` (`^3.13.18`) — `listVirtualizer.measureElement(el)` in the row ref triggers the library's internal `flushSync`.
**Approach:** track an upgrade of `@tanstack/react-virtual` (later v3.x lines reduce/avoid the flushSync path), or, if it persists, evaluate measuring via a `ResizeObserver`/`requestAnimationFrame` outside the synchronous ref. Benign in production (dev-only), so low priority and intentionally not patched here to avoid risky virtualizer changes.
**Status:** Not Started (library follow-up).
