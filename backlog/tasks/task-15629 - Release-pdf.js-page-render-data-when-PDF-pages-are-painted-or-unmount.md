---
id: TASK-15629
title: Release pdf.js page render data when PDF pages are painted or unmount
status: Done
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live measurement (headless Chrome, production WebUI build, 300-page scanned PDF, 1240x1756 JPEG per page) showed the document workspace renderer growing from ~700 MB to ~3.6 GB after scrolling through the document once, and never coming back while the tab sat idle (~10 MB kept per page ever viewed). pdf.js 4.8.69 (the version react-pdf 9.2.1 bundles and the worker copied to public/) keeps a rendered page's operator list and decoded images on its PDFPageProxy until cleanup() is called; it only does that by itself for images over 10 MB (MAX_IMAGE_SIZE_TO_CACHE), and react-pdf calls cleanup() only at the start of the next draw, never after a paint or on unmount. Release the data through the public PDFPageProxy.cleanup() once a page's canvas is painted and when the page unmounts (covers renders cancelled mid-paint), in one wrapper that the main viewer (PdfPage) and the Pages sidebar thumbnails (PagesTab) both use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A PDF page that finishes painting releases its pdf.js render data (PDFPageProxy.cleanup) without waiting for unmount
- [x] #2 A PDF page that unmounts with its render still in flight releases its render data and raises no unhandled rejection
- [x] #3 A page that unmounts and remounts renders again; two mounted consumers of one page proxy do not unmount or re-render each other
- [x] #4 The Pages sidebar thumbnails go through the same release path as the main viewer
- [x] #5 Targeted vitest files for PdfPage, PdfDocument and PagesTab pass; eslint clean on changed files
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. Bug fix inside one WebUI component using pdf.js's public PDFPageProxy.cleanup(), the same call pdf.js's own viewer makes when a page view is destroyed; no durable architecture rule is added or changed (searched Docs/ADR/README.md for pdf/webui/memory decisions; none govern pdf.js page lifecycle).
Root cause verified in source: the viewer and worker both run pdfjs-dist 4.8.69 (react-pdf 9.2.1 pins it; scripts/copy-pdf-worker.mjs copies react-pdf's copy, public/pdf.worker.min.mjs is byte-identical to 4.8.69 legacy). PDFPageProxy keeps _intentStates (operator list) and objs (decoded images, ImageBitmaps) until cleanup(); render completion only schedules it when _maybeCleanupAfterRender is set, which pdf.js sets itself only for an image over MAX_IMAGE_SIZE_TO_CACHE (10e6 bytes), and a 1240x1756 RGBA scan is 8.7 MB. react-pdf's Canvas calls page.cleanup() only at the start of its next draw, so the retained data is never reused, and on unmount it only zeroes the canvas.
Fix: ReleasingPage in PdfPage.tsx wraps react-pdf's Page and calls the public PDFPageProxy.cleanup() in onRenderSuccess and on unmount / pageNumber change; PdfPage (single, continuous and thumbnail modes of PdfDocument) and PagesTab both use it. No private _maybeCleanupAfterRender flag. pdf.js's own viewer uses the same public call (PDFPageView.destroy and the thumbnailrendered handler in web/pdf_viewer.mjs).
Real pdf.js 4.8.69 check (throwaway Node script with node-canvas, scratchpad only): 30 scanned pages +196 MB array buffers without release, +0 MB with cleanup after paint; cleanup-then-cancel and cancel-then-cleanup both end with the data released (immediately, or within pdf.js's 5 s delayed cleanup); cleanup() returns false while a second canvas renders the same proxy, that render still paints correctly, and the data is released 5 s later; a released page renders again.
Tests: red 7 failed (cleanup never called), green 8 passed for PdfPage.release.test.tsx + PagesTab.test.tsx; neighbours PdfDocument.memory, DocumentViewerAlerts.design-system, DocumentWorkspacePageAlerts.design-system, usePdfSearch: 42/42 passed. eslint (lint:ui config) clean; targeted tsc on PdfPage.tsx + PagesTab.tsx clean.
Remaining memory (pdf.js alone in Playwright's Chromium headless shell, 300-page 176 MB scanned PDF opened from a blob: URL, renderer RSS after forced GC and a 20 s settle): without release, load 288 MB, then 1,689 and 1,688 MB after passes 1 and 2. With release: pass 1 adds a one-time ~456 MB (742-839 MB), pass 2 adds nothing (840 MB). Neither a worker-isolate GC nor PDFDocumentProxy.cleanup() frees that step. doc.destroy() plus worker.terminate() frees ~180 MB (the worker's copy of the file) and leaves 560 MB against a 110 MB blank page, so the step is Chromium's working-set high-water mark from decoding and drawing large images, not referenced data. No further fix belongs in this task. While a document is open, the worker holds one copy of the file; the Pages sidebar's own <Document> adds a second worker and a second copy.
PR: https://github.com/rmusser01/tldw_server/pull/3579
Review follow-ups (2026-11-23): added tests that PdfPage still clears its spinner and shows the page note after paint, and that ReleasingPage passes onLoadSuccess/onRenderSuccess through. With the passthrough removed, exactly those 2 tests fail (Unable to find 'Note page 1'; onRenderSuccess not called with the proxy). Renamed the shared-proxy test to what it pins (cleanup per consumer unmount/paint; pdf.js deferral is not exercised with a mocked proxy) and dropped its mount-count assertions, which could not fail. The JSDoc now says a later render() cancels a deferred release and that an unmount mid-render releases up to ~5 s later through pdf.js's timer. Green: 6 files, 44 tests. eslint clean.
Live re-measurement by the coordinator on this branch (production WebUI, headless Chrome): the 300-page scan settles at ~1.12-1.14 GB after repeated scroll passes (base 3.3-3.6 GB), no errors, search works; the 1,500-page text PDF is unchanged (~0.74-0.80 GB). Independent review: ready to merge, no Critical or Important findings.
Possible follow-up, deliberately not done here because open PR #3534 edits PdfDocument.tsx: key the single-page fallback PdfPage by currentPage. On a pageNumber change react-pdf briefly redraws the OLD page after the wrapper's cleanup, so that page's release then relies on pdf.js's cancel path (force-abort or the 5 s delayed cleanup).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PDF pages now release pdf.js's per-page render data (operator list and decoded images) through the public PDFPageProxy.cleanup() once their canvas is painted, and on unmount or page switch. ReleasingPage in PdfPage.tsx does this for every viewer mode and for the Pages sidebar thumbnails. Live on the production WebUI, a 300-page scan settles at ~1.12-1.14 GB after repeated passes instead of 3.3-3.6 GB. In plain pdf.js in Chromium the remaining step (~456 MB) is a one-time renderer high-water mark, not referenced data. PR #3579.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
