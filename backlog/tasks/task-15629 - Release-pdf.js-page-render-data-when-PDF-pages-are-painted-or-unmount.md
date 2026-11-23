---
id: TASK-15629
title: Release pdf.js page render data when PDF pages are painted or unmount
status: In Progress
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
