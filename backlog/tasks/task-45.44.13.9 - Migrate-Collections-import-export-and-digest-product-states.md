---
id: TASK-45.44.13.9
title: Migrate Collections import, export, and digest product states
status: Done
labels:
- design-system
- webui
- extension
- product-state
references:
- TASK-45.44.13
- https://github.com/rmusser01/tldw_server/issues/1670
- Docs/Design/tldw_web_design_system_visual_qa_checklist.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub #1670 with the Collections import and export wizard notices and results and the digest schedule load error. Use shared design-system components while preserving localized content, job progress, and the existing recovery and export actions in WebUI and the browser extension.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migrate the five import/export notices, both import outcomes, and the digest schedule load error to the shared design-system primitives.
- [x] #2 Keep import progress, summary counts, partial-error limits, retry actions, and ZIP validation behavior covered by focused tests.
- [x] #3 Retire exactly the seven captured Collections baseline entries with no new product-state findings or stale entries.
- [x] #4 Record visual evidence for the shared route and verification results.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Collections remains under the long-tail issue #1670 at the requester’s direction. ADR required: no. This applies established Alert and StatePanel conventions without changing API or storage contracts, or introducing a durable design rule.
Base verified against remote dev at eb48892630330498346153332d6d3951fe942f41. The Collections baseline went from seven allowances to zero and the #1670 path map went from fourteen to seven. The repository-wide verifier still exits 1 with the same 97 blocked findings and 28 stale baseline entries as dev. The baseline combines cleanly with the changes in PRs #3671 and #3672.
All 48 focused Vitest tests passed across 15 files. Shared UI type checking and scoped ESLint passed. Bandit found no Python files in the touched Collections scope. The independent code review identified a missing partial-import announcement, which is now tested with a completed job that has errors and zero skipped items.
Visual QA covered Collections at 1440×900 and 390×844 in Chromium with fixtures. The selected source guidance stayed legible, the ZIP warning and its focused link fit at phone width, and Enter switched to JSONL with the selection intact. Desktop also covered upload progress, partial result counts and visible error details, and Import More. The packaged Chrome extension displayed the digest error and recovered on Retry. Saved before and final captures locally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Collections import and export notices and digest schedule recovery to shared Alert components. Used StatePanel for completed, partial, and failed imports. Import progress and terminal outcomes announce in the same region, including the error count for partial imports. The ZIP format switch preserves selected items.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
