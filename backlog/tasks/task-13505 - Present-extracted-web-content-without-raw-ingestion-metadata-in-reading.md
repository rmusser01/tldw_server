---
id: TASK-13505
title: Present extracted web content without raw ingestion metadata in reading
status: In Progress
labels:
- media
- ux
- bug
dependencies:
- TASK-13503
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real URL ingestion of https://example.com stores a [METADATA] JSON wrapper at the start of content.text. Multi-item reading displays that wrapper, including content hash and scraping pipeline, before the article. This crowds the 390px reading viewport and exposes implementation details. See the latest-dev live validation report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default content reading presents the article with readable provenance; retain raw stored content for export or explicit inspection.
- [x] #2 Handle the known metadata envelope safely without removing ordinary article text that happens to mention metadata.
- [ ] #3 Verify actual extracted web payloads in single and multiple-item reading on mobile.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed screenshot: saved-batch-mobile.png. Existing extractMediaDetailContent returns raw content.text. ADR required: no; presentation change within existing ingestion/detail contract. No source fix made during validation.
Plan: IMPLEMENTATION_PLAN_media_live_ux_fixes_20261006.md. Preserve raw content for analysis/export; clean only the reading presentation. ADR required: no; same persistence and detail contract.
Red: both real reader components displayed the stored content_hash envelope. Implemented a presentation-only envelope parser with JSON validation and string/escape-aware object boundary; default single/multi readers and inline comparison use clean text. Raw detail, structured export, editing and analysis inputs remain unchanged. Green: parser/content/comparison/single export suites 31/31; full reading suite 144/144 after correcting a helper-rename import. Real-browser verification pending.
Independent review reproduced hidden metadata spoken by Read full item. Added a real playback regression (red: first audio was [METADATA]); read-along and transcript segmentation now receive the cleaned reading source while export/edit/analysis retain raw content. Full read-along suite and session/localization checks passed (128 tests across 3 files).
Final live browser check found metadata-only generated sections and reading statistics still using the raw envelope. Red: navigation metadata section visible and API raw word count 99 shown; green: 38/38 permalink/export tests after filtering metadata-only character ranges, rebasing article offsets, and computing reading statistics/progress from the clean presentation. Stored content remains the source for editing, analysis, note/flashcard actions and export.
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
