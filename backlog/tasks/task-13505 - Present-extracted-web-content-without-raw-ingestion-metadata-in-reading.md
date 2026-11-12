---
id: TASK-13505
title: Present extracted web content without raw ingestion metadata in reading
status: Done
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
- [x] #3 Verify actual extracted web payloads in single and multiple-item reading on mobile.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed screenshot: saved-batch-mobile.png. Existing extractMediaDetailContent returns raw content.text. ADR required: no; presentation change within existing ingestion/detail contract. No source fix made during validation.
Plan: IMPLEMENTATION_PLAN_media_live_ux_fixes_20261006.md. Preserve raw content for analysis/export; clean only the reading presentation. ADR required: no; same persistence and detail contract.
Red: both real reader components displayed the stored content_hash envelope. Implemented a presentation-only envelope parser with JSON validation and string/escape-aware object boundary; default single/multi readers and inline comparison use clean text. Raw detail, structured export, editing and analysis inputs remain unchanged. Green: parser/content/comparison/single export suites 31/31; full reading suite 144/144 after correcting a helper-rename import. Real-browser verification pending.
Independent review reproduced hidden metadata spoken by Read full item. Added a real playback regression (red: first audio was [METADATA]); read-along and transcript segmentation now receive the cleaned reading source while export/edit/analysis retain raw content. Full read-along suite and session/localization checks passed (128 tests across 3 files).
Final live browser check found metadata-only generated sections and reading statistics still using the raw envelope. Red: navigation metadata section visible and API raw word count 99 shown; green: 38/38 permalink/export tests after filtering metadata-only character ranges, rebasing article offsets, and computing reading statistics/progress from the clean presentation. Stored content remains the source for editing, analysis, note/flashcard actions and export.
Final verification: Docs/Reviews/2026-10-06-media-live-ux-fixes.md and output/playwright/media-live-fixes-20261006/receipts.json. Latest remote dev 1fc353c3; tested source 9a277cf162. 411/411 tests (17 files), WebUI typecheck, lint (0 errors, 180 baseline warnings), WebUI production/token/budget checks and Chrome production build pass. Independent final delta review has no material findings. Bandit invoked on all 23 touched TS files: 0 findings, 23 unsupported-language parse errors; no Python application change and no TypeScript security pass claimed. Raw-source and account ownership protections reviewed/tested. Task-owned services/browser/dependency links/build caches cleaned; original tracked changes and prior untracked entries preserved. Native VoiceOver speech/human participant study remain unverified; full backend/all-pages/packaged-extension matrix not repeated. ADR required: no; existing boundaries retained. Draft PR preparation complete; merge awaits a fresh human-owned Change summary under repository policy.
Draft PR #3204: https://github.com/rmusser01/tldw_server/pull/3204, against dev. Created and attached for review. All implementation stages complete; completed task plan removed. Fresh human-written Change summary remains the merge gate; no merge or auto-merge attempted.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Default single/multiple reading, comparison and read-aloud exclude only a valid leading ingestion envelope. Section targets/statistics/progress align with the clean body; raw editing/analysis/export inputs remain unchanged. Real source 3 shows the article with 25 words/156 characters, while stored metadata/API raw count remain intact. Parser, playback, navigation and export regressions pass; desktop/mobile evidence recorded.
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
