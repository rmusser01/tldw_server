---
id: TASK-13504
title: Normalize real Media detail identity in multi-item reading
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
Latest-dev live validation on 7226596c exposed reading cards and Open items labeled Media 3/4/5 although the actual detail API supplies source.title and source.type. The selected-reading header and list retain correct titles through selectedMetadata; detail enrichment only checks flat title/type and current-page rows, so saved-set/cross-page detail identity is lost. See Docs/Reviews/2026-10-05-media-live-validation.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Normalize source.title/source.type at the shared detail boundary and preserve legacy flat payload compatibility.
- [ ] #2 Reading cards, Open items, comparison and export use the actual source identity even when selected IDs are absent from the current list page.
- [ ] #3 Add a regression check with the actual nested detail payload, then verify it against the real API.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed by saved-batch-mobile.png and nested /media/3 API payload. Code sites: Review/hooks/useMediaReviewActions.tsx fetchDetail/ensureDetail/resolveDetailForCompare; MediaReviewReadingPane.tsx. ADR required: no; reuse existing API contract. No source fix made during validation.
User requested continuation after the live validation report. Implementation follows the report recommendations on latest dev 1fc353c3f67c93ba05102e7b0136ac4acac8f510. Plan: IMPLEMENTATION_PLAN_media_live_ux_fixes_20261006.md. ADR required: no; internal normalization of the existing DTO, retaining API/ownership boundaries.
Shared fetch normalizes request id plus nested source.title/source.type before reading, comparison and export callers. Red: actual nested DTO lost title/type. Green: entire reading-context suite 143/143, including off-page structured export. Real-browser verification pending in final stage.
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
