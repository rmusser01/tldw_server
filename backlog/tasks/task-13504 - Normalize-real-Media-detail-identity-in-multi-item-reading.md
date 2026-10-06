---
id: TASK-13504
title: Normalize real Media detail identity in multi-item reading
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
Latest-dev live validation on 7226596c exposed reading cards and Open items labeled Media 3/4/5 although the actual detail API supplies source.title and source.type. The selected-reading header and list retain correct titles through selectedMetadata; detail enrichment only checks flat title/type and current-page rows, so saved-set/cross-page detail identity is lost. See Docs/Reviews/2026-10-05-media-live-validation.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Normalize source.title/source.type at the shared detail boundary and preserve legacy flat payload compatibility.
- [x] #2 Reading cards, Open items, comparison and export use the actual source identity even when selected IDs are absent from the current list page.
- [x] #3 Add a regression check with the actual nested detail payload, then verify it against the real API.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed by saved-batch-mobile.png and nested /media/3 API payload. Code sites: Review/hooks/useMediaReviewActions.tsx fetchDetail/ensureDetail/resolveDetailForCompare; MediaReviewReadingPane.tsx. ADR required: no; reuse existing API contract. No source fix made during validation.
User requested continuation after the live validation report. Implementation follows the report recommendations on latest dev 1fc353c3f67c93ba05102e7b0136ac4acac8f510. Plan: IMPLEMENTATION_PLAN_media_live_ux_fixes_20261006.md. ADR required: no; internal normalization of the existing DTO, retaining API/ownership boundaries.
Shared fetch normalizes request id plus nested source.title/source.type before reading, comparison and export callers. Red: actual nested DTO lost title/type. Green: entire reading-context suite 143/143, including off-page structured export. Real-browser verification pending in final stage.
Final verification: Docs/Reviews/2026-10-06-media-live-ux-fixes.md and output/playwright/media-live-fixes-20261006/receipts.json. Latest remote dev 1fc353c3; tested source 9a277cf162. 411/411 tests (17 files), WebUI typecheck, lint (0 errors, 180 baseline warnings), WebUI production/token/budget checks and Chrome production build pass. Independent final delta review has no material findings. Bandit invoked on all 23 touched TS files: 0 findings, 23 unsupported-language parse errors; no Python application change and no TypeScript security pass claimed. Raw-source and account ownership protections reviewed/tested. Task-owned services/browser/dependency links/build caches cleaned; original tracked changes and prior untracked entries preserved. Native VoiceOver speech/human participant study remain unverified; full backend/all-pages/packaged-extension matrix not repeated. ADR required: no; existing boundaries retained. Draft PR preparation complete; merge awaits a fresh human-owned Change summary under repository policy.
Draft PR #3204: https://github.com/rmusser01/tldw_server/pull/3204, against dev. Created and attached for review. All implementation stages complete; completed task plan removed. Fresh human-written Change summary remains the merge gate; no merge or auto-merge attempted.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Normalized real nested detail identity at the shared fetch boundary for reading/comparison/export, retaining legacy DTO and owner fencing. Actual saved sources 3/4/5 and fresh 9/10 display correct titles/types; off-page regression and real API verification pass. Final evidence and known validation limits are in the 2026-10-06 report.
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
