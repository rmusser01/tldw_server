---
id: TASK-13507
title: Use singular saved-item wording for one-item ingest results
status: Done
labels:
- media
- ux
- copy
dependencies:
- TASK-13503
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real corrected-PDF import on latest dev succeeds with one saved item but its review action reads Review these 1 saved items. Use count-aware action copy for the common single-item workflow. Screenshot: output/playwright/media-live-validation-20261005/corrected-real-results.png. See validation report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One saved item uses singular wording; multiple items use the correct plural in result and history review actions.
- [x] #2 Verify the accessible button name and visible label for counts 1 and 3 with existing localization patterns.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Minor copy finding from actual latest-dev results, not an ingestion failure. ADR required: no; reuse current translation/count convention. No source fix made during validation.
Plan: IMPLEMENTATION_PLAN_media_live_ux_fixes_20261006.md. Use existing translation plural/count conventions for saved-item CTAs. ADR required: no.
Result and recent-import CTA fallback wording uses singular for one; actual English resources use existing ICU plural formatting. Also corrected Configure eligible-item singular copy in the same flow. Red: rendered one-item CTA and real English resource assertions failed. Green: one/many button and actual ICU resource checks passed with the supporting suites. Existing authoritative handoff/owner-fencing checks were updated for singular labels and remain green. WebUI typecheck/Chrome production build passed. No dependency or localization framework added.
Independent review caught history using review:mediaPage.reviewImportSaved rather than option namespace. Reproduced against the actual consumed English resource, fixed en/review.json and removed unused option entry; one/many actual ICU checks now pass.
Final verification: Docs/Reviews/2026-10-06-media-live-ux-fixes.md and output/playwright/media-live-fixes-20261006/receipts.json. Latest remote dev 1fc353c3; tested source 9a277cf162. 411/411 tests (17 files), WebUI typecheck, lint (0 errors, 180 baseline warnings), WebUI production/token/budget checks and Chrome production build pass. Independent final delta review has no material findings. Bandit invoked on all 23 touched TS files: 0 findings, 23 unsupported-language parse errors; no Python application change and no TypeScript security pass claimed. Raw-source and account ownership protections reviewed/tested. Task-owned services/browser/dependency links/build caches cleaned; original tracked changes and prior untracked entries preserved. Native VoiceOver speech/human participant study remain unverified; full backend/all-pages/packaged-extension matrix not repeated. ADR required: no; existing boundaries retained. Draft PR preparation complete; merge awaits a fresh human-owned Change summary under repository policy.
Draft PR #3204: https://github.com/rmusser01/tldw_server/pull/3204, against dev. Created and attached for review. All implementation stages complete; completed task plan removed. Fresh human-written Change summary remains the merge gate; no merge or auto-merge attempted.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Singular/plural labels now use existing ICU resources and accurate fallbacks in Results and Recent imports. Real single import 8 exposes Review this 1 saved item and Review 1 saved item; the mixed batch exposes plural for two. Actual consumed namespaces and one/three-item accessible names pass regression checks.
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
