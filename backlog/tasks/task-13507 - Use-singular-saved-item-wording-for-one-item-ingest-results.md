---
id: TASK-13507
title: Use singular saved-item wording for one-item ingest results
status: To Do
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
- [ ] #1 One saved item uses singular wording; multiple items use the correct plural in result and history review actions.
- [ ] #2 Verify the accessible button name and visible label for counts 1 and 3 with existing localization patterns.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Minor copy finding from actual latest-dev results, not an ingestion failure. ADR required: no; reuse current translation/count convention. No source fix made during validation.
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
