---
id: TASK-13505
title: Present extracted web content without raw ingestion metadata in reading
status: To Do
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
- [ ] #1 Default content reading presents the article with readable provenance; retain raw stored content for export or explicit inspection.
- [ ] #2 Handle the known metadata envelope safely without removing ordinary article text that happens to mention metadata.
- [ ] #3 Verify actual extracted web payloads in single and multiple-item reading on mobile.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Confirmed screenshot: saved-batch-mobile.png. Existing extractMediaDetailContent returns raw content.text. ADR required: no; presentation change within existing ingestion/detail contract. No source fix made during validation.
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
