---
id: TASK-15660
title: Show review draft save outcomes in the WebUI Quick Ingest receipt
status: In Progress
labels:
- webui
- ux
dependencies:
- TASK-15659
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the draft receipt from review #3633, following #3639. Report pending, saved, empty and failed local writes and keep the extracted result usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Show all four outcomes truthfully and open only confirmed review drafts.
- [x] #2 Read extracted text and retry failed writes without another ingestion request or duplicate drafts.
- [ ] #3 Verify desktop and mobile in Chromium and open a PR against dev with both screenshots.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. This work uses existing browser draft persistence and the current extraction viewer. No new retention or API policy.
Use a source-scoped local write state in the receipt, with per-source text actions that stay available throughout. Verify rollback and retry using the actual browser draft store and check that ingestion requests do not repeat.
Chromium verified pending, failed, saved and empty receipts. On 390px and 1280px, a real IndexedDB transaction was aborted, its rollback verified, and retry produced one saved draft with the original text and a single extraction. An independent code review prompted fixes for extract-only provenance, empty source versus failure, summary-only output, and focus after retry.
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
