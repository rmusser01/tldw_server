---
id: TASK-15660
title: Show review draft save outcomes in the WebUI Quick Ingest receipt
status: Done
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
- [x] #3 Verify desktop and mobile in Chromium and open a PR against dev with both screenshots.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR. This work uses existing browser draft persistence and the current extraction viewer. No new retention or API policy.
Use a source-scoped local write state in the receipt, with per-source text actions that stay available throughout. Verify rollback and retry using the actual browser draft store and check that ingestion requests do not repeat.
Chromium verified pending, failed, saved and empty receipts. On 390px and 1280px, a real IndexedDB transaction was aborted, its rollback verified, and retry produced one saved draft with the original text and a single extraction. An independent code review prompted fixes for extract-only provenance, empty source versus failure, summary-only output, and focus after retry.
PR https://github.com/rmusser01/tldw_server/pull/3640 targets dev and embeds verified saved and failed receipts at 1280×800 and 390×844. 506 scoped unit and integration tests, all 19 Quick Ingest Chromium scenarios, frontend typecheck, scoped lint, translation parity and diff checks passed. Bandit found no Python source in the UI scope. Browser failure was a controlled IndexedDB abort; session tests separately simulate QuotaExceededError.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Published pending, saved, failed and empty review draft receipts with per-source text viewing and local-write retry. Transactional storage and source eligibility preserve the original destination when retrying mixed results.
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
