---
id: TASK-15557
title: Skip Quick Ingest analysis for existing media
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Check the existing media identity before Quick Ingest analysis when overwrite is disabled, and surface the existing item and analysis skip in results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A confirmed duplicate with overwrite off makes no analysis provider call and returns the existing Media ID.
- [x] #2 Quick Ingest explains an analysis skip and opens the existing Media item.
- [x] #3 New items and explicit overwrite still reach analysis.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation follows upstream ADR-020 for owner-scoped Media storage and ADR-003 for the Jobs boundary. No new ADR. Reuse the persisted source URL or upload identity before admission and analysis. Keep save-time dedupe for content-only matches and concurrent first submissions. The synchronous fallback admits analysis once for any inputs that still need it. Direct Quick Ingest web scraping opts in on a persisted single URL. Results retain the existing Media ID and surface the skip warning.
Observed the duplicate worker and result view fail before the fix. Verified 97 backend tests plus 2 provider-quota endpoint cases, 256 focused frontend tests plus 53 wizard integration tests, the generated OpenAPI fingerprint and path union, and zero Bandit findings on the changed modules. Human Change summary and public dev merge are pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed already-saved source matches skip analysis and offer Open in Media. New sources and explicit overwrite still proceed through the analysis path.
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
