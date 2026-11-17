---
id: TASK-15554
title: Cancel Quick Ingest uploads before the job acknowledgement
status: Done
labels:
- quick-ingest
- media-ingestion
- cancellation
references:
- https://github.com/rmusser01/tldw_server/pull/3488
updated_date: 2026-11-17 01:07
documentation:
- Docs/ADR/003-jobs-vs-scheduler-default.md
- Docs/ADR/058-jobs-completion-row-identity.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve owner-scoped Cancel All for native uploads whose job was created before the multipart response reached the browser. Keep the existing Jobs and upload contracts. ADR assessment: reuse the existing media Jobs lifecycle and request scoping. No new ADR is required for the acknowledgement race.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancel All cancels the job while the native multipart response is pending.
- [x] #2 Cancellation before creation prevents publication and existing single-user and multi-user deployments retain their Jobs behavior.
- [x] #3 Record hosted staging Chrome verification and open the public PR against dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Public dev uses the existing SQLite and PostgreSQL Jobs stores for owner-scoped cancellation. Hosted staging Chrome withheld the HTTP 200 from a 44-page, 12,696,025-byte PDF. The same job was cancelled and the UI showed Cancelled in 1.241 seconds. Search found zero saved copies. Public verification covered 466 frontend tests, a 155-test backend slice, and a real PostgreSQL 18 concurrency and owner-isolation test. Scoped Bandit found no new runtime issues. OpenAPI fingerprint and path checks passed. Public PDF extraction, analysis, and save-phase cancellation is tracked in PR #3477.
ADR assessment: ADR-003 and ADR-058 govern the existing Jobs lifecycle and owner-bound row checks. This uses those existing boundaries without a new ADR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cancel All identifies a native upload before its submission response reaches the browser. Jobs publication and cancellation are serialized per owner in the existing Jobs database. The browser confirms cancellation before updating the affected result.
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
