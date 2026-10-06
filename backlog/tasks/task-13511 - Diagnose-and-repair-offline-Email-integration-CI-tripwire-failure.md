---
id: TASK-13511
title: Diagnose and repair offline Email integration CI tripwire failure
status: In Progress
labels:
- ci
- knowledge-followup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up the single optional CI failure on merged Knowledge PR3196: run37412488174 job112104842235 passed239 tests but recorded one forbidden call in the offline nested-email upload fixture. Trace the actual caller and repair the cause without weakening the offline guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify the forbidden operation and reproduce or deterministically exercise its caller
- [x] #2 Preserve rejection of model work, background jobs, DNS and outbound requests
- [ ] #3 Run the affected integration coverage, formatting and touched-scope Bandit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User authorized remaining Knowledge follow-ups on2026-10-06. Related TASK-13453 and PR3196. ADR required:no; focused CI isolation or ingestion repair, unless diagnosis establishes a changed durable rule.
Diagnosis reproduced on latest dev1fc353c3f6: local SQLite four cases pass; configuring PostgreSQL under the unchanged tripwire records forbidden socket calls from quota_resolver.user_quota and ResourceDailyLedger.initialize through persistence._enforce_and_record_media_bytes. Existing PR3016 and PR3084 already isolate that accounting boundary and improve diagnostics. Reuse and validate their fix rather than duplicate it; remote integration remains dependent on those PRs.
Recorded full diagnosis and existing repair ownership in Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md. Disposable accounting-boundary isolation passes 44 Email integration tests with the original offline guards preserved. No CI product/test patch duplicated here; PR3016/PR3084 integration and green remote CI remain required. Bandit is not applicable to this task because it changes no production Python.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Diagnosis complete: PostgreSQL quota/ledger accounting enters sockets under the offline guard. Existing PR3016/PR3084 own the repair. Keep this task In Progress until that repair is integrated and confirmed by CI.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
