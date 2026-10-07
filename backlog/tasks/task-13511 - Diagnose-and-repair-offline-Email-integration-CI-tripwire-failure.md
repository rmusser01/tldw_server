---
id: TASK-13511
title: Diagnose and repair offline Email integration CI tripwire failure
status: Done
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
- [x] #3 Run the affected integration coverage, formatting and touched-scope Bandit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User authorized remaining Knowledge follow-ups on2026-10-06. Related TASK-13453 and PR3196. ADR required:no; focused CI isolation or ingestion repair, unless diagnosis establishes a changed durable rule.
Diagnosis reproduced on latest dev1fc353c3f6: local SQLite four cases pass; configuring PostgreSQL under the unchanged tripwire records forbidden socket calls from quota_resolver.user_quota and ResourceDailyLedger.initialize through persistence._enforce_and_record_media_bytes. Existing PR3016 and PR3084 already isolate that accounting boundary and improve diagnostics. Reuse and validate their fix rather than duplicate it; remote integration remains dependent on those PRs.
Recorded full diagnosis and existing repair ownership in Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md. Disposable accounting-boundary isolation passes 44 Email integration tests with the original offline guards preserved. No CI product/test patch duplicated here; PR3016/PR3084 integration and green remote CI remain required. Bandit is not applicable to this task because it changes no production Python.
Published bounded follow-up fixes/evidence as draft PR3205 against latest dev: https://github.com/rmusser01/tldw_server/pull/3205 . The separate human Change summary, unresolved device/participant qualifications, independent provenance design and existing CI integration remain explicit in the PR/report. Disposable task runtimes/browser profiles are stopped; managed worktree retained for review.
Current dev1e06e03b587310ec023f3810c4480ea4e69b05f5 now includes merged PR3016 (2026-10-06T20:57:17Z), including test(email) isolate offline daily accounting. PR3084 remains open. No duplicate patch added here; affected current-dev verification and corresponding remote CI qualification remain pending.
Verified integrated repair on current dev1e06e03b587310ec023f3810c4480ea4e69b05f5 in an isolated tracked archive: all25 offline Email tests passed with8 warnings in7.11s, including upload/quota pool tripwire. Local Python3.11 is below declared dependency floors; no operator database or primary checkout was modified. Exact merged PR3016 head a6250c853bbb875686a3fc793de4c8749cf41868 has a successful Ubuntu/Python3.12 media-ingestion-new-integration shard:241passed,1skipped,5492warnings,593.71s; log names test_email_offline_ingestion.py. Shard URL https://github.com/rmusser01/tldw_server/actions/runs/37522638581/job/112473175804 . Same head backend-required, security-required and Lint & Type Check succeeded. All existing offline network/model/job/DNS guards remain. No duplicate product patch; production Bandit not applicable to this documentation/investigation task. PR3084 remains separately open but is no longer required to resolve the demonstrated PR3196 failure.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved through upstream PR3016, now merged into dev. Verified25 current-dev offline Email cases and the successful241-case Python3.12 ingestion CI shard at the repair head, with original forbidden-call guards preserved. No duplicate patch or operator-data mutation. Local dependency-floor and existing warning qualifications remain recorded.
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
