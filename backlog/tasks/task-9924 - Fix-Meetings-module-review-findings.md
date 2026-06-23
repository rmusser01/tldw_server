---
id: TASK-9924
title: Fix Meetings module review findings
status: Done
assignee: []
created_date: '2026-06-23 18:40'
updated_date: '2026-06-23 18:41'
labels:
  - meetings
  - review-hardening
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated Meetings module review findings: sanitize SSE framing fields, make status transitions atomic, validate finalizable artifact kinds, and make finalization atomic/idempotent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SSE framing cannot be injected with newline-bearing event ids or event types.
- [x] #2 Session status transitions are atomic and reject stale concurrent transitions.
- [x] #3 Finalize include values reject unsupported final artifact kinds instead of silently skipping them.
- [x] #4 Repeated finalize calls do not create duplicate final artifacts and partial artifact writes are avoided.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-meetings-review-hardening.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replacement task created after an ID collision caused the original Meetings task file to disappear. Red checks: new SSE, stale-transition, unsupported finalize kind, empty include, and repeated-finalize tests failed against the prior implementation for expected reasons. Green checks: focused Meetings suite passed with --confcutdir=tldw_Server_API/tests/Meetings: 37 passed, 6 warnings in 18.23s. Security: Bandit on touched Meetings/API/DB source wrote /tmp/bandit_meetings_review_hardening.json with results_count=0 and no errors. Known skip: did not run test_meetings_routes_smoke.py because the developer guide calls out environment-specific heavy import crashes for that smoke path; focused endpoint coverage was run instead.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Meetings review findings by sanitizing SSE control fields, making service-driven status transitions conditional on the validated current status, rejecting unsupported final artifact kinds before writes, preserving explicit empty include lists, and replacing generated final artifacts atomically/idempotently. Added regression coverage for each behavior and verified the focused Meetings suite plus Bandit on touched source.
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
