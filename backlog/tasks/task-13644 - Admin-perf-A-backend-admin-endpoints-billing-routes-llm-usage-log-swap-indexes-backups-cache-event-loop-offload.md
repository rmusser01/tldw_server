---
id: TASK-13644
title: "Admin perf A: backend admin endpoints \u2014 billing routes, llm_usage_log\
  \ swap, indexes, backups cache, event-loop offload"
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Executed 2026-10-06..09 as PR #3218 (merged 2026-10-14). Implements F1-F7 of Docs/Reviews/ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md per Docs/Plans/2026-10-06-admin-webui-perf-A-backend-endpoints-implementation-plan.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
5/5 stages review-approved. Commits 9ba7897af7, 8a15997d47, c7f2879001, e930e54995, eea72fe0ce+b4640e9e88. Billing admin routes under /api/v1/admin/billing with SQL paging; llm_usage_v2 (never existed) swapped to indexed llm_usage_log; sessions(created_at)+org_members(org_id,user_id) indexes via Migration(101)+PG DDL with verified upgrade path; backups TTL cache + page-scoped stats; event-loop offload incl. SLA metrics read. Evidence: Docs/Reviews/ADMIN_WEBUI_PERF_EXECUTION_SUMMARY_2026_10.md.
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
