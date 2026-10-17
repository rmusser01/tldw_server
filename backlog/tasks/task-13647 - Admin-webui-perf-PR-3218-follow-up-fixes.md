---
id: TASK-13647
title: 'Admin-webui perf PR #3218 follow-up fixes'
status: In Progress
labels:
- backend
- frontend
- admin
- testing
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-ups from merged admin-webui performance PR #3218: (1) getDailyUsage sent start_date/end_date but GET /api/v1/admin/usage/daily takes start/end, so the daily chart ignored the range selector; (2) get_cost_attribution group_by=user double-counted users belonging to multiple scoped orgs via the org_members INNER JOIN; (3) _backup_file_from_row os.stat was unguarded, so a backup deleted between the TTL-cached scan and page-build 500ed GET /admin/backups; (4) flaky ServerAdminPage.design-system password-reset test investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 getDailyUsage sends start/end (page + domain tests pin the wire param names)
- [x] #2 group_by=user counts each usage row once for multi-org members; group_by=org keeps row-per-org attribution; PG + SQLite dialects covered
- [x] #3 list_backup_items skips rows whose file vanished between scan and page build instead of raising
- [ ] #4 Password-reset test root-caused with evidence and stabilized; file passes 3x
- [ ] #5 getDailyUsage sends start/end matching backend /usage/daily params; range selector changes daily chart; test pins param names
- [ ] #6 cost-attribution group_by=user under multi-org scope counts each usage row once (semi-join); group_by=org keeps per-org attribution; test with user in two orgs
- [ ] #7 _backup_file_from_row skips rows whose file vanished (OSError) instead of 500; test covers vanish-mid-window
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no ADR required - three localized bug fixes (query param naming, SQL join shape, fs race guard) and one test-stability fix; no durable architecture rule introduced or changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Items 1-3 fixed with RED-first tests (daily-usage start/end params; cost-attribution org-scope semi-join preserving group_by=org row-per-org and SQLite plan sargability; backups vanished-file skip guard) and item 4 root-caused as wall-clock timeout starvation of a deterministic 8.4s-under-load test, stabilized via per-test 20s timeout + explicit popconfirm wait. 27 backend tests, all touched frontend suites, and Bandit clean. Full detail in .task-13647-report.md.
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
