---
id: TASK-13406
title: Review remaining M5 smoke exceptions before October 8
status: To Do
created_date: 2026-10-01 08:02
references:
- https://github.com/rmusser01/tldw_server/pull/3023
- TASK-13377.9
documentation:
- Docs/Product/Completed/WebUI-related/M5_1_Smoke_Warning_HardGate_Allowlist_Policy_2026_02.md
- Docs/Operations/Email_Real_PST_Validation_2026-09-26.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on TASK-13377.9 and PR3023 before the four remaining exceptions expire 2026-10-08. Preserve their existing WebUI ownership. Repair the kanban Drawer deprecated width prop at its source; replace the minimal-backend moderation list miss with an explicit fixture or supported endpoint; review the two deliberate route-boundary signatures within fixture scope. Do not blanket-renew dates or broaden suppression. This is future frontend maintenance, outside PR3023 merge closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-review the four retained classes before 2026-10-08 using fresh route evidence.
- [ ] #2 Retire removable exceptions through source or fixture repair while preserving strict unexpected-error and expiry enforcement.
- [ ] #3 Record a policy disposition for unavoidable deliberate fixture emissions without widening signatures or route scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate search completed through official CLI after MCP task_search300s timeout. CLI root create failed with Maximum call stack size exceeded; diagnostic /tmp/email_pr3023_ux_followup_create_cli_20261001.log retained. First explicit MCP request rejected unknown status Backlog, with no task created; repository config confirms To Do/In Progress/Done. Explicit ID13406 is above verified largest existing main task13405; official MCP used instead of manual task-file edits. Four existing rule owners are WebUI; no human assignee is invented. Evidence /tmp/email_pr3023_ux_quality_20261001.json and /tmp/email_pr3023_ux_raw_diagnostics_20261001.jsonl. Current repair checks:6focused,101strict production all-pages,16forced recovery fixtures pass; scopes overlap.
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
