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
TASK-13414 (2026-10-01) re-reviewed the four retained classes against fresh full all-pages runs on dev d81c13fddd: standalone build:dev bundle (CI gate profile) and next dev --webpack, backend MINIMAL_TEST_APP single_user, with an uncommitted recorder logging every console/request issue the gate classifies. (1) m5-drawer-width-deprecation-noise: fixed at source (KanbanPlayground ArchivedItemsDrawer width -> size; also the two WritingPlaygroundShell drawers) and removed. With the fix reverted and no entry, the next dev Kanban recovery fixture fails on the warning; with the fix it passes. (2) m5-optional-resource-404-noise (/moderation, 10-08): removed. The GET still 404s on the minimal backend (server log), but the browser 404 lands after the gate classifies, so it matched nothing in the local runs or in CI run 36946184688 (102 passed, no allowlisted hits). The underlying miss (no moderation router in MINIMAL_TEST_APP; the page requests it anyway) is still open here. (3) The two forced route-boundary signatures still fire in next dev on all 16 fixture routes. They cannot fire in the production bundle because shouldForceRouteError is disabled there, and the CI gate does not run that describe. Kept at 2026-10-31 with rationale pointing here. (4) New deliberate signature for AC#3: the Wayfinding 404-recovery test's own document 404 for /__wayfinding-missing-route__. It was covered by the old broad 404 rule and has been unallowlisted since the 27-rule retirement, which made that non-gated test fail. Re-added as the narrow m5-wayfinding-missing-route-document-404 entry, expiring 2026-10-31.
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
