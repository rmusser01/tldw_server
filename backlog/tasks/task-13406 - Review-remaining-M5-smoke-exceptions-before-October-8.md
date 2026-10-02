---
id: TASK-13406
title: Review remaining M5 smoke exceptions before October 8
status: In Progress
assignee: []
created_date: '2026-10-01 08:02'
updated_date: '2026-10-02 01:39'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3023'
  - TASK-13377.9
documentation:
  - >-
    Docs/Product/Completed/WebUI-related/M5_1_Smoke_Warning_HardGate_Allowlist_Policy_2026_02.md
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate search completed through official CLI after MCP task_search300s timeout. CLI root create failed with Maximum call stack size exceeded; diagnostic /tmp/email_pr3023_ux_followup_create_cli_20261001.log retained. First explicit MCP request rejected unknown status Backlog, with no task created; repository config confirms To Do/In Progress/Done. Explicit ID13406 is above verified largest existing main task13405; official MCP used instead of manual task-file edits. Four existing rule owners are WebUI; no human assignee is invented. Evidence /tmp/email_pr3023_ux_quality_20261001.json and /tmp/email_pr3023_ux_raw_diagnostics_20261001.jsonl. Current repair checks:6focused,101strict production all-pages,16forced recovery fixtures pass; scopes overlap.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-10-01: Human authorized completing this task, TASK13407 and the TASK13178 tracking closeout. Reusing the clean isolated email worktree on codex/email-followup-closeout-20261001 from fresh dev d81c13fddd1dac1948b30401af0388932a0af8f2. Official MCP read calls did not return; official CLI fallback is available. Trace and reproduce before minimal source/fixture changes; preserve hard-gate and expiry enforcement. Plan: Docs/Plans/IMPLEMENTATION_PLAN_email_followup_burndown_20261001.md.

TASK13406: all four general exceptions retired. Supported AntD size400 replaces deprecated width; exact GET moderation list stub remains inside the existing minimal-smoke guard and disabled for live-tier UAT; deliberate exact forced-error emissions handled only in their own fixture tests. General allowlist empty; strict ordinary-route classification and UTC metadata enforcement preserved. Policy updated. Local verification completed 2026-10-02 UTC (October 1 local): final production build/token sync/bundle budget passed; relocated standalone includes 324 published server documents, actual manifest/content HTTP200 and traversal/unsupported-source HTTP400. Final strict ordinary smoke105 passed, classifier7 passed and development forced-boundary16 passed; XML has zero failures/errors/skips. Focused unit suite16 passed across5files. Touched-scope ESLint and smoke TypeScript pass; diff check passes. Scopes overlap and are not full repository or hosted CI certification. Production build/ordinary browser runner use local Node26; final development and recovery runner use installed Node20.19.5, unchanged 30-second navigation gates, precompiled admin route and task-local16GB heap. Two prior Node26 development runs each had2 navigation failures/14passes: cold compilation and measured Next memory restart; original logs retained, not green credit. No Python source change; Bandit N/A. Shared installations/Postgres unchanged. Independent source review of immutable patch5cfa514a8f4f7f64b71ca767c7a301fc0ed15890f3c524e40ef5a5737eee122f is clear after actual response-wait RED1/GREEN1; required response assertions remain fail-closed. Original failed/setup/rejected diagnostics retained under /tmp/email_followup_*_20261001. Official CLI interactive editor removed orphaned summary markers without changing historical notes. Local source verified; publication/new PR merge not yet claimed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
