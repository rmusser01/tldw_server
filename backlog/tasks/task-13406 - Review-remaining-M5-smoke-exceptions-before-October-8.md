---
id: TASK-13406
title: Review remaining M5 smoke exceptions before October 8
status: In Progress
assignee: []
created_date: 2026-10-01 08:02
updated_date: 2026-10-03 01:21
labels: []
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/3023
- TASK-13377.9
- https://github.com/rmusser01/tldw_server/pull/3077
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
- [x] #1 Re-review the four retained classes before 2026-10-08 using fresh route evidence.
- [x] #2 Retire removable exceptions through source or fixture repair while preserving strict unexpected-error and expiry enforcement.
- [x] #3 Record a policy disposition for unavoidable deliberate fixture emissions without widening signatures or route scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate search completed through official CLI after MCP task_search300s timeout. CLI root create failed with Maximum call stack size exceeded; diagnostic /tmp/email_pr3023_ux_followup_create_cli_20261001.log retained. First explicit MCP request rejected unknown status Backlog, with no task created; repository config confirms To Do/In Progress/Done. Explicit ID13406 is above verified largest existing main task13405; official MCP used instead of manual task-file edits. Four existing rule owners are WebUI; no human assignee is invented. Evidence /tmp/email_pr3023_ux_quality_20261001.json and /tmp/email_pr3023_ux_raw_diagnostics_20261001.jsonl. Current repair checks:6focused,101strict production all-pages,16forced recovery fixtures pass; scopes overlap.

2026-10-01: Human authorized completing this task, TASK13407 and the TASK13178 tracking closeout. Reusing the clean isolated email worktree on codex/email-followup-closeout-20261001 from fresh dev d81c13fddd1dac1948b30401af0388932a0af8f2. Official MCP read calls did not return; official CLI fallback is available. Trace and reproduce before minimal source/fixture changes; preserve hard-gate and expiry enforcement. Plan: Docs/Plans/IMPLEMENTATION_PLAN_email_followup_burndown_20261001.md.

TASK13406: all four general exceptions retired. Supported AntD size400 replaces deprecated width; exact GET moderation list stub remains inside the existing minimal-smoke guard and disabled for live-tier UAT; deliberate exact forced-error emissions handled only in their own fixture tests. General allowlist empty; strict ordinary-route classification and UTC metadata enforcement preserved. Policy updated. Local verification completed 2026-10-02 UTC (October 1 local): final production build/token sync/bundle budget passed; relocated standalone includes 324 published server documents, actual manifest/content HTTP200 and traversal/unsupported-source HTTP400. Final strict ordinary smoke105 passed, classifier7 passed and development forced-boundary16 passed; XML has zero failures/errors/skips. Focused unit suite16 passed across5files. Touched-scope ESLint and smoke TypeScript pass; diff check passes. Scopes overlap and are not full repository or hosted CI certification. Production build/ordinary browser runner use local Node26; final development and recovery runner use installed Node20.19.5, unchanged 30-second navigation gates, precompiled admin route and task-local16GB heap. Two prior Node26 development runs each had2 navigation failures/14passes: cold compilation and measured Next memory restart; original logs retained, not green credit. No Python source change; Bandit N/A. Shared installations/Postgres unchanged. Independent source review of immutable patch5cfa514a8f4f7f64b71ca767c7a301fc0ed15890f3c524e40ef5a5737eee122f is clear after actual response-wait RED1/GREEN1; required response assertions remain fail-closed. Original failed/setup/rejected diagnostics retained under /tmp/email_followup_*_20261001. Official CLI interactive editor removed orphaned summary markers without changing historical notes. Local source verified; publication/new PR merge not yet claimed.

Published reviewed fixes as d4693fefe37bc75d9d9b54026c77c12de3200c76 on codex/email-followup-closeout-20261001; normal push verified. Draft PR3077 contains the implementation and this tracking closeout. Repair acceptance is locally complete; this new PR is not merged and its human-written Change summary and hosted CI are pending. Source receipt /tmp/email_followup_closeout_receipt_20261001.json binds ten exact source paths and frozen verification artifacts. Browser coverage checks the AuthNZ manifest entry plus selected-document content/render; the real AuthNZ guide HTTP probe is separate. Task-owned servers stopped after identity verification; shared installations and unrelated resources preserved. Completed owned plan retained at /tmp/email_followup_plan_completed_20261001.md after all stages are complete.

Merge-gate audit correction: implementation and local verification are complete, but the repository-wide Definition of Done requires a human-written Change summary for this new AI-authored PR. PR3077 is draft and awaits that human input plus hosted CI. Restored In Progress with this explicit remaining DoD item; no source changes or repeat tests. The prior completed-plan snapshot is retained as /tmp/email_followup_plan_pre_gate_diagnostic_20261001.md, and the owned plan remains Stage3 In Progress until this requirement is satisfied. TASK13178 remains Done because its implementation PR2887 was already merged.
2026-10-02: Requester supplied PR3077 Change summary directly in this chat; published verbatim. Human gate satisfied. Dev 86e287fee7bfa1a1588639232e35db3666851ded advances original d81c13f base across 91 paths with smoke/task merge conflicts. Refresh preserves upstream history and test-local expected emissions. TASK13414 added a deliberate Wayfinding document-404 rule; retire it from the general allowlist within its own exact recovery fixture, with regression evidence. Task remains In Progress pending refreshed verification and actual merge.
TASK-13414 (2026-10-01) re-reviewed the four retained classes against fresh full all-pages runs on dev d81c13fddd: standalone build:dev bundle (CI gate profile) and next dev --webpack, backend MINIMAL_TEST_APP single_user, with an uncommitted recorder logging every console/request issue the gate classifies. (1) m5-drawer-width-deprecation-noise: fixed at source (KanbanPlayground ArchivedItemsDrawer width -> size; also the two WritingPlaygroundShell drawers) and removed. With the fix reverted and no entry, the next dev Kanban recovery fixture fails on the warning; with the fix it passes. (2) m5-optional-resource-404-noise (/moderation, 10-08): removed. The GET still 404s on the minimal backend (server log), but the browser 404 lands after the gate classifies, so it matched nothing in the local runs or in CI run 36946184688 (102 passed, no allowlisted hits). The underlying miss (no moderation router in MINIMAL_TEST_APP; the page requests it anyway) is still open here. (3) The two forced route-boundary signatures still fire in next dev on all 16 fixture routes. They cannot fire in the production bundle because shouldForceRouteError is disabled there, and the CI gate does not run that describe. Kept at 2026-10-31 with rationale pointing here. (4) New deliberate signature for AC#3: the Wayfinding 404-recovery test's own document 404 for /__wayfinding-missing-route__. It was covered by the old broad 404 rule and has been unallowlisted since the 27-rule retirement, which made that non-gated test fail. Re-added as the narrow m5-wayfinding-missing-route-document-404 entry, expiring 2026-10-31.
Current-dev refresh verified 2026-10-02: 91 incoming paths,87 exactdev and4 reviewed intersections; upstream TASK13414 note and Writing drawer fixes preserved. Deliberate Wayfinding document404 is now test-local: require actual response404, remove only exact response URL plus exact browser404 text, retain all unmatched errors. Native unchanged test RED1 after health/persona200; initial missing-backend RED retained separately. GREEN12 includes recovery1/classifier11; fresh relocated current-dev standalone browser106pass with0failures/errors/skips. Current Node20.19.5 build/token-sync/bundle budget (588.1KB/600KB), frontend lint/types/diff pass. Unit84/8files pass after quiescent same-source/gates revalidation; initial2timeouts/2later failures retained without cause/fix claims. Patch80ab627d9c43b712adacc18850159417e549ded2cabd8eda9341f65fec645c8b independently clear. Receipt /tmp/email_followup_devrefresh_receipt_20261002.json binds10source/29frozen artifacts. Real manifest/AuthNZ200 and traversal/source400. Prior16forced-route certificate keeps original inputs, not rerun. No Python delta againstdev; Bandit N/A. Shared installations untouched. Human summary verbatim; source publication/hosted CI/actualmerge pending, task In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Retired general smoke exceptions with source/fixture repairs; intentional forced emissions and deliberate document404 belong only to their own exact tests. Current-dev87paths exact and4intersections reviewed; preserved upstream Writing drawers/task history. Verified native RED1/GREEN12 and fresh106strict browser,84unit/8files, build/lint/types; path/error/expiry guards intact. Requester Change summary supplied verbatim on PR3077. Source publication, current-head hosted CI and actual merge remain pending. Earlier certificates retain original bindings; Bandit N/A: no Python delta againstdev.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 New AI-authored PR has a requester-written Change summary explaining what changed and why these implementation choices were made.
<!-- DOD:END -->
