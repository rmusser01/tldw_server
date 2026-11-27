---
id: TASK-45.44.13.8
title: Migrate Chat Workflows product-state alerts
status: Done
labels:
- design-system
- webui
- chat-workflows
references:
- TASK-45.44.13
- https://github.com/rmusser01/tldw_server/issues/1670
- Docs/Design/tldw_web_design_system_contract.md
- Docs/Design/tldw_web_design_system_visual_qa_checklist.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the four Chat Workflows alerts tracked by #1670. Keep this slice independent of the Kanban and Scheduled Tasks PR. Parent task TASK-45.44.13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library guidance, active run errors, completion, and cancellation use shared alerts while preserving their copy and actions.
- [x] #2 Completion and cancellation update an existing polite status, with focused behavior and guard coverage.
- [x] #3 Remove the four scoped baseline entries and record before/after guard counts, scoped tests, typecheck, and representative browser evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Adopt the existing shared product-state contract without changing component APIs, token policy, or workflow behavior. Base 2053ec5ad4.
Verified directly from latest remote dev 2053ec5ad4. This independent slice removed exactly four allowed Chat Workflows Alert entries. The #1670 area changed from 14 to 10 allowed exceptions, with 26 blocked findings and one stale entry still on this base. All other finding IDs match. The full verifier exits 1 for the remaining blockers.
Static library instructions retain readable copy without a live announcement. Active run failures remain assertive. Completion and cancellation update the same polite text status across a read error and recovery, and the inline alert keeps its visual placement and foreground color. Independent review verified the refresh regression fix.
Passed 72 focused page, API, route, shared-primitive, and guard tests, plus the full shared-UI typecheck, scoped ESLint and diff whitespace. Bandit is inapplicable to the TypeScript, JSON and Markdown scope.
Verified the WebUI library guidance and New Template action before and after at 1440x900 and 390x844 with fixtures. The rendered phone guidance measured 13.89:1 contrast. A freshly built Chrome extension passed the same route and builder probe in an isolated profile against a loopback fixture server. Browser artifacts are in /private/tmp/tldw-tracker-1670.x50vaO.
Read-only three-way merge of the only overlapping file, the baseline JSON, preserved all deletions from this slice and PR #3671. The public #1670 issue remains open for the rest of the migration.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the four legacy Chat Workflows alerts, kept run outcome announcements mounted through refresh failures, and retired their scoped baseline entries. Recorded independent browser and guard evidence.
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
