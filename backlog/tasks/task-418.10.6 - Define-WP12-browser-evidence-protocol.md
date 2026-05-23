---
id: TASK-418.10.6
title: Define WP12 browser evidence protocol
status: In Progress
labels:
- wp12
- webui
- route-governance
- browser-qa
- documentation
priority: High
ordinal: 6
parent_task_id: TASK-418.10
references:
- TASK-418.10
documentation:
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
modified_files:
- apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP12 Task 6 from the WebUI route governance QA plan: define the browser evidence protocol for route-family changes, including before/after observations, viewport requirements, screenshot and DOM naming, triage format, known skips, and Backlog final-summary fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Protocol documents required before/after evidence for changed visual routes.
- [x] #2 Protocol defines desktop, 390px mobile, and sidepanel viewport requirements where relevant.
- [x] #3 Protocol defines screenshot and DOM/browser observation naming conventions.
- [x] #4 Protocol defines console/request triage, known-skip, and Backlog final-summary formats.
- [x] #5 Route governance plan status is updated for Task 6 without changing product code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the WP12 browser evidence protocol as a documentation-only route governance artifact. The protocol defines when route-family evidence is required, where evidence artifacts belong, required desktop/mobile/sidepanel viewports, first-time/returning/degraded/error state coverage, screenshot and DOM/browser observation naming conventions, console/request triage, known-skip format, Backlog final-summary fields, and closure rules. Linked the protocol from the route governance QA plan and marked Task 6 plan steps complete. Verification: git diff --check passed. Bandit was not run because this slice changed Markdown task/plan/protocol documentation only and no Python code or product code was touched. No live browser evidence was required for this protocol-definition slice.
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
