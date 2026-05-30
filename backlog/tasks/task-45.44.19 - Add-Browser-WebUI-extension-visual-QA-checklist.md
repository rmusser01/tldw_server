---
id: TASK-45.44.19
title: Add Browser/WebUI/extension visual QA checklist
status: Done
assignee: []
created_date: '2026-05-14 03:21'
labels:
  - design-system
  - webui
  - extension
  - governance
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1676'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
documentation:
  - Docs/Design/tldw_web_design_system_visual_qa_checklist.md
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_baseline_reporting.md
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns public status.
- [x] #2 Backlog notes record PR links and verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `Docs/Design/tldw_web_design_system_visual_qa_checklist.md` as the durable Browser/WebUI/extension visual QA artifact requested by GitHub issue #1676. The checklist defines when design-system PRs require visual QA, required evidence by change type, default WebUI/mobile/extension sidepanel viewports, extension evidence rules, PR body and Backlog note expectations, known-skip standards, and closure rules. It explicitly complements `apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md` instead of duplicating route-family screenshot and console/request triage rules. Linked the new checklist from the design-system contract enforcement sequence and baseline-reporting workflow so future migration tasks can find it from existing governance docs.

Verification:
- `test -f Docs/Design/tldw_web_design_system_visual_qa_checklist.md`
- `rg -n "Route Evidence Protocol|Visual QA|Extension Evidence|Known Skip|Closure Rule" Docs/Design/tldw_web_design_system_visual_qa_checklist.md`
- `rg -n "tldw_web_design_system_visual_qa_checklist" Docs/Design/tldw_web_design_system_contract.md Docs/Design/tldw_web_design_system_baseline_reporting.md`
- `git diff --check`

Bandit skipped: documentation and Backlog markdown only; no Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the durable design-system visual QA checklist for Browser/WebUI/extension shared surfaces. Migration PRs now have a documented way to decide when visual QA is required, what WebUI and extension evidence to collect, how to record deterministic alternatives and skips, and how to close visual-impact design-system work without relying on static inspection alone.
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
