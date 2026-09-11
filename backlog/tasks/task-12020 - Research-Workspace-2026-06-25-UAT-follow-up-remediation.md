---
id: TASK-12020
title: Research Workspace 2026-06-25 UAT follow-up remediation
status: In Progress
assignee: []
created_date: '2026-06-25 20:05'
updated_date: '2026-08-22 08:11'
labels:
  - research-workspace
  - uat
  - ux
  - webui
milestone: Research Workspace UAT Remediation
dependencies: []
references:
  - TASK-478
  - TASK-12019
  - /private/tmp/tldw_research_workspace_uat_2026-06-25
documentation:
  - Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up epic for the fresh 2026-06-25 CDP UAT of `/research-workspace` across beginner/no-key and power/API-key personas. This is a new follow-up to the closed TASK-478 stream, not a reopening of that completed work. Scope covers newly observed or still-reproducible blockers and UX/HCI findings: frontend setup reliability, first-use onboarding and tour behavior, no-auth add-source recovery, workspace readiness/status contradictions, partially queryable source gating, chat/RAG/Studio prerequisite feedback, advanced share/export/import workflows, templates/folders/destructive-action feedback, accessibility/responsive polish, and final repeatable CDP/Playwright UAT certification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each 2026-06-25 Research Workspace UAT finding is mapped to a child task or explicitly documented as intentionally covered by an existing task.
- [x] #2 Child tasks are scoped to reviewable implementation units with dependencies and verification gates.
- [x] #3 Implementation tasks require fresh browser/session CDP or Playwright validation for the affected persona before closure.
- [ ] #4 Parent remains open until all child tasks are completed, split, or explicitly deferred with rationale.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Planning pass completed under `TASK-12020.1`. Created child tasks `TASK-12020.2` through `TASK-12020.11` and saved `Docs/superpowers/plans/2026-06-25-research-workspace-uat-follow-up-remediation-plan.md`. Parent remains open for implementation and final UAT closure; acceptance criterion #4 remains unchecked until child tasks are completed, split, or explicitly deferred.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
