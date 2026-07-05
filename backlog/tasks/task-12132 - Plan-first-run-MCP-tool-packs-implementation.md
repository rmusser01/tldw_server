---
id: TASK-12132
title: Plan first-run MCP tool packs implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 23:39'
labels:
  - planning
  - mcp
  - setup
  - first-run
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for the approved first-run MCP tool packs setup spec, including backend setup APIs, MCP Hub/profile integration, frontend onboarding UI, tests, verification, and rollout sequencing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans with concrete backend, frontend, test, verification, and commit steps.
- [x] #2 Plan resolves known implementation risks from the approved spec, including MCP Hub profile ID semantics and optional setup completion behavior.
- [x] #3 Plan is reviewed with the plan-document-reviewer workflow and issues are addressed or documented.
- [x] #4 Backlog task records verification results and final planning summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Planning in progress. Current code inspection found that the WebUI-backed MCP Hub service uses numeric database IDs for permission profiles and assignments, while standalone gateway profile models use string IDs. First-run setup should integrate with the existing WebUI MCP Hub endpoints/service and therefore use numeric IDs in its contract unless implementation discovers a deliberate adapter boundary.

Spec correction made during planning: first-run profile/assignment IDs now use existing MCP Hub numeric IDs, and first-run profile provenance is stored under policy_document.first_run_mcp_tools because permission profiles do not have a standalone metadata column.

Plan review loop completed: addressed conflict UI, required MCP Hub recovery, add-on policy boundaries, post-completion auth dependency ordering, and frontend 409 expected-status handling.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Saved reviewed implementation plan at Docs/superpowers/plans/2026-07-04-first-run-mcp-tool-packs-implementation-plan.md. Corrected the approved spec for numeric MCP Hub IDs and policy_document first-run provenance. Reviewer approved after the final 409/admin guard patch. Verification: scoped git diff --check passed for touched planning files; Bandit skipped because this task changed documentation/planning files only.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
