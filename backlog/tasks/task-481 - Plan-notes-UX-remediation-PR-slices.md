---
id: TASK-481
title: Plan notes UX remediation PR slices
status: Done
labels:
- notes
- ux
- planning
- webui
- extension
modified_files:
- Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation plan for the /notes WebUI and directly connected browser-extension capture workflows. Scope: PR-sized remediation slices with acceptance criteria and tests, grounded in observed notes/clipper implementation details. No product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md and ran the requested plan-review subagent. First pass found two blocking coverage gaps: import/export/offline sync and navigation into /notes. Patched the plan to add navigation coverage to PR 3 and a dedicated PR 9 for import/export/offline draft sync, then fixed PR numbering issues found on second pass. Final reviewer pass: Approved, no issues. Verification: git diff --check passed. Bandit not applicable because this is documentation/planning only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated and locally reviewed the /notes UX remediation plan, then created Backlog child tasks TASK-481.1 through TASK-481.11 for each PR slice. The roadmap now starts with backend search and keyword route-contract fixes, maps UX findings N-01 through N-12 to PR slices, distinguishes locally queued offline saves from server-saved status, and links each implementation task to its plan section. Verification: Backlog child task list shows all 11 To Do tasks parented to TASK-481; no trailing whitespace found in the plan/task files checked.
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
