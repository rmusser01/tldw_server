---
id: TASK-478.13
title: 'Gate E: maintain live Research Workspace UAT matrix and regression coverage'
status: To Do
labels:
- research-workspace
- uat
- gate-e
- tests
- playwright
- regression
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Process need: the current UAT found hidden breakages only after running a real backend and WebUI. Future tasks must prove behavior with a live app, not just static code review.

User goal: know which exposed workspace functionality works, which is broken, and which fixes have been verified against the running product.

Scope:
- Convert the live UAT checklist into a maintained matrix covering first-time flow, power-user flow, source ingestion, status, selection, RAG, Studio, My Media, folders, annotations, settings/share, responsive layout, old-route 404/no redirects, and extension handoff.
- Record reproduction steps, expected result, actual result, backend/API evidence, screenshots where useful, and pass/fail status per task.
- Add automated regression tests for high-risk paths as each child task is fixed instead of waiting until the end.
- Keep validation anchored to live backend + WebUI + CDP/Playwright, with configured providers or local llama.cpp when needed.

Acceptance criteria:
- A current UAT matrix exists and is updated as each child task completes.
- Each fixed child task records tests and live verification evidence in its backlog final summary.
- Old `/workspace-playground` remains 404/no redirect and current UI/route metadata use `/research-workspace` names.
- Final matrix shows no critical or high hidden-broken functionality remaining, or explicitly documents unresolved blockers.

Depends on: all functional child tasks for final completion; matrix scaffolding can start immediately.
Parallelization: each task owner updates its own row(s); final consolidation happens last.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
