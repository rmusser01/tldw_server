---
id: TASK-469
title: Address Research Workspace source Jobs review feedback
status: Done
references:
- backlog/completed/task-469 - Enqueue-Research-Workspace-source-ingestion-Jobs.md
- Docs/superpowers/plans/2026-05-23-research-workspace-source-jobs-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-23-research-workspace-source-jobs-plan.md
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/Workspaces/test_workspaces_api.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-469. Fix code-review findings for workspace source job enqueueing: source creation must preserve the source row when Jobs dependency construction fails, and status projection should prioritize workspace_source_ingest Jobs so relevant source lifecycle jobs are not buried behind unrelated media_ingest work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed source Jobs review feedback. Added fail-open job manager resolution so source creation still persists when Jobs construction is unavailable, kept enqueue failures non-destructive, and made status projection query workspace_source_ingest Jobs first before broad legacy media_ingest Jobs. Added regression coverage for dependency construction failures and job-list prioritization. Verification: focused review-regression tests passed, full Workspaces suite passed, Bandit on touched production code reported zero findings, scoped diff check passed, and a live backend smoke verified source add/status plus idempotent job creation.
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
