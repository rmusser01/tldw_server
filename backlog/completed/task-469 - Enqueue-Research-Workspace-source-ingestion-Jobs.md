---
id: TASK-469
title: Enqueue Research Workspace source ingestion Jobs
status: Done
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
modified_files:
- Docs/superpowers/plans/2026-05-23-research-workspace-source-jobs-plan.md
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/Workspaces/test_workspaces_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase A backend follow-up for the Research Workspace hard replacement roadmap. When a source is added to a workspace, enqueue an idempotent user-visible Jobs record for ingestion, extraction, chunking, and indexing status so /api/v1/workspaces/{workspace_id}/sources/status has first-class job-backed progress. Preserve no /workspace-playground aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-research-workspace-source-jobs-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Research Workspace source-add job enqueueing. POST /api/v1/workspaces/{workspace_id}/sources now persists the source row, then submits an idempotent media_ingest/default/workspace_source_ingest Job containing workspace/source/media identifiers, source metadata, URL, and requested lifecycle stages. Job enqueue failures are logged and fail open so the source row remains recoverable. Added integration tests for the job contract and fail-open source preservation. Verification: focused workspace endpoint/status tests, the full Workspaces suite, Bandit on touched production code, scoped diff check, route grep, and a live FastAPI backend smoke test on 127.0.0.1:18001.
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
