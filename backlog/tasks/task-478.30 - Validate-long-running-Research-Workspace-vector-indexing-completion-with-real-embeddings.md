---
id: TASK-478.30
title: Validate long-running Research Workspace vector indexing completion with real
  embeddings
status: Done
labels:
- research-workspace
- rag
- embeddings
- source-status
- uat
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 30
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- TASK-478.3
- TASK-478.5
modified_files:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/superpowers/plans/2026-05-28-research-workspace-vector-indexing-validation-plan.md
- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
- tldw_Server_API/app/core/DB_Management/media_db/legacy_state.py
- tldw_Server_API/app/core/Embeddings/services/jobs_worker.py
- tldw_Server_API/app/core/Embeddings/services/redis_worker.py
- tldw_Server_API/app/core/Workspaces/status_projection.py
- tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py
- tldw_Server_API/tests/Embeddings/test_backpressure_and_quotas.py
- tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a live backend/WebUI validation for a Research Workspace source that progresses through extraction/chunking/vector indexing with a real embeddings configuration until it becomes fully queryable/vector-ready. The goal is to prove the first-class source status projection handles long-running vector completion, not just partial/FTS-ready states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live backend is configured with a real embeddings provider and vector store suitable for a bounded Research Workspace source.
- [x] #2 Ingesting a workspace source shows extraction/chunking/indexing progress through first-class workspace source status APIs.
- [x] #3 The source eventually reaches fully queryable/vector-ready status or records a bounded, diagnosable failure state.
- [x] #4 WebUI source status and grounded RAG behavior agree with the backend projection after vector completion.
- [x] #5 RW-UAT-006 and the high-risk vector-indexing remainder are updated only as far as live evidence supports.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Fixed disabled-Redis backpressure gating so stale localhost Redis streams do not block embeddings when Redis is disabled in config.
- Marked media-backed embeddings complete only after root embeddings job completion, and made terminal media error marking best-effort in worker exception paths.
- Updated Media DB processed-state marking to set `chunking_status=completed` with `vector_processing=1`, and made workspace status projection prefer vector-ready queryable state over stale chunking markers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Live validation completed with backend on 127.0.0.1:18033, WebUI on 127.0.0.1:18034, and a Redis embeddings worker using task-scoped streams. Bounded media source 1 completed embeddings job e6861c66-c3c7-4cfa-965d-0e445078bb91 with embedding_count=1; Media DB reported chunking_status=completed and vector_processing=1; workspace source status for research-workspace-task47830-1779931800375 reported state=queryable, readiness.vector_ready=true, progress_percent=100, and Ready for grounded questions. WebUI/CDP showed the source card as READY and the store source status from workspace-status-projection. CDP-triggered RAG used include_media_ids=[1] and returned citations from the selected source; the seeded numeric token was redacted by content policy, so exact token echo was not claimed. UAT matrix RW-UAT-006 was updated with this evidence. Verification: focused pytest suite 20 passed; git diff --check passed; Bandit over touched production Python reported 0 findings.
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
