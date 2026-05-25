---
id: TASK-478.3
title: 'Gate B: implement first-class workspace ingestion and indexing status'
status: To Do
labels:
- research-workspace
- uat
- gate-b
- backend
- jobs
- ingestion
- indexing
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: source cards show `Ready`, while `/api/v1/workspaces/{id}/sources/status` reports `partially_queryable`, `vector_index_pending`, `progress_percent: 75`, `vector_ready: false`, `summary_ready: false`, and `job: null`. Status remained unchanged after waiting.

User goal: understand exactly what was ingested, what is searchable now, what is still processing, and how to recover from stuck/failed ingestion.

Scope:
- Decide and implement the owner for extraction, chunking, FTS indexing, vector indexing, citation readiness, and summary readiness using the existing Jobs primitives where user-visible progress/admin visibility is needed.
- Add or refine backend/API status fields so they expose per-stage state, progress, job identity, timestamps, retries, error/recovery details, and queryability semantics.
- Ensure status is not only computed on read when that would hide stalled jobs; use a projection/event-updated state where needed, with computed fallbacks for consistency.
- Align UI source-card labels with API readiness: `Ready`, `Partially queryable`, `Indexing`, `Failed`, `Needs retry`, etc.
- Add tests for upload and pasted-text status progression, stuck vector indexing, idempotent retries, and partial success.

Acceptance criteria:
- A source card never says fully `Ready` when vector/citation/summary readiness required by the active feature is incomplete.
- Status API exposes enough detail for UI progress, retry, and diagnostics without leaking unbounded metadata.
- Upload and pasted-text sources progress to a stable queryable state or visible failure state in live backend/WebUI validation.
- Jobs/admin visibility exists for user-facing ingestion/indexing work.

Depends on: none for backend investigation; UI integration depends on Gate A stability.
Blocks: TASK-478.4, TASK-478.5, TASK-478.6, TASK-478.8.
Parallelization: backend/API work can proceed while Gate A frontend model fixes are underway.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Live UAT upload `research-workspace-uat-source.md` reached `partially_queryable` and stayed there after a 10s wait: text_extracted=true, fts_ready=true, vector_ready=false, status_reason=`vector_index_pending`, progress_percent=75, job=null. This confirms the first-class status/job ownership gap remains visible.']
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
