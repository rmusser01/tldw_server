---
id: TASK-478.30
title: Validate long-running Research Workspace vector indexing completion with real
  embeddings
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a live backend/WebUI validation for a Research Workspace source that progresses through extraction/chunking/vector indexing with a real embeddings configuration until it becomes fully queryable/vector-ready. The goal is to prove the first-class source status projection handles long-running vector completion, not just partial/FTS-ready states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Live backend is configured with a real embeddings provider and vector store suitable for a bounded Research Workspace source.
- [ ] #2 Ingesting a workspace source shows extraction/chunking/indexing progress through first-class workspace source status APIs.
- [ ] #3 The source eventually reaches fully queryable/vector-ready status or records a bounded, diagnosable failure state.
- [ ] #4 WebUI source status and grounded RAG behavior agree with the backend projection after vector completion.
- [ ] #5 RW-UAT-006 and the high-risk vector-indexing remainder are updated only as far as live evidence supports.
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
