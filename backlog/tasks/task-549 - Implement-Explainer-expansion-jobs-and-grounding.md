---
id: TASK-549
title: Implement Explainer expansion jobs and grounding
status: To Do
labels:
- backend
- jobs
- explainer
- implementation
priority: High
references:
- TASK-546
- TASK-547
- Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
- Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Explainer/jobs.py
- tldw_Server_API/app/core/Explainer/jobs_worker.py
- tldw_Server_API/app/core/Explainer/prompting.py
- tldw_Server_API/app/core/Explainer/grounding.py
- tldw_Server_API/app/core/Explainer/retrieval.py
- tldw_Server_API/app/core/Explainer/service.py
- tldw_Server_API/app/api/v1/endpoints/explainer.py
- tldw_Server_API/app/api/v1/schemas/explainer.py
- tldw_Server_API/tests/Explainer/test_explainer_jobs.py
- tldw_Server_API/tests/Explainer/test_explainer_endpoints.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement Task 2 from Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md: Jobs-backed expansion, grounding validation, and ownership-checked Explainer job status. Follow TDD and keep LLM/RAG imports out of router startup.
<!-- SECTION:PLAN:END -->

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
