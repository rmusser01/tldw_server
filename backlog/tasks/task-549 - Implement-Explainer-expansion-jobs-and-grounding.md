---
id: TASK-549
title: Implement Explainer expansion jobs and grounding
status: Done
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
Implement Task 2 from Docs/superpowers/plans/2026-06-09-explainer-workspace-implementation-plan.md: Jobs-backed expansion, grounding validation, question answers, and ownership-checked Explainer job status. Follow TDD: write failing service/job/API tests, verify RED, implement minimal jobs/prompting/retrieval/grounding/service/router changes, run targeted tests, Bandit touched scope, update task, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Task 2 with TDD. Added Explainer Jobs constants/enqueue helper, worker handler, local worker entrypoint, deterministic prompt builder, selected-source retrieval validation, and grounding normalization. Service now supports owner-scoped node expansion enqueueing and question-answer persistence. Router exposes `POST /api/v1/explainer/sessions/{session_id}/nodes/{node_id}/expand`, `POST /api/v1/explainer/sessions/{session_id}/nodes/{node_id}/answer-question`, and `GET /api/v1/explainer/jobs/{job_id}`. Job payloads contain IDs/settings only; prompts and excerpts are constructed inside the handler. Source-only insufficient retrieval creates a complete insufficient child node with `outside_knowledge_used=False`, and provider failures mark the target node `error`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 complete. Added Jobs-backed Explainer node expansion, deterministic generation seams, source-context ownership validation, source-only grounding enforcement, ownership/domain checked job status, and answer-question persistence. Verification: RED observed with ModuleNotFoundError for missing Explainer jobs module; targeted Explainer job tests passed (4/4); combined Explainer job+endpoint tests passed (17/17); router group contract passed (173/173); Bandit on touched backend app scope reported zero findings; scoped git diff --check for Task 2 files passed. Global git diff --check is still blocked by unrelated pre-existing trailing whitespace in Docs/Design/Agents.md, which was outside the assigned file set.
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
