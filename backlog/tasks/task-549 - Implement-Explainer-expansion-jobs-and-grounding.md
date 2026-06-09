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

Review fix pass:
- Added an explicitly configured production generator adapter behind `EXPLAINER_GENERATOR_ENABLED`, `EXPLAINER_GENERATOR_PROVIDER`, and `EXPLAINER_GENERATOR_MODEL`, with lazy LLM adapter imports outside router startup.
- Gated open/source-led expansion enqueueing when generation is not configured, and defensively reject non-queued idempotent job rows without mutating the node to queued.
- Split selected-source snapshot metadata from authoritative source context. The default selected-source path now reports insufficient context instead of treating persisted metadata as authoritative excerpts.
- Added a `SourceContextResolver`/retriever seam and kept authoritative ownership validation on injected contexts.
- Tightened source-only citation validation so citations must match retrieved authoritative excerpts by source, excerpt text, snapshot hash, and offsets when present.
- Added stale answer revision checks before generation, duplicate expansion batch detection for retries, and rollback of newly created children if metadata persistence fails mid-batch.
- Wired the worker through `build_explainer_job_handler` so production worker jobs use the configured generator path while tests can inject fakes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 complete with review blockers addressed. Added Jobs-backed Explainer node expansion, deterministic generation seams, source-context ownership validation, source-only grounding enforcement, ownership/domain checked job status, and answer-question persistence. Review fix verification: RED observed first with collection failure for missing `ExplainerGenerationNotConfiguredError`, `current_answer_revision`, `make_configured_explainer_generator`, and `build_explainer_job_handler`; targeted Explainer job tests now pass (14/14); combined Explainer job+endpoint tests pass (28/28); router group contract passes (173/173); Bandit on touched backend app scope reports zero findings. Global `git diff --check` remains blocked by unrelated pre-existing trailing whitespace in `Docs/Design/Agents.md`, which is outside the assigned file set.
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
