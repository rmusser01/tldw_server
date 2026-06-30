---
id: TASK-145.1
title: Write embeddings RAG recipe implementation plan
status: Done
assignee:
  - Codex
created_date: '2026-05-09 03:57'
updated_date: '2026-05-09 04:07'
labels:
  - design
  - evaluations
  - embeddings
  - rag
  - webui
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a detailed implementation plan for the approved productized embeddings model selection recipe flow. The plan must stay grounded in the hardened spec, keep V1 media-scoped unless the backend contract is extended deliberately, split follow-up implementation into reviewable stages, and avoid starting code implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan document is written under Docs/superpowers/plans and references the approved spec.
- [x] #2 Plan decomposes implementation into reviewable staged tasks with exact files, tests, commands, and commit points.
- [x] #3 Plan preserves V1 boundaries: media-ID source labeling, server-owned candidate readiness/apply eligibility, preview/copy fallback before safe config mutation.
- [x] #4 Plan is reviewed and updated before implementation begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write implementation plan in Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md. Keep implementation decomposed into backend contract, backend helper APIs, frontend services/hooks, guided component, RecipesTab integration, apply preview UI, and gated live apply.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan review pass added hardening for secret-free candidate/apply payloads, no FastAPI endpoint calls from helpers, missing-key candidate status, RecipeRunReport/dict normalization, explicit component manifest fixture, env-var override blocking for live apply, and preview/copy fallback if live mutation is not approved or safe.

Verification recorded during planning closeout: ASCII scan of the plan and task file returned no matches for non-ASCII bytes; plan header and tracking task were reread after edits. Bandit is not applicable because this task changed only docs and Backlog tracking files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md for the approved embeddings RAG recipe flow. The plan keeps V1 media-scoped, splits work into backend contract, candidate/apply-preview APIs, shared frontend hooks, guided config UI, report/apply-preview UI, and a separately gated live apply task. Review updates added secret-free payload rules, missing-key candidate readiness, RecipeRunReport normalization, env-var override blocking for live apply, and preview/copy fallback when mutation is not safe.
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
