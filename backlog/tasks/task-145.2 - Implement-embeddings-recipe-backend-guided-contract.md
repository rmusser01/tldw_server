---
id: TASK-145.2
title: Implement embeddings recipe backend guided contract
status: Done
assignee:
  - Codex
created_date: '2026-05-09 04:12'
labels:
  - evaluations
  - embeddings
  - rag
  - backend
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the embeddings RAG recipe implementation plan: backend manifest capabilities, guided dataset validation warnings, media-ID contract enforcement, and recommendation metadata required by apply preview. Use TDD and keep edits limited to the embeddings recipe and its focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Add the requested focused tests in `test_recipe_embeddings_retrieval.py` and run the focused pytest command to confirm the expected failures before production edits.
2. Add additive manifest/default run config metadata to `EmbeddingsRetrievalRecipe` only.
3. Extend `validate_dataset` with optional `run_config`, guided warnings, and media_id expected ID validation while preserving existing labeled/unlabeled/mixed behavior.
4. Extend recommendation slot metadata from already computed candidate summaries and rerun focused pytest, Bandit on the touched recipe source, and `git diff --check`.

Note: Backlog MCP `task_view` returned `TASK_NOT_FOUND` for `TASK-145.2`; this worktree-local task file is being used as the task record.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Embeddings recipe manifest advertises guided UI, media-ID source labeling, candidate discovery, apply preview, and default run config metadata.
- [x] #2 Validation keeps unlabeled guided datasets valid with warnings while rejecting non-integer expected_ids under the media_id contract.
- [x] #3 Recommendation slots include provider/model/apply eligibility metadata without changing recipe execution semantics.
- [x] #4 Focused pytest for test_recipe_embeddings_retrieval.py passes.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- TDD red run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py -q` failed with 4 expected failures for missing guided manifest metadata, missing `run_config` validation support, and incomplete recommendation metadata.
- Green verification: focused pytest passed with 8 tests.
- Security verification: `python -m bandit -r tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py -f json -o /tmp/bandit_embeddings_recipe_task1.json` completed with zero findings.
- Hygiene verification: `git diff --check` completed with no output.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the embeddings retrieval recipe guided backend contract for Task 1. The manifest now advertises guided UI support, media-scoped execution, media-ID source labeling, candidate discovery metadata, apply-preview targeting, and default run config values. Dataset validation now accepts optional run config, warns for guided unlabeled datasets with no expected sources, and returns clear media-ID errors for non-integer expected IDs under the media_id contract. Recommendation slots now include concrete provider/model/apply metadata derived from the candidate summary without adding frontend thresholds, endpoint work, or live apply behavior.
<!-- SECTION:FINAL_SUMMARY:END -->
