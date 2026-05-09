---
id: TASK-145.9
title: Address PR review plan path portability
status: In Progress
assignee: []
created_date: '2026-05-09 17:07'
updated_date: '2026-05-09 17:08'
labels:
  - evals
  - docs
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1421#discussion_r3213461932'
parent_task_id: TASK-145
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR #1421 review feedback by replacing hardcoded user-specific verification paths in the embeddings RAG recipe implementation plan with portable repo-root or virtualenv instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan no longer includes /Users/macbook-dev verification command paths
- [ ] #2 PR review thread is addressed with a pushed commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Search the implementation plan for user-specific absolute paths, replace verification examples with portable commands, run documentation-scoped checks, update this task, commit, push, and reply to the review thread.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced hardcoded /Users/macbook-dev verification command paths in Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md with repo-relative source .venv/bin/activate examples. Verification: rg -n "/Users/macbook-dev" on the plan returned no matches; git diff --check on the plan and task file passed. Bandit not applicable because this review fix only changes markdown/task tracking.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed Gemini PR review feedback by making the embeddings RAG recipe implementation plan portable: verification command examples now use source .venv/bin/activate instead of a user-specific absolute virtualenv path. Recorded doc-scoped verification and noted Bandit as not applicable for this markdown-only review fix.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
