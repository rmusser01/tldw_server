---
id: TASK-406
title: Review conference collections with scoped QA
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 01:21'
labels:
  - bulk-conference-ingest
  - conference-review
  - rag
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RAG search accepts a backend-owned conference collection scope and resolves it to ready media IDs server-side.
- [x] #2 Collection scoped RAG returns no results outside completed or skipped-existing collection media IDs.
- [x] #3 Conference collection review UI shows ordered talks with status/readiness, previous/next navigation, selected comparison, and scoped QA affordance.
- [x] #4 Scoped QA affordance is disabled with clear readiness copy when no collection items are ready.
- [x] #5 Focused backend and frontend tests cover scoped RAG and review UI behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented backend-owned collection_id resolution for scoped RAG; added conference review UI, Knowledge QA scope helper, focused tests, and verification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 6 completed. Scoped Knowledge QA is backend-enforced by collection_id to ready media IDs; conference review UI now supports ordered navigation, status/readiness, comparison, and QA handoff. Focused backend/frontend tests, Bandit, and diff checks passed; frontend typecheck still has unrelated baseline failures.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused backend scoped RAG pytest passes.
- [x] #2 Focused ConferenceCollectionReview and KnowledgeQA Vitest tests pass.
- [x] #3 Bandit runs on touched backend RAG/API files with no new findings.
- [x] #4 git diff --check passes.
- [x] #5 Plan Task 6 and Backlog task are updated before commit.
<!-- DOD:END -->
