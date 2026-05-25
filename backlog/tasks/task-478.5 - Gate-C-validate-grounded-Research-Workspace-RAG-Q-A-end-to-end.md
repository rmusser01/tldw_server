---
id: TASK-478.5
title: 'Gate C: validate grounded Research Workspace RAG Q&A end to end'
status: To Do
labels:
- research-workspace
- uat
- gate-c
- rag
- e2e
priority: High
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: full RAG Q&A could not be trusted because model loading, send guardrails, source readiness, and source selection were broken or inconsistent.

User goal: upload or paste research material, wait until it is queryable, ask questions, and receive grounded answers with source evidence/citations.

Scope:
- Exercise the complete live flow: create workspace, add file source, add pasted source, wait for ingestion/indexing completion, select sources, ask a question, inspect answer evidence.
- Fix any remaining WebUI/API integration gaps in source-bound RAG request construction and response rendering.
- Ensure citations/evidence are visible, inspectable, and clearly tied to source snippets or source records.
- Add e2e coverage for at least one text/file source and one pasted source using configured providers or local llama.cpp where available.

Acceptance criteria:
- A user can ask a source-grounded question after ingestion completes and receives an answer with source evidence.
- The UI distinguishes grounded answers from general chat and does not silently fall back to ungrounded mode when sources were selected.
- Citation/source evidence can be inspected enough to verify why the model answered.
- Live CDP/Playwright UAT records pass/fail evidence in the final matrix.

Depends on: TASK-478.1, TASK-478.2, TASK-478.3, TASK-478.4.
Blocks: final acceptance matrix.
Parallelization: can run in parallel with Studio validation after Gate A/B blockers are resolved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
