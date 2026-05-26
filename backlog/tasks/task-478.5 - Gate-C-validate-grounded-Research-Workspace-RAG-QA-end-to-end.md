---
id: TASK-478.5
title: 'Gate C: validate grounded Research Workspace RAG Q&A end to end'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-25 03:59'
labels:
  - research-workspace
  - uat
  - gate-c
  - rag
  - e2e
milestone: Research Workspace UAT Remediation
dependencies: []
parent_task_id: TASK-478
priority: high
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-05-25 TASK-478.5: Live CDP UAT reproduced the source-bound RAG failure, then verified fixes. Root causes: selected-source RAG allowed backend pre-retrieval clarification to bypass retrieval, required an LLM query rewrite before retrieval in follow-up chats, ignored backend generated_answer, and labelled citations with metadata.source=media_db instead of source titles. Fixed in apps/packages/ui/src/hooks/chat-modes/ragMode.ts. Verification: bun run test:run ../packages/ui/src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts passed 6 tests; bun run test:run ../packages/ui/src/hooks/chat-modes/__tests__ passed 4 files / 11 tests; live WebUI at /research-workspace with backend 127.0.0.1:18002 and frontend 127.0.0.1:8081 returned /api/v1/rag/search documents with include_media_ids [8,7], enable_intent_routing=false, enable_pre_retrieval_clarification=false, visible answer containing PASTE-EVIDENCE-ORION, and expanded citations titled TASK-478.5 Paste Evidence Source plus task-478.5 file.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

Completed scoped Research Workspace RAG fixes and live UAT for TASK-478.5. Selected workspace sources now force media_db scoped retrieval, disable backend pre-retrieval clarification, skip fragile LLM query rewriting before retrieval, use backend generated RAG answers directly when available, block silent ungrounded fallback when no selected-source evidence is found, and attach citation sources using source titles/types instead of generic media_db labels. Docs: no separate docs needed; behavior covered by task notes and tests. Bandit: skipped because touched code is frontend TypeScript only. Known residual: full frontend tsc is blocked by unrelated existing JSX parse errors in WatchlistsPlaygroundPage.tsx; focused chat-mode tests and live CDP UAT passed.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
