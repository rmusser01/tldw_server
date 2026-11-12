---
id: TASK-15488
title: RAG stream quota check refuses for post-answer stages the stream no longer
  runs
status: To Do
labels:
- rag
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On `/api/v1/rag/search/stream`, the quota boundary still refuses (402) an over-limit user on their own key who turns on a post-answer stage, such as suggestions, claims or post-verification. Since TASK-15482 (#3447) the stream's retrieval step runs no generation, so none of those stages makes a call on that route.

`_rag_llm_providers` (`rag_unified.py`) adds a `None` provider for any flag in `_RAG_OTHER_LLM_STAGES`, and a `None` is checked against the user's limit even when every named provider is their own key. The stream route uses the same list as `/search`, which does run those stages.

It fails closed (refuses more, never less), and no hosted deployment exists. Found by #3447's final review (its Review Focus 4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On `/search/stream`, an over-limit user whose named providers are all their own keys, with only stages the stream does not run turned on, is not refused; the same request on `/search` is still refused.
- [ ] #2 Every flag in `_RAG_OTHER_LLM_STAGES` is classified for the stream route as run there or not, and a test pins the classification, so a new flag must be classified (as TASK-15427's `_NO_OTHER_LLM_CALL` does for the schemas).
- [ ] #3 A stage the stream does run (for example query classification or document grading) still gets an over-limit own-key user refused on `/search/stream`.
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
