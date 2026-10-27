---
id: TASK-13903
title: 'Tests: LLM_Adapters/unit or LLM_Calls tests attempt a live DNS lookup of openaipublic.blob.core.windows.net'
status: To Do
labels:
- tests
- llm
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while planning TASK-13884. Under the non-loopback blocker, the plan measured a lookup of openaipublic.blob.core.windows.net (a tiktoken vocabulary download) passing only through a fallback; suspects are LLM_Adapters/unit/test_llm_models_filters.py (5 tests) and test_llm_providers_capabilities_merge.py (about 9 tests) via /llm/models and /llm/providers tokenizer resolution. On the implementation run (617094d014 plus this change, global autouse blocker) the only host recorded was slot-2.example, so re-measure first: the lookup may depend on a cold tiktoken cache.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No test in LLM_Adapters/unit or LLM_Calls attempts a non-loopback lookup on a cold tiktoken cache
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
