---
id: TASK-15503
title: 'Act on #3472''s final-review minors: one fresh-root definition, the embeddings
  test''s context user'
status: Done
labels:
- byok
- usage-quotas
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-15500 (#3472). Its final review left three Minor findings: (1) the overrides that make a fresh root ({prechecked: False, provider_credentials: None}) are written in llm_roots.llm_call_root and again in the media ingest worker's embeddings create_task, so a field added to one could be missed in the other; (2) test_the_embeddings_task_carries_no_key_and_no_admission pins the user_id argument but not current_llm_user_id() inside the task; (3) the committed plan copy Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md says 'replacing 15500 with the id' after its own substitution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One read-only constant in llm_call_context.py defines a fresh root's overrides, and llm_call_root and the worker's embeddings create_task both use it
- [x] #2 The embeddings-task test asserts current_llm_user_id() inside the task is the job owner
- [x] #3 The plan copy's Task 0 Step 3 sentence no longer refers to the id it was substituted with
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Acted on the three Minor findings of #3472's final review. (1) FRESH_ROOT, a read-only MappingProxyType in llm_call_context.py, is the one definition of what a unit of background work never inherits (prechecked, provider_credentials); llm_call_root copies it and the media ingest worker's embeddings create_task unpacks it. No behaviour change. (2) test_the_embeddings_task_carries_no_key_and_no_admission also asserts current_llm_user_id() inside the task is 1; overriding user_id=2 fails it (assert 2 == 1), and dropping _handle_job's scoped_context fails it (assert None == 1, caught by no other assertion). (3) The TASK-15500 plan copy's Task 0 Step 3 no longer names its own substituted id. Measured under the network probe: the 22 test files naming a touched symbol, 638 passed; tests/Docs + lint 527 passed, 1 environment-only error (test_runtime_package_discovery, no setuptools locally); ruff adds no errors; 0 network attempts. The plan review found credential_source is set only around single provider calls, so no root inherits it.
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
