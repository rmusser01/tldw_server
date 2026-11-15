---
id: TASK-15503
title: 'Act on #3472''s final-review minors: one fresh-root definition, the embeddings
  test''s context user'
status: In Progress
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
- [ ] #1 One read-only constant in llm_call_context.py defines a fresh root's overrides, and llm_call_root and the worker's embeddings create_task both use it
- [ ] #2 The embeddings-task test asserts current_llm_user_id() inside the task is the job owner
- [ ] #3 The plan copy's Task 0 Step 3 sentence no longer refers to the id it was substituted with
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
