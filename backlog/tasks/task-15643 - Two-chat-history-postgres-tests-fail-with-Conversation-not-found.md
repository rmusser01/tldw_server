---
id: TASK-15643
title: Two chat history [postgres] tests fail with Conversation not found
status: To Do
labels:
- bug
- chat
- postgres
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13950.25 Task 1 baselining the chat-executors gate at dev ed7a55b606 against Docker postgres:18 (TLDW_TEST_POSTGRES_REQUIRED=1). Both fail the same way: message_store._lock_history_owner raises NotFoundError "Conversation not found." from add_message, which the test calls with a conversation id it just created. Tests: tldw_Server_API/tests/DB_Management/test_chat_retry_system_messages.py::test_retry_context_does_not_expose_another_owners_instructions[postgres] and tldw_Server_API/tests/DB_Management/test_chat_queued_resume_history.py::test_foreign_conversation_is_not_used_as_history_evidence[postgres]. The sqlite variants pass. The chat-executors job deselects both until this is fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both [postgres] tests pass under postgres:18 and are removed from CHAT_EXECUTORS_GATE_DESELECT
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
