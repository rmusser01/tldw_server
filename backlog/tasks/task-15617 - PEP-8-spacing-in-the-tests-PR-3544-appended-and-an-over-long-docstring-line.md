---
id: TASK-15617
title: 'PEP 8 spacing in the tests PR #3544 appended, and an over-long docstring line'
status: To Do
labels:
- tests
- style
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minor M4 of PR #3544's final review (TASK-15582). Eight tests #3544 appended sit one blank line after the previous top-level definition (pycodestyle E302), for example tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py:1034, tests/Usage/test_llm_quota_decision.py:440, app/core/MCP_unified/tests/test_model_completion_integration.py:1465, tests/Chat/integration/test_chat_provider_override_store_boundary.py:569 and tests/Audio/test_tts_credential_runtime_boundary.py:2709. The LLMStoreGroup docstring line in tldw_Server_API/cli/utils/llm_store.py is over 120 characters. Ruff's default rules flag neither.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every top-level definition #3544 added to a test file has two blank lines before it (E302)
- [ ] #2 The LLMStoreGroup docstring in cli/utils/llm_store.py has no line over 120 characters
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
