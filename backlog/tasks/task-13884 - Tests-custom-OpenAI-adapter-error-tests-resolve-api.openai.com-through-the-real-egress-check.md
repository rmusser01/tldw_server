---
id: TASK-13884
title: 'Tests: custom OpenAI adapter error tests resolve api.openai.com through the
  real egress check'
status: To Do
labels:
- tests
- llm
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-13865 plan review. The custom_openai cases of tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_midstream_error_raises.py (CustomOpenAIAdapter, CustomOpenAIAdapter2) and test_adapter_stream_error_normalization.py (custom_openai 400, 401, 429, 500) send their request through the real egress check, which resolves api.openai.com. With DNS blocked they fail with EgressPolicyError('Host could not be resolved') on dev; with DNS they pass, after a live lookup from the test suite.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The six cases pass with every non-loopback connect and DNS lookup blocked, and make no live DNS query
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
