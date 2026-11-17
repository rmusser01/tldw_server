---
id: TASK-13884
title: 'Tests: custom OpenAI adapter error tests resolve api.openai.com through the
  real egress check'
status: Done
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
- [x] #1 The six cases pass with every non-loopback connect and DNS lookup blocked, and make no live DNS query
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED (617094d014, plan measured on e06eeec68c): with no_live_network on both files, 6 failed, 45 passed, 6 errors (teardown assertion): the custom_openai and CustomOpenAIAdapter* cases fail EgressPolicyError('Host could not be resolved'); the only lookup attempted is api.openai.com (config.txt custom_openai_api_ip / custom_openai2_api_ip, read as the server-owned endpoint; the request base_url is stripped).
GREEN: with loopback_custom_openai_endpoints pinning CUSTOM_OPENAI_API_IP and CUSTOM_OPENAI2_API_IP to http://127.0.0.1:9/v1, 51 passed and the fixture records no attempt. test_no_live_network_fixture.py (4 tests) proves the blocker blocks and records dns and connect, allows loopback, and restores. LLM_Adapters/unit + LLM_Calls under a global autouse blocker: 1453 passed, 2 skipped; only non-loopback host recorded: slot-2.example (test_provider_config_resolution, fails closed). Documentation: n/a, no doc describes these tests. Bandit: test-only change, skipped.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added tests/helpers/offline_network.py (block_live_network: patches getaddrinfo, gethostbyname, socket.connect and connect_ex, blocks and records every non-loopback attempt), two LLM_Adapters conftest fixtures (no_live_network asserts no attempt at teardown; loopback_custom_openai_endpoints pins both custom OpenAI slots to a loopback URL via env), and applied both to the two adapter error test files so the six custom OpenAI cases run offline. test_no_live_network_fixture.py proves the blocker itself.
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
