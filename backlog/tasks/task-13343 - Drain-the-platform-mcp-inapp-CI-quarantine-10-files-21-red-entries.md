---
id: TASK-13343
title: 'Drain the platform-mcp-inapp CI quarantine (10 files, 21 red entries)'
status: To Do
assignee: []
created_date: '2026-09-22 06:13'
labels:
  - tests
  - ci
  - mcp
dependencies: []
references:
  - .github/workflows/ci.yml
  - tldw_Server_API/app/core/MCP_unified/tests/test_refresh_token.py
  - tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13291 wired app/core/MCP_unified/tests into CI as the platform-mcp-inapp shard with 10 files quarantined behind --ignore because they were already red. The shard is green at 2948 passed; this task drains the list.

TWO GROUPS.

A. Distribution-building (13 of the 21 entries) - these build sdists/wheels and shell out, and test_runtime_package_boundary.py:29 sets pytestmark = pytest.mark.unit while doing so. They likely do not belong in a unit shard at all; decide between a dedicated packaging job and a non-unit marker:
  test_runtime_package_boundary.py (10 entries)
  test_gateway_protocol_artifact_consumer.py (3 entries)

B. Genuine drift accumulated while the tree was outside CI (8 entries, 8 files). Sampled causes: TypeError: refresh_token() missing 1 required positional argument: request (endpoint signature changed, test not updated); KeyError: rows (result shape changed); gateway status payload gained fields the test does not expect. Each needs a judgement about whether the test or the product is correct:
  test_server_batch_and_formatting.py, test_refresh_token.py, test_gateway_tool_discovery.py,
  test_gateway_fastapi_package.py, test_gateway_admin_auth.py, test_flashcards_module.py,
  test_external_federation_integration.py, test_codegraph_module.py

SEPARATE, DO NOT FOLD IN: test_rag_module.py and test_gateway_protocol_stdio.py are NOT quarantined. They pass in isolation and in the new shard, but fail when run in-process with tests/MCP_unified - cross-test state pollution, the class the 2026-07-04 audit named as its top defect category. That is why the tree got its own shard. Worth its own investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Group A relocated to an appropriate job or marker, not silently ignored
- [ ] #2 Group B triaged: each test either fixed or its product defect filed
- [ ] #3 Quarantine list in ci.yml is empty and the --ignore block removed
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
