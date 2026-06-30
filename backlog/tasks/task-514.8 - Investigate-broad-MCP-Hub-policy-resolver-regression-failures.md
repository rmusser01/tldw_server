---
id: TASK-514.8
title: Investigate broad MCP Hub policy resolver regression failures
status: Done
parent_task_id: TASK-514
documentation:
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_policy_overrides.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_shared_workspace_registry.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_workspace_set_objects.py
- tldw_Server_API/tests/MCP_unified/test_tool_catalogs_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on optional broad MCP Unified regression failures found during TASK-514 closeout. The Notes task MCP tool suite passes, but `python -m pytest tldw_Server_API/tests/Notes_NEW tldw_Server_API/tests/MCP_unified -v` fails three persistent MCP Hub policy resolver assertions where `resolved_policy_document` includes empty `tool_tier_overrides` and `conditions` that are absent from `authored_policy_document`; `test_tool_catalogs_flow` also failed only in the full broad sweep with shutdown_in_progress and passed in isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Resolve or intentionally update the policy resolver authored/resolved document equality expectations.
- [x] #2 Determine whether the tool catalog shutdown_in_progress failure is test order/shared lifecycle leakage.
- [x] #3 Restore the broad Notes_NEW plus MCP_unified pytest sweep or document any accepted skips with focused verification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce the documented broad MCP Hub policy resolver failures from latest dev, inspect the resolver/authored-vs-resolved policy document data flow, add failing focused regression coverage for still-valid mismatches, implement the smallest policy-shape or test-expectation correction, investigate the broad-suite-only tool catalog shutdown_in_progress failure for order/lifecycle leakage, then run focused and broad verification plus Bandit/diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced the original three policy resolver failures and verified the resolver intentionally normalizes `resolved_policy_document` with empty `tool_tier_overrides` and `conditions`; updated stale tests to compare authored keys as a subset and assert the normalized defaults explicitly.
- Verified additional current failures in the broad sweep: stale AuthNZ and telemetry monkeypatch seams after runtime dependency extraction, a static external federation test double missing the current scalar write-flag accessor, and a broad-suite `shutdown_in_progress` readiness leak caused by shared FastAPI app lifecycle state.
- Fixed only still-valid test issues: patched the endpoint-level AuthNZ token classifier seam, injected telemetry through `protocol.dependencies.telemetry_provider`, added `get_virtual_tool_write_flag()` to the static external federation manager double, and reset lifecycle state before the first tool-catalog integration flow.
- PR review follow-up: added explicit resolved-policy key-existence assertions before value comparisons and documented the static external federation write-flag test-double method.
- Validation:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py::test_get_current_user_authnz_revoked_does_not_fallback tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py::test_external_federation_virtual_write_tools_respect_protocol_write_disable tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py::test_protocol_tool_execution_failure_log_omits_raw_exception_and_traceback tldw_Server_API/tests/MCP_unified/test_tool_catalogs_api.py::test_tool_catalogs_flow -q` -> 4 passed.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_policy_overrides.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_shared_workspace_registry.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_workspace_set_objects.py tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py tldw_Server_API/tests/MCP_unified/test_tool_catalogs_api.py -q` -> 49 passed.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Notes_NEW tldw_Server_API/tests/MCP_unified -q` -> 635 passed, 2 skipped.
  - After rebasing onto latest `origin/dev`, reran the touched MCP regression set -> 49 passed, and reran the broad Notes_NEW plus MCP_unified sweep -> 635 passed, 2 skipped.
  - `git diff --check` -> clean.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r <touched test files> -s B101,B105,B106,B108 -f json -o /tmp/bandit_task_514_8_mcp_policy_regressions_filtered.json` -> 0 actionable findings; skipped test-only assert/synthetic credential/sanitizer sentinel warnings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the broad MCP Hub regression follow-up. The policy resolver failures were stale assertions against normalized resolved policy documents, not production resolver defects. Additional current failures were also test-bound and were updated to current seams/contracts. The broad Notes_NEW plus MCP_unified sweep now passes.
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
