---
id: TASK-2374
title: Add MCP policy explain remote admin client and CLI
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-17 07:32'
labels:
  - mcp
  - policy
  - implementation
  - cli
modified_files:
  - mcp_unified/gateway/remote_admin.py
  - mcp_unified/gateway/cli.py
  - tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from the MCP effective permission explain implementation plan: add remote admin client POST methods and local/remote CLI commands for explain-policy and preview-profile-tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RemoteGatewayAdminClient supports POST request bodies and exposes explain_policy plus preview_profile_tools methods.
- [x] #2 CLI registers explain-policy and preview-profile-tools commands with local and remote modes.
- [x] #3 CLI supports --args-json, --args-json-file, and --args-stdin for explain-policy without requiring sensitive JSON on the command line.
- [x] #4 Preview CLI supports category, include/exclude recommendations, include/exclude denied, and limit options.
- [x] #5 Focused CLI/remote client tests cover POST body handling and args-json-file parsing.
- [x] #6 Task 5 focused pytest passes or blockers are documented.
- [x] #7 Changes are committed separately for Task 5.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Task 5 implementation under the approved subagent-driven workflow.

Task 5 added remote policy explain/preview POST client methods and CLI commands. Local preview uses `GatewayPolicyExplainService` with local config/profile storage; a standalone CLI runtime catalog helper is not currently available, so local preview relies on the service's profile/policy fallback and may report degraded runtime catalog state.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Controller follow-up addressed code-quality review findings: policy explain CLI remote mode is now explicit via --gateway-url or --remote, ambient MCP_UNIFIED_GATEWAY_URL no longer silently forces remote mode, --admin-key was removed from policy commands, admin keys are read only from MCP_UNIFIED_GATEWAY_ADMIN_KEY, explicit empty --args-json is rejected, and remote error payloads preserve message fields. Added regression coverage for env/local precedence, env-backed remote mode, empty args JSON, local plus gateway-url rejection, admin-key argument rejection, and message preservation.

Reviews: spec follow-up reported no Critical/Important/Minor issues and marked spec ready; code-quality follow-up reported no Critical/Important issues, prior blockers resolved, and ready to merge.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py -v -> 10 passed, 7 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py -v -> 26 passed, 5 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -k simulate_policy -v -> 4 passed, 90 deselected, 5 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/remote_admin.py mcp_unified/gateway/cli.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py -s B101 -f json -o /tmp/bandit_task2374.json -> 0 results; git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 added remote admin POST methods and standalone CLI commands for policy explain and profile tool preview in both local and explicit remote modes. Review follow-up hardened mode selection and secret handling so local remains the default, remote is selected with --gateway-url or --remote, command-line admin secrets are rejected, and gateway error messages are preserved. Focused tests and Bandit passed.
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
