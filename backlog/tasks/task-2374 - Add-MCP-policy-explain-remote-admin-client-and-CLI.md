---
id: TASK-2374
title: Add MCP policy explain remote admin client and CLI
status: In Progress
labels:
- mcp
- policy
- implementation
- cli
modified_files:
- mcp_unified/gateway/remote_admin.py
- mcp_unified/gateway/cli.py
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_cli.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from the MCP effective permission explain implementation plan: add remote admin client POST methods and local/remote CLI commands for explain-policy and preview-profile-tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RemoteGatewayAdminClient supports POST request bodies and exposes explain_policy plus preview_profile_tools methods.
- [ ] #2 CLI registers explain-policy and preview-profile-tools commands with local and remote modes.
- [ ] #3 CLI supports --args-json, --args-json-file, and --args-stdin for explain-policy without requiring sensitive JSON on the command line.
- [ ] #4 Preview CLI supports category, include/exclude recommendations, include/exclude denied, and limit options.
- [ ] #5 Focused CLI/remote client tests cover POST body handling and args-json-file parsing.
- [ ] #6 Task 5 focused pytest passes or blockers are documented.
- [ ] #7 Changes are committed separately for Task 5.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Task 5 implementation under the approved subagent-driven workflow.

Task 5 added remote policy explain/preview POST client methods and CLI commands. Local preview uses `GatewayPolicyExplainService` with local config/profile storage; a standalone CLI runtime catalog helper is not currently available, so local preview relies on the service's profile/policy fallback and may report degraded runtime catalog state.
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
