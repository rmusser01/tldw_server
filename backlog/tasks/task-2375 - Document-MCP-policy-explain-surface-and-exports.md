---
id: TASK-2375
title: Document MCP policy explain surface and exports
status: Done
labels:
- mcp
- policy
- implementation
- docs
modified_files:
- mcp_unified/gateway/__init__.py
- mcp_unified/README.md
- mcp_unified/USER_GUIDE.md
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the MCP effective permission explain implementation plan: export the policy explain public API from mcp_unified.gateway, add a minimal export smoke test, and document the explain-policy and preview-profile-tools local/remote/admin surfaces in the packaged README and user guide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 mcp_unified.gateway exports GatewayPolicyExplainService, PolicyExplainRequest, ProfileToolPreviewRequest, and GatewayPolicyExplainError without breaking optional dependency behavior.
- [ ] #2 A minimal public export smoke test covers the new gateway exports.
- [ ] #3 mcp_unified/README.md documents explain-policy and preview-profile-tools purpose, local CLI usage, remote CLI usage, admin API examples, and security notes.
- [ ] #4 mcp_unified/USER_GUIDE.md includes the same user-facing policy explain guidance where packaged users can find it.
- [ ] #5 Focused export/docs tests pass or blockers are documented.
- [ ] #6 Bandit is run for touched code where applicable or a docs-only skip is recorded.
- [ ] #7 Changes are committed separately for Task 6.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Task 6 implementation under the approved subagent-driven workflow. Scope: public gateway exports, minimal smoke test, and packaged README/user-guide documentation for the policy explain and profile tool preview surfaces.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 6. Exported GatewayPolicyExplainService, PolicyExplainRequest, ProfileToolPreviewRequest, and GatewayPolicyExplainError from mcp_unified.gateway; added a public export smoke test to the standalone policy explain service tests; documented explain-policy and preview-profile-tools purpose, local CLI use, remote CLI use, admin API calls, and audit/redaction/security guidance in mcp_unified/README.md and mcp_unified/USER_GUIDE.md. Verification: pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v passed 39 tests; pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v passed 34 tests; git diff --check passed. Bandit full touched-file scan was run and reported only pre-existing B101 pytest assert findings in the touched test file, with the new smoke test not flagged; Bandit with B101 skipped reported 0 results and 0 errors.
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
