---
id: TASK-2375
title: Document MCP policy explain surface and exports
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-17 07:49'
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
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the MCP effective permission explain implementation plan: export the policy explain public API from mcp_unified.gateway, add a minimal export smoke test, and document the explain-policy and preview-profile-tools local/remote/admin surfaces in the packaged README and user guide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mcp_unified.gateway exports GatewayPolicyExplainService, PolicyExplainRequest, ProfileToolPreviewRequest, and GatewayPolicyExplainError without breaking optional dependency behavior.
- [x] #2 A minimal public export smoke test covers the new gateway exports.
- [x] #3 mcp_unified/README.md documents explain-policy and preview-profile-tools purpose, local CLI usage, remote CLI usage, admin API examples, and security notes.
- [x] #4 mcp_unified/USER_GUIDE.md includes the same user-facing policy explain guidance where packaged users can find it.
- [x] #5 Focused export/docs tests pass or blockers are documented.
- [x] #6 Bandit is run for touched code where applicable or a docs-only skip is recorded.
- [x] #7 Changes are committed separately for Task 6.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Task 6 implementation under the approved subagent-driven workflow. Scope: public gateway exports, minimal smoke test, and packaged README/user-guide documentation for the policy explain and profile tool preview surfaces.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Review follow-up addressed the remaining spec gap by documenting remote CLI preview-profile-tools usage in both packaged docs and tightened the public export smoke test to assert gateway.__all__ includes the policy explain API exports.

Reviews: final spec follow-up reported no Critical or Important issues and marked spec ready; final code-quality follow-up reported no Critical or Important issues and ready-to-merge.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v -> 39 passed, 7 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v -> 34 passed, 5 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/__init__.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -s B101 -f json -o /tmp/bandit_task2375.json -> 0 results; git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 6. Exported GatewayPolicyExplainService, PolicyExplainRequest, ProfileToolPreviewRequest, and GatewayPolicyExplainError from mcp_unified.gateway; added a public export smoke test to the standalone policy explain service tests; documented explain-policy and preview-profile-tools purpose, local CLI use, remote CLI use, admin API calls, and audit/redaction/security guidance in mcp_unified/README.md and mcp_unified/USER_GUIDE.md. Verification: pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v passed 39 tests; pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v passed 34 tests; git diff --check passed. Bandit full touched-file scan was run and reported only pre-existing B101 pytest assert findings in the touched test file, with the new smoke test not flagged; Bandit with B101 skipped reported 0 results and 0 errors.
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
