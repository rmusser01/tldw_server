---
id: TASK-12750
title: Implement MCP policy explain service foundation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-17 05:09'
labels:
  - mcp
  - policy
  - implementation
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the MCP effective permission explain implementation plan: service request/response models, redaction helpers, strict audit behavior, and focused unit tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mcp_unified/gateway/policy_explain.py exists with service models, GatewayPolicyExplainService, GatewayPolicyExplainError, redaction helpers, and strict audit append behavior.
- [x] #2 Focused service tests cover allow explanation, redacted/sanitized subjects, audit event writing, fail-closed audit append failure, and degraded preview without catalog.
- [x] #3 Task 1 focused pytest command passes or failures are documented with blockers.
- [x] #4 Changes are committed separately for Task 1.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Task 1 service foundation in mcp_unified/gateway/policy_explain.py with typed request/response models, GatewayPolicyExplainService, GatewayPolicyExplainError, subject redaction helpers, effective decision handling, strict audit append behavior, and bounded preview pagination.

Focused tests in tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py cover allow/ask/deny explanations, redaction/sanitization states, audit writes, audit fail-closed behavior, resolver failure auditing, degraded previews, validation error redaction, and preview pagination.

Subagent review results: latest spec compliance review passed with no gaps; latest code-quality review found no Critical or Important issues. Minor follow-up candidates were cursor length hardening, async catalog_provider edge coverage, and richer install metadata in the later catalog task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 1 added the standalone MCP policy explain service foundation and focused regression tests. Verification from this worktree using the root checkout virtualenv:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -v -> 34 passed, 7 warnings
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/policy_explain.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_service.py -s B101 -f json -o /tmp/bandit_task2370.json -> 0 findings
- git diff --check -> exit 0

Known non-blocking follow-ups: cap oversized cursor tokens, add explicit audit_store=None and async catalog_provider tests, and preserve installation metadata in the Task 2 catalog provider.
<!-- SECTION:FINAL_SUMMARY:END -->

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
