---
id: TASK-2394
title: Implement MCP UAT JSON-RPC transport remediation
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-20 07:39'
labels:
  - mcp
  - uat
  - jsonrpc
  - implementation
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved MCP UAT JSON-RPC transport remediation plan across mounted tldw_server MCP, standalone MCP gateway transports, smoke harness alignment, auth compatibility hardening, policy resolver stabilization, and full focused/live UAT validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mounted HTTP and batch JSON-RPC routes use raw parsing, strict response serialization, notification 204 handling, explicit-null-id responses, and post-protocol authz JSON-RPC errors.
- [x] #2 Mounted WebSocket transport uses strict response serialization, notification suppression, exact keepalive handling, parse/invalid-request frames, and auth compatibility parity.
- [x] #3 Standalone gateway HTTP/WebSocket/stdio preserves absent-id notification versus explicit null id and omits invalid optional null response fields.
- [ ] #4 Trusted single-user/test compatibility metadata cannot be forged by client-supplied request metadata.
- [ ] #5 Policy resolver import-cycle remediation is implemented without weakening fail-closed behavior.
- [ ] #6 Smoke harness expectations align with valid ping metadata, unknown-tool semantics, and WebSocket keepalives.
- [ ] #7 Focused pytest suites, standalone/mounted smoke paths where feasible, and Bandit touched-scope validation are run or documented with reasons.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete. Added mounted JSON-RPC transport helper and direct helper tests in commits 93f4d20761 and eab9a44812. Focused helper pytest passed with 22 tests. Bandit on production helper reported no findings. Spec review approved. Code-quality review initially found missing helper coverage; follow-up commit fixed coverage and re-review approved.

Task 2 complete. Mounted HTTP /request and /request/batch now use raw JSON-RPC body parsing and helper-based response serialization in commits 6dee07e6d9 and 3a386d0eb5. Focused route suite passed: 30 passed, 4 warnings. Spec review approved. Code-quality review initially found notification short-circuiting before server processing; follow-up commit routes notifications through the server/protocol path while suppressing responses, and re-review approved. Minor accepted note: _is_jsonrpc_notification_payload is now unused and can be removed during cleanup.

Task 3 complete. Mounted WebSocket JSON-RPC normalization committed as 36824df465. Focused WebSocket suite passed: 16 passed, 4 warnings. Spec review approved. Code-quality review approved with non-blocking maintainability note: consider moving explicit-null-id sentinel helpers into jsonrpc_transport.py in a future cleanup if behavior keeps evolving.

Task 4 complete. Notification and explicit-null id semantics were implemented in commits 7489802a81 and 83a61bc076. Mounted protocol now handles notifications/initialized as a no-op notification, distinguishes omitted id from explicit id:null for raw dict and MCPRequest inputs, and rejects invalid request ids before Pydantic coercion. Standalone gateway JSON-RPC now preserves absent-id notifications versus explicit-null requests across HTTP, stdio, and in-process smoke paths, normalizes runtime context request_id labels, and serializes JSON-RPC responses without invalid optional null fields. Verification: red review-fix tests failed before the fix and passed after; focused Task 4 suite passed with 281 passed, 6 warnings under loopback-enabled execution; git diff --check passed; compileall passed for touched files; Bandit on touched production files reported zero findings. Spec and code-quality re-reviews approved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation plan tasks completed or documented with justified skips.
- [ ] #2 Tests added/updated for new behavior.
- [ ] #3 Focused regression commands and results recorded.
- [ ] #4 Bandit run for touched MCP scopes or documented environment blocker.
- [ ] #5 Final summary added with known residual risks.
- [ ] #6 Changes committed incrementally.
<!-- DOD:END -->
