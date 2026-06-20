---
id: TASK-2394
title: Implement MCP UAT JSON-RPC transport remediation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-20 10:22'
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
- [x] #4 Trusted single-user/test compatibility metadata cannot be forged by client-supplied request metadata.
- [x] #5 Policy resolver import-cycle remediation is implemented without weakening fail-closed behavior.
- [x] #6 Smoke harness expectations align with valid ping metadata, unknown-tool semantics, and WebSocket keepalives.
- [x] #7 Focused pytest suites, standalone/mounted smoke paths where feasible, and Bandit touched-scope validation are run or documented with reasons.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete. Added mounted JSON-RPC transport helper and direct helper tests in commits 93f4d20761 and eab9a44812. Focused helper pytest passed with 22 tests. Bandit on production helper reported no findings. Spec review approved. Code-quality review initially found missing helper coverage; follow-up commit fixed coverage and re-review approved.

Task 2 complete. Mounted HTTP /request and /request/batch now use raw JSON-RPC body parsing and helper-based response serialization in commits 6dee07e6d9 and 3a386d0eb5. Focused route suite passed: 30 passed, 4 warnings. Spec review approved. Code-quality review initially found notification short-circuiting before server processing; follow-up commit routes notifications through the server/protocol path while suppressing responses, and re-review approved. Minor accepted note: _is_jsonrpc_notification_payload is now unused and can be removed during cleanup.

Task 3 complete. Mounted WebSocket JSON-RPC normalization committed as 36824df465. Focused WebSocket suite passed: 16 passed, 4 warnings. Spec review approved. Code-quality review approved with non-blocking maintainability note: consider moving explicit-null-id sentinel helpers into jsonrpc_transport.py in a future cleanup if behavior keeps evolving.

Task 4 complete. Notification and explicit-null id semantics were implemented in commits 7489802a81 and 83a61bc076. Mounted protocol now handles notifications/initialized as a no-op notification, distinguishes omitted id from explicit id:null for raw dict and MCPRequest inputs, and rejects invalid request ids before Pydantic coercion. Standalone gateway JSON-RPC now preserves absent-id notifications versus explicit-null requests across HTTP, stdio, and in-process smoke paths, normalizes runtime context request_id labels, and serializes JSON-RPC responses without invalid optional null fields. Verification: red review-fix tests failed before the fix and passed after; focused Task 4 suite passed with 281 passed, 6 warnings under loopback-enabled execution; git diff --check passed; compileall passed for touched files; Bandit on touched production files reported zero findings. Spec and code-quality re-reviews approved.

Task 5 complete. Mounted HTTP/WS single-user compatibility auth hardening was implemented in commits 4a77edd6db and 1022829683. HTTP and WebSocket now attach trusted server-created metadata only after configured single-user/test API key, test-mode guard, and IP allowlist checks pass. Protocol authorization honors compatibility admin claims only through the server-created sentinel, and forged client metadata with permissions, auth_via, compat_claims_source, or _server_auth_* keys does not bypass RBAC. Verification: red review-fix regressions failed before the fix and passed after; focused Task 5 suite passed with 32 passed, 4 warnings; git diff --check passed; compileall passed; Bandit on touched production files reported zero findings. Spec and code-quality re-reviews approved.

Task 6 complete in commit 61e5796ed3. Mounted MCP policy resolver construction now uses a cycle-safe host adapter loader for the MCP Hub policy resolver, without moving host-specific resolver code into the standalone package. Added coverage for a policy-enabled discovery tool call through the tldw resolver and for resolver runtime failures failing closed before governance preflight. Verification: focused policy/governance pytest passed with 17 passed, 4 warnings; git diff --check passed; compileall passed for tldw_policy.py; Bandit on tldw_policy.py reported zero findings. Red-check caveat: the worker left tests and implementation uncommitted before controller takeover, so the red failure was not reproduced locally after takeover.

Task 7 complete in commit 69fa5c0d8a. Smoke harness expectations now accept ping result metadata while still requiring `pong: true`, accept mounted-style unknown-tool `-32602` only when the message indicates unknown/missing/not-found tool, preserve strict `-32601` handling for unknown-method checks, ignore exact live WebSocket keepalive frames, and include a prefixed standalone FastAPI smoke fixture app. Verification: targeted red check failed for the intended fixture/keepalive/ping/unknown-tool cases before implementation; targeted green check passed with 6 passed; full smoke client suite passed with 81 passed, 5 warnings under loopback-enabled execution; git diff --check passed; compileall passed; Bandit on smoke scenarios/transports reported zero findings.

Task 8 complete. Focused mounted JSON-RPC regression suite passed with 74 passed, 4 warnings. Standalone gateway/smoke focused suite first failed under the sandbox because live WebSocket tests could not bind 127.0.0.1; rerunning the same command with loopback escalation passed with 280 passed, 6 warnings. Auth/policy focused suite passed with 23 passed, 4 warnings. Added test-only stabilization so mounted WebSocket compatibility tests restore singleton server state with monkeypatch instead of leaving a recording protocol installed for later tests. git diff --check passed; compileall passed for the touched test module. Bandit was not run for this child slice because only tests/tracking files changed.

Task 9 complete. Full UAT smoke matrix passed for fixture CLI tests, standalone in-process, standalone stdio subprocess, standalone live HTTP, standalone live WebSocket, mounted tldw_server live HTTP, mounted API-key WebSocket, and mounted JWT WebSocket. Validation found one still-valid mounted smoke gap: the mounted server has separate single and batch HTTP endpoints, so the live HTTP transport now accepts an optional `batch_url` and the CLI exposes `http --batch-url`. Added regression coverage in `test_live_http_transport_uses_batch_url_for_batch_payloads`. Mounted JWT smoke used subject `1` instead of the plan's illustrative `smoke-user`, because live AuthNZ RBAC evaluates DB grants by user id. Verification: `test_smoke_client.py` passed with 82 passed, 5 warnings; live smoke commands all reported PASS; compileall passed; git diff --check passed. The broad MCP Bandit scan wrote `/tmp/bandit_mcp_uat_remediation.json` and reported existing baseline findings outside touched production smoke files; direct Bandit on `mcp_unified/smoke/transports.py` and `mcp_unified/smoke/cli.py` wrote `/tmp/bandit_mcp_uat_touched_production.json` and reported zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the full MCP UAT JSON-RPC transport remediation plan across mounted tldw_server MCP and the standalone MCP gateway/smoke package. The final validation slice added separate HTTP batch endpoint support to the smoke transport/CLI so mounted tldw_server can be exercised through its real `/request` and `/request/batch` endpoints. Focused pytest suites, live UAT smoke paths, compile checks, diff checks, and touched-production Bandit validation are recorded above.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Implementation plan tasks completed or documented with justified skips.
- [x] #2 Tests added/updated for new behavior.
- [x] #3 Focused regression commands and results recorded.
- [x] #4 Bandit run for touched MCP scopes or documented environment blocker.
- [x] #5 Final summary added with known residual risks.
- [x] #6 Changes committed incrementally.
<!-- DOD:END -->
