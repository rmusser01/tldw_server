---
id: TASK-12138
title: Remediate ACP and sandbox WebSocket scoped JWT audit finding
status: Done
created_date: 2026-07-04 00:15
labels:
- audit
- remediation
- mcp
- sandbox
- websocket
- auth
- security
- wave-2
priority: high
references:
- AUDIT-2026-06-27-MCP-WS-001
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-acp-sandbox.md
modified_files:
- Docs/superpowers/plans/2026-07-02-mcp-ws-scoped-jwt-remediation-plan.md
- backlog/tasks/task-12138 - Remediate-ACP-and-sandbox-WebSocket-scoped-JWT-audit-finding.md
- tldw_Server_API/app/api/v1/API_Deps/auth_deps.py
- tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py
- tldw_Server_API/app/api/v1/endpoints/sandbox.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py
- tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py
updated_date: 2026-07-04 00:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for ACP and sandbox WebSocket scoped-JWT enforcement: JWT-authenticated WebSocket connections should enforce the same endpoint/scope policy used by HTTP routes, while preserving existing API-key compatibility behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is present before production code changes on this current-base branch.
- [x] #2 Shared WebSocket token-scope enforcement helper applies endpoint-aware JWT scope checks without weakening API-key compatibility.
- [x] #3 ACP session stream and SSH WebSocket routes enforce scoped JWT claims for bearer-token clients.
- [x] #4 Sandbox run stream WebSocket route enforces scoped JWT claims for bearer-token clients.
- [x] #5 Focused WebSocket authorization tests cover accepted scoped tokens and rejected missing-scope tokens.
- [x] #6 Touched-scope Bandit and focused pytest verification are recorded.
- [x] #7 Residual API-key compatibility or route-scope tradeoffs are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-03 current-base port: applied the worker MCP/sandbox scoped JWT remediation onto origin/dev f2d9be9864 with unique task id TASK-12138. Added AuthNZ enforce_websocket_token_scope() to project WebSocket handshakes into the existing require_token_scope guard. Wired ACP session stream and SSH WebSockets with write scope and endpoint ids acp.sessions.stream/acp.sessions.ssh, and sandbox run stream with read scope and endpoint id sandbox.runs.stream. Existing API-key behavior is preserved; the new scoped guard is applied only in the bearer JWT branch.

Verification recorded: focused pytest command for ACP stream/SSH scoped JWT rejection, existing ACP read-only API-key rejection, and sandbox scoped JWT rejection -> 5 passed, 105 warnings. Bandit production touched scope: PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/api/v1/endpoints/sandbox.py -f json -o /tmp/bandit_task_12138_mcp_ws_scope.json -> 0 results, 0 errors. git diff --check -> clean.

Review notes: HTTPException scope denials are not included in the ACP/sandbox noncritical exception tuples, so endpoint/scope failures propagate instead of falling back to acceptance. Residual tradeoff: sandbox run stream remains read-scoped to match the existing endpoint/compatibility model; API-key compatibility behavior is intentionally unchanged. Two unrelated untracked watchlist template files are present in this worktree and were not staged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Remediated the ACP and sandbox WebSocket scoped-JWT audit finding by adding a shared WebSocket token-scope enforcement helper and applying it to ACP session stream, ACP SSH, and sandbox run stream bearer-token authentication. Added focused regression tests proving scoped JWTs without endpoint permission are rejected while existing API-key scope behavior remains covered. Focused pytest, production Bandit scan, and whitespace verification passed.
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
