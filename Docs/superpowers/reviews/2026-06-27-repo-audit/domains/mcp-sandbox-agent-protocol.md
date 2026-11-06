# MCP, Sandbox, And Agent Protocol Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: MCP, Sandbox, and Agent Protocol
- In scope: MCP tool execution, command runtimes, filesystem/network policies, sandbox runners, agent protocol adapters, audit/reporting hooks, and related tests.
- Out of scope: remediation implementation and new runtime support.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CANDIDATE-mcp-sandbox-agent-protocol-001 | confirmed_issue | static_confirmed | medium | high | security | Scoped AuthNZ JWT restrictions are bypassed on ACP and sandbox WebSocket endpoints | open | validated |
| CANDIDATE-mcp-sandbox-agent-protocol-002 | confirmed_issue | static_confirmed | low | high | reliability | ACP reconnect WebSocket replay leaks WSBroadcaster subscriptions/tasks | open | validated |

## Index Mapping

The coordinator should map these candidates into `findings-index.json` if accepted. This domain report intentionally uses the requested candidate ID prefix and does not edit the shared index.

- `CANDIDATE-mcp-sandbox-agent-protocol-001`
  - `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md`
  - `owner_domain`: MCP, Sandbox, and Agent Protocol
  - `affected_paths`: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`, `tldw_Server_API/app/api/v1/endpoints/sandbox.py`, `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`, `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
  - `recommendation`: Authenticate ACP and sandbox WebSocket bearer tokens through the same scoped-token enforcement used by HTTP endpoints, including `scope`, `allowed_endpoints`, `allowed_methods`, `allowed_paths`, quota, and schedule claims. Add scoped-JWT tests for rejected ACP stream, ACP SSH, and sandbox run stream handshakes.
  - `status`: open
  - `validation_status`: validated
- `CANDIDATE-mcp-sandbox-agent-protocol-002`
  - `source_report`: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md`
  - `owner_domain`: MCP, Sandbox, and Agent Protocol
  - `affected_paths`: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`, `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py`, `tldw_Server_API/app/core/Agent_Client_Protocol/event_bus.py`
  - `recommendation`: Retain the reconnect `WSBroadcaster` and connection id in `acp_session_stream`, remove the connection and call `stop()` in `finally`, and use a per-connection consumer id or a single shared session broadcaster. Add an endpoint-level reconnect-disconnect test that verifies the bus subscriber and broadcaster task are cleaned up.
  - `status`: open
  - `validation_status`: validated

## Confirmed Issues

### CANDIDATE-mcp-sandbox-agent-protocol-001 - Scoped AuthNZ JWT restrictions are bypassed on ACP and sandbox WebSocket endpoints

Scoped AuthNZ bearer tokens are constrained on normal HTTP routes, but ACP and sandbox WebSocket helpers decode or verify the bearer token directly and return a user id without applying the scoped-token contract.

Evidence:

- HTTP scoped-token enforcement is explicit. `require_token_scope()` describes and enforces `scope` checks for JWTs and API keys in `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:2044`. The regular AuthNZ user resolver also rejects scoped tokens unless the route declares scope enforcement, including `scope`, `allowed_endpoints`, `allowed_methods`, `allowed_paths`, `max_calls`, `max_runs`, and `schedule_id` claims in `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py:565`.
- ACP WebSocket auth bypasses that enforcement for JWTs. `_authenticate_ws()` calls `get_jwt_manager().verify_token(token)` and returns `token_data.user_id` at `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:948`. The `required_scope` argument is only used for the API-key path at `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:988`.
- The ACP stream and SSH endpoints request write scope for API keys but rely on the bypassing JWT branch: `acp_session_stream()` passes `required_scope="write"` at `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1220`, and `acp_session_ssh()` does the same at `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1354`.
- The ACP stream is a control channel, not just a passive event feed. It handles `permission_response`, `cancel`, and `prompt` client messages in `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1545`, `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1620`, and `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1637`.
- Sandbox WebSocket auth has the same JWT gap. `_resolve_sandbox_ws_user_id()` verifies an access token and returns `int(sub)` at `tldw_Server_API/app/api/v1/endpoints/sandbox.py:468`, while the API-key branch enforces `required_scope="read"` at `tldw_Server_API/app/api/v1/endpoints/sandbox.py:510`.
- Sandbox run streaming checks run ownership before streaming, but the socket also accepts `stdin` frames and calls `hub.push_stdin()` for interactive runs at `tldw_Server_API/app/api/v1/endpoints/sandbox.py:2195`.
- Existing focused tests cover read-only API-key rejection for ACP stream/SSH, but not scoped AuthNZ JWT claims: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py:371` and `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py:405`.

Impact:

A scoped AuthNZ JWT issued for a narrow endpoint/path/method can be reused against owned ACP or sandbox WebSocket resources because the WebSocket helpers skip the HTTP scoped-token guard. The remaining session/run ownership checks limit cross-user access, but the bypass still expands token authority to ACP prompt execution, permission responses, cancellation, ACP SSH proxying, sandbox log streaming, and sandbox stdin injection.

Recommendation:

Project WebSocket handshakes into a request-like AuthNZ context and reuse `verify_jwt_and_fetch_user()` plus route-specific scope enforcement, or implement an equivalent WebSocket guard that validates all scoped claims before returning a user id. Treat ACP stream/SSH as write-capable. Consider separating sandbox log streaming from stdin-capable interactive control, or require write scope when stdin is enabled.

### CANDIDATE-mcp-sandbox-agent-protocol-002 - ACP reconnect WebSocket replay leaks WSBroadcaster subscriptions/tasks

The ACP reconnect branch creates a new `WSBroadcaster` when `last_sequence > 0`, starts it on the session event bus, and adds a WebSocket connection, but the endpoint never stops that broadcaster or removes its connection on disconnect.

Evidence:

- `acp_session_stream()` creates a local `WSBroadcaster`, starts it, and adds a connection when `bus is not None and last_sequence > 0` in `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1275`.
- The endpoint `finally` block only unregisters the runner callback, stops the `WebSocketStream`, and releases quota in `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py:1335`. There is no call to `broadcaster.remove_connection()` or `broadcaster.stop()`.
- `WSBroadcaster.start()` subscribes to the bus and spawns `_consume_loop()` in `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py:139`. `WSBroadcaster.stop()` is the method that cancels the task and unsubscribes in `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py:147`.
- `WSBroadcaster.consumer_id` is the fixed string `ws_broadcaster` at `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py:57`.
- `SessionEventBus.subscribe()` stores subscribers by consumer id and replaces any existing queue for the same id in `tldw_Server_API/app/core/Agent_Client_Protocol/event_bus.py:85`. Repeated reconnects replace the live bus entry while stale broadcaster tasks remain blocked on their old queues and retain send callbacks.
- Existing reconnect tests exercise `WSBroadcaster.add_connection()` and call `stop()` manually in `tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py:33`, but do not cover endpoint cleanup after the reconnect path.

Impact:

An authenticated session owner can repeatedly reconnect with `last_sequence > 0` and accumulate orphaned asyncio tasks and callback references. The fixed subscriber id also makes concurrent reconnect/live delivery unstable because later subscriptions replace earlier queues at the bus level. This is primarily reliability/resource exhaustion risk, but the surface is remotely reachable by authorized ACP users.

Recommendation:

Keep `broadcaster` and `conn_id` outside the reconnect branch, call `remove_connection(conn_id)` and `await broadcaster.stop()` in the endpoint `finally`, and avoid the fixed consumer id collision by using per-connection consumer ids or one managed session-scoped broadcaster. Add an endpoint test that forces `last_sequence > 0`, disconnects, and asserts that the broadcaster task is cancelled and the event bus subscriber is removed.

## Likely Risks

No additional likely-risk candidates were promoted during this pass.

## Improvement Opportunities

- The standalone `apps/mcp-unified` package includes `StdioProcessPolicy` controls for external stdio MCP transports, but the in-server compatibility adapter uses `ACPStdioClient` directly for configured external servers. I did not find a user-facing API that lets ordinary users write those command definitions, so I did not promote this to a candidate finding. If external server configuration becomes editable through an admin/API surface, the in-server adapter should converge on the standalone process policy.
- Bandit was useful on the reviewed scope, but including `tldw_Server_API/app/core/MCP_unified` also includes `tldw_Server_API/app/core/MCP_unified/tests`. The 17 medium findings from this scoped run were all test-file temp-path or chmod patterns, not production runtime findings.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/dependency-manifest-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- `tldw_Server_API/app/api/v1/endpoints/tools.py`
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/core/MCP_unified/server.py`
- `tldw_Server_API/app/core/MCP_unified/protocol.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- `tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py`
- `tldw_Server_API/app/core/MCP_unified/command_runtime/executor.py`
- `tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py`
- `tldw_Server_API/app/core/MCP_unified/external_servers/manager.py`
- `tldw_Server_API/app/core/MCP_unified/external_servers/config_schema.py`
- `tldw_Server_API/app/core/MCP_unified/external_servers/transports/stdio_adapter.py`
- `tldw_Server_API/app/core/MCP_unified/external_servers/transports/websocket_adapter.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/external_federation_module.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/sandbox_module.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/web_fetch_module.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/config.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/event_bus.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/hardening.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/stdio_client.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/sse_consumer.py`
- `tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py`
- `tldw_Server_API/app/core/Sandbox`
- `apps/mcp-unified/pyproject.toml`
- `apps/mcp-unified/src/mcp_unified/federation/config_schema.py`
- `apps/mcp-unified/src/mcp_unified/federation/process_policy.py`
- `apps/mcp-unified/src/mcp_unified/federation/stdio_transport.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py`
- `tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py`
- `tldw_Server_API/tests/sandbox/test_ws_heartbeat_seq.py`
- Relevant MCP Unified tests under `tldw_Server_API/app/core/MCP_unified/tests`

### Tests Or Scans Run

- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_stream_rejects_read_only_api_key_in_multi_user_mode tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPWebSocketConnection::test_websocket_ssh_rejects_read_only_api_key_in_multi_user_mode tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py`
  - Result: `7 passed, 51 warnings in 11.18s`.
- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/MCP_unified tldw_Server_API/app/core/Sandbox tldw_Server_API/app/core/Agent_Client_Protocol tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/api/v1/endpoints/sandbox.py -f json -o /tmp/bandit_mcp_sandbox_agent_protocol.json`
  - Result: exit code `1`; JSON written to `/tmp/bandit_mcp_sandbox_agent_protocol.json`.
  - Summary: `4418` results, `0` high severity, `17` medium severity, `4401` low severity. The medium findings were in test files under `tldw_Server_API/app/core/MCP_unified/tests`, primarily Bandit `B108` temp-path findings plus one `B103` chmod test case.

### Blocked Or Unverified Areas

- No environment-changing setup was performed: no dependency installation, Docker, service startup, or network access.
- I did not live-exploit scoped JWT WebSocket handshakes because creating realistic scoped AuthNZ tokens and sessions would require broader environment setup. The scoped-JWT finding is static-confirmed against source paths and contrasted with the existing HTTP scoped-token guard.
- I did not run the full backend suite. Verification was limited to focused ACP reconnect/API-key and sandbox WebSocket stdin tests plus scoped Bandit.
- I did not edit `findings-index.json`; candidate mapping is recorded above for coordinator ingestion.
- I did not touch the unrelated untracked watchlist template files.

### Evidence Notes

Commands used for inspection included:

- `find tldw_Server_API/app/core/MCP_unified -maxdepth 3 -type f | sort`
- `find tldw_Server_API/app/core -maxdepth 2 \( -iname '*Sandbox*' -o -iname '*sandbox*' -o -iname '*Agent*' -o -iname '*agent*' \) -print | sort`
- `find tldw_Server_API/tests -type f \( -iname '*mcp*' -o -iname '*sandbox*' -o -iname '*agent*' -o -iname '*tool*' -o -iname '*security*' \) | sort`
- `rg --files apps/mcp-unified | sort`
- `rg -n "websocket|WebSocket|authenticate|Authorization|jwt|scope|verify_token|require_token|allowed_endpoints|RequestContext|handle_websocket|mcp" tldw_Server_API/app/core/MCP_unified tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- `rg -n "def _resolve_sandbox_ws_user_id|stream_run_logs|verify_token_async|decode_access_token|push_stdin|websocket|Authorization|allowed_endpoints|scope" tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- `rg -n "read_only|read-only|scope|WSBroadcaster|last_sequence|broadcaster|stop\(|unregister_websocket|reconnect|subscribe|unsubscribe" tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py tldw_Server_API/tests/Agent_Client_Protocol/test_ws_reconnect.py tldw_Server_API/tests/sandbox/test_ws_stdin_caps.py tldw_Server_API/tests/sandbox/test_ws_heartbeat_seq.py`
- Multiple `nl -ba ... | sed -n ...` reads over the files listed above for line-specific evidence.
- `jq '{metrics: .metrics._totals, issue_count: (.results|length), severities: (.results|group_by(.issue_severity)|map({severity: .[0].issue_severity, count: length})), confidences: (.results|group_by(.issue_confidence)|map({confidence: .[0].issue_confidence, count: length}))}' /tmp/bandit_mcp_sandbox_agent_protocol.json`

Positive coverage notes:

- MCP Unified HTTP and WebSocket paths use stronger AuthNZ integration than ACP/sandbox. `MCPServer.handle_websocket()` calls the host auth provider's `authenticate_authnz_websocket_token()` at `tldw_Server_API/app/core/MCP_unified/server.py:1219`; that adapter projects the handshake into a Starlette request and calls `verify_jwt_and_fetch_user()` at `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py:267`.
- MCP protocol tool execution has layered checks for RBAC, API-key scopes, tool-name scoping, path-scope candidates, runtime approval, and audit reporting in `tldw_Server_API/app/core/MCP_unified/protocol.py`.
- The MCP filesystem module and virtual command runtime use governed tool calls rather than raw shell execution. Network-facing web fetch/search modules are disabled by default and route through policy evaluators when enabled.
