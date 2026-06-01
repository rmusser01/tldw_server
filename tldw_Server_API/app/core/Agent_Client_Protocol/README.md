# Agent Client Protocol

Agent_Client_Protocol is the server-side integration layer for ACP-compatible coding and automation agents. It loads agent profiles and runner configuration, starts local or sandboxed runner sessions, streams structured agent events over HTTP/SSE/WebSocket, gates tool calls through permission policy, and adapts MCP transports so ACP sessions can expose tools and artifacts through the API.

## Start Here

- `agent_registry.py` loads global YAML agent profiles, persists dynamic registrations, and reports runtime availability.
- `runner_client.py` is the local stdio runner client and session manager used by the ACP HTTP API.
- `sandbox_runner_client.py` manages sandbox-backed ACP sessions and SSH connection metadata.
- `events.py`, `event_bus.py`, and `consumers/` define and deliver the structured ACP event stream.
- `governance_filter.py`, `permission_decision_service.py`, `permission_tiers.py`, and `tool_gate.py` implement permission decisions around tool calls.
- `adapters/mcp_adapter.py`, `adapters/mcp_transport.py`, and `adapters/mcp_runners.py` bridge ACP sessions to MCP transports and runners.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`, declared under `/acp`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`.
- Related tests: `tldw_Server_API/tests/Agent_Client_Protocol/`.

## Responsibilities

- Register, list, inspect, and health-check configured agent profiles.
- Create, prompt, cancel, close, fork, reconcile, and diagnose ACP sessions.
- Stream agent events, including thinking, tool calls, tool results, file changes, terminal output, permission requests, status changes, token usage, and lifecycle events.
- Launch local runner processes from `[ACP]` config or environment overrides.
- Launch sandboxed runner sessions when sandbox execution is enabled.
- Apply runtime and MCP Hub permission policy before tool execution.
- Record audit, metrics, checkpoint, replay, and health data for ACP activity.
- Expose MCP-backed tools and agent execution modes through ACP adapters.

## Module Map

- `config.py`: ACP runner and sandbox configuration loading, path normalization, argument parsing, and validation warnings.
- `agent_registry.py`: global and dynamic agent profile registry.
- `runner_client.py`: local ACP stdio process client and session orchestration.
- `sandbox_runner_client.py`: sandbox session orchestration and SSH metadata.
- `stdio_client.py`, `stream_client.py`, `sandbox_bridge.py`: lower-level client and bridge helpers.
- `events.py`, `event_bus.py`: event contracts and in-process fanout.
- `consumers/`: event consumers for audit logs, checkpoints, metrics, replay, SSE, and WebSocket broadcast.
- `governance_filter.py`, `tool_gate.py`, `permission_decision_service.py`, `policy_conditions.py`, `permission_tiers.py`: permission and policy enforcement.
- `adapters/`: ACP adapter implementations, including stdio and MCP support.
- `multiplex/`: protocol and manager support for multiplexed ACP sessions.
- `metrics.py`, `execution_health.py`, `health_monitor.py`: runtime health and metrics helpers.
- `templates.py`, `prompt_utils.py`, `merge_utils.py`, `triggers.py`: prompt, merge, and trigger utilities.

## How It Connects

- FastAPI routes in `agent_client_protocol.py` use AuthNZ dependencies and scoped permission guards for agent, session, run, task, SSE, and WebSocket operations.
- ACP session state and audit data connect to DB management modules such as `ACP_Sessions_DB`.
- `runner_client.py` and `sandbox_runner_client.py` connect to the Sandbox module for isolated execution and to runtime policy services for permission handling.
- MCP adapter files connect ACP sessions to MCP transports, tool presentation, and LLM-driven or agent-driven MCP runs.
- Async prompt execution connects to the Scheduler task surface through the ACP endpoint.
- Workspaces and orchestration code consume ACP capabilities when binding agents to workspace tasks.
- Metrics, checkpoints, replay helpers, and audit consumers observe the event bus rather than duplicating runner logic.

## Extension Points

- Add a new runner transport by starting with `adapters/base.py`, `stdio_adapter.py`, and `mcp_transport.py`.
- Add a new event sink by implementing a consumer under `consumers/` and wiring it to `event_bus.py`.
- Add or adjust agent profile fields in `agent_registry.py` and `schemas/agent_client_protocol.py`.
- Change tool permission behavior by inspecting `governance_filter.py`, `tool_gate.py`, and `permission_decision_service.py` first.
- Extend sandboxed execution through `sandbox_runner_client.py` and `sandbox_bridge.py`.
- Add endpoint behavior in `endpoints/agent_client_protocol.py` only after checking the session manager and schema contracts.

## Testing

- Unit and API coverage lives under `tldw_Server_API/tests/Agent_Client_Protocol/`.
- The test suite includes endpoint, WebSocket, session store, run handler, sandbox runner, governance filter, tool gate, MCP adapter, MCP runner, MCP transport, event bus, audit, metrics, and trigger tests.
- When changing endpoint contracts, also inspect `test_acp_endpoints.py` and `test_acp_websocket.py`.
- When changing tool permission behavior, inspect `test_governance_filter.py` and `test_tool_gate.py`.

## Gotchas

- `runner_cwd` and relative `HOME` values are resolved relative to `Config_Files`, not the process working directory.
- Empty or missing `runner_command` leaves ACP sessions unable to launch even when routes import successfully.
- Permission requests can time out and be denied by policy; do not assume tool calls proceed after an event is emitted.
- Sandbox configuration includes execution quotas, network policy, and SSH port ranges that affect session startup.
- Event payloads can include file paths, terminal output, and raw session data; keep redaction and audit behavior in mind when adding consumers.
