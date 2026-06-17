## Stage 1: Standard Session Close
**Goal**: Move ACP teardown to the v1 `session/close` method while preserving compatibility with existing private runner builds.
**Success Criteria**: Python clients call `session/close` first, fall back to `_tldw/session/close` only on method-not-found style failures, and the Go runner accepts standard `session/close`.
**Tests**: Focused pytest coverage for standard and fallback close paths; Go runner coverage for standard close routing.
**Status**: Complete

## Stage 2: MCP Transport Capability Gates
**Goal**: Prevent unsupported HTTP/SSE MCP server transports from being forwarded to downstream agents that did not advertise the required ACP `mcpCapabilities`.
**Success Criteria**: The runner preserves supported MCP server configs and returns invalid params for unsupported HTTP/SSE transports after downstream initialization.
**Tests**: Go runner tests for supported HTTP transport forwarding and unsupported SSE/HTTP rejection.
**Status**: Complete

## Stage 3: API Session Setup Validation
**Goal**: Align public ACP session setup schemas with ACP v1 transport shapes and reject malformed requests before they reach the runner.
**Success Criteria**: `cwd` must be absolute; MCP server configs support `stdio`, `http`, `sse`, and legacy `websocket`; stdio requires an absolute command; URL-based transports require a URL; dict env is normalized to ACP name/value pairs for compatibility.
**Tests**: Focused schema tests for cwd, stdio, HTTP/SSE, env normalization, and endpoint request preservation.
**Status**: Complete

## Stage 4: Verification
**Goal**: Prove the touched ACP scope works and record any residual risks.
**Success Criteria**: Focused pytest suite, Go runner tests, and Bandit touched-scope run complete; Backlog task has verification notes.
**Tests**: `python -m pytest ...`, `go test ./tools/tldw-agent/internal/acp`, and Bandit on touched Python files.
**Status**: Complete
