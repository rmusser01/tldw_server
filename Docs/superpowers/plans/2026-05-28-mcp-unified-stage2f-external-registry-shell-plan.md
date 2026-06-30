## Stage 1: Package Federation Contracts
**Goal**: Add standalone `mcp_unified.federation` contracts for virtual external tools, transport lifecycle, health state, and execution results without importing `tldw_Server_API`.
**Success Criteria**: Public federation imports are package-local, carry namespaced `ext.<server_id>.<tool>` metadata, and represent lifecycle state without process spawning.
**Tests**: Import-boundary tests and contract/model tests for virtual tool names, caller-owned metadata, and fake transport defaults.
**Status**: Complete

## Stage 2: Registry-Backed Non-Spawning Manager
**Goal**: Implement a small manager that loads enabled `ExternalServerDefinition` rows from `ExternalRegistryStore`, starts/stops fake transports, refreshes virtual tools, reports health, and never launches stdio/websocket processes.
**Success Criteria**: A registry-backed fake server can be started, listed, refreshed, stopped, and isolated per manager instance.
**Tests**: Async tests with an in-memory registry store and fake transport factory.
**Status**: Complete

## Stage 3: Policy-Gated Execution And Audit
**Goal**: Gate fake upstream tool calls on explicit effective-policy grants and emit structured audit events for allow/deny/lifecycle/discovery outcomes when an audit store is provided.
**Success Criteria**: Discovered tools remain non-executable until profile/effective-policy grants allow the server/tool, denied calls carry machine-readable reason codes, and audit records avoid secret values.
**Tests**: Async execution tests for denied server grant, denied tool allowlist, allowed execution, and audit event ordering.
**Status**: Complete

## Stage 4: Compatibility Verification
**Goal**: Keep existing host MCP behavior intact while proving the new package shell stands alone.
**Success Criteria**: Focused standalone tests, existing storage/runtime boundary tests, and Bandit on touched package code pass.
**Tests**: `pytest` focused MCP package tests plus `bandit -r mcp_unified/federation`.
**Status**: Complete
