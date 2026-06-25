# MCP Protocol Tool Execution Refactor Design

## Context

`tldw_Server_API/app/core/MCP_unified/protocol.py` is currently a 4,300-line protocol facade that also owns the security-sensitive `tools/call` execution path. The file combines JSON-RPC request routing, coarse authorization, tool-specific authorization, effective policy checks, external access checks, path-scope enforcement, approval evaluation, governance preflight, hooks, idempotency, rate limiting, execution formatting, metrics, audit, tool-use reporting, resources, prompts, and module handlers.

The refactor goal is not cosmetic file splitting. The goal is easier security review, lower risk for future tool-execution changes, and long-term stability while preserving current public behavior.

## Goals

- Keep `MCPProtocol` as the stable public JSON-RPC facade.
- Make the `tools/call` security pipeline explicit and reviewable.
- Preserve current gate order, error mapping, telemetry, audit, tool-use reporting, nested execution, and compatibility imports.
- Move behavior in small parity-preserving stages backed by characterization tests.
- Avoid extracting resources, prompts, and module handlers in this spec.

## Non-Goals

- Do not rewrite the whole protocol kernel.
- Do not change JSON-RPC response shapes or HTTP error mappings.
- Do not redesign AuthNZ, RBAC, MCP Hub policies, path-scope semantics, approval semantics, hooks, idempotency, or module execution contracts.
- Do not make broad cleanup edits outside the tool-call path.

## Recommended Approach

Use a security-pipeline extraction. `MCPProtocol` remains the public facade and delegates the tool-call path into a new `tool_execution` package. The first implementation should mostly move code, not reinterpret behavior.

The extraction should be stage-first rather than file-count-first. Small files are useful only when the boundaries make the security flow easier to audit.

## Package Shape

Create:

- `tldw_Server_API/app/core/MCP_unified/protocol_types.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/models.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/coordinator.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/security.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/hooks.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/idempotency.py` if `IdempotencyManager` needs to move out of `protocol.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`
- `tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py`

`protocol.py` must keep compatibility re-exports for public symbols that existing tests and modules import from it.

`tool_execution/models.py` is for internal stage result types only. Public shared types should live in `protocol_types.py` to avoid circular imports.

## Shared Types

Move shared protocol/tool-execution types to `protocol_types.py`:

- `RequestContext`
- `PreparedToolCall`
- `InvalidParamsException`
- `GovernanceDeniedError`
- `ApprovalRequiredError`

Keep these importable from `protocol.py` by re-exporting them there.

Keep `MCPRequest`, `MCPResponse`, `MCPError`, and `ErrorCode` in `protocol.py` initially because they belong to the JSON-RPC facade and are not required by extracted tool execution code.

`IdempotencyManager` may remain import-compatible from `protocol.py`, but extracted runtime code must not import `protocol.py` to reach it. During early stages, pass the current manager instance through `ToolExecutionDependencies`. Before or during Stage 5, either keep that dependency-object boundary or move the class to `tool_execution/idempotency.py` and re-export it from `protocol.py` because tests import it there today.

## Tool Execution Dependencies

Introduce a `ToolExecutionDependencies` dataclass. It should hold explicit services and callables rather than the whole `MCPProtocol` object.

Dependencies include:

- module registry
- RBAC policy
- rate limiter
- metrics collector
- telemetry provider
- hook manager
- tool-use recorder
- idempotency manager
- config provider callable
- effective policy resolver
- path scope enforcer
- approval evaluator
- external access evaluator
- reporter facade
- narrow compatibility callables that are still being migrated

Temporary bound method callables are acceptable during migration. The coordinator must not import or store an `MCPProtocol` instance.

Before Stage 3 implementation, create a temporary compatibility callback ledger in the implementation plan. Each callback must have:

- current protocol helper or bound method name
- target module/function that should own the behavior
- removal stage
- parity test that protects the migration

The ledger is part of the security review surface. Do not add a new callback without updating the ledger, and remove ledger entries as callbacks are replaced by extracted module functions.

Introduce a `ToolExecutionReporter` facade in Stage 3 even if its first implementation delegates to existing protocol helpers. The coordinator should depend on this facade for success, denial, failure, replay, audit, and metrics reporting paths from the first extraction stage. Stage 6 then moves the reporter internals rather than changing coordinator call sites again.

## Security Pipeline Contract

The current gate order is a security contract. It should be preserved and covered by tests at the named-gate level.

The preflight/prepare phase is policy-active, not side-effect-free. It may run governance preflight and pre-tool hooks, and hooks can deny execution.

Prepare phase order:

1. Tool name and idempotency key validation.
2. Context allow-list checks.
3. Effective policy lookup.
4. External access checks.
5. Module lookup.
6. Tool definition resolution.
7. Argument hardening and sanitization.
8. Write classification.
9. Module/tool permission checks.
10. Scoped permission and API-key scope checks.
11. Input schema validation.
12. Global write-disable policy.
13. Module argument validator.
14. Idempotency cache key preparation.
15. Path-scope checks.
16. Approval evaluation.
17. Governance preflight.
18. Pre-tool hooks.
19. Prepared-call integrity tag creation.

Execution phase order:

1. Prepared-call integrity verification.
2. Per-tool/category rate limiting.
3. Idempotency binding and cache execution for writes.
4. Circuit-breaker tool execution.
5. Result formatting and execution eval metadata.
6. Audit and metrics.
7. Post-tool hooks.
8. Tool-use reporting for success, failure, and replay cases.

`process_request()` keeps coarse method authorization through `_check_authorization()`. The extracted tool pipeline performs deeper tool-specific authorization. This two-layer model should be documented in code comments near the delegation point.

The authorization boundary is part of the contract. Characterization tests must cover coarse method authorization denial before the tool pipeline starts and deep tool-specific denial inside the prepare pipeline. Both cases should assert caller-visible error mapping and reporting/audit classification where reporting currently occurs.

## Component Responsibilities

`coordinator.py`

- Owns stage ordering.
- Delegates actual logic to `security.py`, `hooks.py`, `runtime.py`, and `reporting.py`.
- Supports both full JSON-RPC calls and direct nested `prepare_tool_call()` plus `execute_prepared_tool_call()` usage.
- Stays small enough to review in one sitting.

`security.py`

- Owns validation, module/tool resolution, argument hardening entry points, write classification, RBAC checks, scoped permissions, API-key gates, effective policy, external access, path scope, approval, and governance preflight.
- Should be organized internally by stage groups before splitting further.

`hooks.py`

- Owns pre-hook and post-hook context creation, payload shaping, hook action coercion, and hook execution.
- Pre-hooks remain part of preflight because they can deny execution.

`runtime.py`

- Owns rate category selection, per-tool/category rate limiting, idempotency binding/cache behavior, circuit-breaker execution, result formatting, and execution eval metadata attachment.
- Receives a config provider instead of importing `get_config()` directly.

`reporting.py`

- Owns tool-use event construction, audit helpers, metrics helpers, and error/reporting classification glue.
- Provides the `ToolExecutionReporter` facade from Stage 3.
- Internal helper extraction should happen late because reporting has many parity-sensitive paths.

`protocol.py`

- Owns JSON-RPC parsing and response objects.
- Owns method dispatch, notifications, top-level rate limiting, and coarse method authorization.
- Owns non-tool handlers: initialize, ping, tools/list, resources, prompts, and modules.
- Re-exports compatibility symbols.

No extracted module should import `MCPProtocol`.

## Testing Strategy

Use characterization tests before moving behavior. Extraction is successful only if behavior stays the same.

Required tests:

- Stage-order tests that assert named security gates run in order: validate, context allow-list, policy/external, resolve/harden/classify, RBAC/API scope, schema/write validation, path scope/approval, governance/hooks, execution/reporting.
- Authorization-boundary tests that separately cover coarse `_check_authorization()` denial before the tool pipeline and deep tool-specific denial inside `prepare_tool_call()`.
- Compatibility import tests for all re-exported symbols.
- Import-boundary tests that assert `tool_execution/*` does not import `MCPProtocol` or `MCP_unified.protocol`.
- Error mapping tests for invalid params, permission denial, governance denial, approval required, rate limit, tool execution failure, and notification behavior.
- Tool-use reporting matrix tests for success, preflight failure, denied, invalid params, execution failure, and idempotency replay.
- Nested execution tests for direct `prepare_tool_call()` plus `execute_prepared_tool_call()` flows, especially the virtual `run` command.
- Lightweight fake-module/fake-policy/fake-hook tests that avoid full FastAPI startup unless HTTP mapping is explicitly under test.

Error mapping tests should check caller-visible JSON-RPC behavior and reporting/audit classification where relevant, because those can drift independently.

Compatibility import tests should include `IdempotencyManager` if it remains publicly importable from `protocol.py`.

## Error Handling Rules

- Preserve existing exception classes and JSON-RPC error codes.
- Preserve `asyncio.CancelledError` behavior; do not accidentally catch it under broad exception handling.
- Keep security denials structured as denials rather than generic failures.
- Avoid moving broad `except Exception` blocks unless the caller-visible response, telemetry attributes, audit classification, and tool-use event behavior remain equivalent.
- Extract error classification only after parity tests cover current behavior.

## Staged Rollout

Stage 1: Characterization and compatibility tests.

Stage 2: Move shared types into `protocol_types.py` and re-export them from `protocol.py`.

Stage 3: Introduce `ToolExecutionDependencies`, `ToolExecutionCoordinator`, `ToolExecutionReporter`, and the compatibility callback ledger. The coordinator may initially use explicit bound-method callables for behavior not yet moved, but it must not store or import the whole protocol object. Each temporary callback must be listed in the ledger with its removal stage and parity test.

Stage 4a: Extract validation, module/tool resolution, argument hardening, and write classification.

Stage 4b: Extract RBAC, scoped permissions, and API-key scope checks.

Stage 4c: Extract effective policy, external access, path scope, approval, governance preflight, and pre-hooks.

Stage 5: Extract runtime behavior: rate limits, idempotency, circuit-breaker execution, result formatting, execution eval metadata. At this point, `IdempotencyManager` ownership must be resolved by either dependency-object injection or moving the class to `tool_execution/idempotency.py` with a `protocol.py` compatibility re-export.

Stage 6: Extract reporting internals behind the Stage 3 `ToolExecutionReporter` facade: tool-use events, audit helpers, metrics helpers, and reporting classifications.

Stage 7: Clean the protocol facade by removing only dead tool-path helpers. Do not refactor non-tool handlers in this stage.

Each stage should run focused MCP protocol/tool tests before proceeding. The final stage should run Bandit on the touched scope.

## Success Criteria

- `protocol.py` keeps the public API stable while delegating tool execution.
- The tool security gate order is readable from coordinator code and covered by tests.
- Extracted modules depend on explicit services/callables, not `MCPProtocol`.
- Existing nested execution and virtual `run` behavior continue to work.
- Existing JSON-RPC and HTTP error behavior remains stable.
- Tool-use reporting, audit, metrics, hooks, idempotency, and cancellation behavior remain equivalent.
- No resource/prompt/module handler refactor is included in this implementation plan.
