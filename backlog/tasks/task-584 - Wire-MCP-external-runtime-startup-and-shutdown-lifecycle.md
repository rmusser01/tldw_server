---
id: TASK-584
title: Wire MCP external runtime startup and shutdown lifecycle
status: Done
labels:
- mcp-unified
- gateway
- external-servers
- runtime
- lifecycle
documentation:
- Docs/superpowers/specs/2026-06-01-mcp-external-runtime-lifecycle-design.md
- Docs/superpowers/plans/2026-06-01-mcp-external-runtime-lifecycle-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-01-mcp-external-runtime-lifecycle-design.md
- Docs/superpowers/plans/2026-06-01-mcp-external-runtime-lifecycle-plan.md
- mcp_unified/gateway/lifecycle.py
- mcp_unified/gateway/bootstrap.py
- mcp_unified/gateway/config.py
- mcp_unified/gateway/fastapi.py
- mcp_unified/gateway/external_runtime.py
- mcp_unified/gateway/cli.py
- mcp_unified/gateway/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
references:
- https://github.com/rmusser01/tldw_server/pull/2206
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next standalone gateway runtime integration slice: opt-in app lifecycle handling for configured external MCP servers so startup can reconcile auto_start-enabled servers and shutdown stops active external transports cleanly without changing safe defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway app/config exposes an explicit safe-default lifecycle option that does not start external servers unless enabled.
- [x] #2 When lifecycle startup is enabled, configured enabled auto_start external servers are reconciled through GatewayExternalRuntimeManager without blocking app startup on individual server failures.
- [x] #3 Gateway app shutdown stops active external transports cleanly and reports/records best-effort errors without leaving runtime state inconsistent.
- [x] #4 Focused tests cover default no-autostart behavior, opt-in startup reconcile behavior, startup failure handling, shutdown stop behavior, and FastAPI lifespan integration.
- [x] #5 Focused pytest, Ruff, Bandit on touched Python source, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-mcp-external-runtime-lifecycle-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a safe-default external runtime lifecycle config and FastAPI lifespan integration. GatewayExternalRuntimeLifecycleConfig defaults to no startup or shutdown action. GatewayExternalRuntimeBootstrapConfig now carries reconcile_on_startup and stop_on_shutdown through GatewayProfileBootstrap so config-driven apps can opt in without ad hoc wiring. create_gateway_app resolves explicit lifecycle config before bootstrap-carried config, records startup/shutdown payloads on app.state, and avoids raw exception messages in lifecycle error payloads. GatewayExternalRuntimeManager.stop_all() snapshots active transports and stops them with deterministic counts/errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added opt-in startup and shutdown lifecycle handling for standalone gateway external runtimes and addressed PR review findings. Startup reconciliation delegates to GatewayExternalRuntimeManager.reconcile(), shutdown cleanup delegates to stop_all(), and existing defaults remain inert unless lifecycle flags are set. Review fixes: stop_all no longer requires registry store lookups for already-active transports, unexpected stop failures are recorded per server without aborting cleanup of other transports, unexpected stop failures are logged with traceback diagnostics, lifecycle exception catches log contextual reason/error type, and config validation rejects lifecycle flags unless external_runtime.enabled is true. Verification: focused pytest passed (233 tests), Ruff passed on touched Python files, Bandit reported zero findings on touched gateway source, and git diff --check passed. No known blockers.
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
