---
id: TASK-583
title: Wire MCP stdio transport factory into gateway bootstrap
status: Done
labels:
- mcp-unified
- gateway
- external-servers
- stdio
- bootstrap
documentation:
- Docs/superpowers/specs/2026-06-01-mcp-stdio-bootstrap-factory-design.md
- Docs/superpowers/plans/2026-06-01-mcp-stdio-bootstrap-factory-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-01-mcp-stdio-bootstrap-factory-design.md
- Docs/superpowers/plans/2026-06-01-mcp-stdio-bootstrap-factory-plan.md
- mcp_unified/gateway/config.py
- mcp_unified/gateway/__init__.py
- mcp_unified/gateway/cli.py
- mcp_unified/gateway/external_runtime.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
references:
- https://github.com/rmusser01/tldw_server/pull/2205
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the package-owned upstream stdio external transport factory into standalone gateway bootstrap/config so configured stdio external servers can be managed by GatewayExternalRuntimeManager without callers manually injecting a transport factory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone gateway bootstrap can create a GatewayExternalRuntimeManager using the package stdio transport factory when external runtime support is configured.
- [x] #2 Configuration loading preserves safe defaults and does not enable unsupported transports silently.
- [x] #3 Package boundary remains clean with no tldw_Server_API imports from mcp_unified integration paths.
- [x] #4 Focused tests cover default bootstrap behavior, injected-factory override behavior, stdio factory wiring, and unsupported transport error behavior.
- [x] #5 Focused pytest, Ruff, Bandit on touched Python source, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-mcp-stdio-bootstrap-factory-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented opt-in external runtime bootstrap wiring for standalone gateway config. Added GatewayExternalRuntimeBootstrapConfig with safe disabled default and stdio-only factory selector validation. bootstrap_profile_gateway_from_config now builds GatewayExternalRuntimeManager from resolved external registry storage only when external_runtime.enabled is true, using the package create_external_transport factory unless callers inject a factory or full runtime manager. Runtime start now wraps transport factory creation errors in the existing external_server_start_failed path so unsupported persisted transports fail at start time instead of escaping raw factory exceptions.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired standalone gateway config bootstrap to create a GatewayExternalRuntimeManager with the package stdio transport factory when explicitly enabled, while preserving disabled defaults and injection overrides. Added regression coverage for default-disabled behavior, stdio factory wiring, injected factory behavior, unsupported factory config rejection, unsupported server transport start failure, CLI safe-default reporting, and package-boundary exports. Verification: focused pytest passed (221 tests), Ruff passed on touched Python files, Bandit reported zero findings on touched gateway source, and git diff --check passed. PR: https://github.com/rmusser01/tldw_server/pull/2205. No known blockers.
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
