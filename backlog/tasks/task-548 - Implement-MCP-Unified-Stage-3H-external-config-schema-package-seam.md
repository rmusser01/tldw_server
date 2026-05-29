---
id: TASK-548
title: Implement MCP Unified Stage 3H external config schema package seam
status: In Progress
labels:
- mcp
- mcp-unified
- standalone
- stage3
modified_files:
- Docs/superpowers/plans/2026-05-29-mcp-unified-stage3h-external-config-package-plan.md
- mcp_unified/federation/config_schema.py
- mcp_unified/federation/__init__.py
- tldw_Server_API/app/core/MCP_unified/external_servers/config_schema.py
- tldw_Server_API/app/core/MCP_unified/external_servers/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- tldw_Server_API/app/core/MCP_unified/tests/test_external_server_config_schema.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move external MCP server registry config schema parsing/loading into the standalone mcp_unified package while preserving the existing host external_servers.config_schema import path and legacy default config behavior. Keep the slice scoped to schema/loading only; do not move spawning transport adapters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone `mcp_unified.federation` exposes external server config schema classes and parse/load functions.
- [x] #2 Existing `tldw_Server_API.app.core.MCP_unified.external_servers.config_schema` imports remain compatible and preserve the legacy default config path.
- [x] #3 Regression tests cover package ownership, host-wrapper compatibility, and existing external config schema behavior.
- [x] #4 Focused pytest, Ruff, Bandit, and diff whitespace verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `mcp_unified.federation.config_schema` as the standalone owner for external MCP server registry schema models, parsing, and config loading.
- Kept the host `external_servers.config_schema` path as a thin wrapper over the package module while preserving the historical default config path `tldw_Server_API/Config_Files/mcp_external_servers.yaml`.
- Added regression coverage for the package export surface, host-wrapper identity, host default path delegation, and extraction-boundary import ownership.
- Verification:
  - RED focused seam tests: 4 expected failures before implementation.
  - Focused seam tests: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_external_server_config_schema_delegates_to_standalone_package_schema tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_host_external_config_schema_shim_reexports_package_contracts tldw_Server_API/app/core/MCP_unified/tests/test_external_server_config_schema.py -q` -> 8 passed.
  - MCP external/federation suite: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_config_schema.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/app/core/MCP_unified/tests/test_external_websocket_adapter.py tldw_Server_API/app/core/MCP_unified/tests/test_external_stdio_adapter.py tldw_Server_API/app/core/MCP_unified/tests/test_external_credential_broker_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_external_federation_integration.py -q` -> 76 passed, 2 skipped.
  - Ruff touched scope: passed.
  - Bandit touched implementation scope: 0 findings in `/tmp/bandit_mcp_stage3h_external_config.json`.
  - `git diff --check`: passed.
- Known non-slice baseline: adding `tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py` to the broad command fails because current `origin/dev` lacks `protocol.get_telemetry_manager`; this slice does not touch protocol telemetry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Moved external MCP server registry config schema parsing/loading into the standalone `mcp_unified.federation.config_schema` package module. The host config module remains a compatibility wrapper and preserves the legacy default config path while standalone consumers use a package-neutral default.
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
