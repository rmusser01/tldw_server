# MCP Unified Stage 3H External Config Schema Package Seam Plan

**Goal:** Move external MCP server registry config schema parsing/loading into the standalone `mcp_unified` package while preserving host imports and host default config behavior.

**Backlog:** TASK-548

## Stage 1: RED Boundary And Compatibility Tests
**Goal:** Capture the desired external-server config schema package seam before implementation.
**Success Criteria:** Tests fail because `mcp_unified.federation.config_schema` does not exist and the host module still owns schema/loading implementation.
**Tests:** Focused pytest for extraction contracts, runtime package boundary, and external server config schema.
**Status:** Complete

- [x] Add a boundary contract requiring `external_servers/config_schema.py` to delegate to `mcp_unified.federation.config_schema`.
- [x] Add standalone export/identity coverage for external server config schema classes and functions.
- [x] Add host-default compatibility coverage for the legacy external server config path.
- [x] Run focused tests and confirm the expected RED failures.

## Stage 2: Move Config Schema Ownership
**Goal:** Put external server schema parsing/loading in the standalone package and leave the host path as a compatibility wrapper.
**Success Criteria:** Existing host imports keep working while standalone consumers can parse/load external server registry configs.
**Tests:** RED tests turn green; existing external config schema tests remain green.
**Status:** Complete

- [x] Add `mcp_unified.federation.config_schema`.
- [x] Re-export schema classes and loader functions from `mcp_unified.federation`.
- [x] Replace host `external_servers/config_schema.py` with a compatibility wrapper that preserves the legacy default config path.
- [x] Re-run focused tests.

## Stage 3: Validation And PR Closeout
**Goal:** Verify the slice and record the result.
**Success Criteria:** Focused pytest, Ruff, Bandit, and diff whitespace checks pass or have documented non-slice blockers.
**Tests:** Focused pytest, Ruff, Bandit, `git diff --check`.
**Status:** Complete

- [x] Run focused pytest for extraction contracts, runtime package boundary, external server config schema, and affected external server manager/transport tests.
- [x] Run Ruff and Bandit on touched code.
- [x] Update TASK-548 with verification and summary; add the PR link after the PR is opened.

## Verification

- RED focused seam tests: 4 expected failures before moving the config schema into `mcp_unified.federation.config_schema`.
- Focused seam tests: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_external_server_config_schema_delegates_to_standalone_package_schema tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_host_external_config_schema_shim_reexports_package_contracts tldw_Server_API/app/core/MCP_unified/tests/test_external_server_config_schema.py -q` -> 8 passed.
- MCP external/federation suite: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_config_schema.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/app/core/MCP_unified/tests/test_external_websocket_adapter.py tldw_Server_API/app/core/MCP_unified/tests/test_external_stdio_adapter.py tldw_Server_API/app/core/MCP_unified/tests/test_external_credential_broker_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_external_federation_integration.py -q` -> 76 passed, 2 skipped.
- Ruff touched scope: `.venv/bin/python -m ruff check ...` -> All checks passed.
- Bandit touched implementation scope: `.venv/bin/python -m bandit -r ... -f json -o /tmp/bandit_mcp_stage3h_external_config.json` -> 0 findings.
- Whitespace: `git diff --check` -> passed.
- Non-slice baseline: adding `tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py` to the broad command fails because `protocol.get_telemetry_manager` is absent on current `origin/dev`; this slice does not touch protocol telemetry.
