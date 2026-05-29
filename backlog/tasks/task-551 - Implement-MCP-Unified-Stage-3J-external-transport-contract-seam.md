---
id: TASK-551
title: Implement MCP Unified Stage 3J external transport contract seam
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 20:45'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage3
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the package-neutral external transport data contracts into the standalone `mcp_unified` package surface while preserving host adapter compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone `mcp_unified.federation` exports `BrokeredExternalCredential` with caller-owned copy behavior.
- [x] #2 Host `external_servers.transports.base` reuses package `ExternalToolDefinition`, `ExternalToolCallResult`, and `BrokeredExternalCredential` contracts instead of duplicating dataclasses.
- [x] #3 Existing host transport imports remain compatible and adapter tests still pass.
- [x] #4 Focused pytest, Ruff, Bandit, and diff whitespace verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-29-mcp-unified-stage3j-external-transport-contracts-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 3J external transport contract seam. Added standalone BrokeredExternalCredential with caller-owned copy behavior, made the host transport base re-export package ExternalToolDefinition, ExternalToolCallResult, and BrokeredExternalCredential, and added package-boundary regression tests.

Verification:
- RED: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -k external_transport -q` failed on duplicate host ExternalToolDefinition identity.
- RED: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -k brokered_external_credential -q` failed because package BrokeredExternalCredential was missing.
- GREEN: `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/app/core/MCP_unified/tests/test_external_stdio_adapter.py tldw_Server_API/app/core/MCP_unified/tests/test_external_websocket_adapter.py tldw_Server_API/app/core/MCP_unified/tests/test_external_credential_broker_runtime.py -q` passed 48 tests.
- Ruff passed on touched Python files.
- Bandit passed on touched runtime Python files with zero findings in `/tmp/bandit_mcp_stage3j_external_transport_contracts.json`.
- `git diff --check` passed.
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
