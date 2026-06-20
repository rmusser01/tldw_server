---
id: TASK-2393
title: Add MCP standalone user guide UAT harness
status: Done
labels:
- mcp
- uat
- docs
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a one-off harness that validates the package-local MCP Unified standalone user guide from a new-user perspective, including package-boundary install behavior, local CLI/config/profile/snapshot commands, smoke-client availability, and clear reporting of guide issues found during UAT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] One-off standalone MCP user-guide UAT harness exists outside the packaged `mcp_unified` module.
- [x] Harness creates an isolated workspace, installs the package-local boundary, runs documented gateway/smoke/profile/admin snapshot flows, exercises in-process/stdio/HTTP/WebSocket smoke transports with local fixtures, and writes a redacted JSON report.
- [x] Package metadata exposes the documented `mcp-unified-smoke` console script and includes smoke package files.
- [x] User guide, README, and smoke-client docs reflect the install/smoke prerequisites discovered during UAT.
- [x] Validation results and known non-required remote-runtime skip are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py` as a one-off harness. Added focused metadata/harness tests in `tldw_Server_API/tests/Helper_Scripts/test_mcp_standalone_user_guide_uat.py`. Fixed `mcp_unified` package metadata so `mcp-unified-smoke` is installed and the gateway extra includes `httpx`/`websockets` for the smoke client. Added local stdio and ASGI smoke fixtures to the harness so it can exercise stdio subprocess, live HTTP, and live WebSocket transports without requiring a deployed gateway. Updated standalone README, USER_GUIDE, and smoke-client documentation based on UAT findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Standalone MCP user-guide UAT completed from an isolated temp workspace. Final harness report: 33 passed, 0 failed, 1 skipped. The passed steps include package install, gateway CLI flows, profile/default assignment, policy explain, external server/grant setup, snapshot import/export, tool-use reporting, in-process smoke, stdio subprocess smoke, live HTTP smoke, and live WebSocket smoke. The only skip is the optional remote-runtime live URL path, which requires --gateway-url or MCP_UNIFIED_GATEWAY_URL. Validation passed: focused pytest, Ruff, Bandit with 0 findings, git diff --check, and full UAT harness run.
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
