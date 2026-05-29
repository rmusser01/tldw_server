---
id: TASK-546
title: Implement MCP Unified Stage 3F protocol catalog provider seam
status: Done
labels:
- mcp
- mcp-unified
- standalone
- stage3
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move protocol tool-catalog lookup behind a runtime dependency provider so MCPProtocol no longer imports the host AuthNZ database directly for catalog filtering. Preserve current catalog name/id resolution behavior and strict-mode fallback semantics with focused boundary and behavior coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCPProtocol catalog filtering no longer imports tldw_Server_API.app.core.AuthNZ.database directly.
- [x] #2 Runtime interfaces expose a host-neutral tool catalog provider contract and default tldw_server dependencies implement it.
- [x] #3 Existing catalog id/name and strict-mode behavior remains covered by focused tests.
- [x] #4 Verification records focused pytest, Ruff, Bandit, and diff whitespace checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED extraction and behavior tests for an injected catalog provider.
2. Extend runtime interfaces and test fakes with a tool catalog provider.
3. Implement the default tldw_server adapter and route MCPProtocol catalog filtering through it.
4. Run focused verification and update this Backlog record.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ToolCatalogProvider` to the standalone and compatibility runtime interfaces and to `MCPRuntimeDependencies`.
- Added `TldwToolCatalogProvider` in the host adapter bundle, preserving legacy catalog id coercion, team/org/global name lookup precedence, entry lookup, and strict-mode empty-set semantics.
- Routed `MCPProtocol._resolve_catalog_tool_names()` through the injected provider instead of importing `tldw_Server_API.app.core.AuthNZ.database`.
- Added a contract test forbidding the direct protocol AuthNZ database import and a behavior test proving protocol catalog resolution delegates to the injected provider with parsed strictness and request metadata.
- Review pass moved tool-catalog SQL out of `TldwToolCatalogProvider` and into `admin_tool_catalog_service.resolve_tool_catalog_filter_names()`.
- Review pass made strict catalog-provider failures fail closed, sanitized catalog lookup logs to exception class names, and added tuple/record row parsing coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3F moved tool-catalog filtering out of `MCPProtocol` and into an injected runtime provider seam. Verification:

- RED: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_protocol_catalog_lookup_uses_runtime_dependencies tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py::test_protocol_catalog_resolution_uses_injected_provider -q` -> 2 failed for the intended direct import and missing provider delegation.
- GREEN/focused pytest: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py -q` -> 41 passed, 3 warnings
- Ruff: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py` -> All checks passed
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/protocol.py -f json -o /tmp/bandit_mcp_stage3f_protocol_catalog_provider.json` -> 0 results, 0 errors
- `git diff --check` -> passed

Review-fix pass after rebase onto latest `origin/dev`:

- Focused pytest: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/tests/Services/test_admin_tool_catalog_service_backend_selection.py -q` -> 56 passed, 5 warnings
- Ruff: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/services/admin_tool_catalog_service.py tldw_Server_API/tests/Services/test_admin_tool_catalog_service_backend_selection.py` -> All checks passed
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/interfaces tldw_Server_API/app/core/MCP_unified/interfaces tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/services/admin_tool_catalog_service.py -f json -o /tmp/bandit_mcp_stage3f_protocol_catalog_provider_review.json` -> 0 results, 0 errors
- `git diff --check` -> passed

No known blockers or verification skips.
<!-- SECTION:FINAL_SUMMARY:END -->

## Modified Files
- `mcp_unified/interfaces/__init__.py`
- `mcp_unified/interfaces/runtime.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py`
- `tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py`
- `tldw_Server_API/app/core/MCP_unified/protocol.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py`
- `tldw_Server_API/app/services/admin_tool_catalog_service.py`
- `tldw_Server_API/tests/Services/test_admin_tool_catalog_service_backend_selection.py`
- `backlog/tasks/task-546 - Implement-MCP-Unified-Stage-3F-protocol-catalog-provider-seam.md`

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
