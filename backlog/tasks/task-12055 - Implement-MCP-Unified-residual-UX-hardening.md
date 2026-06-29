---
id: TASK-12055
title: Implement MCP Unified residual UX hardening
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-29 04:38
labels:
- mcp
- ux
- security
- docs
dependencies: []
references:
- TASK-12054
- TASK-2372
- Docs/superpowers/plans/2026-06-28-mcp-unified-residual-ux-hardening-implementation-plan.md
- Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md
documentation:
- Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md
- Docs/superpowers/plans/2026-06-28-mcp-unified-residual-ux-hardening-implementation-plan.md
modified_files:
- Docs/MCP/Unified/Client_Snippets.md
- Docs/MCP/Unified/Developer_Guide.md
- Docs/MCP/Unified/Modules.md
- Docs/MCP/Unified/README.md
- Docs/MCP/Unified/Smoke_Client.md
- Docs/MCP/Unified/System_Admin_Guide.md
- Docs/MCP/Unified/User_Guide.md
- Docs/MCP/Unified/Using_Modules_YAML.md
- apps/mcp-unified/README.md
- apps/mcp-unified/USER_GUIDE.md
- apps/mcp-unified/src/mcp_unified/README.md
- apps/mcp-unified/src/mcp_unified/gateway/fastapi.py
- tldw_Server_API/Config_Files/mcp_modules.local_opt_in.example.yaml
- tldw_Server_API/Config_Files/mcp_modules.yaml
- tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py
- tldw_Server_API/app/core/MCP_unified/docker/Dockerfile
- tldw_Server_API/app/core/MCP_unified/module_surface.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
- tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
- tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved residual MCP Unified UX hardening work from TASK-2372 and TASK-12054. Scope covers safer high-risk module defaults, richer embedded and package-local status readiness, HTTP/JSON-RPC recovery metadata, docs/quickstart truthfulness, and Docker/package wording without pretending the standalone gateway is shipped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-28-mcp-unified-residual-ux-hardening-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved residual MCP Unified UX hardening work.

Changed files:
- Docs/MCP/Unified/Client_Snippets.md
- Docs/MCP/Unified/Developer_Guide.md
- Docs/MCP/Unified/Modules.md
- Docs/MCP/Unified/README.md
- Docs/MCP/Unified/Smoke_Client.md
- Docs/MCP/Unified/System_Admin_Guide.md
- Docs/MCP/Unified/User_Guide.md
- Docs/MCP/Unified/Using_Modules_YAML.md
- apps/mcp-unified/README.md
- apps/mcp-unified/USER_GUIDE.md
- apps/mcp-unified/src/mcp_unified/README.md
- apps/mcp-unified/src/mcp_unified/gateway/fastapi.py
- tldw_Server_API/Config_Files/mcp_modules.local_opt_in.example.yaml
- tldw_Server_API/Config_Files/mcp_modules.yaml
- tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py
- tldw_Server_API/app/core/MCP_unified/docker/Dockerfile
- tldw_Server_API/app/core/MCP_unified/module_surface.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
- tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
- tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py

Summary:
- Made high-risk MCP modules fail closed by default: filesystem, run_command, and codegraph remain available only through explicit opt-in configuration; status now reports disabled-but-available high-risk modules with next actions.
- Expanded package-local /mcp/status to include non-secret readiness metadata for package publication status, profile store/default profile, admin auth configuration, external registry store, and external server counts.
- Added recovery metadata for HTTP and JSON-RPC errors without changing legacy string details where clients may depend on them.
- Corrected MCP docs, smoke-client examples, package docs, and Docker wording so they distinguish embedded /api/v1/mcp from host-mounted /mcp and do not imply a shipped standalone gateway process.

Verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -> 63 passed, 5 warnings
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -k "gateway_status or basic_jsonrpc_flow" -> 5 passed, 197 deselected, 4 warnings
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py -> 4 passed, 4 warnings
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit tldw_Server_API/app/core/MCP_unified/module_surface.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py apps/mcp-unified/src/mcp_unified/gateway/fastapi.py -f json -o /tmp/bandit_mcp_residual_ux.json -> 0 findings
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
