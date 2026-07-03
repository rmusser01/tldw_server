---
id: TASK-12120
title: Implement rag.* MCP module
status: Done
labels:
- mcp
- rag
references:
- Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md
- Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
modified_files:
- backlog/tasks/task-12120 - Implement-rag.-MCP-module.md
- Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
- Docs/MCP/mcp_tool_catalogs.md
- Docs/MCP/Unified/User_Guide.md
- Docs/MCP/Unified/Client_Snippets.md
- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
- tldw_Server_API/app/core/RAG/rag_service/transport.py
- tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py
- tldw_Server_API/Config_Files/mcp_modules.yaml
- tldw_Server_API/Config_Files/mcp_tool_categories.yaml
- tldw_Server_API/Config_Files/resource_governor_policies.yaml
- tldw_Server_API/app/core/MCP_unified/module_surface.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py
- tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py
- tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py
- tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py
- tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved first-slice rag.* MCP module with capabilities, source_health, search, and answer tools. Keep the implementation aligned to the existing RAG pipeline and MCP security/governance controls; do not add a research.* facade or external-provider research loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Expose rag.capabilities, rag.source_health, rag.search, and rag.answer through MCP with strict schemas and no research.* layer.
- [ ] #2 Preserve existing RAG behavior through shared transport helpers while adding MCP-specific normalization, metadata, truncation, and citation coverage.
- [ ] #3 Enforce MCP module/tool/category/security/quota controls and per-source authorization posture described in the approved plan.
- [ ] #4 Register rag module, mcp.search, and rag_generation governance configuration with fail-closed behavior for missing generation policy.
- [ ] #5 Cover behavior with targeted unit/integration tests, HTTP tools/execute compatibility, JSON-RPC wrapper compatibility, knowledge.search regression, and Bandit on touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a proper rag.* MCP module without a research.* layer. Verification: targeted MCP/RAG suite passed (80 tests), and Bandit completed with zero findings for rag_module.py, rag_service/transport.py, and rag_unified.py.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
