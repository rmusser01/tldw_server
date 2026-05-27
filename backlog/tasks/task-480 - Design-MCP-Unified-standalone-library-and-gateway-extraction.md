---
id: TASK-480
title: Design MCP Unified standalone library and gateway extraction
status: In Progress
labels:
- design
- mcp
- mcp-unified
documentation:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-26-mcp-unified-stage1-adapter-seams-implementation-plan.md
- Docs/MCP/mcp_unified_module_ownership_inventory.md
modified_files:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-26-mcp-unified-stage1-adapter-seams-implementation-plan.md
- Docs/MCP/mcp_unified_module_ownership_inventory.md
- tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py
- tldw_Server_API/app/core/MCP_unified/interfaces/runtime.py
- tldw_Server_API/app/core/MCP_unified/interfaces/policy.py
- tldw_Server_API/app/core/MCP_unified/interfaces/storage.py
- tldw_Server_API/app/core/MCP_unified/adapters/__init__.py
- tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py
- tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/modules/base.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py
- tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
- backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-26-mcp-unified-stage1-adapter-seams-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and revised for a strangler extraction of MCP Unified into an embeddable runtime library first, then a standalone gateway with governance profiles/modes. Stage 1 implementation plan added for adapter seams, module ownership inventory, boundary scaffolding tests, focused compatibility verification, and Bandit gates. Docs-only planning update; Bandit is not applicable until implementation touches Python code. The plan preserves strict tldw_server compatibility while preparing the future standalone package boundary.
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
