---
id: TASK-12120
title: Implement rag.* MCP module
status: In Progress
labels:
- mcp
- rag
references:
- Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md
- Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
modified_files:
- backlog/tasks/task-12120 - Implement-rag.-MCP-module.md
- Docs/superpowers/plans/2026-07-03-rag-mcp-module-implementation-plan.md
- tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py
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
