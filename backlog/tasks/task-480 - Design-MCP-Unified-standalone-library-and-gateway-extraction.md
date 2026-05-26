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
modified_files:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- backlog/tasks/task-480 - Design-MCP-Unified-standalone-library-and-gateway-extraction.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and revised for a strangler extraction of MCP Unified into an embeddable runtime library first, then a standalone gateway with governance profiles/modes. The spec preserves strict tldw_server compatibility while defining a clean package boundary, optional SQLite stores, external MCP lifecycle management, fail-closed policy semantics, conservative presets, audit/provenance, protocol compatibility, dependency/license gates, instance-scoped runtime, module ownership inventory, and package-boundary tests.
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
