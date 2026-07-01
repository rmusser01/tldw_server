---
id: TASK-12080
title: Implement standalone MCP docs Stage 3 server mounting
status: In Progress
labels:
- mcp
- docs
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Stage 3 standalone MCP docs server mounting slice from the approved plan: add a runtime-neutral standalone docs mount/factory, explicit profile defaults, tldw_server docs host adapter boundary, built-in server registration guard, and boundary/packaging regression tests. Keep crawler/sync, embeddings/reranking, browser extraction, Media/RAG bridges, and new required dependencies out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Standalone docs mount/factory enables docs with local SQLite state by default and exposes import/search/context/status behavior.
- [ ] #2 Config profiles are explicit and downgradeable: locked_down hides URL ingest while local_first/online_capable are policy-bound web-capable profiles.
- [ ] #3 tldw_server DocsModule delegates settings/scope translation through a host adapter outside mcp_unified.docs.
- [ ] #4 Built-in MCP server registration is guarded by tests proving docs mounts without Media/RAG dependencies and disabled web acquisition hides docs.ingest_url.
- [ ] #5 Boundary/package tests prove mcp_unified.docs has no tldw_Server_API dependency and no eager optional web dependency import.
- [ ] #6 Focused docs/MCP tests, import smoke, Black check, Bandit, and git diff checks are run or skips are documented.
<!-- AC:END -->

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
