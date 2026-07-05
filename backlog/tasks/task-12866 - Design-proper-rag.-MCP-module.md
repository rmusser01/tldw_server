---
id: TASK-12866
title: Design proper rag.* MCP module
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-03 15:39
labels:
- mcp
- rag
- design
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md
priority: high
modified_files:
- Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md
- backlog/tasks/task-12118 - Design-proper-rag.-MCP-module.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a proper rag.* MCP module that exposes existing RAG functionality through curated MCP tools without adding a research.* facade. Scope for this task is the brainstorming/design spec and review loop before implementation planning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design defines a curated Stage 1 rag.* MCP surface for capabilities, source_health, search, and answer without adding a research.* facade.
- [x] #2 Design requires HTTP and MCP RAG paths to share service-level request resolution, response mapping, quota, source checks, and usage accounting rather than MCP calling FastAPI routes or bypassing controls.
- [x] #3 Design documents per-source MCP authorization/module enablement checks, fail-closed behavior for explicit denied/unavailable sources, and warning/filter behavior for implicit default sources.
- [x] #4 Design explicitly defers unsupported Stage 1 behavior, including batch, streaming, feedback, ingestion, note-writing/export workflows, SQL source support, advanced arguments, and external/search-agent web fallback behavior.
- [x] #5 Design records result, error, testing, rollout, and acceptance contracts sufficient for the implementation handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Code review requested via superpowers:requesting-code-review for range 5ec7168b10ad7ec4556bede67dec687505fb6721..c030227ef4d82115061f4bc587042b7485923976. Review found blockers around source authorization and unsupported scoped retrieval, plus important fixes for search-agent default suppression, /tools/execute wrapper compatibility, task metadata, current enabled module inventory, and SQL source-health semantics. Updated the design spec to add per-source authorization/module enablement requirements, fail-closed unsupported item-scope behavior, forced-off external/search-agent defaults, SQL Stage 1 deferral, current template module inventory, /tools/execute compatibility wording, and expanded testing/acceptance coverage. This is documentation-only work; Bandit is not applicable and is recorded as a non-code skip.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the RAG MCP module design and review loop. The spec now defines a narrow rag.capabilities/source_health/search/answer Stage 1 surface, preserves service-level RAG control sharing, avoids a research.* facade, and incorporates review fixes for source authorization, scoped retrieval, external defaults, SQL deferral, wrapper compatibility, and verification expectations.
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
