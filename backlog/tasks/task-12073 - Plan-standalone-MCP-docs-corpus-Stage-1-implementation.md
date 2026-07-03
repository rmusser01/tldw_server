---
id: TASK-12073
title: Plan standalone MCP docs corpus Stage 1 implementation
status: Done
labels:
- mcp
- docs
- planning
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Stage 1 implementation plan for the standalone MCP document corpus: local imports, SQLite/FTS5 store, retrieval/context tools, collection/keyword metadata, Context7-compatible read aliases, scope enforcement, and boundary tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 1 implementation plan covers local imports, SQLite/FTS5 storage, scoped retrieval/context, collection/keyword metadata, Context7-compatible aliases, package-boundary tests, and MCP config registration.
- [x] #2 Plan explicitly defers URL acquisition, web-scraping extras, embeddings, Playwright, crawler sync, and Media/RAG bridging to later stages.
- [x] #3 Plan path, verification notes, and documentation links are recorded in this Backlog task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Created the Stage 1 implementation plan for the standalone-first MCP docs corpus. The plan creates a top-level `mcp_unified.docs` package for runtime-neutral code, keeps `tldw_server` integration in a thin `DocsModule` shim, and limits this slice to local import, SQLite/FTS5, scoped retrieval/context, collection/keyword metadata, Context7-compatible read aliases, package-boundary tests, and MCP config registration. URL acquisition, web-scraping extras, embeddings, Playwright, crawler sync, and Media/RAG bridging are deferred to later plans.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 1 implementation plan written at `Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md`. Verification: plan marker scan returned no matches for unfinished markers; `git diff --check` passed for the plan and Backlog task files. Bandit skipped because this task only changes documentation and Backlog.md task metadata.
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
