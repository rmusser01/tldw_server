---
id: TASK-12071
title: Design standalone MCP document corpus and Context7-compatible RAG tools
status: In Progress
labels:
- mcp
- docs
- design
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
modified_files:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- backlog/tasks/task-12071 - Design-standalone-MCP-document-corpus-and-Context7-compatible-RAG-tools.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for a standalone-first MCP document corpus that supports document/collection-first SQLite FTS5 retrieval, optional URL acquisition with approval policy, Context7-compatible aliases, and tldw_server mounting through host adapters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design captures standalone-first package boundary with no tldw_server dependency.
- [ ] #2 Design uses document/collection/keyword-first model rather than library/version constraints.
- [ ] #3 Design defines canonical docs.* MCP tools plus Context7-compatible aliases.
- [ ] #4 Design covers URL/source policy, approval flow, egress protections, testing, rollout, and tldw_server host integration.
- [ ] #5 Spec is written under Docs/superpowers/specs and committed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reviewed the draft before implementation planning and tightened the spec around:
- Stage 1 versus full-program acceptance criteria.
- Local `docs.import_path` belonging to Stage 1, while Stage 2 is optional URL acquisition only.
- Baseline standalone install not requiring the web-scraping pipeline.
- Existing tldw_server scraping code being a reference/copy-adapt source or host-adapter implementation, not a standalone runtime dependency.
- Store-level owner/profile scope enforcement.
- Current tldw_server module-loader shim requirements.
- Config-backed standalone source profiles and future policy storage.
- Explicit collection/keyword read and write tools.
- Context7 alias authorization through canonical docs.* operations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Design spec written and revised after review for the standalone-first document corpus/RAG MCP module. The revision separates Stage 1 from full-program acceptance, moves local `docs.import_path` into Stage 1, makes URL/web acquisition optional, requires the baseline standalone install to work without the web-scraping pipeline, treats existing tldw_server scraping code as a reference/copy-adapt source or host-adapter implementation rather than a mandatory runtime dependency, adds store-level scope enforcement, documents the current tldw_server shim path, clarifies source-policy storage, splits collection/keyword tools by read/write behavior, and maps Context7 aliases to canonical docs.* authorization. Verification: unfinished-marker scan returned no matches in the spec/task. Bandit skipped because this task only changes documentation and Backlog.md task metadata.
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
