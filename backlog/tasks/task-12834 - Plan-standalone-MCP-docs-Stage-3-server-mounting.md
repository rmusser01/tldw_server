---
id: TASK-12834
title: Plan standalone MCP docs Stage 3 server mounting
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-01 03:14'
labels:
  - mcp
  - docs
  - planning
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
  - >-
    Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Stage 3 implementation plan for standalone MCP docs server mounting: ensure the standalone MCP server enables the docs module with local SQLite state by default, keep the built-in tldw_server MCP shim thin, define host adapter boundaries, and avoid Media/RAG bridge coupling unless explicitly planned later.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan scopes Stage 3 to standalone server mounting and tldw_server shim/adapter boundaries, not crawler/sync/embedding work.
- [x] #2 Plan identifies current baseline gaps before prescribing code changes.
- [x] #3 Plan preserves the standalone package boundary and optional web acquisition dependency model.
- [x] #4 Plan includes TDD tasks with exact files, commands, expected red/green outcomes, and focused verification including Bandit where applicable.
- [x] #5 Plan is saved under Docs/superpowers/plans and linked from this Backlog task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started baseline investigation for Stage 3 server mounting plan. Initial finding: Stage 1/2 added the runtime-neutral `mcp_unified.docs` package plus the built-in `tldw_server` `DocsModule` shim and default config, but no separate top-level standalone MCP server mounting layer is visible under `mcp_unified` yet.

Wrote the Stage 3 implementation plan at `Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md`. The plan scopes Stage 3 to a runtime-neutral standalone docs mount/factory, explicit locked_down/local_first/online_capable profile defaults, a small tldw_server docs host adapter boundary, and a built-in server registration guard. It explicitly defers crawler/sync, embeddings/reranking, browser extraction, and Media/RAG bridges.

Review note: the writing-plans workflow recommends a plan-document-reviewer subagent, but the available multi-agent tool rules prohibit spawning unless the user explicitly asks for subagents. I performed a local review instead. Issues found and fixed during review: local_first should be web-capable but policy-bound while locked_down is the downgrade that hides URL ingestion; and the host adapter package needs a parent `adapters/__init__.py` so imports and packaging discovery are explicit.

Verification for this planning slice: placeholder scan found no unfinished markers in the plan; `git diff --check` passed. Bandit is not applicable because this task only changes documentation and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 server mounting implementation plan completed and saved under `Docs/superpowers/plans`. It documents baseline gaps, defines the standalone docs mount/factory and profile behavior, keeps the tldw_server integration behind a host adapter shim, preserves optional web acquisition and package boundaries, and provides TDD-oriented tasks with exact files, commands, red/green expectations, verification, and Bandit instructions.
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
