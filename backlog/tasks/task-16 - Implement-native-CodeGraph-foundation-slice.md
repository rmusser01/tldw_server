---
id: TASK-16
title: Implement native CodeGraph foundation slice
status: Done
assignee: []
created_date: '2026-05-03 21:16'
updated_date: '2026-05-03 22:01'
labels:
  - codegraph
  - mcp
  - planning
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and prepare the first reviewable native CodeGraph implementation slice for Unified MCP. Scope is the Stage 1 foundation only: optional dependency gate, core package skeleton, SQLite schema/repository foundation, trusted workspace/index path resolution, language registry with supported/planned status, bounded foreground index/sync skeleton, and initial Unified MCP module/status/tool surface. This task must not include deep Python or JS/TS extraction unless the approved plan explicitly expands scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written under Docs/superpowers/plans.
- [x] #2 Plan scopes the first slice to foundation/status/bounded foreground skeleton and defers deep extractors.
- [x] #3 Plan includes TDD steps, exact files, focused verification commands, Bandit guidance, and commit checkpoints.
- [x] #4 Backlog task is updated with the plan path and review/verification notes before implementation starts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan document: Docs/superpowers/plans/2026-05-03-native-codegraph-foundation-implementation-plan.md

Approved design input: Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md

Scope for this task: Stage 1 foundation only. Build dependency health, trusted workspace resolution, tldw-managed SQLite index storage, language registry, bounded foreground file-inventory index/sync, and initial MCP tools: codegraph.status, codegraph.index, codegraph.sync, and codegraph.files.

Explicitly deferred: symbol extraction, Python/JS/TS parser depth, JS/TS path-alias resolution implementation, graph query tools, context/impact tools, Jobs integration, file watching, and real C/C++/C#/Java/Kotlin extractors.

Execution rule: no production CodeGraph code starts until this plan is explicitly approved. Implementation must follow TDD task-by-task and keep TASK-16 notes current.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-03-native-codegraph-foundation-implementation-plan.md. Plan keeps Stage 1 narrow to foundation/status/file-inventory indexing and prevents overclaiming graph tools before extractors land.

Plan includes TDD red/green steps, exact files, focused pytest commands, Bandit command, git diff --check, dependency-matrix gate before pyproject optional extra, and commit checkpoints.

Subagent plan review was not run because the active tool policy allows subagents only when the user explicitly asks for subagent/parallel agent work; plan includes a local checklist review note instead.

Addressed plan review findings before implementation: status is now specified as inspect-only/read-only when no DB exists; indexer tests now cover foreground byte and wall-clock bounds; MCP module implementation must offload blocking index/sync/files work through asyncio.to_thread; planned-language files must be skipped instead of persisted; repository tests must seed and clean future graph rows to verify stale-edge cleanup behavior.

Implemented Stage 1 foundation in the codegraph worktree: CodeGraph settings/dependency probe/language registry/workspace resolver/schema/repository/indexer plus Unified MCP CodeGraphModule with status/index/sync/files.

Verified parser dependency matrix in /private/tmp/codegraph-matrix-venv: tree-sitter 0.25.2, tree-sitter-python 0.25.0, tree-sitter-javascript 0.25.0, and tree-sitter-typescript 0.23.2 parsed Python, JavaScript, TypeScript, and TSX smoke snippets.

Verification: pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q => 24 passed; pytest test_filesystem_module.py test_dynamic_module_catalog.py -q => 16 passed; Bandit touched CodeGraph scope => 0 results; git diff --check => clean.

PR opened against dev: https://github.com/rmusser01/tldw_server/pull/1244

2026-05-04 PR #1244 second review pass: verified GitHub reviewThreads had 0 unresolved items. Addressed Qodo's remaining literal-prefix and DB placement comments by moving the SQL-backed repository/schema under `tldw_Server_API/app/core/DB_Management/codegraph`, leaving a compatibility export at `app/core/CodeGraph/repository.py`, and adding a regression test proving `%` and `_` in `path_prefix` are treated literally.

Verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q -> 47 passed.

Verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/CodeGraph tldw_Server_API/app/core/DB_Management/codegraph tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py -f json -o /tmp/bandit_codegraph_1244_second_pass.json -> 0 results.

Verification: git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the native CodeGraph Stage 1 foundation: bounded foreground file-inventory indexing/sync, stable SQLite repository/schema, trusted workspace/index-path resolution, foundation/planned language registry, read-only status/files behavior, and a disabled Unified MCP CodeGraph module entry. Added focused tests for config, dependencies, registry, repository cleanup, bounded index limits, MCP metadata/offloading/protocol validation, and default disabled catalog registration. Deep symbol extraction, graph query tools, Jobs integration, file watching, and non-foundation language extractors remain intentionally deferred.
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
