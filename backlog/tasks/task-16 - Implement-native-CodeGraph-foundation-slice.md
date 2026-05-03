---
id: TASK-16
title: Implement native CodeGraph foundation slice
status: In Progress
assignee: []
created_date: '2026-05-03 21:16'
updated_date: '2026-05-03 21:28'
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
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
