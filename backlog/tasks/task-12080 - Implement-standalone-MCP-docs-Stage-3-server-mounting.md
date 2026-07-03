---
id: TASK-12080
title: Implement standalone MCP docs Stage 3 server mounting
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-01 03:24'
labels:
  - mcp
  - docs
  - implementation
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
  - >-
    Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Stage 3 standalone MCP docs server mounting slice from the approved plan: add a runtime-neutral standalone docs mount/factory, explicit profile defaults, tldw_server docs host adapter boundary, built-in server registration guard, and boundary/packaging regression tests. Keep crawler/sync, embeddings/reranking, browser extraction, Media/RAG bridges, and new required dependencies out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Standalone docs mount/factory enables docs with local SQLite state by default and exposes import/search/context/status behavior.
- [x] #2 Config profiles are explicit and downgradeable: locked_down hides URL ingest while local_first/online_capable are policy-bound web-capable profiles.
- [x] #3 tldw_server DocsModule delegates settings/scope translation through a host adapter outside mcp_unified.docs.
- [x] #4 Built-in MCP server registration is guarded by tests proving docs mounts without Media/RAG dependencies and disabled web acquisition hides docs.ingest_url.
- [x] #5 Boundary/package tests prove mcp_unified.docs has no tldw_Server_API dependency and no eager optional web dependency import.
- [x] #6 Focused docs/MCP tests, import smoke, Black check, Bandit, and git diff checks are run or skips are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed from `Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md` in isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/codex-mcp-docs-stage1` on branch `codex/mcp-docs-stage1`.

TDD evidence:
- Task 1 red: focused pytest failed during collection with `ModuleNotFoundError: No module named mcp_unified.docs.standalone`; green: `test_docs_standalone_mount.py` passed 4 tests.
- Task 2 red: focused pytest failed during collection with `ModuleNotFoundError: No module named tldw_Server_API.app.core.MCP_unified.adapters`; green: host adapter and shim tests passed 8 tests.
- Task 3 registration guard passed immediately as an existing supported path; full dynamic module catalog file passed 7 tests.
- Task 4 boundary tests passed 6 tests; package discovery reported `missing= []`.

Verification:
- `python -m pytest tldw_Server_API/tests/MCP_unified/docs -q --tb=short` -> 153 passed, 6 warnings.
- `python -m pytest tldw_Server_API/tests/MCP_unified tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -k "docs or write_tools or validator or dynamic_module_catalog" -q --tb=short` -> 184 passed, 419 deselected, 66 warnings.
- Standalone import smoke -> `True False loaded_optional= []`.
- Black check on touched Python scope -> 19 files would be left unchanged.
- Bandit on touched production scope -> errors: [], results: [].
- `git diff --check` passed.

Known skips/blockers: none. Warnings are existing test-suite warnings/logging behavior and did not fail the focused verification.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 3 standalone MCP docs server mounting. Added `mcp_unified.docs.standalone` with a local SQLite-backed docs mount factory and explicit locked_down/local_first/online_capable profile defaults, exported the public standalone mount API, introduced a tldw_server host adapter package for docs settings and context-scope translation, updated `DocsModule` to delegate through that adapter, and added registration/boundary/package regression tests. No Media/RAG bridge, crawler/sync, embeddings, browser extraction, or new required dependency was added.
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
