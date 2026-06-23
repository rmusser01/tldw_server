---
id: TASK-9931
title: Harden Notes Graph review findings
status: Done
assignee: []
created_date: '2026-06-23 18:54'
updated_date: '2026-06-24 04:23'
labels:
  - notes-graph
  - security
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated review findings in tldw_Server_API/app/core/Notes_Graph and related endpoint/DB support. The compacted context mentioned TASK-2418, but that ID belongs to an unrelated Persona task in this checkout; this task records the Notes Graph work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Graph requests clamp effective max_nodes, max_edges, and max_degree before traversal.
- [x] #2 allow_heavy graph requests require elevated notes.graph.admin permission before larger caps are honored.
- [x] #3 Graph cursor pagination validates cursor payloads and advances within a large neighbor list.
- [x] #4 Tag and source filters use direct seed lookups instead of sampling recent notes.
- [x] #5 Graph node ordering and timezone-aware time filtering are deterministic.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: add failing regression coverage, harden service and endpoint limit handling, add direct tag/source seed DB lookups, run focused pytest and Bandit verification.

Implemented request cap clamping before traversal, gated elevated allow_heavy limits behind notes.graph.admin/admin/*, validated graph cursor payloads, preserved neighbor offsets across cursor pages, added direct tag/source graph seed queries, ordered note nodes deterministically, and normalized timezone-aware time filters to UTC before comparison.

Verification: py_compile on touched Python files passed. TEST_MODE=1 MINIMAL_TEST_APP=1 ULTRA_MINIMAL_APP=1 python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_graph_service.py tldw_Server_API/tests/Notes_Graph/unit/test_graph_db_queries.py -q => 62 passed. python -m pytest tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py::test_allow_heavy_requires_graph_admin_permission -q => 1 passed. Bandit on touched backend Python files => 0 findings.

Worktree PR prep verification on branch codex/notes-graph-review-hardening: py_compile passed, focused Notes Graph unit/DB tests passed (62 passed), endpoint allow_heavy permission regression passed (1 passed), Bandit JSON output had 0 results, and git diff --check passed.

PR review follow-up: rebased codex/notes-graph-review-hardening onto origin/dev, dropped the unrelated local-only Claims_Extraction design commit from the branch, addressed Qodo comments for _to_utc_naive typing, AuthNZ-compatible heavy graph admin detection, and deterministic tag/source edge ordering. Verification after follow-up: py_compile passed, focused Notes Graph unit/DB tests passed (63 passed), endpoint heavy-permission tests passed (8 passed), Bandit had 0 findings, and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Notes Graph against the validated review findings: bounded graph expansion, admin-only heavy graph caps, valid advancing cursors, complete tag/source seed lookup queries, deterministic ordering, and correct timezone instant comparisons. Added focused regression coverage and recorded passing pytest/Bandit verification.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused regression tests pass.
- [x] #8 Bandit runs clean on touched backend Python files.
<!-- DOD:END -->
