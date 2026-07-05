---
id: TASK-12141
title: Address PR 2598 review feedback
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 17:57'
labels:
  - mcp
  - docs
  - review
dependencies: []
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/2598'
  - >-
    Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR 2598 on latest dev and address actionable review feedback for standalone MCP docs source discovery. Scope is limited to review items on sitemap/XML robustness, URL origin handling, source registration lifecycle, sync item reason reporting, small docstring/type/style comments, tests, Bandit, and PR push.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased on latest origin/dev and pushed.
- [x] #2 Actionable PR review issues are fixed in code/tests or explicitly documented as not applicable.
- [x] #3 Focused tests cover XML parse failure propagation, default-port origin equivalence, query sitemap registration denial, and sync skipped reason propagation.
- [x] #4 MCP docs tests, import-boundary tests, Bandit, and diff hygiene pass.
- [x] #5 Backlog task records final verification and PR status.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use the existing PR branch. Rebase on origin/dev, inspect review threads, implement the smallest root-cause fixes with focused tests, run MCP docs tests plus Bandit, update this Backlog task, and force-push the rebased PR branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased PR branch on origin/dev, inspected GitHub review threads, and fixed the actionable review items in the shared docs discovery/sync paths. Implemented controlled defusedxml exception handling, propagated sitemap parse failures from docs.discover_source, normalized default ports in same-origin checks, denied register/register_and_ingest for query-bearing sitemap seeds when query persistence is disabled, preserved skipped sitemap candidate reason codes, added safe missing-document hash handling, made include_seed functional for page-link discovery, reported the one-hop depth cap, and added small docstring/type/style fixes. Did not add docstrings to every pytest function because that is inconsistent with the existing test style and adds no safety.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 2598 review feedback addressed and branch rebased on latest origin/dev. Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q -> 302 passed, 4 warnings; /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -q --tb=short -> 6 passed, 3 warnings; Bandit on apps/mcp-unified/src/mcp_unified/docs and tldw_Server_API/tests/MCP_unified/docs -> 0 findings; git diff --check -> clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run for touched code when applicable or documented skip
- [x] #4 Final summary added
- [x] #5 Known skips or blockers documented
<!-- DOD:END -->
