---
id: TASK-12875
title: Implement standalone MCP docs Stage 4B source discovery
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-03 23:26
labels:
- mcp
- docs
- implementation
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md
- Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
priority: high
modified_files:
- Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
- apps/mcp-unified/src/mcp_unified/docs/__init__.py
- apps/mcp-unified/src/mcp_unified/docs/discovery.py
- apps/mcp-unified/src/mcp_unified/docs/mcp_module.py
- apps/mcp-unified/src/mcp_unified/docs/models.py
- apps/mcp-unified/src/mcp_unified/docs/settings.py
- apps/mcp-unified/src/mcp_unified/docs/sync.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py
- tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py
- backlog/tasks/task-12124 - Implement-standalone-MCP-docs-Stage-4B-source-discovery.md
references:
- https://github.com/rmusser01/tldw_server/pull/2598
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4B bounded source discovery for the standalone MCP docs corpus from the approved plan. Scope includes discovery settings/models/status, docs.discover_source, sitemap and page-link parsing, dry-run and apply modes, optional BeautifulSoup/trafilatura behavior without required dependencies, url_sitemap sync refresh, host shim exposure, policy/redaction/security tests, Bandit, and Backlog closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discovery settings, models, status surface, and provider advertisement follow the approved plan.
- [x] #2 docs.discover_source supports bounded sitemap/page-link dry-run and apply flows with policy, redaction, dedupe, and metadata propagation tests.
- [x] #3 Registered url_sitemap sources can refresh through docs.sync_source with dry-run/apply/stale handling tests.
- [x] #4 Standalone MCP import boundaries and optional dependency behavior remain covered by tests.
- [x] #5 Focused MCP docs tests, import-boundary tests, Bandit, diff hygiene, and Backlog closeout are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md inline using TDD. Commit after each task slice where tests are green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 4B source discovery from the approved plan using test-first slices. Added safe discovery settings/status, DiscoverSourceRequest models, lazy optional page-link extraction with stdlib fallback, bounded sitemap/page-link discovery, dry-run and apply modes, url_sitemap registration and sync refresh, provider/host shim exposure, same-origin and prefix filtering, query redaction, metadata propagation, and import-boundary coverage. Bandit initially flagged stdlib ElementTree sitemap parsing; the parser now uses defusedxml.ElementTree, which is already a project dependency.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4B standalone MCP docs source discovery is implemented and verified. docs.discover_source is disabled by default and only advertised when web acquisition and source discovery are enabled. It supports bounded sitemap and page-link discovery with dry-run, register, ingest, and register_and_ingest flows. Registered url_sitemap sources refresh through docs.sync_source with dry-run/apply/stale handling. PR: https://github.com/rmusser01/tldw_server/pull/2598. Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q -> 295 passed, 4 warnings; Bandit on apps/mcp-unified/src/mcp_unified/docs and tldw_Server_API/tests/MCP_unified/docs -> 0 findings; import-boundary test -> 6 passed; git diff --check -> clean.
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
