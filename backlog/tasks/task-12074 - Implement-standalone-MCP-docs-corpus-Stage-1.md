---
id: TASK-12074
title: Implement standalone MCP docs corpus Stage 1
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-30 14:56'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
  - Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 1 of the standalone MCP document corpus: runtime-neutral docs package, SQLite/FTS5 store, local file/tree ingestion, retrieval/context tools, collection and keyword metadata, Context7-compatible read aliases, tldw_server MCP shim, configuration registration, tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created the standalone `mcp_unified.docs` package boundary and committed it in `cdf549bdc8`.
- Added the SQLite/FTS5 docs catalog store with scoped documents, chunks, sections, collections, keywords, aliases, status reporting, package-resource schema loading, and package-friendly import boundaries.
- Addressed Task 2 review findings before commit: quoted user FTS terms to avoid punctuation-triggered SQLite syntax errors, added migration backfill for legacy NULL owner/profile scope rows, and added regression coverage for punctuation queries, supported filters, and legacy NULL-scope migration.
- Verification for Task 2: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v` passed 16 tests; Bandit on `mcp_unified/docs/store` and `test_docs_schema_store.py` exited 0 with JSON written to `/tmp/bandit_task_12074_task2.json`.

- Added local Markdown/MDX/text/static HTML importers with trusted-root path enforcement, symlink escape denial through resolved paths, max import file-size checks, and deterministic directory traversal.
- Verification for Task 3: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v` passed 9 tests; Bandit on `mcp_unified/docs/importers` and `test_docs_importers.py` exited 0 with JSON written to `/tmp/bandit_task_12074_task3.json`.

- Added store-backed retrieval and bounded context-pack services with collection/keyword filter passthrough, scope-isolated search, document/collection/keyword listing, and character-budget enforcement.
- Verification for Task 4: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v` passed 19 tests; Bandit on `mcp_unified/docs/retrieval` and `test_docs_retrieval_context.py` exited 0 with JSON written to `/tmp/bandit_task_12074_task4.json`.

- Added Context7-compatible read aliases, the runtime-neutral DocsMCPToolProvider, provider exports, and store-backed collection/keyword management methods for the advertised write tools.
- Verification for Tasks 5-6: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v` passed 13 tests; Bandit on `mcp_unified/docs` and `test_docs_mcp_provider.py` exited 0 with JSON written to `/tmp/bandit_task_12074_task5_6.json`.

- Added the host MCP `DocsModule` adapter and default `mcp_modules.yaml` registration with web acquisition disabled by default.
- Verification for Task 7: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_docs_module_without_web_acquisition tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py::test_write_tools_have_ingestion_or_management_category -v` passed 4 tests; Bandit on `docs_module.py` and `test_docs_module_shim.py` exited 0 with JSON written to `/tmp/bandit_task_12074_task7.json`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
