---
id: TASK-12093
title: Implement standalone MCP docs Stage 4A source sync
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 07:53'
labels:
  - mcp
  - docs
  - source-sync
dependencies: []
references:
  - TASK-12091
  - TASK-12092
documentation:
  - >-
    Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md
  - >-
    Docs/superpowers/plans/2026-07-02-standalone-mcp-docs-stage4a-source-sync-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Stage 4A.1 standalone MCP docs source-sync plan: source-sync settings/status, source registry schema/store helpers, local and URL source population, docs.list(kind="sources"), docs.sync_source for local files/directories and URL pages, host exposure, focused tests, docs test slice verification, and Bandit on touched Python paths. Sitemap registration/sync remains deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-02-standalone-mcp-docs-stage4a-source-sync-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 4A.1 bounded source sync for the standalone MCP docs corpus.

What changed:
- Added source-sync settings/status and public source/sync model contracts.
- Added SQLite source registry tables, source-document links, sync runs, lifecycle columns, migration guards, scoped source helpers, active-only document counts, and sync-aware document upserts that preserve user keywords/collections while merging source defaults.
- Populated refreshable sources from local imports and approved URL ingests, with query-bearing URL sources disabled by default unless `persist_url_query_strings=true`.
- Added query/credential-safe source utilities and public response redaction so query-bearing URL tokens are not exposed in docs.list or docs.sync_source responses.
- Exposed `docs.list(kind="sources")` and `docs.sync_source` through the standalone provider and host DocsModule shim.
- Implemented local_file/local_directory sync with strict dry-run immutability, apply mode, source-link hashing, metadata preservation, max-document/stale-link limits before mutation, report/tombstone stale policy, symlink escape prevention, and lifecycle-aware search/get/resolve/facet filtering.
- Implemented url_page sync for existing sources only, using the Stage 2 policy/resolver/transport/extraction seams, preflight policy denial before fetch, dry-run/apply behavior, sync-run recording only for apply, redirect/canonical retarget reconciliation, active-only source document counts, and redacted public summaries.
- Kept sitemap registration/sync disabled/unsupported and did not add crawling, browser automation, jobs/scheduler, Media DB, ChromaDB, RAG, or host service bridges.

Verification:
- Focused host/boundary tests: `16 passed, 4 warnings`.
- Full docs test slice: `253 passed, 4 warnings`.
- Final lifecycle focused review tests: `57 passed, 3 warnings`.
- Bandit: exit 0, no findings; report written to `/tmp/bandit_mcp_docs_stage4a.json`.
- `git diff --check`: clean.

Known deferrals:
- Sitemap source registration/sync remains deferred to Stage 4A.2 or later.
- Optional scraping pipeline dependencies remain out of the baseline standalone install.
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
