---
id: TASK-12122
title: Plan standalone MCP docs Stage 4B source discovery implementation
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-03 19:38
labels:
- mcp
- docs
- planning
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md
- Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
priority: high
modified_files:
- Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
- backlog/tasks/task-12121 - Design-standalone-MCP-docs-Stage-4B-bounded-source-discovery.md
- backlog/tasks/task-12122 - Plan-standalone-MCP-docs-Stage-4B-source-discovery-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for Stage 4B bounded source discovery in the standalone MCP docs corpus. The plan should translate the approved design into test-first, reviewable implementation slices for docs.discover_source, sitemap/page-link discovery, url_sitemap sync_source refresh, optional BeautifulSoup/trafilatura behavior, policy/redaction/security tests, and host shim exposure without tldw_Server_API runtime imports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is saved under Docs/superpowers/plans with the required superpowers plan header.
- [x] #2 Plan decomposes Stage 4B into test-first tasks with exact files, commands, and expected outcomes.
- [x] #3 Plan preserves standalone MCP import boundaries and optional web dependency behavior.
- [x] #4 Plan includes tests for source discovery policy, redaction, sitemap parsing, page-link extraction, metadata propagation, sync_source url_sitemap refresh, and host shim exposure.
- [x] #5 Plan records verification and docs-only Bandit skip rationale.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Saved the Stage 4B source discovery implementation plan to Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md. The plan breaks implementation into eight test-first slices: settings/models/status, parser helpers, dry-run service, apply modes, MCP provider wiring, url_sitemap sync refresh, host shim/config safety, and full verification closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Stage 4B implementation plan for standalone MCP docs source discovery. The plan covers bounded docs.discover_source support, sitemap and page-link discovery, optional BeautifulSoup/trafilatura behavior without required dependencies, source policy and redaction handling, apply/register/ingest modes, url_sitemap sync_source refresh, host shim exposure, and required verification. Docs-only verification performed for this planning task: git diff --check, ASCII scan, and placeholder scan. Bandit is documented as skipped for this docs-only planning change and required for the later Python implementation.
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
