---
id: TASK-12124
title: Implement standalone MCP docs Stage 4B source discovery
status: In Progress
labels:
- mcp
- docs
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md
- Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4B bounded source discovery for the standalone MCP docs corpus from the approved plan. Scope includes discovery settings/models/status, docs.discover_source, sitemap and page-link parsing, dry-run and apply modes, optional BeautifulSoup/trafilatura behavior without required dependencies, url_sitemap sync refresh, host shim exposure, policy/redaction/security tests, Bandit, and Backlog closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Discovery settings, models, status surface, and provider advertisement follow the approved plan.
- [ ] #2 docs.discover_source supports bounded sitemap/page-link dry-run and apply flows with policy, redaction, dedupe, and metadata propagation tests.
- [ ] #3 Registered url_sitemap sources can refresh through docs.sync_source with dry-run/apply/stale handling tests.
- [ ] #4 Standalone MCP import boundaries and optional dependency behavior remain covered by tests.
- [ ] #5 Focused MCP docs tests, import-boundary tests, Bandit, diff hygiene, and Backlog closeout are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md inline using TDD. Commit after each task slice where tests are green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
