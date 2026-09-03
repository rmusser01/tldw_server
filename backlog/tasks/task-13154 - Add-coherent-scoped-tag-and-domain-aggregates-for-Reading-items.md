---
id: TASK-13154
title: Add coherent scoped tag and domain aggregates for Reading items
status: To Do
assignee: []
created_date: '2026-09-03 02:29'
updated_date: '2026-09-03 02:30'
labels:
  - collections
  - reading-list
  - api
  - facets
  - pagination
dependencies: []
references:
  - 'tldw_chatbook:TASK-18919'
  - >-
    tldw_chatbook:Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose bounded, deterministic, fully pageable tag and domain aggregate results for an explicitly documented Reading scope. Use distinct `capture_q` and `facet_q` parameters. Aggregate rows and exact totals use one snapshot with accepted capture search/status/favorite/date/tag/domain filters and self-excluding facet semantics. Results are user-scoped, normalize values with a stable tie-break order, and reveal neither capture content nor sensitive URL components. Advertise exact `hasReadingAggregateFacetsV1=true` only with the complete SQLite/PostgreSQL contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Authenticated callers can page every tag or domain aggregate through deterministic bounded results with exact aggregate totals and server-side `facet_q` search.
- [ ] #2 Aggregate count and rows use one SQLite/PostgreSQL snapshot and apply `capture_q` plus all non-self facet filters before aggregation.
- [ ] #3 Self-excluding tag/domain semantics are documented and verified for combined filters, concurrent writers, empty scopes, deep pages, and normalization collisions.
- [ ] #4 Responses reveal no capture body, note, highlight, sensitive URL component, or other user's values.
- [ ] #5 Docs-info advertises `hasReadingAggregateFacetsV1=true` only with the complete contract, and the ADR check plus focused API/database/security tests are recorded.
<!-- AC:END -->
