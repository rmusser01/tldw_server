---
id: TASK-13156
title: Attest bounded Reading digest schedule and output management
status: To Do
assignee: []
created_date: '2026-09-03 02:31'
updated_date: '2026-09-03 02:32'
labels:
  - collections
  - reading-list
  - digests
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
Keep existing digest routes/shapes unchanged and add an exact bounded schedule-page route beside the bare-list route; retain the output-page envelope. Use deterministic ordering and user scoping. Create accepts a user-scoped `client_request_id`: identical key/payload returns the original schedule, mismatched payload conflicts, and exact key lookup reconciles lost responses. Preserve last status/run/next-run/output history as the only run evidence. Document refresh-based reconciliation for uncertain update/delete outcomes and separate worker availability. Advertise exact `hasReadingDigestManagementV1=true` only with these additive guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing digest routes and response shapes remain compatible, and an additive schedule-page route returns exact totals with deterministic bounded rows.
- [ ] #2 Schedule creation uses a user-scoped `client_request_id`; identical replays return the same schedule, payload mismatch conflicts, and exact lookup reconciles a lost response.
- [ ] #3 Schedule and output reads never cross users and expose only `last_status`, `last_run_at`, `next_run_at`, and bounded output history rather than claiming a complete run ledger.
- [ ] #4 Update/delete outcomes and scheduler/worker availability are documented separately so clients can refresh after uncertainty without inventing optimistic conflicts.
- [ ] #5 Docs-info advertises `hasReadingDigestManagementV1=true` only when the additive paging and idempotent-create contract is active.
- [ ] #6 Focused SQLite/PostgreSQL, API, concurrency, compatibility, and security tests pass; the ADR check is recorded.
<!-- AC:END -->
