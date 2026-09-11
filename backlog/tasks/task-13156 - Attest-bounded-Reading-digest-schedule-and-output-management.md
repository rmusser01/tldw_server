---
id: TASK-13156
title: Attest bounded Reading digest schedule and output management
status: To Do
assignee: []
created_date: '2026-09-03 02:31'
updated_date: '2026-09-03 02:41'
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
Keep all existing digest routes and response shapes unchanged. Add an exact bounded schedule-page
route beside the current bare-list route; the existing output-page route retains its envelope. Both
use deterministic ordering and user scoping. Schedule creation accepts a caller-generated,
user-scoped `client_request_id`: repeating the same key and normalized payload returns the original
schedule, while reusing it with a different payload fails with a bounded conflict. Exact lookup by
that key lets a client reconcile a lost create response. Preserve schedule `last_status`,
`last_run_at`, `next_run_at`, and output history as the only run evidence; do not invent a distinct
run-history API. Update/delete responses remain non-optimistic and are documented for refresh-based
reconciliation after transport uncertainty. Docs-info advertises exact
`hasReadingDigestManagementV1=true` only when these additive guarantees are active and configured
scheduler/worker availability is reported separately.
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
