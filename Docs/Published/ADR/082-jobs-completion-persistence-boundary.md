# ADR-082: Completion persistence belongs in DB_Management

**Status:** Accepted
**Date:** 2026-10-04
**Backfilled from:** not backfilled
**Decision owner:** Human requester, ownership option A approved in this task
**Related task:** TASK-13421
**Related spec/plan:** Docs/superpowers/specs/2026-10-04-jobs-completion-extraction-design.md; foundation and routing plans require separate review before implementation

## Decision

Jobs single-job completion uses one completion-specific backend persistence
operation under `core/DB_Management`, on the facade's existing connection/RLS
cursor, while `JobManager` retains public policy, context managers, and
post-commit observers.

## Context

PR #3092 established locked row identity and atomic completion bookkeeping in
[ADR-058](058-jobs-completion-row-identity.md), but intentionally left SQL in
the existing facade. The next strict extraction must satisfy the repository's
SQL ownership guidance without distributing the lock, mutation, counter, and
outbox protocol across independently owned connections or transactions.

Existing Jobs admission/lease operations also contain SQL. Moving those paths
or introducing a generic terminal-transition engine is not part of this decision.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Jobs orchestration over granular database methods | Spreads the ordered completion protocol and makes transaction ownership easier to split. |
| SQL in Jobs operation modules | Retains ownership debt and requires an exception to the repository database-layer guidance. |
| New repository-owned connection or transaction | Changes RLS, context-exit, locking, commit, and observer failure ordering. |
| Generic engine for every terminal lifecycle path | Expands strict completion extraction into uncharacterized failure/cancellation semantics. |

## Consequences

- Completion contracts and backend operations live in `DB_Management` without
  runtime imports of `JobManager` or Jobs observer/metric/tracing modules.
- `JobManager` opens and closes the connection and retains its current
  connection/cursor contexts. Applied operations perform the current explicit
  commit; early replay/no-transition paths retain implicit context-exit behavior.
- One supplied executor carries authoritative locking, identity guards, enabled
  counters/outbox, and optional SLA savepoints. No replacement connection,
  independent transaction, or second RLS cursor is introduced.
- Lazy data-only policy callbacks preserve dynamic setting/clock and exception
  timing. Returned facts are prepared before commit; non-durable dispatch remains
  after commit and before context exit, with exact existing suppression.
  The existing separately connected read-only SLA policy lookup remains lazy;
  it never becomes an owner of completion writes or a new operation connection.
- Shared helper relocation is mechanical and verified against all affected
  lifecycle callers. Completion-only SQL moves with backend routing; other Jobs
  SQL ownership debt remains separate.
- ADR-058 remains accepted and unchanged. This decision adds a module boundary;
  it does not alter the public API, row-identity rules, or completion behavior.

## Follow-up

Implementation is not present in this decision commit. Review the written design,
then deliver a characterized contracts/helper foundation PR with production
routing unchanged. After it merges, use a separate task and plan for backend
routing. Newly validated ordinary defects remain separate follow-up PRs.
