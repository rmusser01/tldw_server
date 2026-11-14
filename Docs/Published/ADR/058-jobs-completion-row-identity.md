# ADR-058: Jobs completion uses locked row identity

**Status:** Accepted
**Date:** 2026-10-02
**Backfilled from:** Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
**Decision owner:** Human requester, approved design in this task
**Related task:** TASK-13215
**Related spec/plan:** Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md; Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md

## Decision

Jobs completion binds its mutation and bookkeeping to the visible row loaded
under a transaction lock, with an optional caller-supplied UUID precondition.

## Context

A completion lookup that misses can race with insertion of the same numeric
ID. The previous update can then complete the new row without facts needed for
counters and the completion event. Durable state and enabled bookkeeping must
refer to the same row incarnation and commit together.

PostgreSQL uses the established RLS-aware cursor with `SELECT ... FOR UPDATE`.
SQLite starts `BEGIN IMMEDIATE` before reading. A miss returns `False`. Every
completion mutation and defensive replay query guards the numeric ID and raw
stored UUID with null-safe equality. `expected_uuid` is optional; WorkerSDK
passes the acquired job's UUID for ordinary success.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Update with RETURNING as the only boundary | Expands terminal replay, queued policy, and backend behavior changes during a focused fix. |
| Advisory locks keyed by numeric ID | Requires a new protocol across inserters and terminalizers without helping an immediate return on a missing row. |
| Reuse the slides terminal-result operation | Its result and correlation contract does not cover general completion. |

## Consequences

Enabled lifecycle counters and completion outbox records remain mandatory
parts of the completion transaction. SLA statement failures remain best effort
inside an established savepoint; savepoint control failures abort completion.
Observers run after commit.

SQLite reserves the writer slot even for a missing-row attempt. Existing
numeric-ID callers remain compatible but cannot reject replacements that
precede their call. Null and empty historical UUIDs receive only the protection
of the in-operation lock; values are preserved without normalization.

## Follow-up

- TASK-13216: migrate other acquired-job completion callers to UUID preconditions.
- TASK-13217: assess historical bookkeeping drift.
- Resume strict completion extraction after this fix merges.
