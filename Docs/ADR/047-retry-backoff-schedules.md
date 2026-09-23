# ADR-047: Two retry backoff schedules, not one

**Status:** Accepted
**Date:** 2026-09-22
**Backfilled from:** not backfilled
**Decision owner:** repository owner (decided 2026-09-22 during core-module review remediation)
**Related task:** TASK-13330, TASK-13319
**Related spec/plan:** `Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md` (F30)

## Decision

Outbound HTTP retries use **decorrelated jitter**; in-process contention retries
(principally SQLite `database is locked`) use **short capped exponential**. Both live in
`core/Utils/backoff.py` as separate named functions. One algorithm is not imposed on both.

## Context

The 2026-09-21 core-module review found eight independent backoff implementations plus 28
inline copies of one loop in a single file, with three internal inconsistencies among the
28: one compared `"database is locked"` without `.lower()`, one hardcoded `attempt < 4`
inside a `range(5)` loop, and five omitted the jitter term the other 23 had.

Ranking the three adopted implementations:

| Implementation | Schedule | Verdict |
| --- | --- | --- |
| `http_client._decorrelated_jitter_sleep` | `min(cap, uniform(base, prev*3))`, plus `Retry-After` (delta-seconds *and* HTTP-date) and a DNS-permanent classifier | best for HTTP |
| `RAG/rag_service/resilience.py` | symmetric ±25% jitter, no `Retry-After`, no classifier | keeps clients clustered — rejected as the shared default |
| `DB_Management/transaction_utils.py` | `0.1 * 2**n`, **no jitter** | weakest |

The obvious consolidation — promote the best one everywhere — was considered and rejected.
The two populations solve different problems. HTTP retries cross a network to a shared
remote endpoint, where the purpose of jitter is to de-synchronise a *fleet* of clients.
SQLite lock contention is single-process, the window is typically milliseconds, and the
existing call sites sleep 0.05–1.2s in total. Decorrelated jitter's `prev*3` growth is
faster than capped exponential, so adopting it there would have materially lengthened lock
retries to solve a problem those call sites do not have.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| One schedule (decorrelated jitter) everywhere | Lengthens SQLite lock retries for no benefit; fleet de-synchronisation is not a property of single-process lock contention. |
| One schedule (capped exponential) everywhere | Loses decorrelated jitter on HTTP, which is the one place a retrying fleet genuinely needs it, and would discard `Retry-After` handling. |
| Promote `resilience.py` as canonical | Symmetric ±25% jitter keeps clients in a narrow band — the failure mode jitter exists to prevent — and it has neither `Retry-After` parsing nor a retriability classifier. |
| Leave all eight implementations in place | The three observed inconsistencies are already live defects, and each new call site re-derives the schedule. |

## Consequences

- `core/Utils/backoff.py` owns delay computation and knows nothing about *what* is retried.
  `decorrelated_jitter_delay` and `parse_retry_after_seconds` were **moved out of**
  `core/http_client.py` rather than copied, so that 6,600-line module shrinks; it re-exports
  them under their original private names because tests monkeypatch those.
- Migrations are **behaviour-preserving**: each call site keeps its current schedule,
  including `jitter=False` where that is what it does today. Enabling jitter on the two
  DB-contention helpers is a deliberate follow-up, not part of the consolidation.
- `is_sqlite_locked_error` is case-insensitive, fixing the copy that was not.
- A caller needing a third schedule passes its own bounds rather than adding a ninth
  implementation.

## Follow-up

- TASK-13319: the 28 inline loops in `PromptStudioDatabase.py` still carry their own copies.
  They are deliberately left to the decomposition in TASK-13318 rather than edited in place,
  since that work restructures the file. The shared helper now exists for them.
- TASK-13319 also covers the absence of any retry on the PostgreSQL side of that class,
  which this ADR does not address.
- 2026-09-23: `core/DB_Management/retry_policy.py` is the shared contention policy. Both SQL
  backends now raise the typed `TransientContentionError` (SQLite "database is locked";
  PostgreSQL SQLSTATE 40001, 40P01, 55P03), so PostgreSQL contention is recognisable for the
  first time. Prompt Studio repositories use it as they move out of the two classes; the
  remaining inline loops go with their aggregates.
