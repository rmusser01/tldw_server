# ADR-078: Concurrency leases count per worker while Redis is unreachable

**Status:** Accepted
**Date:** 2026-10-20
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13859
**Related spec/plan:** `Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`
**Amends:** ADR-076 (Consequences, "Lease-store failures": a reserve after Redis drops no longer admits without a lease)

## Decision

The Resource Governor's Redis backend treats `redis.exceptions.RedisError` as a Redis failure and answers it by its fail mode. So while Redis is unreachable, a per-user concurrency lease is counted in the worker that holds it, whether Redis was down when the worker first connected or dropped later. When Redis answers but cannot run the lease script, leases are counted in Python, and the script loads again on the next reserve:
- after a Redis restart (NOSCRIPT), against Redis, so the limit holds across workers;
- while Redis rejects writes (OOM, READONLY), in each worker on its own.

## Context

ADR-076 says that if Redis drops after a worker has connected, "a reserve that fails admits the request". It did, because RG's best-effort guards (`_RG_NONCRITICAL_EXCEPTIONS` in `governor_redis.py`) listed only builtin exceptions, and redis-py's `ConnectionError` and `TimeoutError` derive from `RedisError`, not from the builtins. The error escaped RG, and `concurrency_quota` admitted the request with no lease. The same escape made ingress (`RGSimpleMiddleware`) answer every governed request with HTTP 500 once Redis dropped, whatever the fail mode, and again after every Redis restart, when the cached script SHAs no longer exist (TASK-13859).

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep admitting quota requests with no lease while Redis is down | No RG fail mode admits a lease without counting it: `fallback_memory` counts it in the worker's memory governor, and `fail_open` and `fail_closed` count it in RG's in-process lease map. Keeping ADR-076's behaviour needs a lease path that bypasses RG for quotas alone, and lets a user start unlimited concurrent work for as long as Redis is down. |
| Refuse quota requests while Redis is down | Turns a Redis outage into an outage of media ingest, audio streaming and transcription. |
| Deny, or skip, a lease whose script fails while Redis still answers | After a Redis restart every reserve fails the script, so denying refuses every quota request until the worker restarts, and skipping drops the limit. |
| Catch `RedisError` only in RG's callers (ingress, `concurrency_quota`) | RG's fail modes would still never run for a Redis error, and every new caller would have to know to catch it. |

## Consequences

- **Per-worker counting.** While Redis is unreachable or rejects writes, each worker counts its own leases, so with N workers a user can run up to N times the limit.
  - That is how the memory backend works, and what ADR-076 already says for a worker that never connected.
  - Under the default `fallback_memory`, the worker's memory governor holds the lease. Under `fail_open` and `fail_closed`, RG's in-process lease map does. While Redis rejects writes, the lease is in that map but not in Redis.
  - `fail_closed` does not refuse a lease during an outage.
  - While Redis rejects writes, each reserve also loads the lease script and fails it again.
- **Leases on the other side.** A worker counting on its own does not see leases taken before Redis dropped. When Redis returns, Redis does not see the leases the memory governor took during the outage, until those requests end. Either way, a user can briefly exceed the limit by the leases the other side holds.
- **Release and renewal.** A lease released or renewed during the outage acts on the worker's own copy of the lease. Redis leases the worker could not release expire within 120 s.
- **Logging.** Every Redis error that leaves the worker enforcing a limit on its own is logged: unreachable, READONLY or OOM. Each worker process logs one WARNING per Redis namespace at most once a minute. A restart (NOSCRIPT) is not logged, because the limit still holds.
- **No more 500s.** A Redis error no longer answers a request with HTTP 500. Ingress answers by fail mode:
  - `fail_open` and `fallback_memory` admit, within the requests limit each worker keeps on its own;
  - `fail_closed` answers 429 (`Retry-After` up to 60 s) while Redis is unreachable, and for up to 60 s after, while each worker's deny floor runs out.
