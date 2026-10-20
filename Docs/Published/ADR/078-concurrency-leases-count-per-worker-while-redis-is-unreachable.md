# ADR-078: Concurrency leases count per worker while Redis is unreachable

**Status:** Accepted
**Date:** 2026-10-20
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13859
**Related spec/plan:** `Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`
**Amends:** ADR-076 (Consequences, "Lease-store failures": a reserve after Redis drops no longer admits without a lease)

## Decision

The Resource Governor's Redis backend answers every redis-py error (`RedisError`) by its fail mode, so while Redis is unreachable or rejects writes each worker counts per-user concurrency leases on its own, instead of admitting the request without a lease.

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
  - Under the default `fallback_memory`, the worker's memory governor holds the lease while Redis is unreachable or read-only (READONLY), because counting leases starts with a write. Otherwise RG's in-process lease map holds it: under `fail_open` and `fail_closed`, and under `fallback_memory` while Redis is out of memory (OOM). A lease taken while Redis rejects writes is not in Redis.
  - `fail_closed` does not refuse a lease during an outage.
- **Lease scripts Redis cannot run.** When Redis answers but cannot run the lease script, leases are counted in Python, and the script loads again on the next reserve.
  - After a Redis restart (NOSCRIPT), they are counted against Redis, so the limit holds across workers.
  - While Redis rejects writes (OOM, READONLY), each worker counts on its own, and each reserve loads the script and fails it again.
  - If the script ran but its reply was lost (a redis-py `TimeoutError`, or a connection reset), the Python count includes the lease the script just took. With a limit of 1 the request is refused, and that lease blocks the user's slot until it expires, within 120 s. With a higher limit, the request uses one extra slot until then. The multi-key rate script's members are charged again the same way and never refunded. Before this ADR, the request was admitted with no lease, and the script's lease was stranded the same way.
- **Leases on the other side.** A worker counting on its own does not see leases taken before Redis dropped. When Redis returns, Redis does not see the leases the memory governor took during the outage, until those requests end. Either way, a user can briefly exceed the limit by the leases the other side holds.
- **Release and renewal.** A lease released or renewed during the outage acts on the worker's own copy of the lease. A lease taken during the outage in RG's lease map is written to Redis when it is renewed after Redis returns. A lease the memory governor holds (`fallback_memory`, with Redis unreachable or read-only) is not: it stays in the memory governor until its request ends. Redis leases the worker could not release expire within 120 s.
- **Rates.** A `requests` or `tokens` window that Redis cannot read or write gets the fail mode, whether Redis is unreachable or rejects writes:
  - `fallback_memory` counts both in each worker's memory governor;
  - `fail_open` admits, and holds only `requests` per worker, in its acceptance window;
  - `fail_closed` answers 429 (`Retry-After` up to 60 s) while the failure lasts, and for up to 60 s after, while each worker's deny floor runs out.
- **Logging.** Every Redis error that leaves the worker enforcing a limit on its own is logged: unreachable, READONLY or OOM. Each worker process logs one WARNING per Redis namespace at most once a minute. A restart (NOSCRIPT) is not logged, because the limit still holds.
- **No more 500s from RG.** With Redis unreachable, read-only or out of memory, no Redis error escapes RG's `reserve`, `check`, `renew`, `commit` or `peek_with_policy`. So none answers a request with HTTP 500, at ingress or at the endpoints that call RG directly, such as chat's tokens reserve. That holds even though chat's guard (`_CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS`) does not list `RedisError`.

## Follow-up

- TASK-13882: a connect timeout, or a circuit breaker, for the RG Redis client. A governed request now makes several Redis calls during an outage, and redis-py 8.1's `from_url` waits 5 s per call on a black-holed Redis. That adds up to about 30-70 s per request.
- TASK-13892: before counting in Python after a failed script, `reserve` removes the members it planned for that script, so a lost reply does not count the script's own lease.
