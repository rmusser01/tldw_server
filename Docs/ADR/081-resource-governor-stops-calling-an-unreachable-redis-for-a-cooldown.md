# ADR-081: The Resource Governor stops calling an unreachable Redis for a cooldown

**Status:** Accepted
**Date:** 2026-10-23
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13882
**Related spec/plan:** `Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md`
**Amends:** ADR-078 (Consequences: when the fail modes apply, and the `fail_closed` bound after Redis returns)

## Decision

Once a Redis command and a PING right after it both fail to reach Redis, the Resource Governor's Redis backend stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (10 s by default, `0` turns it off) and answers by its fail mode, then probes Redis with one command at a time.

## Context

ADR-078 made RG answer every redis-py error by its fail mode instead of raising. A governed request then goes on making Redis calls during an outage, and with a Redis that does not answer (a network partition, or a host gone without a reset) each call waits redis-py's 5 s timeout. Measured on a black-holed address, an ingress request took 30 s under `fallback_memory` and 70 s under `fail_open`, and a quota request 10 s and 65 s, for every request of the outage (TASK-13882).

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| A shorter socket timeout only | At 0.5 s a `fail_open` ingress request still waits 7 s, every request, and a shorter reply timeout turns more slow replies into lost ones (TASK-13892). |
| Start the cooldown on any one failed command | A connection Redis or a proxy closed while idle, or one stall past the timeout, would start it. Under `fail_closed` that refused every new user for 10 s, where RG had admitted them, and each refused user kept a 60 s `Retry-After`. |
| Start the cooldown on one timeout, and confirm only a refused or reset command with a PING | One stall between 5 and 10 s then refuses every new user under `fail_closed` for about 10 s, and a script reply lost to such a stall cannot be cleaned up (TASK-13892). It saves 5 s on the first request of an outage. |
| The shared `Infrastructure/circuit_breaker.py` | It logs a WARNING on every trip, for each of the nine governors in a worker, against ADR-078's one warning per worker a minute; a registry breaker reads a database on every call. |

## Consequences

- **When the fail modes apply.** ADR-078 applies them while Redis is unreachable or rejects writes. With the cooldown:
  - a single failed command still gets the fail mode for that call alone, as before;
  - once a PING confirms the failure, every call of that governor gets the fail mode for the cooldown, without reaching Redis;
  - after Redis returns, they go on for up to the cooldown plus one socket timeout: about 15 s by default.
- **`fail_closed` after Redis returns.** A user refused in that time keeps a `Retry-After` of up to 60 s, so 429s can last about 75 s after Redis recovers. ADR-078 said up to 60 s.
- **Latency while Redis does not answer.** Requests in the first 10 s of an outage wait up to 10 s each, a call's timeout and its PING's. Then one request per governor waits up to 10 s every 20 s or so, as it probes Redis.
- **Timeouts.** RG's client waits at most 5 s to connect and for each reply, redis-py 8's defaults, set explicitly so older releases the `redis>=4.5.0` pin allows wait no longer. `socket_timeout` and `socket_connect_timeout` options in `REDIS_URL` override them.
- **Release and renewal.** ADR-078's "renewed after Redis returns" now means after the cooldown ends: a lease in RG's lease map reaches Redis with its first renewal after that. A lease released in the lag window is released in the worker's copy only; if Redis holds it, it lapses there within 120 s.
- **Per governor.** Each governor (ingress, the quota lease store, each module's limiter) has its own client, cooldown and probes.
- **The knob.** `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` is read when each governor is built: at startup, or on a limiter's first use. A shorter cooldown shortens the lag after recovery; below 10 s, probes can overlap on a Redis that does not answer. `0` restores the ADR-078 behaviour.
- **Unchanged from ADR-078:** what each fail mode does, per-worker lease counting, the once-a-minute warning, and the error replies (NOSCRIPT, READONLY, OOM), which never start the cooldown and end one.

## Follow-up

- TASK-13892 removes a failed script's own members before RG counts in Python. Because one stall does not start the cooldown, that cleanup reaches Redis for a reply lost to a stall shorter than about two timeouts.
