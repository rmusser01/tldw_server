# ADR-076: Per-user concurrency quotas are usage quotas held as Resource Governor leases

**Status:** Accepted
**Date:** 2026-10-18
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13435
**Related spec/plan:** `Docs/Design/2026-10-02-usage-quota-posture-design.md` (Non-goals, deferred), `Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md`
**Amends:** ADR-064 (synchronous concurrency is now a usage quota) and ADR-056 (`RG_ENABLED` no longer covers media and audio concurrency)

## Decision

Per-user synchronous concurrency (media ingest, audio streams, direct transcription) is a usage quota: a `limits.*` value resolved and switched like every other (ADR-064), enforced as one lease held for the life of each request in a Resource Governor lease store that the request carries its own limit into, built from `RG_BACKEND` and not turned off by `RG_ENABLED`.

## Context

Spec 2 stopped the media and audio sites reserving `jobs` and `streams` leases from RG policies, because those were defaults nobody had assigned. That left synchronous concurrency unlimited, and the per-user limit was deferred because `RGRequest` could not carry one (TASK-13435). The owner's commercial offering needs it per user, team or org, blank by default. The governors read a lease limit only from the policy's category block; the Redis backend released a lease only when that block existed; and a policy listing `global` charged every lease to the server-wide bucket as well.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| A request-level `RGRequest.max_concurrent` field | One value for a request that can carry several categories, and no place for the lease TTL. Per-category `max_concurrent`/`ttl_sec` in `RGRequest.categories` mirror how policies already spell the limit. |
| Reserve on the app's governor (`app.state.rg_governor`) | It is `None` when `RG_ENABLED` is off, so an assigned limit would silently stop applying; the operator's policy store could add a `global` bucket; the quota keys are not policies, so lookups log "policy not defined". |
| Encode the limit in the policy id, with no governor change | Changing a user's limit opens a new, empty bucket, so the user briefly runs the old plus the new limit. |
| A separate lease store (a dict plus a Redis ZSET) outside RG | Duplicates the TTL purge, the atomic Lua acquire and the retry-after logic RG already runs on both backends. |
| Lease TTL only, no renewal | A long ingest or stream outlives a short TTL and stops counting; a long TTL locks a user out that long after a crash. |
| Turn the limits off with `RG_ENABLED` | Every other `limits.*` quota works with RG off (ADR-064); an assigned limit would silently stop applying. |

## Consequences

- **Keys:**
  - `limits.media_concurrent_jobs` covers `/media/add`, including its research-discovery handoff. It does not count queued ingest jobs (`/media/ingest/jobs`) or `/media/process-*`.
  - `limits.audio_concurrent_streams` covers the audio WebSockets, the realtime sockets and `POST /audio/chat`.
  - `limits.audio_concurrent_transcriptions` covers `/audio/transcriptions` and `/audio/translations`.
  - Unset is unlimited and costs nothing; `0` blocks.
- **The request carries the limit.** A `streams`/`jobs` entry in `RGRequest.categories` may carry `max_concurrent` and `ttl_sec`. They replace the policy's values and charge only the caller's bucket, on both backends (`policy_eval.lease_limits`); renewing such a lease refreshes only that bucket.
- **Memory vs Redis.** On the memory backend a limit holds per worker process, so N workers admit up to N times the limit, as for every memory-backend limit (ADR-056). `RG_BACKEND=redis` makes it exact across workers.
- **Release and renewal.** A request releases its lease as the last step of its `finally`, on success, error, cancellation and graceful shutdown, shielded from the request's cancel scope. A held lease is renewed every 30 s while its task runs.
- **Unreleased leases lapse within 120 s.** That covers a killed worker, a missed release, a cancellation that lands after the store granted the lease, and an event loop blocked for two minutes. The worst case is a two-minute lockout of one slot, or briefly one extra admission.
- **Retry-After is the lease TTL.** A refusal's `Retry-After` is when the oldest held lease would lapse without renewal, up to 120 s. A slot usually frees sooner, when a request finishes.
- **Lease-store failures.** If Redis is unreachable when a worker's lease store first connects, that worker counts on its own for the life of the process, as on the memory backend. If Redis drops later, a reserve that fails admits the request, and a renewal or release that fails is logged and the lease lapses within 120 s. Redis errors never fail the request.
- **Switches.** `RG_ENABLED=false` no longer turns off media or audio concurrency limits; `USAGE_QUOTAS_ENABLED` does. The cost: the lease store is a governor that `RG_ENABLED` does not remove, which ADR-056 had ruled out for RG enforcement. With RG off and `RG_BACKEND=redis`, a worker connects to Redis once a user has a value set.
