# Jobs Manager: Acquisition Ordering and Backend Differences

This note clarifies how `JobManager.acquire_next_job` selects the next job to process, and why ordering can differ between SQLite and PostgreSQL (PG) paths.

## Priority and FIFO semantics

- Priority is evaluated first in all backends. Lower numeric values represent lower urgency by default; higher numeric values represent higher urgency only when the caller assigns them so. The current implementation orders by `priority ASC` (smaller number first) across both backends unless otherwise documented by specific tests or toggles.
- After priority, time-based ordering is applied. For jobs with the same priority, the selection is based on `COALESCE(available_at, created_at)` and then by `id` as a stable tiebreaker.

## SQLite (default local backend)

- Ordering: `priority ASC`, then `COALESCE(available_at, created_at) ASC` (oldest first), then `id ASC`.
- Effect: FIFO among jobs with equal priority. This provides predictable behavior in tests and typical single-node usage.

## PostgreSQL

Two acquisition paths exist:

1) Single-update SKIP LOCKED path (recommended for fairness and fewer races)
- Enabled via `JOBS_PG_SINGLE_UPDATE_ACQUIRE=true`.
- Ordering: `priority ASC`, then `COALESCE(available_at, created_at) ASC` (oldest first), then `id ASC`.
- Effect: Matches SQLite FIFO behavior among equal-priority jobs.

2) Two-step pick + update path (legacy/compat)
- Used when `JOBS_PG_SINGLE_UPDATE_ACQUIRE` is not enabled.
- Ordering of the pick step: `priority ASC`, then `COALESCE(available_at, created_at) DESC`, then `id DESC` (newest-first tie-breaking).
- Rationale: This path historically favored newest-first tie-breakers to minimize certain race windows when not using a single `UPDATE ... RETURNING` statement.

## Practical guidance

### Resource-Required Claims Work

`claims_rebuild_clusters` uses `admit_resource_job` rather than bare creation or
independent receipt admission. Its resource is `(claims, owner, cluster-rebuild)`
across queues/methods. Heads precede exact Jobs-row locks; reactivation retains
immutable generation and cannot replace a successor. Resource heads/counters
survive ordinary pruning; unexpired request receipts pin their exact UUID target
for 24 hours, bounded to 128 aliases. Media publication receipts are independent
domain output metadata, not another execution history.

Workers finalize with acquired UUID/worker/lease authority even when optional
completion enforcement is disabled. Owner-safe status projection does not grant
admin controls; foreign admin links additionally bind `job_uuid`. Cancellation
does not stop a physical thread or undo committed domain output. The recovery
candidate window can underfill under contention; no post-row-lock refill or
global fairness guarantee is claimed. Eligible bulk maintenance may hold its
selected heads through commit.

See the [Claims operations runbook](../Operations/Claims_Cluster_Rebuild_Jobs.md)
for rollout allowlists, safe rollback ordering and coordinated restore evidence.
False routing enables legacy sync and is not an admission/restore barrier.

- Prefer enabling `JOBS_PG_SINGLE_UPDATE_ACQUIRE=true` in PG deployments to align behavior with SQLite FIFO semantics and reduce acquisition contention.
- When tests require strict FIFO semantics across backends, ensure the PG single-update mode is enabled in the test environment.
- Scheduled jobs (future `available_at`) are not considered “ready”; FIFO applies to ready jobs where `available_at IS NULL OR available_at <= now`.

## Admin counters and gauges

- Admin endpoints adjust `job_counters` and gauges on batch operations (cancel, reschedule, requeue) best-effort. Ordering changes do not affect aggregate counts, but can affect which specific job is acquired next within a group.

If you notice any discrepancy between this note and behavior in code or tests, check:
- `JobManager.acquire_next_job` docstring and the two PG code paths
- Environment toggles such as `JOBS_PG_SINGLE_UPDATE_ACQUIRE`
