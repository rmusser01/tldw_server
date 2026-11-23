# ADR-097: Watchlist Schedule Generation Ownership

**Status:** Proposed
**Date:** 2026-11-19
**Backfilled from:** Not backfilled
**Decision owner:** Watchlists and Scheduler maintainers; review pending
**Related task:** TASK-15568.1
**Related plan:** `IMPLEMENTATION_PLAN_pr3071_rebase_qodo_merge_2026_10_02.md`

## Decision

The owner-scoped Watchlist job is authoritative for the active recurring
schedule. A definition change creates a new scheduler row, then attaches it
only if the job revision and previous attachment still match. A private
managed-job ID distinguishes the row from independently configured workflows
and survives renames or a failed deletion.

## Context

Watchlists and Collections store job definitions separately from the
recurring Scheduler. Two successful Watchlist edits can reach the scheduler
in reverse order if both write the same row. The link then points to an old
cron that can no longer dispatch successfully. Using display names to
identify managed rows can also turn an orphaned cron into an unfenced task.
This decision retains Scheduler as the execution mechanism under ADR-003.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Re-read the Watchlist job before updating in place | A later write can still finish first between the stores. |
| Identify ownership by name or consumer inputs | Both can change independently of the owning Watchlist job. |
| Replace the Scheduler with a separate queue | Existing recurring dispatch, leadership and job handling already serve the required runtime. |

## Consequences

- The managed marker is a nullable, additive scheduler field. Do not infer
  ownership by bulk rewriting names or unlinked records. A previously linked
  scheduler row can be marked before retirement using verified owner and
  attachment evidence.
- A managed occurrence must still match the owner, attached row and current
  job revision at admission. Explicit Run Now continues to use the manual
  path. Independent workflow schedules retain their existing behavior.
- A winning attachment makes the old generation ineligible and records the
  exact owner, job and superseded schedule ID in the same content-store
  transaction. Cleanup validates the scheduler row against that record and
  acknowledges successful deletion or confirmed absence. A failed cleanup
  leaves the record for retry; an unattached row alone is not proof of
  retirement. The scheduler lease holder retries recorded retirements using
  existing owner storage.
- A job ID remains reserved in SQLite while run history, an undo record, or
  a retirement record references it. A reversible delete cannot replace a still-present undo
  record. Collections hard-delete commits the feed, linked job, and any
  verified retirement record together before attempting scheduler cleanup.
- A Collections job created for a feed has a private binding to that source
  ID and creation timestamp in its owner's store. Undo preserves the binding.
  Editable source settings and job scope/output preferences are not proof of
  ownership. Existing unverified links are not backfilled or used to edit or
  delete jobs. A user can still manage the source and job independently in
  Watchlists. Hard-delete can consume only a matching undo record; its final
  check reads both live and undo state in the same snapshot.

## Follow-up Required Before Rollout

Specify how to reconcile unattached rows and queued legacy payloads that lack
provenance without misclassifying manual runs. Verify both scheduler stores,
including upgrades of existing PostgreSQL databases. This proposal does not
authorize any cleanup, migration or queue operation on live data.
