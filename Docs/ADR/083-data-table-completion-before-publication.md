# ADR-083: Data Table Completion Before Publication

**Status:** Proposed
**Date:** 2026-10-27
**Backfilled from:** not backfilled
**Decision owner:** UAT requester and implementation review under authorized S03 repair
**Related task:** TASK-13260.278.18.83.45
**Related spec/plan:** [Design](../Design/Data_Tables_Completed_Job_Publication_20261027.md), [S03 plan](../Plans/IMPLEMENTATION_PLAN_uat_s03_artifact_editors_20261026.md)

## Decision

Persist generated Data Table output as an exact-attempt Jobs result before publishing it to Media, then replay publication through an atomic owner/version/receipt guard until the artifact is ready or explicitly conflicted.

## Context

The handler currently replaces Media rows before Jobs completion. Real SQLite regressions demonstrate dedicated, manager and admin cancellation accepting that still-processing job after saved rows were replaced. A later rejected completion cannot undo the replacement. Separate Jobs and Media stores cannot share a portable transaction.

ADR-023 retains Media as table storage and Jobs as generation authority. ADR-058 supplies exact locked Jobs completion. This proposal defines how those existing boundaries coordinate; it does not amend their accepted rationale.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Another status check immediately before persistence | Cancellation can still win between that read and the Media commit. |
| UI abort or dedicated endpoint lock only | Generic manager/admin cancellation and sidecars bypass that local boundary. |
| Treat progress percentage as a publication fence | Progress is mutable telemetry, may survive retries and is not exact-attempt authority. |
| Trust only an on_completed callback | Existing callbacks may fail, time out or be skipped; their original result may differ from stored/truncated data. |
| A new shared transaction or generation database | Adds a competing store and does not solve deployment boundaries portably. |

## Consequences

- A completed stored Jobs receipt is the sole publication authority. Cancelled/rejected/stale attempts never replace Media content.
- Durable result-size validation, exact owner/table/job/lease binding and safe public projection are mandatory.
- Publication uses the existing Media transaction, owner lock and version. Stable generated column/row UUIDs recognize an already-applied receipt, including its retained tombstones; replay never rewrites later edits.
- Authenticated job reads/waits supply recovery when best-effort completion callbacks do not publish. User-facing success requires applied or verified publication, not only computation completion.
- Publication conflicts and missing/purged/malformed receipts fail explicitly. This is replayable coordination, not a distributed atomic transaction claim.

## Follow-up

Implement and independently review the bounded design and SQLite/official PostgreSQL regressions, then package and validate real S03 workflows. This proposed record is not implementation or UAT acceptance.


### Recovery refinement under implementation

A nullable internal generation-job JSON reservation stores the generation key, original Jobs owner/queue and exact UUID when acquired. It enables authorized table reopen to discover the same active/archive job and replay publication without re-enqueueing. The key is reserved before admission and guards all worker/publication/error state changes, so stale generations cannot overwrite newer intent. This remains Proposed until the backend and native checks qualify it.


### Attempt identity and failure replay refinement

The acquired reservation also includes a SHA256 lease witness; terminal replay derives the same witness from stored completion_token. Exact Jobs processing-lease authentication precedes a full captured-reservation CAS under the Media parent lock. A replacement lease may bind over metadata-only old-attempt updates, but delayed old binds cannot match the replacement witness. Later status/error writes require both their exact witness and captured running version. A transient bind CAS miss is retryable, not a false missing-table failure.

Admin remount recovery opens the actual Media owner's managed session rather than treating the Admin's handle as the owner. A definitive rejected admission marks only its own reserved table failed while keeping the lookup record; uncertain commits do not clear or re-enqueue. Publication-conflict failure marking is version-guarded and idempotent. The proposal remains unaccepted until PostgreSQL and native qualification complete.
