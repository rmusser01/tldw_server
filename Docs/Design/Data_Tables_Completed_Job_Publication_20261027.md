# Data Tables completion before publication

Task: TASK-13260.278.18.83.45. S03 remains open. This is the bounded backend repair design, not runtime acceptance. The human authorized necessary fixes/diagnostics and the three additional S03 capabilities; no new approval is pending.

ADR required: yes. Proposed [ADR-089](../ADR/089-data-table-completion-before-publication.md) adds the completion/publication rule while retaining ADR-003 Jobs, ADR-023 Media storage/exports and ADR-058 exact locked completion identity.

## Problem and causal evidence

A real regeneration handler replaces saved Media rows before WorkerSDK attempts Jobs completion. Dedicated DELETE, direct JobManager cancellation and admin batch cancellation can each then acknowledge cancellation while replacement content is already visible. All three normal SQLite lifecycle regressions fail the original saved columns/rows comparison; cancellation and acquired identity assertions pass. No PostgreSQL or live claim follows from these tests.

The final pre-write status lookup also treats caught database errors as permission to continue. The corrected SQLite OperationalError regression fails because no error is raised. An earlier Media DatabaseError fixture was outside the caught tuple and does not prove fail-open behavior.

## Chosen flow

1. The handler resolves sources and generates/validates output as today. Preserve prior columns/rows. Capture the owner/table UUID/internal ID and current Media version atomically with the running-state update. Give every generated column and row a stable UUID before finalization.
2. Return a versioned private publication receipt in the Jobs result, alongside the existing public summary. Bind it to the acquired job UUID, lease and effective owner/table identity. Validate encoded size against the real Jobs result limit before completion; never allow truncation to become a publishable receipt.
3. WorkerSDK completes the exact acquired UUID/lease through existing Jobs completion. Cancellation wins -> completion rejected -> no Media replacement. Completion wins -> every existing cancel path sees a terminal job and refuses cancellation. No percentage/progress field is authority. Lease enforcement is required for this worker, and the completion token binds its lease.
4. The existing on_completed callback reads the authoritative stored completed result and publishes it. It never trusts only the callback argument. on_completion_rejected never publishes. Callback failure, skip or process interruption cannot lose the completed receipt.
5. Authenticated job GET and wait-for-completion paths replay the same publisher before exposing artifact success. They validate the job owner and open only the receipt-bound owner's Media store through existing managed factories. This supplies durable recovery without another poller or in-memory-only journal.

## Atomic Media application and replay

Use the existing Media transaction abstraction: SQLite BEGIN IMMEDIATE; PostgreSQL locks the owner-bound data_tables row FOR UPDATE. A DB-owned helper verifies live table UUID/client identity and applies the expected-version predicate in that same transaction as generated columns, rows, sources and ready metadata.

Existing generated column/row UUIDs are durable receipt identities: their records are inserted atomically and ordinary edits retain or soft-delete them. A bounded lookup of the staged first column and first row, including tombstones and with exact table/client binding, recognizes prior application before checking the expected version. Return already-applied without rewriting later edits. Missing/deleted table, mismatched owner, malformed receipt or a changed version without that receipt is an explicit publication conflict. If historical data was purged, fail closed rather than republish over later data.

This is not a distributed transaction. A completed Job may temporarily have an unpublished result; the stored receipt permits replay. An acknowledged cancellation cannot have that completed receipt, and cannot publish. Publication conflict is an explicit workflow failure, not a successful artifact.

## Public API and frontend

The job-status result projects only the existing safe summary plus publication state/error; never return internal rows, source snapshots, lease IDs or the receipt. Preserve raw Jobs lifecycle status. The client must wait for ready publication before claiming generation success, and show conflict/unavailable publication explicitly. Legacy completed jobs without the versioned receipt retain their existing read behavior. New admission payloads carry an exact integer protocol marker outside the result; a truncated, missing or malformed new receipt cannot become legacy success. Acquired legacy pending jobs upgrade only through exact processing/worker/lease/payload guards.

Keep existing permissions, job-owner/Admin checks, per-owner Media access, quota/storage boundaries, normal request scope and numeric job/UUID table API conventions. No raw SQL outside DB abstractions, generic Jobs cancel rewrite, new persistence service or dependency.

## Validation before runtime

- Real dedicated, manager and admin cancellation before exact completion preserves prior saved columns/rows.
- Successful exact completion publishes; stale UUID/lease and rejected completion do not.
- Skip/fail on_completed then replay stored receipt; replay twice remains idempotent.
- Later cell/column edits survive replay; a different generation/version conflicts without overwrite.
- Size/truncation/malformed/foreign-owner receipts fail closed; public job status never exposes staging.
- Use real SQLite and official isolated restricted PostgreSQL fixtures, including concurrent publisher replays and exact owner refusals.
- Affected worker/API/Media tests, scoped lint/compile/Bandit, independent actual/staged reviews and normal hooks precede packaging/native generation.

## S03 capability additions remain required

After Data Tables and Presentations, add AI Kanban generation/Cancel, document-page generation/Cancel and combined annotated-document export. Map and reuse their current storage, Jobs, file/render/export and UI contracts in a separate concrete feature design before implementation. None is marked unavailable, implemented or accepted by this backend design.


## Durable remount recovery refinement

One nullable `data_tables.generation_job_json` column holds an internal reservation: a fresh generation key and the actual Jobs submitter/queue, then the acquired exact Job UUID. The API stores the reservation before admission and includes the key as the existing Jobs idempotency key and payload field. Admission uncertainty never causes GET to enqueue or clear the reservation. Regeneration no longer performs a queued-state write after admission, which could race a fast worker.

Worker running-state/version capture, cancellation/error markers and admission-failure cleanup are conditional on that same key inside the owned Media transaction. Publication checks it under the parent row lock after recognizing an already-applied receipt. A later user generation therefore supersedes old work without stale metadata writes. Legacy acquired jobs may reserve an empty legacy table or their own key only; their actual UUID provides discovery without pretending the idempotency column changed.

Authorized table GET resolves the reservation through existing scoped active/archive key or UUID lookups, verifies domain/type/owner/table/key binding, and replays the canonical stored result before returning saved content. The publisher uses exact active/archive UUID authority rather than a potentially reused numeric ID. Ordinary reads never create another generation. Add lost-callback remount, fast-worker admission, superseded reservation, archive replay and foreign-pointer refusals to the acceptance checks.


### Attempt identity and failure replay refinement

The acquired reservation also includes a SHA256 lease witness; terminal replay derives the same witness from stored completion_token. Exact Jobs processing-lease authentication precedes a full captured-reservation CAS under the Media parent lock. A replacement lease may bind over metadata-only old-attempt updates, but delayed old binds cannot match the replacement witness. Later status/error writes require both their exact witness and captured running version. A transient bind CAS miss is retryable, not a false missing-table failure.

Admin remount recovery opens the actual Media owner's managed session rather than treating the Admin's handle as the owner. A definitive rejected admission marks only its own reserved table failed while keeping the lookup record; uncertain commits do not clear or re-enqueue. Publication-conflict failure marking is version-guarded and idempotent. The proposal remains unaccepted until PostgreSQL and native qualification complete.
