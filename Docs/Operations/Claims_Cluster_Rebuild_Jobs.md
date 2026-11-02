# Claims Cluster Rebuild Jobs

## Contract And Configuration

This opt-in Stage 2C moves `POST /api/v1/claims/clusters/rebuild` to shared
Jobs type `claims_rebuild_clusters`. ADR-086 is Accepted and composes with
ADR-003/058; SDK storage-lifetime ADR-087 remains Proposed. Claims owns
calculation and publication, not a queue, lease loop, cancellation state,
shutdown gate or restore-repair service.

Environment overrides file settings in `[ClaimsMonitoring]`. Producers require
both `CLAIMS_JOBS_ENABLED` and `CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED`. The default
remains legacy synchronous HTTP 200 with existing result fields. Worker-disabled
means visible queued work, never local fallback. Reuse the existing
`CLAIMS_JOBS_WORKER_ENABLED`, `CLAIMS_JOBS_QUEUE`, lease and backoff settings.

| Setting | Default / Hard Maximum |
| --- | --- |
| CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED | false |
| CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD | 3 |
| CLAIMS_CLUSTER_REBUILD_MAX_EXACT_CLAIMS | 50000 |
| CLAIMS_CLUSTER_REBUILD_MAX_EMBEDDING_CLAIMS | 5000 |
| CLAIMS_CLUSTER_REBUILD_MAX_TEXT_BYTES | 33554432 |
| CLAIMS_CLUSTER_REBUILD_MAX_VECTOR_WIDTH | 4096 |
| CLAIMS_CLUSTER_REBUILD_MAX_EMBEDDING_COMPONENTS | 2000000 |
| CLAIMS_CLUSTER_REBUILD_MAX_SIMILARITY_COMPONENT_COMPARISONS | 50000000 |
| CLAIMS_CLUSTER_REBUILD_MAX_CLEANUP_REFERENCES | 50000 |

Retries permit integers 0..100; limits permit integers 1..the listed hard maximum,
not booleans. Lower limits fail safely, never silently truncate. Limits bound
input/work, not hard RSS or deadlines. Payload/algorithm versions are 1;
`min_size` is an integer 1..1000; embedding threshold is finite 0..1; exact
threshold is null and semantically ignored. Jobs stores no claim text, vectors,
paths, credentials or notification bodies.

One active rebuild per owner spans queues and methods. Equivalent effective
settings reuse it; conflicting active settings or retained keys return 409 with
an authorized existing reference. No replacement is queued. HTTP 202 returns
`job_id`, `job_uuid`, `status_url`, `reused`; `Location` equals `status_url`.
Uncertain/lost admission acknowledgement is sanitized 503, never synchronous
execution. Quotas/rate limits still apply; ingress includes replays.

Own status: `/api/v1/jobs/owned/{job_uuid}`. An authorized foreign-owner platform
administrator receives `/api/v1/jobs/{job_id}?domain=claims&job_uuid={job_uuid}`
only after Jobs role/domain authorization. A Claims permission does not grant
Jobs controls. Missing/foreign owned UUIDs return 404; either link remains bound
to the exact UUID through archival and numeric-ID reuse, then returns 404 after
Jobs retention. Never substitute a numeric-only link or reconstruct status in
Claims.

Optional async-only `Idempotency-Key` is 1..128 printable ASCII characters with
no surrounding whitespace. Jobs stores only its digest for 86400 seconds (24
hours), with at most 128 retained aliases per logical Job. Existing aliases
replay at the ceiling; a new alias is quota-rejected without mutation. A retained
key replays terminal/live/archive work; differing settings conflict. An expired
key is fresh but can still coalesce with equivalent active work. A keyless
retry after completion can create a new rebuild: transport exactly-once is not
promised. Request receipts and domain publication receipts have different
retention rules.

## Rollout

1. Keep ingress quiescent for maintenance. Deploy Jobs resource migrations and
   Media migration 27 through the normal migration workflow before producers.
   Retain `job_resource_heads`, `job_resource_request_receipts`,
   `claims_cluster_publication_heads` and `claims_cluster_publication_receipts`.
2. Deploy compatible workers first. Inspect the Claims worker's existing
   supported-type acquisition allowlist (`CLAIMS_SUPPORTED_JOB_TYPES`); old
   workers must not acquire `claims_rebuild_clusters`. All processes must use
   one Jobs authority/database and compatible algorithm/embedding collection
   configuration. Independent Jobs stores are unsupported.
3. Run owner/status/backend probes on isolated qualification data: owner-safe
   reads and authorized foreign UUID links, unchanged Jobs admin denial, one
   matching admission/replay, conflicting settings, actual WorkerSDK completion,
   nonadmin Media/Jobs owner scope, and SQLite plus official restricted-role
   PostgreSQL tests. Missing fixture results are not parity evidence.
4. Drain in-flight synchronous rebuilds and all old/noncooperating vector SDK
   clients. No mixed local/Jobs writer interval is supported. Check actual
   handlers/processes, not only queue counters.
5. Uniformly enable the two producer flags across every API process, then reopen
   ingress. Keep compatible workers active. Do not toggle one owner/process
   independently while work is active. A captured async request may still admit
   after a flag change; there is no atomic toggle guarantee.

The qualified vector profile is fresh Rust Chroma 1.5.5/MD5 quiescent storage,
including proven never-persisted buffered/empty logical reads. Materialized
targets use the SDK after structural validation; missing materialized files are
not repaired. Unknown profiles fail closed, not `no_embeddings`. Existing SDK
bookkeeping may change: existing-only does not mean universally byte-read-only.
Canonical clients coordinate whole native/thread SDK lifetimes with a lock
sidecar outside the owner tree. Use fresh spawn/exec after fork; inherited
sessions fail closed. Failed SDK shutdown quarantines the handle/exclusion until
process restart. Do not unlink lock files or force-unlock a potentially live
SDK. Windows, arbitrary external filesystem deletion, uncooperative SDK clients,
all SDK versions, hard memory/deadline and throughput guarantees are unqualified.

## Rollback

1. Quiesce ingress using existing operator controls; keep async true while
   doing so. Existing Jobs pause/drain may stop new admission/acquisition as
   appropriate; it does not stop an in-flight HTTP request or physical writer.
2. Drain or cancel through existing Jobs controls, and wait for actual
   physical handlers/threads/processes to finish. A terminal status alone is
   not proof of physical drain. Cancellation cannot stop an already-running
   synchronous thread, undo committed output, or atomically fence a Media
   transaction after its last Jobs check. A newer applied/skip domain generation
   fences older output, but this is not a substitute for rollback drain.
3. Only after ingress and physical work are settled, uniformly flip the async
   routing flags to false across all producers; disable workers as appropriate.
4. Only then reopen synchronous ingress. **false starts LEGACY SYNC; it is NOT stop-admission**.

Retain migrations and immutable history: never drop heads or receipts. There is
no new Claims shutdown state. Leave Stage 3 fallback removal and metrics cleanup
to their separately approved workstreams.

## Restore

Quiesce **both async and legacy writers**, including in-flight API requests,
Jobs handlers and vector clients, before a coordinated paired checkpoint.
Setting flags false alone is not a restore barrier. Stop affected workers and
retain ingress quiescence until the pre-start evidence below is approved.

Record checkpoint provenance outside the databases: checkpoint identifier,
consistent Jobs and each affected owner Media backup, migration/config versions,
resource authority/counters, applied head UUID/generation and a complete
per-owner publication receipt inventory (UUID, authority, generation,
fingerprint and counts), plus integrity hashes and acquisition procedure.
Include associated embedding configuration/storage provenance. Validate the
backup method's transaction consistency and physical drain; independently
copied live files are not a coordinated checkpoint.

Before restarting producers or workers, operators must establish:

- checkpoint provenance proves Jobs and Media belong to the same quiescent pair;
- authority equality for each `(claims, owner, cluster-rebuild)` and domain head;
- Jobs counter >= domain watermark; counters/authorities were not reset;
- exact equal-generation UUID identity matches the immutable Jobs history or
  checkpoint manifest, never merely a reused numeric ID (a terminal Jobs head
  can have null active UUID);
- receipt completeness against the independent checkpoint inventory, including
  historical receipts even if the head and newest receipt still agree.

**matching UUID alone is insufficient**: a same-authority old Jobs backup can
rewind counters; a Media backup can lose historical receipts without changing
the newest head. Runtime guards reject some inconsistent heads/identities but
cannot prove that every independent restore is detected. Test-only evidence
checks are not a production restore validator. An unprovable pair **stays disabled**
for separately reviewed recovery: no automatic restore repair, authority rebind,
counter reset, fence deletion or job-ID comparison to unblock it.

## Outcomes And Incident Triage

An absent owner DB is permanent `claims_cluster_owner_database_missing`, not a
skip: maintenance never provisions its directory/schema. Explicitly reactivating
that failed Job after external owner DB creation can proceed under existing
resource arbitration; a new request is normally preferred. Missing Watchlists
schema skips best-effort evaluation without bootstrap. Access errors are
sanitized failures/diagnostics, not evidence of absence.

Exact empty input replaces clusters with empty output. Legitimately absent
embeddings in an existing owner DB preserve clusters with a durable
`skipped/no_embeddings` receipt and advance the generation. Partial coverage
keeps current partial semantics/counts. Storage/vector failures are never
successful empty/no-embedding outcomes. Foreign old references fail permanently
without cross-owner mutation; transfer repair is separate. Snapshot drift is
retryable through Jobs, not a Claims retry loop.

Crash after commit but before Jobs completion replays the same receipt and
cluster IDs. Only a newly committed publication evaluates Watchlists; crash
between commit and evaluation can lose that best-effort side effect. There is
no durable notification delivery promise or second worker. A fresh rebuild may
change cluster IDs and invalidate links/subscriptions; dangling relationship
metadata may remain. No stable-ID, relationship repair or automatic subscription
remapping migration is included. Receipt replay does not restore
older output after a newer publication.

Publication receipts live for the owner dataset independently of Jobs pruning;
Jobs resource heads retain their high-water counters. Unexpired Jobs request
receipts pin exact replay targets, not domain receipt history. Do not compact
domain receipts by guessing a Jobs TTL. Explicit owner dataset deletion removes
domain history with the dataset.

For operational failures use existing Jobs status/admin controls and sanitized
failure codes. Never log raw SQL, database URLs, claim text, vectors, keys or
provider/notification bodies. The bounded oldest-candidate recovery window may
underfill or recover zero under contention without post-row-lock refill;
eligible bulk maintenance can retain its selected heads through commit. This
qualification is not a fairness/throughput certification.

## Qualification And Release Boundary

Run the approved Claims/Jobs, storage, restricted PostgreSQL, Watchlists,
configuration/router and Backlog commands with `RUN_JOBS=1`; retain skips with
actual fixture reasons and warning category/location evidence. Compile, configured
full-file Ruff, Bandit and applicable hooks use actual branch merge-base touched
paths plus new files. Inherited debt is not zero total findings; require zero
new findings/errors without suppressing rules or weakening gates.

The controller owns the single repository-wide pytest attempt after Task11's
final source commit; that attempt and independent task/whole-branch reviews
remain release prerequisites. This runbook is not deployment authorization.
No push/PR/merge/rebase is implied. A materially AI-authored PR needs the human
requester's own Change summary explaining what changed and why; required checks
and merge-queue policy remain mandatory.
