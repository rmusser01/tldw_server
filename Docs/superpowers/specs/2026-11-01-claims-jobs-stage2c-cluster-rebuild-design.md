# Claims Jobs Stage 2C: Cluster Rebuilds

**Status:** Approved by the requester on 2026-11-01; revised implementation plan approved and execution authorized on 2026-11-01
**Date:** 2026-11-01
**Task:** TASK-15460
**Baseline:** dev `54b321dcec8cdbb6358581f42c1548696260570b`
**Parent:** [Claims Jobs operational design](2026-06-24-claims-jobs-operational-control-plane-design.md)
**ADR check:** A new ADR is required for resource-scoped Jobs admission and
domain publication fencing. [ADR-094](../../ADR/094-jobs-resource-admission-and-domain-publication.md)
is Accepted with this design. ADR-003 governs Jobs ownership; ADR-058 continues
to govern Jobs completion identity. Neither accepted decision is superseded.

## Intent And Approved Choices

Move owner-scoped exact and embedding cluster rebuilds onto shared Jobs without
building a second execution system inside Claims. Administrators gain durable
status, retries and existing Jobs controls; Claims retains clustering and data.

The requester approved these product choices in the conversation:

- One active cluster rebuild per owner across both methods.
- Equivalent effective settings reuse the active Job; different settings return
  HTTP 409 and leave the active Job unchanged. Do not queue a pending replacement.
- A new request after the previous rebuild finishes can create a fresh Job.
- Default behavior remains synchronous HTTP 200 with existing result fields.
- Opt-in Jobs mode returns HTTP 202 with a Job reference and status link.
- A successfully admitted Job never falls back to local execution.
- All admission and execution lifecycle mechanisms belong to shared Jobs.

The requester approved this revised written design, including its engineering
proposals, after the follow-up corrections in commit `92ac607403`. The shared
resource protocol, publication receipts, read-only owner status route, optional
request key and initial resource ceilings were engineering proposals, not prior
product choices. Approval authorizes implementation planning, not implementation.

## Current Implementation

`endpoints/claims.py` delegates `/api/v1/claims/clusters/rebuild` to
`claims_service.rebuild_claim_clusters`. The service checks Claims administration,
resolves the actual owner and method, runs the rebuild synchronously, and then
best-effort evaluates watchlist notifications. Only platform administrators may
override another owner's ID.

The exact and assignment rebuild operations in
`DB_Management/media_db/runtime/claims_cluster_*_rebuild_ops.py` replace existing
owner clusters within Media DB transactions. Embedding calculation reads Media
DB claims and a per-owner Chroma collection before replacing assignments.
Rebuilds can change cluster IDs and remove cluster relationships today.

`claims_job_contracts.py` has media rebuild, notification, alert, export and
review-metrics types, but no cluster rebuild type. `claims_jobs_worker.py` uses
WorkerSDK and handlers generally offload synchronous work with `asyncio.to_thread`.

`JobManager.admit_job` already has typed admission and transactional
`OwnerScopeAdmissionPolicy`. Its scope includes domain, queue, job type and owner;
its active limit rejects rather than semantically reusing a different request.
The ordinary idempotency index includes queue and job type and is not limited to
active jobs. Neither mechanism alone implements the approved policy.

WorkerSDK fences Jobs completion with UUID and acquired lease. That does not
undo a Media DB commit or stop a thread after cancellation. Jobs and Media DB
can be separate databases; they cannot share an assumed transaction.

`/api/v1/jobs/{job_id}` currently belongs to `jobs_admin.py`, whose router requires
the admin role. Claims administration also accepts `CLAIMS_ADMIN` permission;
it must not silently grant that principal global Jobs administration.

## Goals, Non-Goals And Anti-Goals

Goals are safe owner isolation, atomic admission, replay-safe domain publication,
observable asynchronous outcomes, bounded work, backend parity and controlled
rollout with unchanged disabled-mode behavior.

Non-goals are Stage 3 fallback removal, metrics changes, scheduled cluster rebuilds,
automatic pending refresh, a new clustering algorithm, stable cluster-ID migration,
new embedding generation, a distributed Media DB/Chroma snapshot, and a new Claims
dashboard or queue-control API. Do not publish the old PR's local closeout as part
of this feature.

Failure conditions include Claims-owned queues or lease loops, list-then-create
admission, using the media rebuild job type for cluster work, mutating an active
Job's settings, retries rebuilding already-published clusters, leaking claim text
into Jobs, or reporting storage failures as successful `no_embeddings` outcomes.

## Approaches

| Approach | Tradeoff | Decision |
| --- | --- | --- |
| Reuse equivalent active work; reject conflicting settings | Predictable cost and no replacement backlog; callers must explicitly retry after completion | Selected by requester |
| Queue every request serially | Preserves every request but permits redundant expensive work and stale settings | Not selected |
| Keep one latest pending request | Provides automatic refresh but adds pending-replacement policy and recovery | Not selected |

Use existing Jobs admission infrastructure with an opt-in resource policy rather
than implementing the selected approach in Claims. A permanent owner-only
idempotency key would incorrectly reuse a completed job forever. A key based only
on settings would admit different settings concurrently. An in-process mutex
would not protect multiple API processes or workers.

This is a project-sized change: a shared Jobs capability, Claims integration and
cross-backend verification. The implementation plan must keep those boundaries
separately reviewable, starting with Jobs. No implementation starts with this spec.

## API And Authorization

Keep the current query parameters: `method`, `min_size`, `similarity_threshold`
and optional platform-admin `user_id`. Resolve the owner from authenticated context;
never from a raw DB path, blank value or synthetic fallback. Check authorization
before looking up an existing request or exposing its reference.

| Mode or outcome | Response |
| --- | --- |
| Jobs disabled | Existing HTTP 200 result, including current best-effort watchlist fields |
| Newly admitted or equivalent active Job | HTTP 202, `job_id`, `job_uuid`, `status_url`, `reused` |
| Same retained request key replayed | HTTP 202 reference, even if that Job has since become terminal |
| Conflicting active settings or request-key reuse | HTTP 409 with stable code and authorized existing Job reference |
| Invalid input | Existing validation behavior; validate all effective settings before admission |
| Unauthorized owner or operation | HTTP 403, no admission and no foreign Job reference |
| Jobs quota/rate rejection | Existing Jobs/API quota mapping; not a false active-conflict response |
| Jobs storage/admission unavailable or outcome uncertain | Sanitized HTTP 503; never execute locally |

HTTP 202 is acceptance, not a promise that status is still queued when the client
reads it. Return `Location` equal to `status_url`; do not copy mutable Jobs status
into a Claims-owned lifecycle field. Use typed response models and document the
200/202/409 alternatives in OpenAPI.

Add a minimal read-only status projection in the Jobs module, proposed route
`GET /api/v1/jobs/owned/{job_uuid}`. It authenticates the caller and reads only a
Job whose stored owner equals the caller's canonical user ID. It exposes UUID,
numeric ID, domain/type, canonical status, safe progress/count results and stable
failure codes, but no arbitrary payload, raw errors, paths or secrets. Unowned
UUIDs return 404. Domain result projection must be explicitly allowlisted by Jobs,
not an unrestricted serialization of other domains' result dictionaries.

No queue-control permission is conferred by this read route. Existing Jobs admin
RBAC and control endpoints remain unchanged. Extend the admin detail read with
an optional `job_uuid` query precondition, using the existing manager's exact-UUID
active/archive lookup. A platform administrator targeting another owner receives
`/api/v1/jobs/{job_id}?domain=claims&job_uuid={job_uuid}` only after the same Jobs
read/domain authorization succeeds; otherwise reject the async request before
admission. Never return a numeric-only admin status link: SQLite can reuse a
pruned numeric ID even while its original UUID remains archived. With the UUID
precondition, return only that logical Job or 404, never its numeric replacement.
Legacy detail calls without the optional precondition retain their behavior.
Permission-only Claims administrators retain their own-owner rebuild ability and
use the owned read route. No Claims status endpoint or Claims proxy for Jobs
controls is added.

The owned read route must live in a separate Jobs-owned authenticated router,
not inherit `jobs_admin.py`'s global admin dependency. Resolve UUIDs in active and
archived storage with the exact owner predicate; preserve UUID identity rather
than resolving an old link to a reused numeric ID. Initially allowlist only the
cluster-rebuild safe result projection; other job types are not exposed by
accident. Register the route through the usual router groups and ingress policy.

An optional `Idempotency-Key` header is accepted only for async admission:
1..128 printable ASCII characters, no surrounding whitespace, no raw-key logging
or storage. Jobs retains an owner/resource-scoped digest receipt for 24 hours by
default, including alias keys that reused an already-active Job. Same key and
settings replays the same UUID; different settings conflicts. Receipt retention
and archive protection belong to Jobs. An expired key is a fresh request.
Without a key, each HTTP request is new, but an equivalent active Job still
coalesces. A lost response followed by a keyless retry after completion may create
a new rebuild; do not promise transport exactly-once semantics in that case.
Bound retained key aliases to 128 per logical Job; existing aliases still replay,
but a new alias over that ceiling receives a quota response without changing the
Job. Keyless active reuse requires no alias receipt.

## Immutable Contract And Effective Settings

Introduce `claims_rebuild_clusters` under domain `claims`, using the configured
Claims Jobs queue and the existing worker. Payload version 1 contains only:

```json
{
  "version": 1,
  "owner_user_id": "1",
  "method": "embeddings",
  "min_size": 2,
  "similarity_threshold": 0.85,
  "algorithm_version": 1
}
```

Owner IDs use existing strict routable-owner validation. Method is normalized to
`exact` or `embeddings` before admission, including resolving its configured
default. `min_size` is an integer in 1..1000, not a boolean. An embedding threshold
is finite, in 0..1, and resolved to a concrete value before admission. For exact
mode the threshold is always null and excluded from semantic comparison.

Canonical settings are version, method, min_size, effective threshold and
algorithm_version. The resource scope separately contains owner and domain.
Jobs persists a versioned canonical encoding and SHA-256 fingerprint, and checks
the canonical settings as well as the digest before reuse. Operational batch
size, retry budget and request tracing are not semantic settings. The worker uses
the admitted settings, not newly changed process defaults. Queued work requires
compatible algorithm and embedding collection configuration on API and worker
deployments; do not silently reinterpret it after a deployment.

No claim text, vectors, paths, collection credentials or notification bodies enter
Jobs payloads or results. Jobs assigns resource identity/generation metadata
outside the client payload; callers cannot forge it or select another resource.

## Shared Jobs Resource Admission

Add an opt-in resource policy to the shared admission pipeline and backend
operations. Proposed public facade: `admit_resource_job`, with typed outcomes
created, active-reused, request-replayed, active-conflict, request-conflict or
ordinary admission rejection. Names may follow existing contracts in the plan;
the following transaction guarantees may not be weakened.

The unique resource is `(domain="claims", owner_user_id, resource="cluster-rebuild")`.
Queue, method and job-type version are deliberately not part of uniqueness.
Admission serialization is database-backed: SQLite `BEGIN IMMEDIATE`; PostgreSQL
a shared resource-head row lock and the existing RLS-aware operation patterns.

Jobs owns a persistent resource-head record containing an opaque authority UUID,
monotonic generation counter and optional active Job UUID. New Jobs carry their
immutable resource identity, generation and canonical settings. The resource
record is Jobs admission state, not a Claims table. It contains no source data.

Register the new job type as resource-required in Jobs. Bare `create_job` or
`admit_job`, batch import, admin reactivation and future versioned type aliases
must not create an unbound cluster-rebuild Job that evades this policy. Claims
declares its contract; shared Jobs enforces the registered admission requirement.
Other job types retain their existing opt-out behavior.

Inside one transaction, after caller/owner authorization and canonical validation:

1. Lock or safely create the head, retaining its authority and counter forever
   across ordinary job/archive/receipt pruning.
2. Resolve a retained request receipt first, checking full scope and settings.
3. Inspect the active UUID against the authoritative locked Jobs row. Queued and
   processing jobs, including future retries and paused queues, occupy the slot.
4. Equal canonical settings return that Job; unequal settings return a conflict.
   Never change its payload, queue or retry controls to fit the new request.
5. If no active Job exists, apply ordinary queue policy, quotas, fair-share,
   payload safety and admission bookkeeping; allocate the next generation,
   insert the Job and attach request receipt/head atomically.

Replays do not consume another Jobs submission quota or emit another creation
event. HTTP ingress rate limiting still applies to every request. Do not bypass
existing quotas or fork an independent create pipeline. Any ambiguous legacy row,
missing active target or authority corruption fails closed for this resource.

All transitions that reactivate an opted-in Job, including admin requeue/retry,
must participate in this protocol; no admin helper may insert a second active
resource Job. Reactivation retains that Job's logical generation. Published Jobs
replay their domain receipt rather than forcing another rebuild. A new rebuild
requires a new request, not requeueing an already-published logical Job.

Terminalization clears only the matching active UUID, with the same locked-row
identity discipline as ADR-058. Completed, failed, cancelled and quarantined are
terminal Jobs states. A stale completion cannot clear a successor's pointer.
Archive/prune/repair and receipt retention must preserve counters and identities;
active or receipt-referenced jobs cannot disappear under the protocol.

All resource-mutating Jobs paths use one lock order: resource head, then exact
Jobs rows in stable order. Existing terminal/requeue implementations that lock a
job first must be adapted for opted-in rows, revalidating any preliminary lookup
under those locks. No row-then-head mutation is allowed to deadlock with admission.
Ordinary acquisition/renewal that does not touch a head need not lock that head.
In the retention rule above, "receipt-referenced" means an unexpired Jobs request
receipt, not a Media DB publication receipt. Publication history does not pin
Jobs rows indefinitely. After ordinary Jobs retention expires, a status link may
return 404; no Claims endpoint reconstructs a second Jobs history.

The generation is not a numeric job ID, wall-clock timestamp, lease or retry
counter. Queue changes cannot evade admission. Independent owners may proceed
concurrently, subject to backend/global quotas.

## Domain Publication, Replay And Stale Workers

Logical single-active admission does not guarantee only one physical thread is
running: a cancelled or lease-expired thread may finish after a successor starts.
Do not add Claims lease renewal, job polling loops or cancellation state to fix
that. Use Jobs identity checks plus Media DB-owned publication fencing.

Add two source-free Media DB structures through existing migrations and helper
bindings, with SQLite/PostgreSQL parity and PostgreSQL owner isolation:

- An owner cluster-publication head: Jobs resource authority UUID, highest
  applied generation and that generation's Job UUID. This is domain output
  ordering, not a job status table.
- A publication receipt keyed by `(owner_user_id, job_uuid)`: authority,
  generation, settings fingerprint, bounded result counts and applied outcome.
  It has no queued/running/retry/lease fields and is never a queue to execute.

Retain publication receipts for the lifetime of this owner feature's data in
Stage 2C, independently of Jobs retention; they contain only bounded metadata and
counts. Ordinary Jobs pruning neither consults nor removes them. Explicit owner
data deletion removes them with the domain dataset. Future receipt compaction
requires its own proven replay horizon; no automatic compaction is part of this
stage. Accept the small per-rebuild history growth rather than guessing a safe TTL.

The worker validates row owner, payload owner, job type and Jobs-assigned immutable
resource metadata. It derives an existing-only Media DB session from the owner;
no absent database, directory, schema or Chroma collection is created by rebuild
maintenance. PostgreSQL worker scope uses the real owner plus explicit owner
predicates, not an unrestricted synthetic administrator.

In async mode, request dependencies must not eagerly call a creating Media DB
constructor before this decision. Resolve routing/authentication first and use
existing-only inspection; the disabled synchronous branch retains its current DB
dependency behavior. If the owner DB disappears before a worker opens it, fail
non-retryably with `claims_cluster_owner_database_missing`. This is not a durable
domain skip: no publication receipt or generation advance is claimed for storage
that does not exist. Jobs records the failure. An explicitly reactivated failed
Job may proceed if the DB later exists, subject to resource/fence checks; a new
request remains the normal way to request a new rebuild. This intentionally
differs from metrics maintenance's missing-database skip.

Before expensive calculation, read a publication receipt if present and return
it without rebuilding. Otherwise read a bounded owner-scoped input snapshot,
calculate clusters without holding a long database write lock, and attempt
publication through one DB_Management-owned transaction:

1. Serialize the owner publication head (row/advisory transaction lock on
   PostgreSQL; SQLite writer transaction).
2. Recheck the receipt; another attempt may already have published it.
3. Reject authority mismatch or reuse of an equal generation by a different
   UUID. If a higher generation has already applied,
   return `outcome=skipped, reason=superseded` without changing clusters.
4. Recheck the canonical digest of clustering-relevant source rows and ownership
   under the transaction. Include claim ID, media ID, chunk index and claim text,
   and detect additions/deletions/ownership changes. If it changed since the
   snapshot, roll back and let Jobs retry with `claims_cluster_source_changed`.
5. Validate the bounded existing claim assignments and memberships referencing
   clusters to be deleted, including references outside the current source set.
   A transfer completed before the snapshot can leave another owner's claim
   pointing to an old cluster. If any foreign reference exists, fail without
   changing clusters, memberships, assignments, publication head or receipt;
   use non-retryable `claims_cluster_foreign_reference`, not a source-change retry.
   Do not clear another owner's claim or silently leave a dangling reference.
6. Atomically replace the owner's clusters, membership and claim assignments,
   advance the publication head, and save the result receipt. No helper commits
   per cluster. Validate all representative/member IDs against the scoped source.

The reference guard belongs to DB_Management and must remain effective under
restricted-role PostgreSQL/RLS. An owner-filtered query returning no foreign rows
is not proof that no foreign references exist. Select an integrity guard or
constraint that enforces this check through commit without broadening the worker
role or returning foreign claim data. Bound guard/cleanup input sets as below;
if integrity cannot be established safely, fail closed before replacement.
Repairing ownership transfers or remapping their cluster references is separate
work; this rebuild does not acquire cross-owner mutation authority.

On PostgreSQL, the digest read is the defined publication input point, not a
promise to include every insertion until commit. Lock the bounded source Claims
rows and their Media ownership rows in deterministic order through replacement
and commit, using the existing transaction connection. Also cover rows whose old
cluster references are cleared. This prevents deletion, source edits or owner
transfer of a row actually being assigned between validation and commit. Use
owner predicates on writes as well as validation. Guard/cleanup input sets are
bounded; legacy oversized memberships fail safely rather than requiring an
unbounded lock or parameter list. New claims or incoming ownership that arrive
after the digest snapshot may be omitted until the next rebuild. Do not assign
or broadly update those unlocked arrivals. No global source-writer protocol or
table-wide PostgreSQL lock is required. SQLite's writer transaction gives a
stronger serialization point without changing this minimum cross-backend
contract. Include explicit concurrent insert, edit, delete and transfer tests.

If a crash follows commit but precedes Jobs completion, the retry reads the receipt
and returns the same result without deleting/recreating clusters or repeating
watchlist evaluation. If a newer publication commits first, older late attempts
cannot overwrite it. The receipt handles duplicate attempts of the same Job;
the generation handles different Jobs. No distributed transaction is claimed.

Embedding vectors are read once into a bounded execution snapshot and validated
for equal dimensions, finite components and nonzero norms. This is not an atomic
snapshot with Chroma or a promise to include claims added after calculation starts.
Preserve current clustering rules and ordering; do not silently change threshold
comparison, normalization or representative-selection semantics in this migration.

Legitimately absent embeddings preserve existing clusters and complete with a
durable skipped result. Partial missing embeddings retain current partial-coverage
semantics and expose counts. Legitimate skip publication also advances the domain
generation, so an older late attempt cannot supersede the newer skip decision.
Exact mode with no eligible claims retains its current empty-rebuild semantics.
Storage/access errors must not be collapsed into missing embeddings.

Before dispatch and publication, use a shared Jobs helper to validate current
UUID/worker/lease authority; any required helper belongs to Jobs. This is an
extra rejection opportunity, not an atomic cross-database cancellation fence.
Cancellation after that check can race a domain commit. Document that cancel is
best effort, does not undo committed output, and may be followed by receipt-backed
domain completion even while Jobs correctly rejects late completion. WorkerSDK
remains responsible for lifecycle transitions and renewal.

Resource authority UUIDs and publication heads cannot be independently reset.
The supported restore is a quiescent coordinated checkpoint of Jobs and the
affected Media databases, including heads and receipts. Before restarting these
producers/workers, operators must verify checkpoint provenance, matching resource
authorities and non-regressed Jobs counters against domain watermarks. Same-UUID
backups can rewind counters or lose receipts; UUID mismatch checks do not detect
all such restores. Automatic independent-restore detection/recovery is not
promised. A mismatched or unprovable pair stays disabled pending separately
reviewed recovery; do not delete the fence, auto-rebind authority or compare job
IDs to unblock it. Equal-generation/different-UUID publication is a permanent
corruption error, not a superseded success.

## Watchlists, Results And Failure Semantics

Only the attempt that newly commits a domain publication performs the existing
best-effort watchlist evaluation, through an existing-only Watchlists accessor
bound to the validated owner/backend. The async path must not call the current
creating `WatchlistsDatabase.for_user` constructor or any schema/bootstrap helper.
Missing Watchlists storage or required schema skips this side effect without
creating a database, directory or table; it does not fail or repeat the committed
rebuild. Access/storage errors log a sanitized diagnostic rather than being
misreported as absent storage. The disabled synchronous path keeps its existing
access behavior. A crash between publication and this side effect can omit
evaluation; Stage 2C does not promise durable notification delivery or create a
second side-effect worker. Existing delivery Jobs stay unchanged. Replay returns
the immutable publication counts without recalculation.

Fresh rebuilds may change cluster IDs and invalidate old subscriptions/links as
they do today. Preserve and document this behavior, test it explicitly, and do
not imply stable identities or automatic subscription remapping. A durable
identity migration is separate work, not an accidental side effect of retries.

Results contain only bounded counts, method, effective threshold where relevant,
outcome and stable reason codes. Claims owns the receipt; Jobs owns lifecycle.
Use ClaimsJobError's existing `retryable`, `failure_code` and optional backoff
attributes. Transient DB locks, connection deadlines, Chroma availability and
source-snapshot conflicts are retryable through Jobs. Invalid owner/payload,
malformed vectors, resource limits and unsupported algorithm are permanent.
Foreign references to clusters selected for deletion are the non-retryable
integrity failure defined above; fixing them is not an automatic retry side effect.
An absent owner DB is the non-retryable failure defined above, not a skip or
database provisioning. Legitimate missing embeddings in an existing DB remain
receipt-backed domain skips.
Unknown storage errors are failures, never successful empty output. Preserve
metadata-only logging; do not log DB URLs, SQL, claim text or raw provider errors.

## Resource Bounds And Configuration

Proposed routing flag: `CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED=false`, conjunctive
with `CLAIMS_JOBS_ENABLED`. Reuse `CLAIMS_JOBS_WORKER_ENABLED`, queue and existing
worker lease/backoff settings. Add `CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD=3`
using the current bounded retry parser. Worker-disabled means visible queued work,
not local fallback. Do not create a separate Claims worker runtime.

Initial async-only engineering defaults, exposed as documented positive bounded
configuration and enforced without silent truncation:

| Limit | Initial proposal |
| --- | --- |
| Exact source claim count | 50,000 |
| Embedding source claim count | 5,000 |
| Total retained UTF-8 source text | 32 MiB |
| Embedding width | 4,096 finite components |
| Total retained embedding components | 2,000,000 |
| Similarity component comparisons | 50,000,000 |

Count and text ceilings are checked while reading pages (at most limit+1), not
after building an unbounded list. Batch Chroma and SQL reads with bounded key
sets; count computation budget inside clustering loops. Over-budget work fails
before replacement and leaves existing clusters intact. These limits bound
inputs/work, not an exact process RSS or wall-clock deadline. Tests inject low
limits. The plan must define the concrete config names and validators consistently
with these semantics, without introducing a new generic configuration system.

## Rollout And Rollback

Deploy Jobs migrations/resource admission, Media DB migrations, and workers before
enabling any producer. Verify both backends, owned status reads and Jobs admin
permissions. All processes must use the same Jobs database/resource authority and
embedding configuration. Old workers must not acquire the new unsupported type;
use the existing Jobs job-type acquisition allowlist mechanism during deployment.

Enable producer routing uniformly only after in-flight synchronous rebuilds stop.
Do not mix flags across API processes or switch an owner between local/Jobs mode
while work is active. Admission is global across queues, but incompatible worker
versions and independent Jobs databases are not a supported rollout topology.

Rollback stops new producers, drains/cancels existing Jobs through Jobs controls,
waits for physical handlers to settle, then disables workers/async routing before
resuming synchronous calls. A cancelled status alone does not prove a thread
has stopped. Do not drop resource heads or publication receipts during rollback.
Keep the local path for this opt-in stage; removal belongs to Stage 3 after
recorded parity evidence, not to TASK-9935.2's distinct metrics-only cleanup.

## Verification Contract

Use test-first regression coverage in the implementation, including these cases:

- Concurrent matching admissions from separate managers/processes create one Job;
  conflicting methods/thresholds get one stable conflict, across queues as well.
- Canonical default/explicit settings compare equally; exact ignores threshold;
  nonfinite values, booleans, invalid owners and unsupported versions are rejected.
- Optional request-key retry survives completion/archive for its retention window;
  alias-key replay, conflicting key settings and keyless fresh requests differ.
- Quotas, queue policy, RLS, encryption, safe payload limits and creation events
  remain correct; duplicates create no additional event or submission count.
- Terminalization, retry, manual requeue, quarantine, pruning and queue changes
  cannot violate the resource invariant or reduce/reuse its generation.
- Exact and embedding API -> Jobs -> WorkerSDK -> Media DB flows return the
  approved 200/202/409 contracts; owned status lookup and privileged override
  checks do not broaden Claims or Jobs administration.
- Owner and cross-owner admin status links stay UUID-bound across archival,
  numeric-ID reuse and pruning. An admin link returns its original Job or 404,
  never a replacement; the optional precondition does not bypass admin/domain
  authorization or change legacy reads without that precondition.
- Missing DB/collection does not provision it. Missing DB fails without a domain
  receipt; explicit reactivation after DB creation is distinguished from replay
  of a published Job. Owner, foreign representative,
  membership and team/shared visibility tests prove exact owner scoping.
- Crash after domain commit replays the same receipt/cluster IDs; duplicate
  leases do not republish; older cancelled/expired threads cannot overwrite a
  newer applied generation; cancellation/commit races have the documented result.
- Concurrent source edits/deletes/transfers cannot invalidate assigned rows
  through commit, while post-snapshot insertions can await a later rebuild.
  Snapshot changes, atomic replacement failures and over-budget vectors leave
  prior clusters untouched; exact empty and embedding missing/partial outcomes
  preserve the stated semantics. No storage failure becomes `no_embeddings`.
- Transfer-before-snapshot and transfer-followed-by-source-retry tests cover old
  foreign assignments/memberships under restricted-role PostgreSQL as well as
  SQLite. The integrity guard cannot mistake RLS-hidden references for absence;
  failure leaves domain data, head and receipts unchanged, with no foreign writes.
- Watchlist evaluation is best effort only for a new publication, never a reason
  to rebuild committed clusters. Existing unstable-ID behavior is not hidden.
- Existing Media DB with missing Watchlists storage or schema causes no file,
  directory or table creation during async evaluation. Backend/owner pinning,
  sanitized access-error handling and unchanged disabled-mode behavior are tested.
- SQLite and official restricted-role PostgreSQL fixtures cover admission,
  publication, isolation and recovery. Skip only when the official fixture says
  PostgreSQL is unavailable; do not create a custom database harness.
- Property tests exercise normalization, replay, interleaved generation ordering
  and resource ceilings; fault tests cover ack loss, retained/expired request keys,
  bounded aliases, publication history independent of Jobs pruning, lock-order
  races and mandatory coordinated restoration including same-UUID old backups.

Run scoped Claims/Jobs/storage/config/API regressions, syntax/lint, OpenAPI and
route-policy contracts, and touched-scope Bandit before implementation completion.
Required remote checks, human Change summary and merge rules remain mandatory.

## Builder Discretion And Approval Boundary

The builder may choose helper/class names, migration numbers, config key spelling,
index names and test file layout consistent with repository patterns. The builder
may not substitute queue-local dedupe, reduce owner checks, drop replay receipts,
weaken cancellation caveats, change approved HTTP behavior, or place Jobs lifecycle
inside Claims. If the generic Jobs prerequisite grows beyond this bounded
resource policy, stop and revise the design rather than importing a second engine.

## Independent Builder Check

The independent builder check identified four material questions: PostgreSQL
source-write protection through commit, missing-DB outcomes without storage,
publication receipt retention versus Jobs pruning, and same-UUID old-backup
restoration. The clarified contracts above answer all four. The same reviewer
rechecked the revision and reported "nothing important left".

A follow-up review found three additional gaps: foreign
cluster references left by pre-snapshot transfers, schema creation by the existing
Watchlists evaluator, and numeric-only cross-owner admin status links. The
requester approved correcting all three. This revision adds explicit fail-closed
reference integrity, non-creating Watchlists access and UUID-bound admin reads,
with implementation regression requirements for each. A fresh independent
focused recheck reported "nothing important left". These design checks are not
evidence that the new contracts have been implemented or tested.

This is design review, not verification of an implemented resource protocol.
The requester approved the revised written design and its engineering proposals
on 2026-11-01. The implementation plan is tracked in TASK-15283 at
[the Stage 2C plan](../plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md).
That plan still requires review before implementation; no unreviewed execution
or merge is authorized by design approval.
