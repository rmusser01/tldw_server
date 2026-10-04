# Embeddings Stage 3: Durable Store And Jobs Runner

Date: 2026-10-02
Revised: 2026-10-04
Task: TASK-13450 (canonical replacement for archived, colliding TASK-13421)
Status: Consolidated written spec approved by requester on 2026-10-04.
Predecessor: TASK-12973, completed through merged PR #2790.

## Purpose And Scope

Build a dormant internal durability foundation for the Embeddings workflow. Jobs
owns the root operational lifecycle; Embeddings owns execution detail. The
foundation must be independently testable before any production entrypoint uses
it. This spec consolidates the approved architecture, persistence, execution,
recovery, verification, and the subsequent review corrections.

In scope:

- An additive domain-neutral prepared Jobs capability and exact identity proofs.
- An Embeddings repository on the configured Jobs SQLite or PostgreSQL database.
- Immutable reference-only source manifests, pinned generations, bounded batches,
  provider attempts, output receipts, and a typed transactional operations outbox.
- An injectable runner, resolver, provider executor, sink, accounting boundary,
  and reconciler, exercised with certified test doubles.
- Explicit migrations, concurrency/security tests, and compatibility evidence.

Out of scope:

- Public or admin endpoints, production admission, worker registration, recurring
  reconciliation, startup migrations, and changes to application lifespan.
- Migrating the inline API, media Redis pipeline, vector-store batches, re-embed
  services, or compaction. These remain later workstream stages.
- Production billing integration, new billing policy, real source/sink adapters,
  provider implementation changes, and removal of legacy code or rollback flags.
- Exactly-once provider execution, per-item Jobs fan-out, and a second durable DLQ.

The foundation is installed but inert. Imports and normal application startup
must not create its tables, admit roots, start workers, or resolve content.

## ADR Check And Alternatives

ADR required: yes. This establishes durable persistence and execution/recovery
ownership rules. Proposed governing record:
[ADR-060](../../ADR/060-embeddings-durable-workflow-foundation.md).

[ADR-003](../../ADR/003-jobs-vs-scheduler-default.md) governs Jobs selection;
[ADR-022](../../ADR/022-embeddings-api-and-media-pipeline.md) remains governing for
the unchanged API and Jobs-root/Redis-stage media pipeline;
[ADR-051](../../ADR/051-retry-backoff-schedules.md) governs retry schedules.
This foundation supplements ADR-022; it does not activate a replacement pipeline
or supersede its accepted rationale.

| Approach | Disposition |
| --- | --- |
| Jobs root plus Embeddings detail tables in the configured Jobs database | Selected: shared durability backend, clear lifecycle ownership, and transactional fence validation. |
| Independent Embeddings database and worker lifecycle | Rejected: additional operational authority and cross-database checkpoint fencing. |
| Durable Jobs record per item or internal stage | Rejected: unnecessary fan-out, retry/control ambiguity, and increased state volume. |

## Component Boundaries

| Component | Responsibility | Depends On |
| --- | --- | --- |
| Jobs prepared capability | Exact owner/root identity, acquisition, live lease proofs, continuation, retry/terminal dispositions, and retained operation acknowledgement proofs | Jobs-owned contracts and database abstractions |
| Embeddings repository | Manifests, generations, attempts, checkpoints, operation claims, and retention | DB_Management abstractions and transaction-enlisted Jobs fence validation |
| Durable coordinator | One bounded batch per acquisition; preparation, pinned execution, and checkpoint orchestration | Narrow repository and adapter ports |
| Source resolver | Authorize and retrieve one bounded batch of exact source revisions, preserving item order | Injected source-specific authority |
| Provider executor | One reported provider dispatch with a certified embedding-space identity | Injected provider adapter |
| Output sink | Bounded batch lookup/writes, authoritative receipt absence, first-accepted values, sealing/closure, and scoped cleanup | Injected sink adapter |
| Accounting boundary | Atomically ordered reservation and absorbing terminal settlement per attempt | Injected boundary-owned accounting implementation |
| Reconciler | Resolve existing operation intents and receipts; never compute embeddings | Repository, Jobs capability, sink, and accounting ports |

New domain SQL and migrations live under `app/core/DB_Management/`; Embeddings
coordinates through a repository interface. Generic Jobs code must not import
Embeddings schemas or query Embeddings tables. No general plugin registry or
cross-domain workflow engine is introduced.

Reuse the Stage 2 preparation and vector-processing boundaries where their
contracts fit. Execute one pinned provider attempt, not the inline coordinator's
automatic fallback path. Disable shared cache reads/writes unless an adapter can
certify the pinned embedding-space identity; the initial fake-tested foundation
uses a cache that always misses and performs no shared writes.

## Identity, Sources, And Generations

Durable workflow and attempt IDs are canonical UUIDv4 values. They are separate
from the inline trace's `emb-wf-...` identifier contract. A Jobs payload contains
only `workflow_id`; the fixed new root scope is
`embeddings/workflow/embedding_workflow`, distinct from legacy media roots.
Queue admission still requires explicit Jobs configuration/capability approval;
this stage does not enable that queue in production.

Every persisted row carries a nonblank, exact `owner_user_id`. Root identity is
the immutable Jobs UUID plus owner, domain, queue, job type, idempotency key, and
exact payload. Integer Jobs IDs are lookup hints only. A previously bound root
must never be replaced, recreated, or reassigned during resume.

Admission explicitly selects `ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT` for this new
root scope. Persist the selected expiry policy, retry/quarantine controls, and
no-progress limit as immutable admission facts. Validate the stored controls on
idempotent admission replay, binding, and resume, as well as the root identity.
A matching payload or idempotency key with different controls is a typed conflict,
not a reusable root.
Do not inherit `admit_job`'s `CONSUME_RETRY` default, silently repair an existing
root, or copy Admin Webhook origin/identity to obtain retry-neutral expiry. Legacy
Jobs defaults, existing roots, and Admin Webhook restrictions remain unchanged.

The manifest is ordered and immutable: `(item_index, source_kind,
source_reference, source_revision)`. References are server-issued opaque handles
from a closed source-kind set, never arbitrary URLs, filesystem paths, raw text,
token arrays, or caller metadata. Duplicate references may occupy different
indexes intentionally. Reject empty manifests and unsupported kinds at admission.

The resolver must authorize the owner and return the exact revision in one
source-consistent read. It must bound data while reading, not after loading it.
Changed, deleted, unavailable, or unauthorized revisions stop execution; never
substitute the latest content or silently rebuild the manifest. Missing sources
do not prevent receipt-only accounting or cleanup recovery.
The coordinator submits one bounded batch-resolution call and validates the
returned index order and revisions. This is not 64 sequential per-item calls;
adapter admission requires the full batch to meet its certified deadline.

Each generation pins provider, model, trusted backend/model revision, dimensions,
dimension/vector policy, transform identity, batch size, and policy fingerprint.
The embedding-space fingerprint covers every certified property that affects the
vector space, including tokenizer/normalization revisions where applicable.
Runtime backend names or matching dimensions alone are not certification. Check
the identity before and after execution; uncertified or changed identity fails
closed, without writing output under the old fingerprint.

Only provider-origin failures classified by the frozen routing policy can select
fallback. A different embedding space supersedes the old generation, increments
the generation number, and restarts at item zero. Resolver, sink, accounting,
cache, policy, and database failures never choose another provider. Superseded
generations are never authoritative; only the current completed generation can
serve a completed workflow result. Cancelled partial output is retained as
non-authoritative material, not presented as a complete result.

## Persistence Model

Six Embeddings-owned tables comprise the domain store:

| Table | Essential Facts |
| --- | --- |
| `embedding_workflows` | Owner/workflow identity; exact root binding; manifest fingerprint/count; current generation; execution phase; terminal intent; reconciliation health; row version; admission capacity reservation; timestamps |
| `embedding_workflow_items` | Owner/workflow/index; immutable source kind, opaque reference, and revision |
| `embedding_workflow_generations` | Owner/workflow/generation; pinned provider/model/configuration and embedding-space fingerprints; dimensions; batch size; cursor; generation state; closure reason |
| `embedding_workflow_outputs` | Owner/workflow/generation/index; sink identity; immutable output handle and receipt fingerprint; dimensions; accepted state |
| `embedding_workflow_attempts` | Owner/workflow/generation/attempt; bounded batch range and missing-item set; claim epoch; reservation identity; dispatch intent/time/deadline; known/unknown outcome; safe usage counts; failure category; settlement state |
| `embedding_workflow_operations` | Owner/workflow/generation/attempt where applicable; closed operation kind; stable effect key; immutable bounded facts/fingerprint; dependencies; delivery claim/expiry; retry count/schedule; acknowledgement proof; resolution state |

Child foreign keys include the owner and parent identity, so a child cannot refer
to another owner's workflow, generation, or attempt. Outputs are unique by
owner/workflow/generation/item index. Root UUID bindings are unique. Operation
effect keys are unique within exact owner/workflow scope. Enforce ranges and
closed state vocabularies with database constraints as well as DTO validation.

### State Vocabulary

Jobs remains the coarse lifecycle authority. Domain execution phases only describe
progress: `admitting`, `ready`, `executing`, `finalizing`, and `settled`. Phases move
forward; failure/cancellation may skip phases, and fallback remains in `executing`.
Terminal intent is `none`, `complete`, `fail`, or `cancel`; reconciliation health
is `clean`, `pending`, or `exhausted`, independent of the root's terminal status.
These are private durable-store vocabularies, not additions to
`EmbeddingWorkflowPhase` or the inline trace status literals. Stage 3C planning
must specify an exhaustive mapping to the existing closed trace contracts and
tests for every durable phase and terminal outcome. Never pass `admitting`,
`ready`, or `settled` directly into the inline trace DTOs or infer completion from
a private phase before Jobs acknowledgement.

Generation states are `building`, `sealed`, `completed`, `superseded`, `cancelled`,
and `failed`. Only an exact Jobs completion acknowledgement can promote the
current sealed generation to completed. Terminal generations never return to
building. Sink lifecycle is separately `open`, `sealed`, or `closed`, proven by
acknowledged operations rather than inferred from the generation's domain state.
Persisted outputs are accepted receipts, never pending vectors.

Operations are `pending`, `claimed`, `acked`, `superseded`, or `exhausted`. Expired
claims return to pending by CAS without changing immutable logical effect facts.
Attempts progress through `prepared`, `reserved`, `dispatch_intent`, an
`outcome_known` or `outcome_unknown` classification, and `settled`. Unknown evidence
may become known before terminal settlement; afterward late evidence cannot
change the mutually exclusive terminal accounting decision.

A definitely rejected admission can settle the local intake without a Jobs root.
Unknown admission outcomes remain pending exact idempotent reconciliation, never
become rejection merely because lookup failed. Previously bound roots are never
recreated. Terminal domain work is not automatically revived if a generic Jobs
admin action requeues the root; manual retry/replay activation is later design work.

No foreign key points to the active Jobs table: roots may move to archive.
Use an exact active/archive UUID lookup that rejects duplicate/conflicting
authority. Do not interpret a transient lookup error as absence or expiry.

Embeddings-owned versioned migrations are explicitly invoked against the Jobs
backend. Generic Jobs schema initialization neither imports nor installs them.
Capability checks verify versions, required constraints/indexes, and security
configuration before admission or execution. Interrupted migration is recoverable
transactionally or fails with an explicit unsupported-schema result; normal
runtime must not opportunistically repair schema.

PostgreSQL enables and forces RLS on all six tables, including outbox reads,
updates, and cleanup. Runtime roles must be neither superuser nor `BYPASSRLS`.
Set owner scope transaction-locally and deny missing/mismatched scope. Maintenance
also acts under explicit owner scope, not a blanket admin bypass. SQLite requires
the same owner predicates and composite constraints, with foreign keys enabled.
Never use a default owner or infer owner identity from an integer Jobs ID.

## Repository Transitions And Fencing

Use named transitions, not a generic arbitrary-field update API:
`admit_manifest`, `bind_root`, `claim_batch`, `record_reservation`,
`mark_dispatch_intent`, `checkpoint_batch`, `supersede_generation`,
`prepare_terminal_intent`, `claim_operation`, `resolve_operation`, and
`reconcile_attempt_evidence`.

Every normal execution mutation requires expected workflow version, generation,
claim epoch, and a current exact unexpired Jobs worker/lease. Check and lock the
Jobs row and domain rows in the same database transaction; a separate manager
read followed by a domain write is insufficient. Lock Jobs, workflow, generation,
attempt/output, and operation rows in that order. SQLite uses a write transaction;
PostgreSQL locks rows and compares expiry against the database clock. Never await
an adapter call while holding these locks. Capacity admission is serialized before
publishing the manifest, and reserves future row capacity so ordinary checkpoints
do not need a conflicting admission lock order.

Recovery authority is distinct from execution authority. It is owner-scoped,
claim-fenced, and limited to an existing attempt's immutable identity, dispatch
facts, sink receipts, and accounting effect keys. It may record evidence and
settlement after lease loss or cancellation, but cannot start a provider call,
advance normal execution, choose fallback, reopen a generation, or revive a root.
Concurrent recovery claims are bounded and CAS-protected; an expired recovery
claim cannot acknowledge another claimant's operation.

## Admission And Batch Execution

1. Validate capabilities, policy, manifest, and aggregate capacity. Persist the
   workflow, immutable items, initial generation, and `jobs.admit` operation in
   one transaction. No provider work is eligible yet.
2. Deliver owner-scoped idempotent Jobs admission; bind its exact UUID and proof.
   A root acquired before binding may only reconcile binding or defer without
   consuming provider/failure budgets. Never compute from an unbound root.
   Distinguish typed definitive rejection from an unknown admission outcome.
3. Under the acquired live lease, resolve required outstanding operations and
   recover accepted output receipts before planning new paid work. The batch
   range is the contiguous interval starting at the committed cursor; only
   missing indexes within that range reach the executor.
4. Resolve/authorize the batch's exact source revisions, preserve index order,
   enforce byte limits, and prepare against the pinned policy. Create or resume
   the bounded attempt; persist and acknowledge its idempotent accounting
   reservation.
5. Atomically revalidate live execution authority and record dispatch intent
   before calling the provider. Once intent exists, absence of a response is an
   unknown dispatch, not proof that the provider was never called. An executor
   may not hide retries that bypass attempt identities, budgets, or accounting.
6. Validate count, finite numeric shape, dimensions, and postprocessing. Obtain
   first-accepted sink receipts through one bounded batch write for missing items
   under bounded write permits. Partial acceptance is recovered by batch lookup.
7. When every batch item has a valid receipt, atomically checkpoint the receipts,
   contiguous cursor, attempt outcome, settlement intent, and logical Jobs
   continuation/terminal intent. Never advance across a missing item.
8. Required accounting and transition acknowledgements gate further execution.
   Successful batches yield the root without consuming failure retries. A final
   generation is sealed; completion requires every output and required accounting
   acknowledgement before applying Jobs completion and recording its proof.

Provider calls are at least once. A crash after dispatch may require another paid
attempt for still-missing output, with a new attempt/accounting identity. Accepted
output and operation effects are idempotent, not the provider invocation itself.

## Jobs Intents And Reconciliation

Current `find_job_by_identity`, `ensure_lease_horizon`, prepared DTOs, and prepared
disposition application enforce Admin Webhook-specific identity/origin rules.
Do not remove those checks or reuse AuthNZ origin as a generic workflow authority.
Stage 3A adds a narrow domain-neutral capability while preserving that path.

The capability must provide exact root lookup/admission proof, transaction-enlisted
live fencing, budget-neutral continuation, retry/terminal application, and retained
exact acknowledgement proofs. Jobs owns its support schema; these records are
additional to, not part of, the six Embeddings tables.

Persist a lease-independent logical intent: effect key, exact owner/root identity,
expected generation/cursor or terminal facts, action, reason, and schedule.
Worker/lease values are delivery authority, not the stable intent identity. The
neutral Jobs capability applies a transition and its exact effect-key/fingerprint
receipt atomically. Retain the receipt through later root transitions and archival
until reconciliation acknowledges it; the latest root `result` marker is not an
operation journal. Receipt release/acknowledgement must itself be recoverable and
bounded, without Jobs reading Embeddings tables.

| Observation | Resolution |
| --- | --- |
| Exact effect receipt exists with matching facts | Acknowledge without repeating the transition or consuming another retry. |
| No receipt; a new live lease owns the same still-valid intent | Revalidate locked root/domain facts, then deliver using the new authority and the same logical effect key. |
| Root/generation/terminal facts make the unapplied intent obsolete | Mark superseded with bounded proof; do not hold subsequent recovery forever or apply a different action under the old key. |
| Authority/fingerprint conflicts or lookup/schema is unavailable | Fail closed; retry only classified transient failures, then mark exhausted. |
| Root is cancelled, failed, quarantined, or archived terminal | Stop normal execution; only settle existing obligations and resolve obsolete intents. Never recreate the root. |

Operations include Jobs admission/transitions and proof acknowledgements, accounting
reservation/settlement, sink opening/sealing/closure, and closed-generation cleanup.
Dependencies are explicit: reservation acknowledgement precedes dispatch; closure
precedes destructive cleanup; accounting precedes completion. Claim eligible
operations in bounded pages under owner scope. A reconciler never calls a provider
or dereferences source text.

Use fixed reason codes and immutable allowlisted facts. Retry only known transient
failures with finite budgets, using ADR-051's existing contention/HTTP policies as
appropriate. Normal continuation, binding waits, and receipt-only recovery do not
increment provider attempts or failure retries. Lease expiry is infrastructure
recovery, not evidence of another provider call: both acquisition-time reclaim and
the integrity sweep must preserve Jobs `retry_count` and failure streak under the
pinned `REQUEUE_NO_ATTEMPT` policy. Reconcile dispatch intent and accepted receipts
before deciding whether a new provider attempt is needed. Provider retry decisions
and Jobs retry application must share one durable intent so an acknowledgement
loss does not spend the same failure budget twice.

### No-Progress Exhaustion

Retry-neutral expiry must not permit unlimited automatic cycling. Admission pins
a separate no-progress limit: default and maximum 5 consecutive acquisitions
ending in yield or expiry without semantic progress. Trusted construction may
lower this limit. The neutral Jobs capability owns durable episode accounting by
exact root/acquisition identity, including expiry before the domain handler starts;
both reclaim paths must honor it. Replayed expiry evidence counts once, and process
restart or admission replay cannot reset the limit. This support state is Jobs-owned,
not a seventh Embeddings table or an Embeddings query in generic Jobs code.

Only acknowledged root binding, newly accepted output evidence, a committed cursor
advance, or a newly acknowledged required accounting/sink lifecycle operation may
establish semantic progress and reset the consecutive count. The reset requires an
exact, retained progress proof, fenced with the domain transition; replaying that
proof cannot reset a later count. Lease renewal, a new delivery claim/claim epoch,
Jobs continuation/deferral acknowledgement, proof-release bookkeeping, or elapsed
time is not progress. Provider attempts and operation deliveries retain their own
limits.

On the fifth no-progress episode, stop new paid work and record typed
`no_progress_exhausted` evidence. Jobs alone applies the root's non-retryable failure
or its existing quarantine policy under locked current authority, with retained
acknowledgement proof; exhaustion does not manufacture a provider attempt or spend
a provider retry. Cancellation or an already terminal root wins over this pending
disposition. Receipt-only recovery may still settle existing obligations within
its finite operation budgets. Exhausted reconciliation health is inspectable and
retains obligations/capacity, not a separate domain DLQ or automatic root revival.

## Cancellation And Accounting Settlement

Jobs can cancel a processing root immediately and clear its lease. Treat that as
revocation of execution authority; do not change legacy cancellation semantics.
The durable coordinator stops scheduling batches as soon as revocation is observed.
Already dispatched work may finish within its preissued bounded sink permit.
Queue pause/drain prevents subsequent acquisition under existing Jobs semantics;
in-flight work may settle. No new per-workflow paused lifecycle or admin surface
is implemented in this foundation.

Recovery can reconcile its accepted receipts and settle the existing reservation
without the former worker lease. It must verify exact owner, workflow, generation,
attempt, dispatch facts, and effect identity, and cannot declare a cancelled root
completed. Retained partial outputs do not authorize a resumed batch. Settle or
expire outstanding writer permits before sealing cancelled material. A late
receipt cannot reopen a closed generation or produce a second settlement.

Accounting distinguishes reserved/not-dispatched, dispatch-intent/unknown,
known-outcome, and terminal-settled states. Reservation, release for proven
non-dispatch, known settlement, and unknown settlement are idempotent by attempt
identity. Permit only one terminal settlement decision per attempt. Persist safe
usage estimates/counts needed for recovery, never text, credentials, prices, or
provider response bodies. Unknown dispatch uses the accounting adapter's certified
conservative settlement contract; it is never implicitly refunded. Production
units/pricing/adjustments remain boundary-owned and out of scope. Tests use a fake
ledger that rejects conflicting terminal settlements and detects duplicate effects.

Reservation and settlement are atomically ordered at the accounting boundary
under exact owner/workflow/generation/attempt identity and immutable effect facts.
Terminal settlement is absorbing: a delayed first reservation or replay after
release/settlement returns `already_terminal` with the existing proof, never a
new hold or a dispatch-eligible reservation acknowledgement. Release for proven
non-dispatch may create a terminal tombstone even if the reservation has not yet
arrived; it cannot be an unrecorded no-op that allows a late hold to appear.

Persist terminal settlement intent by attempt CAS before delivering it, preventing
concurrent dispatch or new reservation delivery authority. Previously issued
reservation deliveries are time-bounded. Keep terminal proof through their maximum
validity horizon; after proof pruning, reject expired delivery authority rather
than implicitly recreating the reservation. A stale delivery claim cannot reopen
the ledger even when it is too old to acknowledge its domain operation. If the
adapter cannot certify this ordering and retention, it is unavailable. This adds
no production billing hookup or pricing policy.

## Sink Receipt And Generation Contract

The sink contract is stronger than stable keys or ordinary `upsert`:

- Explicitly open a generation under exact owner/workflow/generation/sink/space
  identity. Writes never implicitly create or reopen it.
- Atomically accept the first valid write for an item key. Return typed `accepted`,
  `already_accepted`, or `identity_conflict` results. A competing payload cannot
  overwrite the accepted value. If the original receipt matches the exact owner,
  source revision, generation, sink, space, and dimensions, that winner remains
  authoritative even when a new computation differs numerically. Validate the
  receipt and its stored payload digest, not equality with the losing computation;
  mismatched identity fails closed. Settle the losing provider attempt separately.
- Receipts bind owner, workflow, generation, item index, source-revision identity,
  sink identity, embedding-space fingerprint, dimensions, and accepted payload
  digest. Receipt lookup must distinguish accepted, absent, unavailable, and closed
  or conflicting identity. Unavailable lookup is not a cache miss.
- Batch lookup must provide writer-authoritative, linearizable accepted/absent
  decisions for each item. A write accepted before lookup begins cannot be hidden
  behind `absent`. Lagging replicas, unresolved writes, and uncertain visibility
  return `unavailable`; halt new paid work for that batch and reconcile under the
  existing finite budgets. Concurrent acceptance after a certified absence still
  uses first-accepted-write arbitration; it does not weaken the lookup guarantee.
- Enforce expiring writer permits and irreversible sealing/closure at the sink,
  not only in a pre-write database check. Seal rejects new values but preserves
  receipt lookup. Closure cannot race with a write that recreates deleted output.
- Before superseded cleanup, close the generation, acknowledge closure, and wait
  for any issued permit horizon required by the certified sink. Keep closure proof
  until no previously valid writer/opening permit can still be accepted. Reject
  expired permits and prohibit implicit generation recreation after proof pruning.
- Delete only the exact closed generation's owned staging material. Never perform
  collection-wide deletion or delete another workflow's completed output.

Fake adapters must model concurrent first writes, conflicting replays, delayed
receipt visibility, closure/write races, expiry, and cleanup replay. Existing Chroma
write/delete helpers do not certify this contract and must not be wired in during
this foundation stage. Selecting ChromaDB versus another production sink is not
part of this decision.

## Concrete Resource Limits And Deadlines

The following are foundation defaults and maximums, not changes to inline API
limits. Trusted construction may lower them, but cannot silently raise them;
raising a maximum requires a reviewed policy change and updated capacity tests.
Freeze the selected policy fingerprint at admission. Combined limits can reject a
manifest even when each individual field fits.

| Resource | Default / Maximum |
| --- | --- |
| Manifest items | Maximum 10,000; nonempty |
| Batch size | Default 64; maximum 64 items |
| Planned batches per generation | Maximum 512; reject larger plans at admission |
| Generations | Maximum 3 including primary |
| Dispatch intents per generation/batch | Maximum 3; at most 4,608 per workflow |
| Dimensions | Maximum 8,192; positive rectangular finite vectors |
| Numeric vector slots per provider result | Maximum 524,288; enforce actual missing-item count times dimensions |
| Resolved UTF-8 text | Maximum 64 KiB per item and 1 MiB per batch |
| Provider wire response | Maximum 16 MiB, enforced before unbounded decoding |
| Owner/reference/revision identity | Maximum 512/512/128 UTF-8 bytes; closed kind identifiers maximum 64 bytes |
| Encoded manifest | Maximum 8 MiB, including framing and field overhead |
| Operation or acknowledgement facts | Maximum 4 KiB per encoded record; no arbitrary nested metadata |
| Output receipt handle | Maximum 512 UTF-8 bytes |
| Domain operation rows | Maximum 32,768 per workflow, including acknowledgement/cleanup operations |
| Retained workflows / unresolved rows | Maximum 32 retained workflows per owner; 100,000 owner rows and 500,000 global domain rows |
| Concurrent roots | Maximum 4 per owner and 32 globally; Jobs may impose stricter limits |
| Consecutive no-progress acquisitions | Maximum 5, pinned at admission; yield/expiry evidence counts once independently of provider retries |
| Operation delivery / scan | Maximum 5 automatic deliveries; scan maximum 100 rows |
| Batch resolver / provider / batch sink / accounting call | 10 / 120 / 10 / 10 seconds; lookup and write limits cover the full batch |
| Normal batch calls | Maximum 1 resolver, 1 receipt lookup, 1 write, 1 provider; at most 9 non-provider port calls including first/final-batch lifecycle and Jobs proof delivery |
| Preflight recovery / local preparation and DB work | Maximum 60 / 20 seconds per acquisition |
| Batch acquisition quantum | 300 seconds end to end, including reconciliation and cleanup of child tasks |
| Adapter cancellation grace | 5 seconds, included in the 300-second quantum |

### Batch Progress Budget

Resolver, sink receipt lookup, and sink write ports accept at most one 64-item
batch each. Their 10-second limits cover the whole call, not each item. An adapter
that can only resolve items serially within separate 10-second limits is not
certified for the default batch. It must supply a bounded batch implementation or
be admitted with a smaller batch size and an explicitly certified timing plan.
Batch size and call plan remain pinned; no runtime resizing or persistent raw-text
checkpoint is introduced. Internal adapter fan-out, when used, must also be
bounded and share the batch deadline without hidden retries.

The conservative normal first/final-batch plan allocates 60 seconds to preflight
recovery, 20 to local preparation/DB work, at most nine 10-second non-provider
calls, 120 to one provider call, and 5 to child-task teardown: 295 seconds, leaving
5 seconds of scheduling margin. Non-provider calls comprise batch lookup, opening,
resolution, reservation, writing, settlement, sealing, Jobs disposition, and Jobs
proof release. Skip unnecessary calls rather than repurposing their allowance for
extra attempts or synchronous recovery retries. Perform additional recovery in a
later bounded acquisition; if required obligations remain after preflight, persist
progress and yield without dispatch. Successful baseline execution must reach a
batch checkpoint under this plan, including first/last and partial batches.

Before dispatch, reserve the provider's full selected deadline plus the remaining
critical-path call/local budgets and cancellation grace. An undersized remaining
quantum must not be turned into a predictably truncated paid attempt. Repeated
pre-dispatch quantum exhaustion without durable progress is a typed certification
or policy failure, subject to the separate no-progress limit rather than unbounded
budget-neutral continuation. Tests use a virtual monotonic clock, not long sleeps,
to prove the normal plan and reject the original 64-times-6-second serial-resolution
counterexample before provider dispatch.

Admission reserves a conservative upper bound for future domain rows, accounting
for items, all permitted generations/attempts, output receipts, operations, and
proof acknowledgement/cleanup. Serialize capacity checks so simultaneous admission
cannot oversubscribe owner/global budgets. Recovering or exhausted records count
against these limits until safely released; a cancelled Jobs root does not make
its unsettled state free. Test every maximum and one-over boundary, including
multibyte identifiers, framing overhead, partial batches, and combined bounds.

Stage 3B planning must select a concrete atomic global-reservation primitive that
works with FORCE RLS and non-bypass, owner-scoped runtime roles. Summing visible
workflow rows under one owner's scope cannot enforce the global cap. The primitive
must serialize owner/global reservations and safe release without exposing other
owners' rows or requiring a runtime RLS bypass. Include concurrent cross-owner
admission, crash/replay, and exhausted-obligation retention in backend tests; a
per-owner-only test is not global-capacity certification.

Adapters must enforce bounded reads/decoding and respond to deadlines. Merely
wrapping blocking synchronous work in an async timeout is not certification.
Each call receives the lesser of its configured deadline and the remaining batch
quantum minus required cancellation grace. Multiple resolver/sink calls share the
same quantum; they cannot reset it. Do not dispatch when insufficient budget
remains. If the quantum expires during receipt/acknowledgement delivery, persist
the existing recovery obligations and stop normal execution. Bound both native
and postprocessed dimensions/scalar slots before unbounded materialization.
Do not launch replacement paid work while an earlier dispatch is still live;
unknown work must settle and its execution/permit deadline must pass first.
Cancellation-resistant adapters are unavailable to this foundation until they
provide an isolation and bounded-completion guarantee. No real model/process
adapter is certified or added here.

Use a monotonic clock for local elapsed-time deadlines and the database clock for
leases, claims, persisted schedules, and permit expiry proofs. Check sufficient
lease horizon before dispatch and renew within the configured Jobs cap. If the
cap cannot support the selected quantum, fail before provider dispatch; never
silently extend beyond the cap or revive an expired lease. Strict durable fences
remain mandatory even if legacy Jobs lease-enforcement flags are relaxed.

## Retention And Privacy

Ordinary workflow retention follows the exact active or archived root UUID. After
both stores expire, delete domain history only when required operations and writer
permits are resolved. Unsettled accounting, disposition acknowledgement, or cleanup
pins minimal owner-scoped recovery state even after root expiry. Exhausted state
stops automatic retries, remains inspectable through the internal repository, and
continues consuming capacity; never silently discard obligations to reclaim space.

Jobs acknowledgement proof and sink closure proof must remain available for their
recovery horizons. Retention must not strand a pending intent with only a missing
root/latest-marker lookup. Release proofs through explicit bounded acknowledgement
contracts. Do not infer root status from unrelated integer IDs or recreate a root
whose history was pruned. Archived proof reads, including compressed archives,
must use the existing bounded decoding and exact-identity facilities.

Pruning Jobs/domain history does not delete current completed-generation sink
output. Its lifetime remains sink/content-owner policy. Superseded staging cleanup
is a separate closed-generation operation. Preserve cancelled partial material
under the existing approved retention choice; no public retrieval API is added.

Persist and log only allowlisted bounded metadata. Use fixed failure/reason codes,
not exception messages, representations, provider bodies, raw inputs, credentials,
headers, arbitrary paths/URLs, or caller-supplied observability tags. Metrics do not
label by workflow, owner, model, source reference, or other unbounded identifiers.
Privacy tests use sentinel content/secrets and inspect state, proofs, logs, and
trace output, including failure and recovery paths.

## Required Verification

Use existing pytest/Hypothesis patterns and Jobs backend fixtures. PostgreSQL tests
use `jobs_pg_dsn` / `pg_temp_db`, not hand-built database provisioning. Explicitly
enable lease enforcement and RLS instead of relying on Jobs fixture defaults.
Validate runtime role flags and transaction-local owner reset; a superuser run is
not RLS certification. Only unavailable shared fixtures justify local PostgreSQL
skips, and skips leave backend certification incomplete until CI supplies evidence.

| Area | Required Evidence |
| --- | --- |
| State machine | Closed vocabularies and monotonic phases; property-based acquire/renew/expire/retry/fallback/cancel/replay sequences; cursor prefix, generation isolation, immutable manifests, and terminal non-revival even after admin requeue |
| Admission | Duplicate creates; crash after manifest, Jobs admission, and binding; definitive rejection versus unknown outcome; early acquisition; exact owner/UUID/payload and persisted expiry/retry/quarantine/no-progress control conflicts; atomic cross-owner global reservations under non-bypass RLS |
| Concurrency | Same-version checkpoints; cancellation between validation and write; expired execution/recovery claims; fixed lock ordering on real SQLite/PostgreSQL transactions |
| Provider/sink | Partial receipt recovery; no recomputation of accepted output; writer-authoritative absence after lost acknowledgement; lagging visibility returns unavailable and blocks new paid work; same-identity competing vectors preserve the original winner and settle both attempts; identity conflicts; space changes; close/delete versus late writes; permit expiry |
| Accounting | Crash before/after reservation and dispatch intent; known/unknown outcomes; cancellation and timeout; terminal release before delayed first reserve; reserve replay after settlement; absorbing terminal proof and expired-delivery rejection after pruning; conflicting settlement rejection; no duplicate reserve/terminal effects |
| Jobs intents | Applied-but-unacknowledged transitions; stale unapplied commands under a new lease; obsolete intent supersession; replay after a later transition and after archival; exact retained proofs |
| Expiry/no-progress | Crash before handler start/dispatch and after receipt acceptance/checkpoint; acquisition reclaim and integrity sweep preserve retry/failure counters; fifth no-progress episode stops automatic execution through Jobs; exact progress proof resets once, stale proof/replay/restart/deferral acknowledgements cannot reset; cancellation/terminal state cannot revive |
| Retention | Missing/error/conflicting root lookup; prune before acknowledgement; exhausted operations retain obligations/capacity; completed output survives domain pruning; scoped cleanup only |
| Migration/security | Fresh/upgrade/repeat/interrupted migration; missing schema; all six RLS tables; denied missing/cross-owner CRUD and cleanup; foreign-key mismatches; connection reuse isolation |
| Bounds/deadlines | Every limit and one-over case; native/postprocessed bounds and malformed/oversized responses before materialization; full 64-item bulk path checkpoints within the 295-second timing plan; reject 64 serial 6-second resolutions and undersized pre-dispatch budget; cumulative call deadlines within one quantum; hanging/cancellation-resistant adapters; no lease revival or hidden retry multiplication |
| Compatibility/dormancy | Inline flag on/off endpoint parity; exhaustive private durable-phase/terminal mapping to unchanged trace literals; legacy media/storage and Jobs expiry defaults unchanged; Admin Webhook prepared regression; no startup migration/worker/admission/adapter calls |

Implementation completion requires both backend integration/parity evidence,
focused endpoint regression, changed-scope lint/format/compile checks,
`git diff --check`, and Bandit on touched production code. Record existing warnings,
baseline findings, and genuine environment skips explicitly. This documentation
task does not claim that future Stage 3 tests or production adapters already pass.

## Review Corrections And Evidence

| Validated Gap | Incorporated Correction |
| --- | --- |
| Cancellation clears the processing lease, preventing ordinary late checkpoints | Separate execution from receipt-only recovery authority; no root revival or next batch after cancellation |
| Unapplied old-lease dispositions are rejected; latest root marker loses historical proof | Lease-independent logical intents, current-authority revalidation, supersession rules, and exact retained Jobs receipts |
| Stable sink keys and ordinary upsert cannot prevent overwrite or post-cleanup resurrection | Atomic first-accepted receipts, explicit generation opening, sink-enforced sealing/closure, bounded permits, and closure-proof retention |
| Item counts do not bound payloads, vectors, or handler lifetime | Concrete combined resource limits, atomic capacity reservation, adapter certification, and per-call/quantum deadlines |

Review baseline: 48 existing runner/type tests passed. The subsequent focused Jobs
review passed 79 SQLite prepared-disposition/cancellation tests, with 7 PostgreSQL
cases deselected. Direct temporary-database probes confirmed cancellation clears
all lease fields; a late canonical completion returned `backend_conflict`; an
unapplied old-lease intent returned `stale_lease`; and replay of an earlier marker
after a later transition returned `wrong_status`. These establish existing boundary
behavior, not successful execution of the unimplemented Stage 3 contracts.

The 2026-10-03 written-spec review added four approved corrections: absorbing
accounting terminal records, a certified batch-level progress budget,
writer-authoritative sink absence, and ADR renumbering. Read-only countermodels
demonstrated the missing ordering, serial-timeout, and lagging-lookup guarantees;
these were requirements gaps, not observed production failures. ADR-058 is now
the accepted Jobs completion row-identity decision in `origin/dev`; this proposal
was renumbered to ADR-059 in that revision.

The 2026-10-04 written-spec review corrected admission policy: generic Jobs
admission defaults to retry-consuming lease expiry, so the retry-neutral promise
requires explicit `REQUEUE_NO_ATTEMPT` admission and stored-control validation.
Finite no-progress episode accounting bounds infrastructure recovery independently
from provider retry accounting. Stage 3B/3C planning must also resolve the concrete
RLS-compatible global reservation primitive and the exhaustive private-phase trace
mapping. Existing Jobs lease-budget tests establish the underlying expiry-policy
behavior; they do not certify the new neutral capability or its no-progress budget.
The upstream ADR inventory has since accepted ADR-059 for the backlog-py task
editor cutover. This proposal is now ADR-060, with active references updated.
Recheck numbering before integration and preserve accepted ADR-058 and ADR-059.

## Delivery And Approval Gates

Deliver as sequential, reviewable units: Stage 3A neutral Jobs capability/proofs;
Stage 3B repository/migrations/security; Stage 3C fake-certified adapters and bounded
coordinator/reconciler; Stage 3D crash/property/backend/compatibility certification.
Each unit receives its own Backlog task and scoped implementation plan. These
units are requirements for later planning, not authorization to start coding.

The requester approved this written spec and ADR on 2026-10-04. Stage 3A planning
is tracked separately by TASK-13451; runtime implementation still requires plan
review and execution-method selection. Future PRs
must meet the repository's human-authored Change summary merge gate. Production
activation and real adapter certification require separate designs, explicit
approval, and the later media/vector-store migration stages.
