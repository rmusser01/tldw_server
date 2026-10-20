# Stage 3A Prepared Jobs Capability

Internal contract for the dormant, domain-neutral Jobs substrate implemented
under TASK-13900 and ADR-060. This is not durable Embeddings execution or
production activation. Jobs does not import Embeddings, call providers, schedule
a worker, or start a reconciler. Stages 3B (domain repository and capacity), 3C
(runner and ports), and 3D (domain certification) remain deferred.

## Installation And Roles

Ordinary `JobManager` construction, imports, startup, and legacy migrations do
not install support, enroll existing Jobs, or admit roots. Installation is an
explicit administrative operation on an already initialized Jobs database:

```python
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.workflow_capability import (
    inspect_workflow_support,
    install_workflow_support,
)

installer = JobManager(db_path="/absolute/path/jobs.db", backend="sqlite")
install_workflow_support(installer)
readiness = inspect_workflow_support(installer)
```

For PostgreSQL, construct the installer with `backend="postgres"` and
`db_url=installer_dsn`. `install_workflow_support` owns one atomic transaction;
`install_workflow_support_in_transaction(connection, backend=...)` is the
lower-level explicitly enlisted migration entrypoint. Installation adds
`jobs.workflow_capability_version` and the version-1 tables
`job_workflow_schema`, `job_workflow_roots`, and
`job_workflow_effect_receipts`. Existing Jobs retain a null enrollment marker.
Complete installation is idempotent. Partial, altered, or unsupported schemas
are rejected, not automatically repaired. Readiness inspection is read-only and
returns `APPLIED` or `UNAVAILABLE`.

The PostgreSQL runtime must be a real non-superuser, non-`BYPASSRLS` role.
Use a separate runtime login, `JOBS_PG_SKIP_SCHEMA_INIT=true`, and
`JOBS_PG_RLS_ENABLE=true`; an explicitly configured `JOBS_PG_RLS_ROLE` is entered
with `SET LOCAL ROLE`. Known PostgreSQL permission/configuration denials during
role/owner/RLS setup raise `WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)` and
roll back. The original driver exception remains the in-memory cause for
internal classification; raw error text is not logged or persisted by this
boundary. Unrelated programming errors and cancellation are not reclassified.
There is no best-effort privileged fallback. Do not grant schema creation, broad domain
access, or RLS bypass to make a runtime connection work.

Grant schema `USAGE`, required ordinary Jobs table operations and associated
sequence usage, support-table `SELECT/INSERT/UPDATE/DELETE`, and version-table
`SELECT`. Domain privileges belong to the future domain repository's separate
contract. Both owned support tables have ENABLE and FORCE RLS with owner-only
USING/WITH CHECK policies. `app.is_admin` and domain allowlists cannot bypass
those policies. Missing/blank owners see nothing. Runtime transactions set
owner scope locally, require `row_security=on`, and validate the role, transaction
ID, and scope again before use/commit. Commit, rollback, exceptions, and reused
sessions must not retain another owner's authority.

## Admission And Inspection

Bind trusted policy, not policy inferred from a request. A capability constructor
is inert. The following is a Jobs-only admission example, not domain enrollment
or an installed Embeddings repository:

```python
from uuid import uuid4
from tldw_Server_API.app.core.Jobs.operations.contracts import ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand, WorkflowControls, WorkflowPolicy, WorkflowScope,
)

policy = WorkflowPolicy(
    WorkflowScope("embeddings", "workflow", "embedding_workflow"),
    "workflow_id",
    frozenset({"provider_transient", "permanent", "no_progress_exhausted"}),
)
capability = installer.workflow_capability(policy)  # SQLite example only
command = WorkflowAdmissionCommand(
    "owner-a", str(uuid4()), "workflow-key",
    WorkflowControls(3, None, ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT, 5),
)
admitted = capability.admit(command)
```

The queue must be permitted by Jobs configuration, for example
`JOBS_ALLOWED_QUEUES_EMBEDDINGS=workflow`. Ordinary queue restrictions,
admission quotas, pause/drain behavior, counters, and events still apply.
An admitted root's payload contains only `workflow_id`. The owner is nonblank,
exact, and at most 512 UTF-8 bytes; reference and Jobs identities are canonical
UUIDv4 strings. The idempotency key is at most 200 UTF-8 bytes. Numeric Jobs IDs
are hints, never authority.

Admission serializes the exact owner/scope/key and retains immutable controls.
An exact replay returns `ALREADY_APPLIED` before mutable admission policy is
re-evaluated. Changed reference/controls conflict. A previously bound root is
never replaced or revived, including after archive/prune. Legacy correlations
are not enrollment. `lookup(root)` returns active/archive status or a closed
missing/conflict/unavailable result. No history is not permission to recreate.
Canonical Admin Webhook identity and prepared-disposition APIs remain separate;
this capability cannot enroll the reserved Admin Webhook delivery scope.

## Transactions, Leases, And Proofs

Acquire through public `JobManager.acquire_next_job`, passing domain, queue,
job type, owner, worker, and lease duration. Successful enrolled acquisition
records its durable epoch before committing, in both acquisition modes and on
both backends. Construct `WorkflowLease(row["worker_id"], row["lease_id"])`
from that delivery, not an old cached lease.

`capability.transaction(owner_user_id)` yields one synchronous
`WorkflowTransaction` and owns its outer commit/rollback. The transaction
exposes `connection` and `backend` for enlisted domain work. Lock Jobs first
with `tx.lock_live_root(root, lease)`, then capability metadata, then future
workflow/generation/attempt/output/operation rows. Do not call an adapter or
`await` inside it. Do not nest a committing manager wrapper.

The opaque fence requires the exact UUID, owner, policy, controls, processing
status, worker/lease identity, and an unexpired database-clock lease. Legacy
lease enforcement switches cannot weaken it. `tx.ensure_horizon(fence,
minimum_seconds)` extends only live authority, bounded by
`JOBS_LEASE_MAX_SECONDS` (default 3600); a request beyond the cap returns
`INSUFFICIENT_HORIZON`, not a promise. Database time governs leases/schedules;
callers must use local monotonic deadlines for their own remaining time.

Effect APIs:

- `lookup_effect(root, effect)` checks retained exact proof without a lease.
- `tx.apply_effect(fence, effect, verified_guard_fingerprint)` applies one
  lifecycle intent after the caller's locked, same-connection domain CAS.
- `tx.record_progress(fence, WorkflowProgress(proof_key, kind, fingerprint))`
  records typed durable progress. It is not a heartbeat or arbitrary counter.
- `release_effect(root, effect)` releases a matching proof but retains its
  immutable key, fingerprint, facts, and slot tombstone.

This document intentionally supplies no runnable domain repository/CAS example:
Stage 3B has not implemented one. A matching digest is the caller's assertion of
actual enlisted domain revalidation, not a replacement for that work.

Effects use a stable 64-lowercase-hex key, typed action, guard fingerprint, an
allowlisted reason only for retry/fail, and an aware UTC-normalized schedule
only for continue/retry. Logical fingerprints include exact root identity,
controls, and original schedule, but exclude delivery lease claims. An exact
replay remains acknowledged after later transitions, restart, archive, and
Jobs-history pruning. Changed facts under the same key conflict. Terminal
states are absorbing. Jobs alone owns retry/quarantine and lifecycle authority.

Native SQLite transaction-control authorizers retire stale fences. Both
backends validate SQL proof obligations for consumed effects, progress resets,
and closed episodes before commit. Rolling a savepoint back must not leave a
cached Python acknowledgment, domain-only commit, progress reset, or observer
event. Observers run only after the successful outer commit. Preserve these
checks when adding domain operations.

## Independent Budgets And Privacy

Provider retries, no-progress episodes, and receipt capacity are distinct:

- `RETRY` charges once per logical effect, up to immutable `max_retries`;
  a retry beyond that budget fails without another retry increment.
  Optional quarantine counts consecutive fixed failure reasons independently.
- `CONTINUE` and expired-lease recovery do not spend provider retries. Shared
  acquisition/sweep recovery closes each epoch once. Each empty epoch charges
  the no-progress budget. Its immutable limit is 1-5 (default 5).
- A new typed semantic progress proof resets no-progress once. Replaying it,
  including after restart or in another epoch, cannot reset again.
- Reaching the configured no-progress limit (the fifth empty epoch at the
  default and maximum limit of 5) fails with `no_progress_exhausted` and its own
  committed failure proof. A continuation call returns `EXHAUSTED` without
  acknowledging its original continuation key. This is not a capacity-denial
  rollback.
- Action and progress proof namespaces each have 32768 slots per root. Keys
  share the root namespace; release never recycles a slot. Capacity denial
  raises `WorkflowFenceError(EXHAUSTED)` and makes the transaction sticky
  rollback-only, even if caught, undoing the preceding domain CAS.

Logical JSON facts and acknowledgments are bounded to 4096 bytes. Decoders
reject malformed UUIDs, digests, timestamps, duplicate keys, nonfinite values,
and excessive nesting. Archive payload validation bounds encoded and expanded
gzip data, rejects trailing members/truncation, and compares all present
payload projections. Receipts hold compact fixed facts, not provider responses,
credentials, input text, paths, URLs, arbitrary exception messages, or metric
identifiers. Never put secrets in owner/idempotency identities or trusted fixed
reason vocabularies either.

## Dormant Rollback And Certification

Rollback means stop capability callers and leave support inert. Preserve root
identity and retained proof obligations even if ordinary Jobs history is
pruned. There is no destructive down migration, automatic receipt cleanup, or
implemented domain/support retention cleanup in Stage 3A. Do not delete proof
tables or strip enrollment markers to simulate rollback. Follow-on aggregate
capacity/retention policy belongs to Stage 3B.

Certification uses real SQLite and PostgreSQL stores, non-bypass runtime roles,
bounded stateful command sequences, shared deterministic traces, concurrent
cancel/prune/recovery races, and explicit legacy/endpoint/CI parity suites.
Use the project virtual environment and `RUN_JOBS=1`; PostgreSQL uses only
`jobs_pg_dsn`/`pg_temp_db`. Fixture unavailability is incomplete certification,
not backend parity. Existing warnings must remain visible. Passing the Jobs
substrate tests is not production activation or Stage 3B/C/D certification.

No PR/merge is authorized by this work. Any materially AI-authored PR remains
blocked until the human requester writes their own **Change summary** explaining
what changed and why these choices were made, per
`Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md`.
