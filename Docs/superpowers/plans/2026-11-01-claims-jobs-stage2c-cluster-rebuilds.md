# Claims Jobs Stage 2C Cluster Rebuilds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, owner-exclusive exact and embedding cluster rebuild Jobs without moving execution lifecycle into Claims or changing the default synchronous API.

**Architecture:** Shared Jobs arbitrates resource admission, request replay and every lifecycle transition. DB_Management owns bounded source reads, atomic cluster replacement and publication receipts; Claims owns calculation and integration through the existing WorkerSDK worker. Jobs and Media DB never require a distributed transaction.

**Tech Stack:** Existing Python, FastAPI/Pydantic, SQLite, PostgreSQL, Chroma, pytest/Hypothesis, Loguru and WorkerSDK; no new runtime dependency.

**Spec:** [Approved Stage 2C design](../specs/2026-11-01-claims-jobs-stage2c-cluster-rebuild-design.md).

**Status:** Draft implementation plan; awaiting requester review before execution.

**Tracking:** TASK-15283 plans the implementation; TASK-15282 completed the design. Before executing each reviewable task, create or find its own Backlog execution record with backlog-py and link this plan. This document is not execution approval.

**ADR check:** No additional ADR is required. Accepted [ADR-085](../../ADR/085-jobs-resource-admission-and-domain-publication.md) governs resource admission and domain publication; ADR-003 and ADR-058 continue to govern Jobs ownership and row identity. A departure from these decisions requires design review, not an implementation shortcut.

**Baseline:** `origin/dev` at `54b321dcec8cdbb6358581f42c1548696260570b`; approved design corrections at `92ac607403`. Planning uses `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/claims-jobs-stage2b-review-metrics-design`, branch `codex/claims-jobs-stage2c-cluster-rebuild-design`. Preserve the old Stage 2B closeout and unrelated main-checkout changes.

## Global Constraints

- One active rebuild per owner across both methods; equal effective settings reuse the active UUID, unequal settings return 409, and no replacement is queued.
- Default behavior remains synchronous HTTP 200 with existing result fields; opt-in Jobs mode returns HTTP 202 with a Job reference and status link.
- A successfully admitted Job never falls back to local execution. Uncertain admission returns sanitized 503, never synchronous execution.
- Resource identity is `(domain="claims", owner_user_id, resource="cluster-rebuild")`; queue, method and job-type version do not partition uniqueness.
- Payload version and algorithm version are 1. `min_size` is an integer in 1..1000, not a boolean. Embedding threshold is finite in 0..1; exact threshold is null and semantically ignored.
- `Idempotency-Key`: 1..128 printable ASCII characters, no surrounding whitespace; digest-only storage, 24-hour retention, at most 128 retained aliases per logical Job.
- Resource head precedes exact Job-row locks. No opted-in create, reactivation, terminalization, batch, repair or prune path may bypass this rule or regress identity/counters.
- Claims has no queue, job registry, lease loop, retry counter, cancellation state or status proxy. Domain heads/receipts contain output metadata, not lifecycle.
- No claim text, vectors, paths, credentials or notification bodies enter Jobs payloads/results/logs. Owner status is explicitly allowlisted and foreign UUIDs return 404.
- Maintenance never provisions missing Media DB, directory, schema, Watchlists storage or Chroma collection. Missing owner DB is permanent `claims_cluster_owner_database_missing`, not a domain skip.
- Publication is one Media DB transaction. Replays preserve cluster IDs; newer applied generations fence older output. Foreign references fail without cross-owner writes or changes to domain data, head or receipts.
- Cancellation is best effort across stores and cannot stop an already-running thread or undo committed output. WorkerSDK remains the lifecycle authority.
- Exact empty input replaces with empty output; absent embeddings preserve old clusters with a receipt-backed skip; partial embeddings retain current partial-coverage semantics. Storage failures are not missing embeddings.
- Publication receipts survive Jobs pruning for the lifetime of the owner dataset. Jobs heads survive ordinary pruning. Restore requires a coordinated quiescent checkpoint and operator provenance/watermark checks.
- Rollout deploys migrations and compatible type-filtered workers before producers; uniform flags and a drained synchronous path are mandatory. Rollback waits for physical handlers, not merely terminal statuses.
- No Stage 3 removal, metrics fallback cleanup, stable cluster-ID migration, transfer repair, new clustering algorithm or prior-PR closeout is included.

## Review Focus

1. Queue/policy changes between first admission and HTTP retry must not turn a valid retained-key replay into a fresh submission or consume another quota; pinned in Task 2.
2. A transfer completed before the snapshot, with foreign Media hidden by RLS, must block destructive cleanup without exposing or mutating foreign claims; pinned in Tasks 5 and 7.
3. An archived Job whose numeric ID was reused must remain the target of both owner and authorized admin status links, or return 404; pinned in Task 4.
4. An absent collection/schema and an access/storage error must remain distinguishable without initialization side effects; pinned in Tasks 6 and 8.
5. Crash-after-publication, cancellation and delayed worker threads must never republish committed clusters or undo newer output; pinned in Tasks 7, 9 and 11.

---

## File And Interface Map

All source/test paths below are repository-relative, not replacements for the public package entry points. Keep new helpers focused; do not restructure existing large modules wholesale.

| Boundary | New files | Existing integration points |
| --- | --- | --- |
| Shared immutable Claims contract | `tldw_Server_API/app/core/claims_cluster_rebuild_contract.py` | `Claims_Extraction/claims_job_contracts.py`; Jobs registry imports only the neutral leaf contract |
| Jobs resource types and policy | `tldw_Server_API/app/core/Jobs/resource_contracts.py`, `resource_policy.py` | `Jobs/manager.py`, `operations/contracts.py`, `models.py`, `migrations.py`, `pg_migrations.py` |
| Jobs backend resource operations | `tldw_Server_API/app/core/Jobs/operations/sqlite/resource.py`, `operations/postgres/resource.py` | Existing backend admission and lifecycle operations, `Jobs/admin_operations.py` |
| Jobs safe reads | `tldw_Server_API/app/core/Jobs/owned_status.py`, `api/v1/endpoints/jobs_owned.py`, `api/v1/schemas/jobs_owned.py`, `api/v1/API_Deps/jobs_admin_deps.py` under `tldw_Server_API/app/` | `Jobs/manager.py`, `endpoints/jobs_admin.py`, `router_groups/admin.py`, `router_groups/minimal.py` |
| Media publication contracts/storage | `tldw_Server_API/app/core/DB_Management/media_db/claims_cluster_rebuild_types.py`, `runtime/claims_cluster_rebuild_snapshot.py`, `runtime/claims_cluster_rebuild_publication.py` | `media_db/media_database_impl.py`, schema migration registry and RLS helpers, `runtime/claims_cluster_ops.py` |
| Media migrations | `tldw_Server_API/app/core/DB_Management/migrations/027_claims_cluster_rebuild_publication.sql`, `media_db/schema/migration_bodies/postgres_claims_cluster_rebuild_publication.py` | `media_db/schema/migrations.py`, `media_db/media_database_impl.py`, `media_db/schema/features/postgres_rls.py` |
| Bounded calculation/vector reads | `tldw_Server_API/app/core/Claims_Extraction/claims_cluster_rebuild.py`, `claims_cluster_vectors.py` | `claims_clustering.py`, `Embeddings/ChromaDB_Library.py`, `DB_Management/db_path_utils.py` |
| Existing-only Watchlists evaluation | `tldw_Server_API/app/core/Claims_Extraction/claims_cluster_watchlists.py` | `DB_Management/Watchlists_DB.py`, `Claims_Extraction/claims_service.py` |
| Claims admission/worker/API | No additional execution runtime | `claims_jobs.py`, `claims_job_handlers.py`, `claims_job_contracts.py`, `services/claims_jobs_worker.py`, `endpoints/claims.py`, `schemas/claims_schemas.py`, `API_Deps/DB_Deps.py`, `core/config.py`, `Config_Files/config.txt` |
| Operations | `Docs/Operations/Claims_Cluster_Rebuild_Jobs.md` | `Docs/Code_Documentation/Claims_API_and_Schema.md`, `Claims_Extraction.md`, `Jobs_Manager.md` |

The migration number 27 is free at the baseline. Recheck before execution and change both backend registrations together if another migration landed. This is an implementation numbering adjustment, not a reason to rewrite accepted ADR rationale.

### Shared Types

Task 1 defines frozen Jobs dataclasses in `Jobs/resource_contracts.py`:

- `ResourceScope(domain: str, owner_user_id: str, resource: str)`.
- `ResourceAdmissionRequest(scope: ResourceScope, canonical_settings: dict[str, object], request_key: str | None = None)`; raw keys are transient, never persisted or logged.
- `ResourceBinding(scope: ResourceScope, authority_uuid: str, generation: int)`.
- `ResourceAdmission(kind: Literal["created", "active_reused", "request_replayed", "active_conflict", "request_conflict", "rejected"], job: dict[str, Any] | None, rejection: AdmissionResult | None)`.
- `ResourceAttempt(job_id: int, job_uuid: str, worker_id: str, lease_id: str, binding: ResourceBinding)`.

Task 1 also defines `ClusterRebuildSettings(version: int, method: Literal["exact", "embeddings"], min_size: int, similarity_threshold: float | None, algorithm_version: int)` and `ClusterRebuildLimits(exact_claims: int, embedding_claims: int, text_bytes: int, vector_width: int, embedding_components: int, similarity_component_comparisons: int, cleanup_references: int)` in the neutral Claims contract. Jobs does not import Claims services, Chroma or Media DB to enforce registration. Copy settings mappings defensively on construction; frozen dataclasses alone do not freeze nested dictionaries.

Task 5 defines frozen domain dataclasses in `media_db/claims_cluster_rebuild_types.py`:

- `ClusterSourceRow(claim_id: int, media_id: int, chunk_index: int, claim_text: str)` and `ClusterSourceSnapshot(owner_user_id: str, rows: tuple[ClusterSourceRow, ...], digest: str)`.
- `ClusterMember(claim_id: int, similarity: float)` and `ClusterAssignment(canonical_claim_text: str, representative_claim_id: int, members: tuple[ClusterMember, ...])`.
- `ClusterCalculation(assignments: tuple[ClusterAssignment, ...], outcome: Literal["ok", "skipped"], reason: str | None, claims_skipped: int)`.
- `ClusterPublicationIdentity(owner_user_id: str, job_uuid: str, authority_uuid: str, generation: int, settings_fingerprint: str)`; no Jobs class is imported into the domain type module.
- `ClusterPublicationResult(result: dict[str, int | float | str | None], newly_published: bool)`; the flag is returned to the current caller, not stored as lifecycle state.

### Configuration Decisions

These are builder choices within the approved ceilings, not additional product approvals. Each numeric ceiling is configurable from 1 through its listed default in this stage; raising a hard ceiling requires revisiting resource evidence. Accept neither booleans, nonfinite values nor malformed strings as numeric limits. Environment overrides config.txt through the existing loader.

| Key | Default / hard maximum |
| --- | --- |
| `CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED` | false; conjunctive with `CLAIMS_JOBS_ENABLED` |
| `CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD` | 3; existing bounded retry parser, 0..100 |
| `CLAIMS_CLUSTER_REBUILD_MAX_EXACT_CLAIMS` | 50,000 |
| `CLAIMS_CLUSTER_REBUILD_MAX_EMBEDDING_CLAIMS` | 5,000 |
| `CLAIMS_CLUSTER_REBUILD_MAX_TEXT_BYTES` | 33,554,432 |
| `CLAIMS_CLUSTER_REBUILD_MAX_VECTOR_WIDTH` | 4,096 |
| `CLAIMS_CLUSTER_REBUILD_MAX_EMBEDDING_COMPONENTS` | 2,000,000 |
| `CLAIMS_CLUSTER_REBUILD_MAX_SIMILARITY_COMPONENT_COMPARISONS` | 50,000,000 |
| `CLAIMS_CLUSTER_REBUILD_MAX_CLEANUP_REFERENCES` | 50,000; bounds old cluster IDs, assignment rows, membership rows and their unique claim-ID union separately |

Jobs request retention and alias constants are 86,400 seconds and 128; keep them Jobs-owned. Limits are execution controls, not canonical semantic settings. The worker uses its validated limits and the Job's admitted method/threshold/algorithm, never recalculates semantic defaults.

### Test And Commit Conventions

Before every Python command activate `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate`. Run commands below from the worktree root with that venv active. Test bodies show the essential assertions; define local setup fixtures of the stated interface types rather than building a second test framework.

For PostgreSQL use existing `jobs_pg_dsn` / `pg_jobs` tests for Jobs and `pg_restricted_backend` / `postgres` tests for Media DB. Read their current registration in `tldw_Server_API/tests/conftest.py` and official AuthNZ fixtures. Skip only on their reported unavailability; no custom cluster, superuser-only proof or fabricated owner scope.

Every task runs RED, minimal GREEN, relevant characterization regressions, focused lint and Bandit on its touched production files before its commit. Use normal hooks, stage only listed files plus its Backlog record, and commit as `feat(jobs): ... (TASK-<execution-id>)` or `feat(claims): ... (TASK-<execution-id>)`. Record this plan path, evidence and remaining stages in each task. A task is not complete merely because its new test passes.

## Stage 1: Shared Jobs Resource Protocol

**Goal:** Ship a reusable Jobs-owned resource capability before any Claims producer uses it.
**Success Criteria:** Both backends enforce admission/replay and every relevant lifecycle path without changing opted-out Jobs behavior.
**Tests:** Tasks 1-3 plus existing admission, lease, retry, quarantine, pruning, RLS, counter and event regressions.
**Status:** Not Started

### Task 1: Contracts, Registration And Durable Schema

**Files:** Create the neutral Claims contract, Jobs resource contracts/policy and backend resource modules listed above. Modify `Jobs/migrations.py`, `pg_migrations.py`, `models.py`, `operations/contracts.py` and backend `__init__.py` exports. Create `tldw_Server_API/tests/Jobs/test_jobs_resource_contracts.py`, `test_jobs_resource_migrations.py` and `tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_contract.py`.

**Interfaces:** Produces the shared types above, `canonical_cluster_settings(settings: ClusterRebuildSettings) -> dict[str, object]` in the neutral Claims contract, `settings_fingerprint(settings: dict[str, object]) -> str` in Jobs resource contracts, and `resource_requirement_for(domain: str, job_type: str) -> str | None` in Jobs resource policy. Canonical JSON uses sorted keys, compact separators, UTF-8 and `allow_nan=False`; normalize negative zero. Exact settings omit threshold from semantic encoding.

- [ ] **Step 1: Write failing contract and migration tests.** Pin exact/default threshold equality, explicit exact threshold omission, integer/owner/version validation and frozen binding fields. Migration tests cover fresh/upgrade/repeated runs and archived resource metadata.

```python
def test_exact_threshold_is_not_a_semantic_setting():
    first = ClusterRebuildSettings(1, "exact", 2, None, 1)
    assert "similarity_threshold" not in canonical_cluster_settings(first)
    assert resource_requirement_for("claims", "claims_rebuild_clusters") == "cluster-rebuild"
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_resource_contracts.py tldw_Server_API/tests/Jobs/test_jobs_resource_migrations.py tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_contract.py`; expect missing new contracts/schema, not fixture failure.
- [ ] **Step 3: Implement the contracts and additive migrations.** Create `job_resource_heads` keyed by domain/owner/resource with authority UUID, nonnegative generation and nullable active UUID. Add immutable nullable resource metadata/settings encoding/fingerprint to `jobs` and `jobs_archive`; all-null means opted out, partial bindings are invalid. Add `job_resource_request_receipts` keyed by full scope/key digest with settings encoding/fingerprint, Job UUID and expiry; index UUID and expiry. Apply owner/domain RLS to both new tables. No lifecycle/result copy goes in receipts. Register the exact type and reserve its `.v<positive integer>` alias family; unsupported aliases fail closed until explicitly supported. Other unrelated types remain opted out.
- [ ] **Step 4: Run GREEN and characterization.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py tldw_Server_API/tests/Jobs/test_jobs_migrations_compat_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py`; expect no new failure and only official-fixture availability skips.
- [ ] **Step 5: Verify, review and commit.** Check constraints reject partial bindings and queue changes cannot change the resource key. Record migration/RLS evidence; commit only this independently usable prerequisite.

### Task 2: Atomic Admission And Retained Request Replay

**Files:** Modify `Jobs/manager.py`, `operations/sqlite/admission.py`, `operations/postgres/admission.py` and Task 1 resource modules. Create `tldw_Server_API/tests/Jobs/test_jobs_resource_admission_sqlite.py`, `test_jobs_resource_admission_postgres.py` and `property/test_jobs_resource_admission_properties.py`.

**Interfaces:** Consumes Task 1 and existing `CreateJobCommand` with explicit `priority=5`. Produces `JobManager.admit_resource_job(command: CreateJobCommand, *, request: ResourceAdmissionRequest) -> ResourceAdmission`. Ordinary `admit_job` and `create_job` continue their existing public contracts and reject resource-required types without the resource protocol.

- [ ] **Step 1: Write failing admission tests.** Use two managers/connections and barriers, not sleeps. Exercise equal/conflicting requests across queues/methods, independent owners, full canonical equality despite forced digest collision, retained-key replay after terminal/archive, conflicting keys and expired keys. Pin receipt aliases, quota/event/counter rollback and ambiguous/missing head targets. The sample `request` fixture has `request_key="request-1"`; keyless reuse instead expects `active_reused`.

Include `test_retained_replay_survives_queue_policy_change`: admit with a key, make new submissions disallowed by queue policy/quota, replay that key, and assert `kind == "request_replayed"`, identical UUID and exactly one creation event/admission count. Mismatched scope owner, declared resource or settings versus the validated payload must fail before revealing another Job.

```python
def test_active_reuse_does_not_submit_again(manager, command, request):
    first = manager.admit_resource_job(command, request=request)
    second = manager.admit_resource_job(command, request=request)
    assert (first.kind, second.kind) == ("created", "request_replayed")
    assert first.job["uuid"] == second.job["uuid"]
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_resource_admission_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_resource_admission_postgres.py`; expect missing facade/atomic policy failures.
- [ ] **Step 3: Implement the facade inside the existing pipeline.** Share its preparation/serialization rather than copying queue policy, secret scanning, encryption, quotas, fair-share and creation events. The registered policy verifies command owner/domain/type/resource and derives the same canonical settings from the neutral validated payload; callers cannot disguise different payloads with an unrelated settings map. Extend backend admission with resource-before-row hooks, including post-insert head/receipt attachment in the same transaction. Under `BEGIN IMMEDIATE` or a safely created locked PostgreSQL head: retained-key lookup first, exact active-row check second, new-work policy/quota checks only for fresh creation, generation allocation and insert last. Replays must remain available when current creation quotas/queue policy would reject new work; they create no event or admission count. Encrypted/truncated payload processing must not alter the immutable admitted contract: reject destructive truncation. Raw request keys never enter ordinary idempotency fields, spans or events. Enforce 128 unexpired aliases, allowing old alias replay at the ceiling.
- [ ] **Step 4: Run GREEN and properties.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Jobs/property/test_jobs_resource_admission_properties.py tldw_Server_API/tests/Jobs/test_jobs_manager_admission_facade.py tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_admission_operations_postgres.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py`; assert one creation event/submission and no regressions in opted-out Jobs.
- [ ] **Step 5: Verify, review and commit.** Fault between insert/head/receipt/event writes must roll back all four. Public imports/batch creates of the registered type cannot bypass the resource requirement.

### Task 3: Lifecycle Closure, Retention And Attempt Authority

**Files:** Modify `Jobs/manager.py`, `admin_operations.py`, backend resource modules, and `operations/postgres/lifecycle.py` where it handles opted-in transitions. Create `tldw_Server_API/tests/Jobs/test_jobs_resource_lifecycle.py` and `test_jobs_resource_retention.py` with SQLite/PostgreSQL parametrization through official fixtures.

**Interfaces:** Consumes `ResourceAttempt`. Produces `JobManager.validate_resource_attempt(attempt: ResourceAttempt) -> bool`, an exact read checking domain/owner/type/UUID/resource authority/generation, matching active-head UUID, processing state, worker/lease identity and live lease. It is not a cross-store transaction or cancellation promise. Existing lifecycle APIs retain their public results.

- [ ] **Step 1: Write failing transition tests.** Cover same-Job retries retaining generation, failed/cancelled/quarantined admin reactivation, published logical requeue, a successor active conflict, stale completion, numeric-ID replacement, TTL, lease exhaustion, batch cancellation/failure and repair/prune races.

```python
def test_old_completion_cannot_release_successor(resource_scenario):
    old, successor = resource_scenario.cancel_then_admit_successor()
    resource_scenario.try_stale_completion(old)
    assert resource_scenario.active_uuid() == successor["uuid"]
    assert not resource_scenario.manager.validate_resource_attempt(old)
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_resource_lifecycle.py tldw_Server_API/tests/Jobs/test_jobs_resource_retention.py`; local `resource_scenario` fixture wraps public Jobs APIs and reads persisted heads for assertions only.
- [ ] **Step 3: Route every opted-in mutation through resource locks.** Inventory `complete_job`, `fail_job`, `cancel_job`, `release_job`, acquired-lease recovery, retry admission, `retry_now_jobs`, quarantined requeue, batch complete/fail/cancel, TTL/integrity sweep, reschedule and archive/prune/import repair. Re-read preliminary row identity after head-then-row locking. Automatic queued retries keep the active pointer; terminalization clears only its matching UUID. Reactivation claims an empty slot or conflicts without changing another Job, retaining the old logical generation. Validate attempt authority through Jobs, never in a Claims polling loop. Preserve normal counters/outbox/lease behavior.
- [ ] **Step 4: Run GREEN and lifecycle characterization.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py tldw_Server_API/tests/Jobs/test_failed_job_requeue_admission.py tldw_Server_API/tests/Jobs/test_jobs_ttl_prune_transaction_boundaries.py tldw_Server_API/tests/Jobs/test_jobs_prune_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_prune_postgres.py`. Unexpired resource key receipts keep an exact active/archive target; pruning expired receipts never resets the head counter or consults Media receipts.
- [ ] **Step 5: Verify, review and commit.** Run barriers for admission versus terminalization/reactivation/prune in both backends. Missing/ambiguous targets fail closed; never repair by lowering counters or substituting numeric IDs. Stage 1 is complete only after this inventory is covered.

## Stage 2: Jobs-Owned Safe Status Reads

**Goal:** Return usable owner/admin status links without expanding administration.
**Success Criteria:** Exact UUID identity survives archival and numeric-ID reuse; permission-only Claims admins receive only their own safe projection.
**Tests:** Task 4 and existing Jobs admin/RBAC/router contracts.
**Status:** Not Started

### Task 4: Owned Projection And UUID-Bound Admin Detail

**Files:** Create `tldw_Server_API/app/core/Jobs/owned_status.py`, `tldw_Server_API/app/api/v1/endpoints/jobs_owned.py`, `tldw_Server_API/app/api/v1/schemas/jobs_owned.py`, `tldw_Server_API/app/api/v1/API_Deps/jobs_admin_deps.py`. Modify `Jobs/manager.py`, `endpoints/jobs_admin.py`, `router_groups/admin.py`, `router_groups/minimal.py`. Create `tldw_Server_API/tests/Jobs/test_jobs_owned_status_api.py` and `test_jobs_admin_uuid_detail.py`; extend `tests/Services/test_router_groups_contract.py`.

**Interfaces:** Produces `get_owned_job_status(manager: JobManager, *, job_uuid: str, owner_user_id: str) -> OwnedJobStatus | None` and `GET /api/v1/jobs/owned/{job_uuid}`. Add optional `job_uuid: UUID | None` to existing admin `get_job_detail`; pass the canonical UUID to the existing manager's UUID-precondition argument. Define `OwnedJobStatus` in the schema file and `async def authorize_jobs_admin_domain_read(principal: AuthPrincipal, *, domain: str) -> None` in `API_Deps/jobs_admin_deps.py`. Existing owner-safe UUID lookup is reusable; numeric ID alone is never used for owned reads.

- [ ] **Step 1: Write failing API tests.** Create a minimal authenticated router app with existing dependencies. Own UUID succeeds, foreign/missing/disallowed type returns 404, and a permission-only Claims admin does not acquire admin controls. Archive old ID 1, create replacement ID 1, then assert old owner/admin links still resolve the original UUID or 404 after final retention expiry. Preserve legacy admin detail behavior without a precondition.

```python
def test_admin_uuid_link_cannot_follow_reused_id(client, archived_and_reused):
    old, replacement = archived_and_reused
    response = client.get(f"/api/v1/jobs/{old['id']}",
                          params={"domain": "claims", "job_uuid": old["uuid"]})
    assert response.status_code == 200
    assert response.json()["uuid"] == old["uuid"] != replacement["uuid"]
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_owned_status_api.py tldw_Server_API/tests/Jobs/test_jobs_admin_uuid_detail.py`; expect missing route/schema/query precondition.
- [ ] **Step 3: Implement allowlisted reads and registration.** The separate owned router requires a real authenticated principal, strict owner equality and ingress guards, not `jobs_admin`'s global role dependency. Register it with its own `jobs-owned` route key and `default_stable=True`, plus minimal-app registration, without making admin controls stable by association. Initially project only `claims_rebuild_clusters`: identity, canonical status, numeric progress, bounded count/result fields and recognized failure codes. Never forward arbitrary progress messages, payloads, raw errors or provider data. Admin detail retains `RequireRole("admin")` and domain/RLS checks. Move its domain authorization helpers into the Jobs API dependency module with compatibility imports where existing tests require them. The new async guard awaits `RequireRole("admin")(principal)` and runs the same domain checks; do not replace its admin-claim bypass with a simplistic role-string test. Claims calls this at the endpoint, avoiding core-to-endpoint imports.
- [ ] **Step 4: Run GREEN and compatibility.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_get_job_owner_scope.py tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py tldw_Server_API/tests/AuthNZ_Unit/test_jobs_admin_permissions_claims.py tldw_Server_API/tests/Services/test_router_groups_contract.py`. Pin OpenAPI/auth dependency metadata, Postgres owner predicates and sanitized failure projection.
- [ ] **Step 5: Verify, review and commit.** Stable-route toggles must not leave an admitted caller with an unregistered status route; test registration alongside Claims Jobs mode. Queue administration is unchanged.

## Stage 3: Bounded Domain Calculation And Publication

**Goal:** Produce replay-safe cluster output without holding a database write lock during calculation.
**Success Criteria:** Both backends preserve ownership, previous output on failure, receipt identity and generation ordering; maintenance access never initializes absent storage.
**Tests:** Tasks 5-8 and current clustering, cluster helpers, Watchlists and Media DB regressions.
**Status:** Not Started

### Task 5: Source Snapshots, Cleanup Integrity And Domain Schema

**Files:** Create the Media domain types, snapshot helper and both migration files in the map. Modify `media_db/media_database_impl.py`, `schema/migrations.py`, `schema/features/postgres_rls.py`, `runtime/claims_cluster_ops.py`. Create `tldw_Server_API/tests/DB_Management/test_claim_cluster_rebuild_snapshot.py` and `tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_snapshot_postgres.py`.

**Interfaces:** Produces `MediaDatabase.read_claim_cluster_source(*, owner_user_id: str, claim_limit: int, text_byte_limit: int) -> ClusterSourceSnapshot` and `read_claim_cluster_source_on_connection(db: MediaDatabase, conn: Any, *, owner_user_id: str, claim_limit: int, text_byte_limit: int, lock: bool) -> ClusterSourceSnapshot` in the snapshot module. The public wrapper opens a read transaction; publication uses the connection-taking function, never an independently committed nested transaction. Also produce transaction-internal `lock_claim_cluster_cleanup(db: MediaDatabase, conn: Any, *, owner_user_id: str, limit: int) -> tuple[int, ...]`. It returns only scoped cleanup claim IDs to the publisher, never foreign data to Jobs/API. Register public helpers by imports/bindings in `media_database_impl.py`, following review-metrics helpers.

- [ ] **Step 1: Write failing source/schema tests.** Pin owner predicates despite team/org visibility, active claim eligibility, stable claim-ID order, digest additions/deletions/text/chunk/media changes, read limits, and an existing owner DB opened without initialization. Seed A's old cluster with a claim transferred to B before reading A's source; under A's restricted RLS context cleanup must fail, not conclude the hidden Media is absent from all references.

```python
def test_transferred_reference_is_not_hidden_by_owner_snapshot(transferred_db):
    with pytest.raises(ClusterRebuildStorageError) as error:
        with transferred_db.transaction() as conn:
            lock_claim_cluster_cleanup(transferred_db, conn, owner_user_id="42", limit=50_000)
    assert error.value.code == "claims_cluster_foreign_reference"
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/DB_Management/test_claim_cluster_rebuild_snapshot.py tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_snapshot_postgres.py`.
- [ ] **Step 3: Implement the snapshot, schema and cleanup lock.** Use a bounded consistent read transaction (SQLite deferred read; PostgreSQL repeatable-read read-only) with keyset claim-ID pages and length-delimited digest fields (not separator concatenation). Add `claims_cluster_publication_heads` and `claims_cluster_publication_receipts` with the spec's exact fields and owner policies to fresh bootstrap DDL as well as upgrade migrations; provision only through schema deployment, never worker access. Keep authority nullable only before the first publication, and initialize it in that first publication transaction. Define `ClusterRebuildStorageError(code: str, *, retryable: bool)` in domain types; include source-change, foreign-reference, corruption and resource-limit codes without row content.
- [ ] **Step 4: Implement bounded reference protection before destructive writes.** Lock the owner's old cluster rows, then enumerate assignment/membership reference IDs without an owner-filtering Media join; resolve and lock their Media ownership rows separately, then Claims rows in deterministic ID order. At this baseline Claims/membership references remain visible while foreign Media may be RLS-hidden: missing/foreign owner proof fails closed. Do not use an inner join that drops those references. Check actual table RLS/visibility assumptions before trusting reference enumeration; prove them with the restricted fixture and fail closed if a deployed policy can hide references. Do not broaden the worker role. Adapt `add_claim_to_cluster` to lock/revalidate its destination cluster before membership/assignment changes, using the same cluster -> Media -> Claims order, so a late insert cannot target a deleted cluster. This is bounded cluster-writer coordination, not a global source-writer protocol. Bound old clusters, membership rows, assignment rows and their unique source IDs independently; exceeding any ceiling fails before deletion.
- [ ] **Step 5: Run GREEN and review the boundary.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_existing_only.py tldw_Server_API/tests/DB_Management/test_media_db_claims_cluster_ops.py tldw_Server_API/tests/DB_Management/test_media_db_postgres_claims_collection_structures.py`. Characterize concurrent `add_claim_to_cluster`, transfer-before-snapshot and transfer-after-lock; no new permission, notification or HTTP contract. Verify, review and commit.

### Task 6: Bounded Exact/Embedding Calculation And Non-Creating Vector Access

**Files:** Create `tldw_Server_API/app/core/Claims_Extraction/claims_cluster_rebuild.py`, `claims_cluster_vectors.py`; modify `claims_clustering.py` only for small shared pure helpers, `Embeddings/ChromaDB_Library.py` and `DB_Management/db_path_utils.py` for explicit non-creating access. Create `tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_calculation.py`, `test_claims_cluster_vectors_existing_only.py`, and `property/test_claims_cluster_rebuild_limits.py`.

**Interfaces:** Consumes Task 5 types. Produces `calculate_claim_cluster_rebuild(snapshot: ClusterSourceSnapshot, *, settings: ClusterRebuildSettings, limits: ClusterRebuildLimits, read_vectors: Callable[[tuple[str, ...]], Mapping[str, Sequence[float]]] | None) -> ClusterCalculation` and `open_claim_cluster_vector_reader(*, owner_user_id: str, embedding_config: Mapping[str, Any]) -> ContextManager[Callable[[tuple[str, ...]], Mapping[str, Sequence[float]]] | None]`. `None` means proven absent collection/storage, not generic failure.

- [ ] **Step 1: Write failing behavioral tests.** Compare exact normalization/first source representative and embedding greedy claim-ID order, strict `sim > best_sim` behavior and current centroid/member similarity rules to existing characterization tests. Pin empty exact, all/partial missing vectors, dimensions, finite components and finite positive computed norms. Inject tiny limits and verify counts, retained bytes/components and loop comparison budgets fail before publication.

Include `test_vector_access_error_is_not_missing_embeddings`: make the existing-only reader raise a connection/access error and assert a retryable `ClusterRebuildStorageError`, not `None`, empty vectors or a successful skip. Its permanent malformed-schema counterpart also fails without creating files/tables.

```python
def test_missing_vectors_are_a_preserving_skip(snapshot, settings, limits):
    result = calculate_claim_cluster_rebuild(snapshot, settings=settings,
        limits=limits, read_vectors=lambda ids: {})
    assert result.outcome == "skipped" and result.reason == "no_embeddings"
    assert result.assignments == ()
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_calculation.py tldw_Server_API/tests/Claims/test_claims_cluster_vectors_existing_only.py`.
- [ ] **Step 3: Implement calculation and the existing-only reader.** Keep legacy synchronous behavior intact; no clustering redesign. Characterize colliding embedding IDs and preserve existing successful mapping semantics rather than silently expanding claim coverage; SDK rejection is a classified failure, never `no_embeddings`. Use bounded ID batches, validate result IDs/cardinality and preserve characterized missing counts. Add keyword-only `existing_only: bool = False` to `ChromaDBManager.__init__`, retaining its existing arguments/default behavior, and `get_existing_collection_strict(collection_name: str) -> Collection`. Add `DatabasePaths.resolve_user_chroma_directory(user_id: int | str, *, base_dir_override: str | Path | None = None) -> Path` using the existing non-creating user-directory resolver. The async reader uses those paths and strict lookup, with sanitized error translation. Do not reuse the legacy wrapper that converts arbitrary `ValueError` into collection absence, enable stub fallback after provider failure, or invoke get-or-create. Validate persisted schema before any client initialization; access errors fail, version/schema incompatibility fails, proven absence skips. Test disappearance between resolution and open too: if the SDK cannot enforce the no-provisioning contract, stop for a narrowly reviewed storage-access design rather than silently create or weaken it.
- [ ] **Step 4: Run GREEN and properties.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Claims/property/test_claims_cluster_rebuild_limits.py tldw_Server_API/tests/Claims/test_claims_clustering_embeddings.py tldw_Server_API/tests/Claims/test_claim_cluster_upsert_idempotency.py tldw_Server_API/tests/ChromaDB/unit/test_chromadb_manager.py tldw_Server_API/tests/ChromaDB/unit/test_chromadb_dimensions_and_list.py`. Memory/CPU ceilings are inputs/work limits, not a claimed hard process RSS/deadline.
- [ ] **Step 5: Verify, review and commit.** Confirm all malformed/error paths retain prior domain output because this task has no write path. Existing-only vector access is a prerequisite, not optional worker cleanup.

### Task 7: Atomic Publication, Receipts And Generation Fencing

**Files:** Create `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_cluster_rebuild_publication.py`; bind its helpers in `media_database_impl.py`. Create `tldw_Server_API/tests/DB_Management/test_claim_cluster_rebuild_publication.py`, `tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_publication_postgres.py`, and `tldw_Server_API/tests/Claims/property/test_claim_cluster_publication_order.py`.

**Interfaces:** Produces `MediaDatabase.load_claim_cluster_publication(*, identity: ClusterPublicationIdentity) -> ClusterPublicationResult | None` and `MediaDatabase.publish_claim_cluster_rebuild(*, identity: ClusterPublicationIdentity, settings: ClusterRebuildSettings, snapshot: ClusterSourceSnapshot, calculation: ClusterCalculation, limits: ClusterRebuildLimits) -> ClusterPublicationResult`. Uses Task 5 transaction-scoped cleanup locks, not public legacy rebuild methods that open/commit their own transactions.

- [ ] **Step 1: Write failing publication tests.** Pin one publication under duplicate attempts, receipt replay preserving IDs/counts, source mismatch rollback, late older generation, skip generation advance, authority mismatch and equal-generation/different-UUID corruption. Fault every destructive/receipt write. Exercise foreign references before snapshot and after a source-change retry, plus oversized old memberships.

```python
def test_replay_does_not_rebuild(publisher, publication_request):
    first = publisher.publish(**publication_request)
    ids = publisher.cluster_ids()
    replay = publisher.publish(**publication_request)
    assert replay.result == first.result and not replay.newly_published
    assert publisher.cluster_ids() == ids
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/DB_Management/test_claim_cluster_rebuild_publication.py tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_publication_postgres.py`.
- [ ] **Step 3: Implement exactly one publication transaction.** Serialize owner head, recheck exact receipt metadata, reject authority/collision corruption, return superseded without mutation for a higher applied generation, lock/re-digest the bounded source and cleanup references, then replace only validated owner rows and insert the receipt/head advance together. PostgreSQL locks old clusters -> Media -> Claims in stable order through commit, using the same connection; later source arrivals are excluded, not broadly cleared. An embedding skip still validates source and advances its generation but does not delete clusters. Store result counts/method/effective threshold/outcome/reason only; `newly_published` is not a receipt field.
- [ ] **Step 4: Run GREEN and ordering properties.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Claims/property/test_claim_cluster_publication_order.py tldw_Server_API/tests/DB_Management/test_media_db_claims_cluster_assignment_rebuild_ops.py tldw_Server_API/tests/DB_Management/test_media_db_claims_cluster_exact_rebuild_ops.py`. Barriers prove edits/deletes/transfers of assigned rows cannot pass through commit; post-snapshot inserts may await a new rebuild. Domain history must survive Jobs receipt/job pruning.
- [ ] **Step 5: Verify, review and commit.** A committed receipt is the replay authority even after a newer publication; it reports that Job's immutable result without restoring its old output. No failure path writes a success receipt or lowers a watermark.

### Task 8: Existing-Only Best-Effort Watchlists

**Files:** Modify `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`; create `tldw_Server_API/app/core/Claims_Extraction/claims_cluster_watchlists.py`, `tldw_Server_API/tests/Watchlists/test_watchlists_existing_only.py`, and `tldw_Server_API/tests/Claims/test_claims_cluster_watchlists_existing_only.py`.

**Interfaces:** Produces `WatchlistsDatabase.open_existing(*, user_id: str, backend: DatabaseBackend) -> WatchlistsDatabase | None` using an already validated existing Media DB backend, and `evaluate_published_cluster_watchlists(db: MediaDatabase, *, owner_user_id: str) -> None`. Missing required schema returns `None`; permission/storage errors propagate to a sanitized best-effort boundary. Do not close a caller-owned backend or refresh to a global backend.

- [ ] **Step 1: Write failing no-provisioning tests.** Existing Media DB with no Watchlists tables remains unchanged after evaluation. Verify full owner/backend pinning, no directory/database/table creation, no constructor/bootstrap call, sanitized access errors, normal notifications when schema exists and unchanged legacy `for_user` behavior.

```python
def test_missing_watchlists_schema_is_not_initialized(media_db):
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    before = media_db.execute_query(query).fetchall()
    assert WatchlistsDatabase.open_existing(user_id="42", backend=media_db.backend) is None
    assert media_db.execute_query(query).fetchall() == before
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Watchlists/test_watchlists_existing_only.py tldw_Server_API/tests/Claims/test_claims_cluster_watchlists_existing_only.py`.
- [ ] **Step 3: Implement a non-creating construction path and evaluator.** Reuse existing subscription/count/notification logic, not `_get_watchlists_db` or a `for_user` constructor. Verify every table queried/written by this evaluator, not merely the generic six-table startup sentinel. Skip absent schema; catch access errors once without raw SQL/provider details. No separate delivery worker, retry queue or notification durability protocol is introduced.
- [ ] **Step 4: Run GREEN and characterization.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_watchlist_notifications.py tldw_Server_API/tests/Watchlists/test_watchlists_db_user_scope.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py`; include official PostgreSQL owner/backend tests in the new file.
- [ ] **Step 5: Verify, review and commit.** Invoke evaluation only on `newly_published=True` later in Task 9. A failure cannot invalidate publication or trigger a rebuild; a crash before evaluation can omit it, as documented.

## Stage 4: Claims Worker And API Adoption

**Goal:** Connect the proven shared/domain APIs without creating a Claims execution system.
**Success Criteria:** Real WorkerSDK execution and 200/202/409 contracts work with strict ownership and existing-only async dependencies.
**Tests:** Tasks 9-10 and existing Claims worker/config/API suites.
**Status:** Not Started

### Task 9: Immutable Job Contract, Configuration And Worker Handler

**Files:** Modify `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`, `claims_job_handlers.py`, `claims_jobs.py`, `tldw_Server_API/app/services/claims_jobs_worker.py`, `tldw_Server_API/app/core/config.py`, `tldw_Server_API/Config_Files/config.txt`. Create `tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_jobs_worker.py` and `tldw_Server_API/tests/Config/test_claims_cluster_rebuild_config.py`.

**Interfaces:** Produces `enqueue_claims_cluster_rebuild(*, owner_user_id: str, settings: ClusterRebuildSettings, request_key: str | None, job_manager: JobManager, settings_obj: Mapping[str, Any]) -> ResourceAdmission`, `claims_cluster_rebuild_jobs_enabled(settings_obj: Mapping[str, Any]) -> bool` and `claims_cluster_rebuild_limits(settings_obj: Mapping[str, Any]) -> ClusterRebuildLimits`. Extend existing `process_claims_job(job: dict[str, Any]) -> dict[str, Any]`; its new synchronous calculation/publication helper runs in `asyncio.to_thread`.

- [ ] **Step 1: Write failing real-worker tests.** Enqueue through Task 2, acquire through WorkerSDK and inspect actual domain output. Pin strict payload keys/owner/resource metadata, missing owner DB permanent failure, unsupported versions, transient source/DB/Chroma retries, malformed vectors permanent failure, worker-disabled queued visibility and every config default/bound. Cancel while a thread is paused after authority validation, then publish a successor and resume: no newer output may be overwritten.

```python
def test_resource_flag_is_conjunctive():
    assert not claims_cluster_rebuild_jobs_enabled({
        "CLAIMS_JOBS_ENABLED": False, "CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED": True})
    assert claims_cluster_rebuild_limits({}).exact_claims == 50_000
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_jobs_worker.py tldw_Server_API/tests/Config/test_claims_cluster_rebuild_config.py`.
- [ ] **Step 3: Implement admission and handler integration.** Enqueue an explicit priority-5 `CreateJobCommand` with only version/owner/method/min_size/threshold/algorithm fields. Validate stored owner/payload/resource identity before access, use Task 3 authority checks before dispatch and publication, open owner-scoped existing-only Media DB sessions without synthetic admin, replay receipt before vector work, calculate/publish, then invoke Task 8 only for a new receipt. Map domain errors to `ClaimsJobError` with stable codes and retryability. No handler calls `complete_job`, polls cancellation or renews leases. Use WorkerSDK's existing `run(job_types=...)` allowlist for supported Claims types; no second worker/loop.
- [ ] **Step 4: Run GREEN and compatibility.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py tldw_Server_API/tests/Claims/test_claims_review_metrics_worker_e2e.py tldw_Server_API/tests/Config/test_claims_env_int_parsing.py`. Add official restricted-role Media DB worker coverage and confirm context is restored after exceptions.
- [ ] **Step 5: Verify, review and commit.** Crash after domain commit but before completion must replay without vector loading, new cluster IDs or watchlist evaluation. This includes legitimate skip receipts and explicitly reactivated already-published Jobs.

### Task 10: Producer Routing, Authorized Links And API Compatibility

**Files:** Modify `tldw_Server_API/app/api/v1/endpoints/claims.py`, `schemas/claims_schemas.py`, `API_Deps/DB_Deps.py` and `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`. Create `tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_jobs_api.py`; extend `test_claims_clusters_api.py` and `tests/Config/test_claims_cluster_rebuild_config.py`.

**Interfaces:** Define frozen `ClaimsClusterRebuildRouteContext(async_enabled: bool, db: MediaDbSession | None)` in `API_Deps/DB_Deps.py`. Produce `async def get_claim_cluster_rebuild_context(request: Request, current_user: User = Depends(get_request_user)) -> AsyncIterator[ClaimsClusterRebuildRouteContext]`, capturing the routing flag once and delegating to the existing request DB dependency only in synchronous mode. Async mode yields a context with `db=None`. Preserve the existing service's synchronous signature; add `admit_claim_cluster_rebuild(*, min_size: int, method: str | None, similarity_threshold: float | None, user_id: int | None, request_key: str | None, principal: AuthPrincipal, current_user: User, job_manager: JobManager) -> ResourceAdmission`. New Pydantic types are `ClaimsClusterRebuildAccepted` and `ClaimsClusterRebuildConflict`; leave the legacy 200 result fields unchanged.

- [ ] **Step 1: Write failing API tests.** Default/disabled response matches legacy 200, Jobs returns 202 plus `Location=status_url`, equal active requests reuse, differing settings return stable 409, keys replay terminal Jobs, quotas/rate limits keep their existing mappings, and uncertain admission is 503 without local execution. A forbidden cross-owner request reveals no Job reference; a platform admin needs the existing Jobs role/domain authorization before admission. Test malformed/raw-key logging, Unicode/whitespace/129-character keys, NaN/booleans at contract boundaries and ignored exact threshold.

```python
def test_async_routing_does_not_open_creating_media_db(client, no_creating_db):
    response = client.post("/api/v1/claims/clusters/rebuild", params={"method": "exact"})
    assert response.status_code == 202
    assert response.headers["location"] == response.json()["status_url"]
    no_creating_db.assert_not_called()
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_jobs_api.py`; make creating DB/local rebuild collaborators raise if the async branch calls them.
- [ ] **Step 3: Implement routing/authentication before DB construction.** Use the captured route context, not a second flag read. Make this endpoint async so it can await Task 4's admin guard and the ingress limiter; run synchronous legacy service/admission work in the existing threadpool utility, never block the event loop. Resolve Claims permission and strict owner first; only platform admins may target another owner. For a foreign-owner admin link await the Jobs-owned guard before admission, then generate `/api/v1/jobs/{id}?domain=claims&job_uuid={uuid}`. Own-owner responses use `/api/v1/jobs/owned/{uuid}`. If route toggles omitted the required status route, fail with sanitized 503 before admission rather than publish a broken link. Resolve JobManager lazily only in async mode after authorization, not an eager dependency on the disabled path. Apply existing ingress rate limiting to every async attempt including replay; leave disabled synchronous behavior intact. Invalid async request keys return 422. Map conflict outcomes to 409 codes `claims_cluster_active_conflict` and `claims_cluster_request_conflict` with an authorized reference; retain ordinary Jobs rejection mapping, never conflate quotas with conflicts. Document 200/202/409/403/503 in OpenAPI without copying mutable Jobs status into Claims.
- [ ] **Step 4: Run GREEN and API characterization.** Rerun RED command plus `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_clusters_api.py tldw_Server_API/tests/Claims/test_claims_cluster_links_and_search.py tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py tldw_Server_API/tests/Services/test_router_groups_contract.py`. Prove absent owner storage is never created by dependencies and an admitted Job never invokes synchronous fallback, even after acknowledgement loss. Change the routing flag after dependency resolution and assert the request still uses its captured branch, with no second creating dependency or lost synchronous cleanup.
- [ ] **Step 5: Verify, review and commit.** Check route toggles, OpenAPI unions, sanitized references and the permission-only Claims-admin case end to end. Keep local rebuild code and its current default response intact.

## Stage 5: Cross-Backend Qualification And Operational Handoff

**Goal:** Demonstrate failure safety and document a controlled opt-in rollout.
**Success Criteria:** Fault/property tests cover all approved invariants; SQLite and official PostgreSQL evidence and a usable runbook are recorded before release.
**Tests:** Task 11 and the full scoped verification below.
**Status:** Not Started

### Task 11: Fault Matrix, Runbook And Whole-Branch Verification

**Files:** Create `tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_jobs_e2e.py`, `test_claims_cluster_rebuild_recovery.py`, `property/test_claims_cluster_rebuild_recovery.py`, `tldw_Server_API/tests/Config/test_claims_cluster_rebuild_docs.py` and `Docs/Operations/Claims_Cluster_Rebuild_Jobs.md`. Modify the three existing code-documentation files in the map. Update execution Backlog tasks and this plan's stage statuses as evidence is obtained.

**Interfaces:** Consumes the completed Tasks 1-10 without a new production service. The recovery scenario fixture uses real Jobs/domain APIs, separate managers/connections and barrier/fault hooks at transaction boundaries; it is test code, not a Claims coordinator.

- [ ] **Step 1: Write failing recovery/docs tests.** Pin lost acknowledgements, all retained/expired key cases and the 128-alias ceiling, process-separated admission races, crash-after-receipt, old cancelled/expired thread versus newer applied/skip generation, failed reactivation after owner DB reappears, archive/prune independence and exact UUID status retention. For restore tests show matching UUID alone is insufficient after counter rewind or lost receipts; verify operator pre-start evidence rejects an unproven pair rather than claim automatic detection. Docs tests pin actual flag names/defaults, both safe URLs and the physical-handler rollback warning.

```python
def test_commit_then_crash_replays_without_new_clusters(recovery_scenario):
    job, first_ids = recovery_scenario.commit_then_lose_completion()
    recovery_scenario.recover_same_job(job)
    assert recovery_scenario.cluster_ids() == first_ids
    assert recovery_scenario.watchlist_evaluations() <= 1
```

- [ ] **Step 2: Run RED.** `python -m pytest -q tldw_Server_API/tests/Claims/test_claims_cluster_rebuild_recovery.py tldw_Server_API/tests/Config/test_claims_cluster_rebuild_docs.py`; missing runbook/cases must fail meaningfully, not merely import successfully.
- [ ] **Step 3: Complete qualification and write the runbook.** Document migrate -> compatible worker allowlists -> owner/status/backend probes -> drain sync -> uniform producer enable. Rollback stops producer admission, drains/cancels through Jobs, waits for actual handlers/threads, then returns to sync; never drop heads/receipts. Restore is a quiescent paired checkpoint with provenance, authority equality, Jobs counter >= domain watermark, exact equal-generation UUID checks and receipt completeness evidence. An unprovable pair stays disabled for separately reviewed recovery. Explicitly describe unstable cluster IDs/links/subscriptions, best-effort watchlist crash loss, owner DB failure versus no-embedding skip and keyless transport limits. Do not automate broad restore repair or add fallback cleanup here.
- [ ] **Step 4: Run GREEN and the scoped suite.** Run every new test plus the baseline commands below, then test both Jobs backends and Media backends with official fixtures, including restricted roles. Record exact counts, skips and environment reasons; a missing fixture is not parity proof. Run touched-file compilation, the repository's focused lint, applicable full-branch pre-commit hooks and touched-production Bandit with zero new findings/errors. Stop and fix validated regressions before declaring a stage complete.
- [ ] **Step 5: Review and commit the handoff.** Whole-branch review checks approved spec coverage, all three design corrections, complete Jobs mutation inventory and absence of Claims lifecycle state. Mark stages complete only with evidence, finalize execution tasks, and remove only this plan when every stage is actually complete per repository policy. Leave the approved spec and accepted ADR as durable records. Opening/merging a PR requires separate authorization and a genuinely human-written Change summary; never fabricate it, bypass hooks, cancel CI or push directly to dev.

### Scoped Verification Commands

Run from the worktree with the root venv active. The new files named above are mandatory additions to these baseline directories, not optional tests to omit when absent.

```bash
python -m pytest -q tldw_Server_API/tests/Claims tldw_Server_API/tests/Jobs
python -m pytest -q tldw_Server_API/tests/DB_Management/test_claim_cluster_rebuild_snapshot.py tldw_Server_API/tests/DB_Management/test_claim_cluster_rebuild_publication.py tldw_Server_API/tests/DB_Management/test_media_db_existing_only.py
python -m pytest -q tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_snapshot_postgres.py tldw_Server_API/tests/MediaDB2/test_claim_cluster_rebuild_publication_postgres.py
python -m pytest -q tldw_Server_API/tests/Watchlists/test_watchlists_existing_only.py tldw_Server_API/tests/Claims/test_claims_cluster_watchlists_existing_only.py
python -m pytest -q tldw_Server_API/tests/Config/test_claims_cluster_rebuild_config.py tldw_Server_API/tests/Config/test_claims_cluster_rebuild_docs.py tldw_Server_API/tests/Services/test_router_groups_contract.py
PYTHONPATH=tools/backlog-py/src python -m pytest -q tools/backlog-py/tests tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py tldw_Server_API/tests/CI/test_check_backlog_task_ids.py
python Helper_Scripts/ci/check_backlog_task_ids.py .
git diff --check
```

For compile/lint/Bandit/hooks, obtain the reviewed branch's exact touched-file list relative to its merge base; pass those paths explicitly, including new modules, rather than scanning unrelated main-checkout changes. Use `python -m py_compile` for touched Python, existing Ruff configuration, `python -m bandit <touched-production-files> -f json -o /tmp/bandit_claims_stage2c.json`, and `pre-commit run --files <explicit-reviewed-files>`. Stage canonical task corrections before hooks so pre-commit cannot stash the verified version and test an older record.

If subsequently authorized to publish, the seven required checks are `backend-required`, `security-required`, `coverage-required`, `frontend-required`, `e2e-required`, `container-build-check`, and `frontend-license-policy/trusted/dev`. Follow the current merge-queue policy; never rebase an armed PR. No automatic merge/rebase monitor is created by this plan.

## Coverage And Self-Review

| Approved requirement | Owning tasks |
| --- | --- |
| Neutral settings, owner validation, no sensitive payload and supported type family | 1, 2, 9 |
| Queue-independent admission, quotas, keys, aliases and no extra events | 1, 2, 3 |
| Every lifecycle/import/repair/prune path and exact attempt authority | 3 |
| Owner read projection, unchanged admin permissions and UUID links | 4, 10 |
| Existing-only source access and pre-snapshot foreign-reference guard | 5, 7, 9 |
| Exact/embedding semantics, vectors and every resource ceiling | 1, 5, 6, 9 |
| Atomic output, receipts, source locks, stale generations and replay | 5, 7, 9, 11 |
| Non-creating Watchlists and best-effort evaluation only after new publication | 8, 9, 11 |
| 200/202/409, authorization-before-lookup, rate limits and no local fallback | 9, 10 |
| Cancellation caveats, physical threads, owner deletion and coordinated restore | 3, 7, 9, 11 |
| Official PostgreSQL parity, RLS, properties, fault injection and security gates | 1-11 |
| Uniform opt-in rollout, rollback, remaining workstream and human merge gate | 11 |

Before handoff, verify every interface name/type above has exactly one defining task, each consuming task uses that spelling, all Review Focus items have named tests, and no step leaves a product decision or lifecycle mechanism to Claims. All stages remain Not Started at planning handoff. A storage SDK limitation, an unprovable integrity guard or a generic Jobs change larger than this policy is a stop-and-review condition, not permission to weaken the approved design.

**Planning self-review (2026-11-01):** Completed the spec coverage, step, type/signature, Review Focus and proportion checks. The plan has five unstarted stages and eleven red/green review units. Removed a prospective embedding-ID coverage change to preserve the approved migration scope, made the Jobs admin guard reuse asynchronous and endpoint-owned, pinned non-creating Chroma/Watchlists access, and made publication reuse its transaction connection. No execution checkbox is complete; tests in the tasks describe future implementation evidence, not passing implementations.
