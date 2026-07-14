# Protected Service Prompt Job Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee that every prompt-bearing background job executes the exact authenticated full prompt bundle resolved at enqueue time.

**Architecture:** Extend both Jobs backends with a `held` state and four protected tables for owner-scoped authenticated components, submission-bound authenticated component manifests, authenticated pin sets, and authenticated job bindings. Enqueue declares every finite prompt candidate, resolves the complete bundles, commits one multi-definition pin set, creates a held job, then atomically binds and releases it to queued. WorkerSDK verifies the binding, authenticators, owner/submission, operator mode, contracts, selection policy, and component digests before invoking a handler. No prompt text is placed in the ordinary job payload.

**Tech Stack:** Jobs SQLite/PostgreSQL migrations, existing JobManager/WorkerSDK, Context Integrity HMAC key, optional existing AES-GCM at-rest encryption, pytest including standard PostgreSQL fixtures.

---

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate.

## Task 1: Add `held` to Jobs lifecycle safely

**Files:**

- Modify: `tldw_Server_API/app/core/Jobs/models.py`
- Modify: `tldw_Server_API/app/core/Jobs/migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/pg_migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_status_guardrails.py`

- [ ] Create the implementation Backlog task and link this plan.
- [ ] Write failing SQLite/PostgreSQL tests for creating held jobs, excluding held jobs from acquisition/ready counters, compare-and-swap release `held→queued`, moving acquired prompt jobs `processing→held` when operator mode blocks stored overrides, cancellation, and stale release rejection.
- [ ] Add `held` to the shared status model and both database constraints. For existing SQLite databases, rebuild the checked table transactionally while preserving all columns, indexes, triggers, and rows; test v-old migration with representative data.
- [ ] Add narrow `create_held_job(...)`, `release_held_job(job_uuid, expected="held", connection=None)`, and lease-checked `hold_prompt_job(...)` methods. Do not make arbitrary initial statuses public.
- [ ] Ensure list/admin responses can display held jobs but normal queued/processing quota counts exclude them.
- [ ] Rerun the focused lifecycle tests and commit: `feat: add held jobs for atomic prompt binding (<task-id>)`.

## Task 2: Add protected execution tables and integrity format

**Files:**

- Modify: `tldw_Server_API/app/core/Jobs/migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/pg_migrations.py`
- Create: `tldw_Server_API/app/core/Jobs/service_prompt_store.py`
- Create: `tldw_Server_API/app/core/Service_Prompts/execution_integrity.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py`

- [ ] Write failing migration and integrity tests for clean/repeated setup, backend parity, snapshot deduplication, owner isolation, manifest substitution, binding substitution, snapshot tampering, wrong key/key ID, and optional encryption on/off.
- [ ] Add backend-equivalent tables:
  - `service_prompt_components(scope_key, digest, content_blob, content_encrypted, auth_tag, key_id, created_at, PRIMARY KEY(scope_key, digest))`
  - `service_prompt_component_manifests(uuid PK, owner_user_id, submission_id, definition_id, part_id, contract_version, component_scope_key, component_digest, canonical_manifest, manifest_digest, auth_tag, key_id, created_at, unbound_expires_at)`
  - `service_prompt_pin_sets(uuid PK, owner_user_id, submission_id UNIQUE, canonical_manifest, manifest_digest, set_digest, auth_tag, key_id, created_at, unbound_expires_at)`
  - `service_prompt_job_bindings(job_uuid UNIQUE, pin_set_uuid, owner_user_id, retention_until, binding_auth_tag, key_id, created_at)`
- [ ] Use `owner:<owner_id>` scopes for user-authored and explicit-request components. Trusted immutable server assets may use `system:<asset_id>` scope for global digest deduplication. APIs never reveal cross-owner/system deduplication.
- [ ] Every component row's authenticator binds scope, content digest, stored bytes/encryption metadata, creation time, and key ID. Every component-manifest authenticator additionally binds owner/submission, definition/part IDs, contract, component scope/digest, locked/hidden/source flags, creation/expiry, and key ID. A colocated digest alone is never trusted.
- [ ] Canonical pin manifests bind one or more definition manifests, ordered part/component digests, selection-policy ID/version where applicable, source flags, contract versions, owner, one-time submission ID, set digest, creation/unbound-expiry, and key ID. Binding MACs bind envelope/set digest, pin UUID, submission ID, exact job UUID, owner, bound timestamp, and `retention_until`.
- [ ] Require the externally configured Context Integrity MAC key even when Jobs payload encryption is disabled. Reuse existing Jobs AES-GCM helpers only for optional content encryption; MAC verification is always required.
- [ ] Keep SQL inside `core/Jobs/service_prompt_store.py`; expose backend-neutral methods and parameterize every query. Accept an active signing key and retained verification key ring; require prior verification keys for at least the greater of 30 days or the configured maximum Jobs retention.
- [ ] Rerun backend tests and commit: `feat: store authenticated service prompt pin sets (<task-id>)`.

## Task 3: Implement held-bind-release enqueue

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/job_pinning.py`
- Modify: `tldw_Server_API/app/core/Jobs/service_prompt_store.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py`

- [ ] Write failing tests for declared finite requirements, multi-definition all-or-nothing sets, data-dependent selection policy pinning, explicit overrides in components, atomic pin creation failure, held-job creation failure, bind failure, duplicate submission/enqueue, owner mismatch, and concurrent bind attempts.
- [ ] Implement `ServicePromptJobPinner.enqueue(...)`: require the producer's finite definition/selection-policy declaration, resolve each candidate once, commit all authenticated components and the pin set in one protected-store transaction, create a held job whose payload contains only pin-set UUID, submission ID, and set digest, then call store `bind_and_release`.
- [ ] Snapshot templates and assembly metadata, not rendered source documents or runtime variable values. Store the exact part text only for request-bound `literal` overrides, under the owner/submission retention boundary.
- [ ] Implement `bind_and_release` as one backend transaction: verify held job/owner before `unbound_expires_at`, insert the unique authenticated binding with `retention_until = max(30 days, configured Jobs retention)`, update status with `WHERE status='held'`, update counters/events, and commit. Roll back the entire binding/release on any mismatch.
- [ ] On pre-bind failure, cancel or leave the job held for the one-hour reconciler; never queue it. Preserve existing Jobs idempotency semantics and return the existing job only if submission ID, set digest, owner, and authenticated binding match.
- [ ] Do not write prompt content, rendered variables, MACs, or encryption keys to job events/logs.
- [ ] Rerun backend concurrency tests and commit: `feat: bind prompt snapshots before job enqueue (<task-id>)`.

## Task 4: Verify pins in WorkerSDK

**Files:**

- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Modify: `tldw_Server_API/app/core/Jobs/service_prompt_store.py`
- Modify: `tldw_Server_API/app/core/Service_Prompts/job_pinning.py`
- Test: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py`

- [ ] Write failing tests proving handlers are never called for missing binding, invalid component/set/binding authenticator, owner/submission mismatch, missing component, digest mismatch, wrong contract/selection policy, bound retention expiry, unavailable retained key, or undecryptable content. Also prove a valid bound job remains verifiable after its one-hour unbound deadline.
- [ ] Add one built-in WorkerSDK guard that activates only when `service_prompt_pin_set_uuid` is present. It verifies and loads the full render-ready bundle before the domain handler, renders runtime variables with the same constrained renderer/budgets, and passes the immutable result through the worker context rather than mutable payload fields.
- [ ] Before content load, compare current operator mode to source flags. When `bypass_stored_overrides` would block a pin containing approved stored content, atomically move the job to `held` without substitution; a reconciler releases it only when mode permits the original pin.
- [ ] Verification uses `unbound_expires_at` only before a binding exists. Once a valid binding exists, ignore the unbound deadline and enforce the binding's authenticated `retention_until`; bound held/queued/processing/retained jobs keep every referenced manifest/component until that boundary.
- [ ] Classify transient store connectivity as retryable without handler execution. Classify authentication/digest/contract failures as non-retryable integrity failures and quarantine through existing poison-message behavior.
- [ ] Confirm ordinary jobs take the unchanged fast path.
- [ ] Rerun WorkerSDK tests and commit: `feat: fail closed on invalid prompt job pins (<task-id>)`.

## Task 5: Enforce retention and quotas

**Files:**

- Modify: `tldw_Server_API/app/core/Jobs/service_prompt_store.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Test: `tldw_Server_API/tests/Jobs/test_service_prompt_pinning_retention.py`

- [ ] Write failing clock-controlled tests for one-hour unbound cleanup, terminal retention at `max(30 days, configured Jobs retention)`, live/nonterminal preservation, owner-scoped shared-component reference preservation, and 256 MiB per-user cap.
- [ ] Add a reconciler/cleanup method called from the existing Jobs prune service: repair verifiable unbound submissions only before their one-hour deadline, delete/cancel expired never-bound submissions, release operator-mode-held bound jobs when safe, remove bound metadata only after authenticated `retention_until` and active/retained job checks, then garbage-collect components with zero references.
- [ ] Calculate quota from owner-scoped stored content bytes attributed through each user's live/retained pin sets; deduplicated content counts once within an owner and never across owners. Account trusted global server assets against a separate server budget. Never delete live data to satisfy a new enqueue.
- [ ] Return a typed quota failure before creating a held job when the cap cannot be met after safe cleanup.
- [ ] Rerun retention and existing prune tests; commit: `feat: retain and prune protected prompt snapshots (<task-id>)`.

## Task 6: Security verification

- [ ] Run `python -m pytest -q tldw_Server_API/tests/Jobs/test_service_prompt_pinning_sqlite.py tldw_Server_API/tests/Jobs/test_service_prompt_pinning_postgres.py tldw_Server_API/tests/Jobs/test_service_prompt_worker_guard.py tldw_Server_API/tests/Jobs/test_service_prompt_pinning_retention.py tldw_Server_API/tests/Jobs/test_worker_sdk.py`.
- [ ] Run `python -m bandit -r tldw_Server_API/app/core/Jobs/models.py tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/worker_sdk.py tldw_Server_API/app/core/Jobs/service_prompt_store.py tldw_Server_API/app/core/Service_Prompts/job_pinning.py tldw_Server_API/app/core/Service_Prompts/execution_integrity.py -f json -o /tmp/bandit_service_prompt_pinning.json` and review the JSON.
- [ ] Run `git diff --check`, inspect schema parity, update the Backlog task, and commit: `test: verify protected service prompt job execution (<task-id>)`.
