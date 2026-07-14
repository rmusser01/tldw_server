# Service Prompt Persistence, API, and Private Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist per-user service-prompt bundle revisions safely, expose the complete typed API, and add private account backup/import semantics that never transfer trust.

**Architecture:** Add service-specific tables to schema v6 of each user's existing Prompts DB and access them through a narrow DB_Management repository. The service layer validates bundles against the registry, delegates trust decisions to Context Integrity, and performs compare-and-swap state transitions with idempotent mutation receipts. A dedicated FastAPI router exposes the exact catalog/detail/preview/override/reset/ack/history/revision/restore contract. Chatbook private account archives serialize revisions and non-operative state provenance logically and import all rows as unapproved history.

**Tech Stack:** SQLite, existing `PromptsDatabase`, FastAPI/Pydantic, AuthPrincipal, existing audit/rate-limit dependencies, Chatbooks archive framework, pytest/Hypothesis.

---

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate.

## Task 1: Add Prompts DB schema v6

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`
- Create: `tldw_Server_API/app/core/DB_Management/Service_Prompts_DB.py`
- Test: `tldw_Server_API/tests/Prompt_Management/test_service_prompts_db_migration.py`
- Test: `tldw_Server_API/tests/Prompt_Management/test_service_prompts_repository.py`

- [ ] Create the implementation Backlog task and link this plan.
- [ ] Write failing migration tests for new DB creation, v5→v6 upgrade, repeat initialization, rollback on failure, foreign-key checks, and unchanged ordinary prompt/FTS behavior.
- [ ] Add schema v6 tables:
  - `ServicePromptRevisions`: immutable UUID and per-definition sequence, definition ID, canonical editable-parts JSON, bundle digest, contract version, registry-schema digest, locked-assembly digest, trusted-default digest, origin action (`save`, `restore`, or `private_backup`), immutable trust origin (`local_unapproved` or `unapproved_import`), creator metadata, and created timestamp.
  - `ServicePromptState`: one row per definition with active UUID, pending UUID, acknowledged trusted-default digest, monotonic `generation`, and updated timestamp.
  - `ServicePromptStateEvents`: immutable reset, acknowledgement, approval, rejection, supersession, stale, and restore provenance with generation, operative/non-operative origin, and content-free metadata.
  - `ServicePromptMetadata`: one row holding monotonic per-user `catalog_generation`.
  - `ServicePromptMutationReceipts`: owner-local client mutation ID, request digest, completed result JSON, and created timestamp for safe retries.
- [ ] Add constraints/indexes for unique revision UUID/digest identity and fast definition/history lookup. User identity remains implicit in the per-user DB; do not duplicate it into every row.
- [ ] Increment `_CURRENT_SCHEMA_VERSION` to 6 and implement the explicit `current_db_version == 5` migration branch.
- [ ] Keep service tables out of ordinary prompt FTS and sync-log semantics.
- [ ] Run focused migration tests and commit: `feat: add service prompt revision schema (<task-id>)`.

## Task 2: Implement transactional repository behavior

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/Service_Prompts_DB.py`
- Test: `tldw_Server_API/tests/Prompt_Management/test_service_prompts_repository.py`
- Test: `tldw_Server_API/tests/Prompt_Management/test_service_prompts_repository_properties.py`

- [ ] Write failing tests for first save, atomic multipart save, pending supersession, generation conflict, idempotent mutation retry, mutation-ID body mismatch, approval activation, rejection, safe reset, acknowledgement, restore-as-new-pending, combined history ordering, catalog-generation increments, 50-history retention, and the 200-event cap after each of save/approve/reject/reset/acknowledge/restore/import. Include a stateful/property test that mixes those mutations and never observes more than 200 retained events per definition.
- [ ] Property-test random save/approve/reject/reset sequences: at most one active and one pending pointer exist, pointers reference the same definition, no rejected/imported revision resolves active, and revision rows never change after insertion.
- [ ] Implement `ServicePromptRepository` over an injected `PromptsDatabase`; do not subclass it and do not add service methods to the already-large ordinary prompt class.
- [ ] Save canonical editable parts, current registry/locked/default baseline digests, pending revision, supersession event, mutation receipt, definition generation, and catalog generation in one transaction. Reject partial or unknown parts before opening the transaction.
- [ ] A newer save changes only the pending pointer and appends a supersession event for the old pending revision; it does not mutate either revision, activate either one, or return a generic one-pending error.
- [ ] Activate or reject only when expected pending UUID and `generation` match. Activation changes pointers and appends an approval event without modifying revision rows; it also requires Context Integrity to prove that the current signed manifest contains the exact owner-scoped asset/digest and expected manifest sequence. Rejection clears the pending pointer and appends an event.
- [ ] Provide a repository `locked_transition(expected_generation)` context used by the service for approval/reset coordination. Its reset commit step accepts an already verified reset-baseline/manifest result, then clears active/pending pointers, appends reset/supersession events and mutation receipt, and increments both generations; repository code never imports Context Integrity.
- [ ] Acknowledge only the caller-supplied current trusted-default digest under expected generation.
- [ ] Restore revalidates a historical bundle, copies it into a new pending UUID, and appends a restore event; it never reuses or trusts the old approval.
- [ ] After every successful state mutation and private-import transaction, keep active/pending plus the newest 50 historical rows and 200 state events per definition. Prune older content revisions/events in the same transaction, deterministically by `(created_at DESC, id DESC)`; never prune a revision named by active/pending state. Retain mutation receipts for 24 hours and at most 1,000 rows per account, pruning oldest expired/completed rows in the mutation transaction.
- [ ] Rerun repository tests and commit: `feat: persist service prompt revisions atomically (<task-id>)`.

## Task 3: Add service and API schemas

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/service.py`
- Create: `tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py`
- Create: `tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py`
- Test: `tldw_Server_API/tests/Service_Prompts/test_service.py`

- [ ] Write failing tests for registry validation, pending-save/supersession behavior, locked-part rejection, size limits, stale contracts/baselines, mutation receipts, unavailable integrity store, mode behavior, prompt-body redaction across logs/audit/errors/metrics/notifications, bounded metric labels, private catalog ETags, and capability calculation.
- [ ] Implement `ServicePromptService` as the only coordinator between registry/resolver/repository and Context Integrity manifest services. Endpoint code must not issue SQL, read key material, or mutate manifests directly.
- [ ] Implement safe reset in the service: resolve a trusted server default, enter the repository's locked transition, acquire the manifest-store lock second, recheck generation/default, advance the stable state asset to `no_override`, then commit repository pointer/events/receipt. Preserve state when the default is unavailable/quarantined and reconcile an indeterminate post-manifest DB commit idempotently.
- [ ] Define response/request models with `extra="forbid"`; include stable IDs, labels/descriptions, part contracts, provenance, generation tokens, pending review links, mode, availability, contract version, stable error codes, and `can_approve_pending`.
- [ ] Capability rules: route exists whenever compiled in; `availability` explains missing integrity key/config error; `read_only` allows catalog/detail/preview/history; `bypass_stored_overrides` allows mutation/history but resolution skips active user rows; `can_approve_pending` derives from the authenticated principal.
- [ ] Rerun service tests and commit: `feat: coordinate service prompt lifecycle (<task-id>)`.

## Task 4: Add Context Integrity review and signed-manifest decisions

**Files:**

- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py`
- Test: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`

- [ ] Write failing tests for review privilege, non-reviewing-admin denial, owner isolation, escaped canonical diff, no rich HTML/model use, stale revision, changed baseline, changed manifest sequence, single-user explicit approval, reject, retry after manifest-only completion, and audit redaction.
- [ ] Add a dedicated `context_integrity.approve` permission to the existing admin route policy. Single-user principals receive the same privilege but still call a distinct explicit action.
- [ ] Add:
  - `GET /api/v1/admin/context-integrity/reviews/service-prompts/{user_id}/{revision_id}`
  - `POST /api/v1/admin/context-integrity/reviews/service-prompts/{user_id}/{revision_id}/approve`
  - `POST /api/v1/admin/context-integrity/reviews/service-prompts/{user_id}/{revision_id}/reject`
- [ ] Review reloads the current definition and returns an escaped, content-size-capped canonical diff only to the approval principal.
- [ ] Approval validates the draft, re-resolves registry schema/locked assembly/trusted default, and marks it `pending_revision_stale` on baseline change. Acquire locks in the documented order (per-user Prompts transaction, then global manifest-store lock), recheck pending revision/generation, then atomically add the immutable revision asset and replace the stable state asset using expected manifest sequence/digest.
- [ ] After manifest success, publish the newly verified manifest snapshot and commit the same pending revision/generation as active. If DB commit is indeterminate, leave resolution fail-closed and let an idempotent retry/reconciler compare manifest entry and mutation receipt before completing activation.
- [ ] Rejection compares pending revision/generation, clears pending, records a rejected event, and leaves prior active unchanged. Reset uses a service-only manifest transition to `no_override` containing the verified reset-baseline digest before repository pointer changes.
- [ ] Return 409 for stale generation/manifest/baseline, 422 for contract mismatch, 503 for unavailable/indeterminate trust store, and no prompt content in normal response logs. Emit signed/hash-chained audit evidence using IDs/digests only.
- [ ] Rerun focused auth/API tests and commit: `feat: review service prompts through signed manifests (<task-id>)`.

## Task 5: Expose the user API

**Files:**

- Create: `tldw_Server_API/app/api/v1/endpoints/service_prompts.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Test: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py`
- Test: `tldw_Server_API/tests/Services/test_router_groups_contract.py`

- [ ] Write failing API tests for auth, cross-user isolation, path-like/encoded definition IDs, validation/error mapping, rate limits, mode restrictions, expected-generation writes, mutation receipts, private ETag/304 catalog reads, cursor pagination, private/no-store detail caching, hidden-part redaction, and OpenAPI registration in full/minimal profiles.
- [ ] Add these routes under `/api/v1/service-prompts`:
  - `GET /capability`
  - `GET /catalog`
  - `GET /{definition_id}`
  - `POST /{definition_id}/preview`
  - `PUT /{definition_id}/override`
  - `POST /{definition_id}/reset`
  - `POST /{definition_id}/acknowledge-default`
  - `GET /{definition_id}/history`
  - `GET /{definition_id}/revisions/{revision_id}`
  - `POST /{definition_id}/revisions/{revision_id}/restore`
- [ ] Use claim-first `AuthPrincipal`, the per-user Prompts DB dependency, existing pagination envelopes, and `rbac_rate_limit("service_prompts.<action>")`.
- [ ] Require `expected_generation` and `client_mutation_id` for mutations. Map unknown/unavailable definitions to 404, invalid bundles/contracts to 422, stale generation/incompatible restore/unsafe reset/stale acknowledgement to 409, mode/authorization to 403, body limits to 413, rate limit to 429, and unavailable integrity/config to sanitized 503.
- [ ] Return prompt content only to its owning user and never include hidden part content.
- [ ] Emit safe audit metadata (actor, definition ID, action, revision/digest, result, timestamp) and bounded metrics using registry definition ID, source kind, validation outcome, and latency only.
- [ ] Link pending detail responses to the Task 4 review endpoint and rerun router/API tests.
- [ ] Commit: `feat: expose service prompt settings API (<task-id>)`.

## Task 6: Add deterministic private account export/import

**Files:**

- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_account_inventory.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_service_prompts_private_backup.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_export_contract.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`

- [ ] Write failing tests proving service prompts are absent from shareable chatbooks, present only in private full-account scope, export only user revisions and state-event provenance, and never export packaged/deployment/hidden content, signed-manifest artifacts, mutation receipts, active pointers, or pending pointers.
- [ ] Add a distinct private-account content category such as `service_prompt_revisions`; do not merge it with ordinary `prompts`.
- [ ] Serialize definition ID, contract version, editable parts, bundle digest, source timestamp, archive row ID, and content-free state events explicitly marked non-operative. Validate every row against archive size/count/path rules.
- [ ] On import, sort revisions/events by `(source_timestamp, archive_row_id)`, create fresh local UUIDs with immutable trust origin `unapproved_import`, import events as provenance only, and leave active/pending state null. Do not replay generations or select any imported row as current pending.
- [ ] Preserve every size-valid imported revision, including unknown definitions and future/incompatible contracts, as non-operative opaque history after verifying its archived digest and strict JSON shape. Mark its compatibility reason; never feed it to the renderer/resolver until a current local registry definition validates it.
- [ ] Reject digest mismatches, duplicate archive row IDs, malformed/oversized bundles, and path traversal. For currently known definitions, flag hidden/locked/part-contract mismatches as incompatible history rather than dropping the revision.
- [ ] Surface imported rows in history with source `private_backup`; a user may explicitly restore one only after current-contract validation, creating a new pending revision. Unknown/future rows remain visible and return an actionable incompatibility report.
- [ ] Rerun the focused Chatbooks tests and commit: `feat: back up service prompt history without trust (<task-id>)`.

## Task 7: Documentation and verification

**Files:**

- Create: `Docs/API/service-prompts.md`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`

- [ ] Document modes, limits, approval workflow, status codes, backup trust behavior, strict managed override failures, and curl examples with non-sensitive sample content.
- [ ] Run `python -m pytest -q tldw_Server_API/tests/Prompt_Management/test_service_prompts_db_migration.py tldw_Server_API/tests/Prompt_Management/test_service_prompts_repository.py tldw_Server_API/tests/Service_Prompts/test_service.py tldw_Server_API/tests/Prompt_Management_NEW/integration/test_service_prompts_api.py tldw_Server_API/tests/Chatbooks/test_chatbooks_service_prompts_private_backup.py tldw_Server_API/tests/Services/test_router_groups_contract.py`.
- [ ] Run `python -m bandit -r tldw_Server_API/app/core/DB_Management/Prompts_DB.py tldw_Server_API/app/core/DB_Management/Service_Prompts_DB.py tldw_Server_API/app/core/Service_Prompts/service.py tldw_Server_API/app/core/Chatbooks tldw_Server_API/app/core/AuthNZ/permissions.py tldw_Server_API/app/api/v1/API_Deps/service_prompt_deps.py tldw_Server_API/app/api/v1/endpoints/service_prompts.py tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py tldw_Server_API/app/api/v1/endpoints/admin/__init__.py tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py tldw_Server_API/app/api/v1/schemas/admin_schemas.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/api/v1/router_groups/minimal.py -f json -o /tmp/bandit_service_prompt_api.json` and review new findings in touched code.
- [ ] Run `git diff --check`, update the Backlog task, and commit: `docs: document service prompt lifecycle API (<task-id>)`.
