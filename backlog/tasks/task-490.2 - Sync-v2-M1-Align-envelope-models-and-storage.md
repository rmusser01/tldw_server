---
id: TASK-490.2
title: 'Sync v2 M1: Align envelope models and storage'
status: Done
assignee:
  - '@Codex'
created_date: ''
updated_date: '2026-05-23 08:13'
labels:
  - sync
  - sync-v2
  - m1
  - backend
  - database
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
  - Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
parent_task_id: TASK-490
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the existing Sync v2 schemas, core models, Sync DB schema, and store facade with the M1 envelope contract, including M1 domains, server_trusted_v1, base-state metadata, payload_hash, object state, apply status, and default personal dataset lookup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API and core models expose only M1 domains and default to server_trusted_v1.
- [x] #2 Sync DB persists base-state metadata, payload_hash, created/received timestamps, object revisions, apply status, object state, and idempotency keys.
- [x] #3 Model/store tests cover envelope validation, object state, idempotency, and default personal dataset lookup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-2-align-sync-v2-models-schemas-and-storage
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated API schemas to the M1 contract: public domains are `notes.note`, `chat.conversation`, `chat.message`, and `attachment.ref`; default encryption is `server_trusted_v1`; legacy capability domain inputs normalize to the M1 set.
- Updated core models with M1 constants and transition aliases for `server_cursor`/`server_sequence`, `object_id`/`entity_id`, and `payload`/`payload_clear`. Direct `payload_clear` construction no longer copies private cleartext into the new `payload` field.
- Extended `sync_envelopes` with M1 base-state metadata, object revisions, client sequence/profile, payload JSON/hash, created/received timestamps, tombstone flag, encryption metadata, and projection apply status/error fields.
- Added `sync_object_state` keyed by dataset/domain/object ID and store helpers for object state, apply-status updates, failed applies, accepted-envelope replay, and default personal dataset lookup.
- Added fail-closed store validation for non-M1 domains, unsupported M1 operations, non-`server_trusted_v1` envelope/dataset policy, non-empty object IDs and payload hashes, complete base metadata sets, whole-object update/tombstone base requirements, `chat.message` append identity/hash requirements, and required `attachment.ref` metadata.
- Fixed pre-M1 SQLite migration ordering by adding M1 envelope columns before creating indexes when an old `sync_envelopes` table exists; `payload_hash` is now included in the migration column set.
- Fixed envelope idempotency fingerprints to ignore mutable apply status so retries after `applied` reuse the stored row without drift conflict.
- Updated conflict-resolution schemas to the locked M1 batch shape: request-level `dataset_id`, `device_id`, `resolutions`, and actions limited to `overwrite`, `duplicate_rename`, and `skip`.
- Touched `tldw_Server_API/app/api/v1/endpoints/sync.py` only to strip API/server-only fields in `_core_envelope_from_api()` before constructing `SyncEnvelopeCreate`.
- Verification used `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv` because this worktree does not contain its own `.venv`.
- Review red test evidence: focused new tests initially failed with required `server_cursor`, leaked `envelope_id`, legacy conflict schema, `no such column: client_sequence`, idempotency drift after `applied`, and missing store validation.
- Focused regression rerun: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest` on the 14 new regression tests -> 14 passed, 5 warnings.
- Final targeted tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_object_state.py -q` -> 65 passed, 5 warnings.
- Security focused check: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_security.py::test_envelope_redaction_never_leaks_ciphertext_or_private_clear_payload -q` -> 1 passed, 5 warnings.
- Broad Sync suite status after review fixes: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync -q` -> 52 failed, 127 passed, 5 warnings. Remaining failures are concentrated in later-task legacy service/endpoints/media compatibility tests still using pre-M1 domains (`notes`, `chat`, `media`, `source_cache`), `client_private_v1`, old protocol/capability fields, old conflict actions/routes, and service/factory adapter drift.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/api/v1/endpoints/sync.py -f json -o /tmp/bandit_sync_v2_task2.json` -> 0 findings (`results: []`).

- Re-review fixes after `db22ffd1a`: added focused tests for malformed object-map validation and the M1 batch conflict-resolution endpoint shape.
- `_normalize_object_map()` now raises `ValueError`, so Pydantic v2 reports malformed object-map inputs such as `payload=[]` as normal `ValidationError`s instead of leaking raw `TypeError`.
- Replaced the stale single-conflict endpoint handler with `POST /api/v1/sync/conflicts/resolve`, which accepts the locked M1 request shape (`dataset_id`, `device_id`, `resolutions[]`) and returns per-conflict `resolved`/`rejected` outcomes. Public actions remain `overwrite`, `duplicate_rename`, and `skip`; the endpoint maps `skip` to the current service-layer dismiss action internally.
- Re-review red test evidence: the two new focused tests initially failed with raw `TypeError: value must be an object` and `ImportError: cannot import name resolve_sync_v2_conflicts`.
- Re-review focused tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py::test_sync_envelope_rejects_non_object_payload_as_validation_error tldw_Server_API/tests/Sync/test_sync_v2_models.py::test_conflict_batch_endpoint_resolves_locked_m1_request_shape -q` -> 2 passed, 5 warnings.
- Updated final targeted tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_object_state.py -q` -> 67 passed, 5 warnings.
- Updated Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/api/v1/endpoints/sync.py -f json -o /tmp/bandit_sync_v2_task2.json` -> 0 findings (`results: []`).

- Re-review fixes after `c11ed989a`: added focused real-service tests for `duplicate_rename` with a distinct resolution object ID, same-user cross-dataset conflict scoping, and public response serialization using `envelope_id` rather than `resolved_by_envelope_id`.
- `SyncV2Service.resolve_conflict()` now accepts an expected `dataset_id`, rejects conflict IDs from other datasets before mutation, passes dataset scope through `SyncV2Store.resolve_conflict()`/`SyncDatabase.resolve_conflict()`, and allows `duplicate_rename` resolution envelopes to target a distinct object ID within the same domain.
- `SyncConflictResolveResolvedItem` now exposes public `envelope_id`; `resolved_by_envelope_id` remains accepted as an input alias only for internal transition mapping.
- Added M1 domains to the sync adapter registry allowlist so the real service path can register and evaluate `notes.note`, `chat.conversation`, `chat.message`, and `attachment.ref` adapters. This was an extra touched production file required by the real-service regression path.
- Re-review red test evidence: focused tests initially failed with missing `dataset_id` propagation in the endpoint fake-service call and `ValueError: Unknown Sync adapter domain: notes.note` before the service could exercise M1 conflict semantics.
- Re-review focused tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py::test_conflict_batch_endpoint_resolves_locked_m1_request_shape tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_duplicate_rename_accepts_distinct_object_id tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_rejects_expected_dataset_mismatch_without_mutation -q` -> 3 passed, 5 warnings.
- Updated final targeted tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_object_state.py -q` -> 67 passed, 5 warnings.
- Updated Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/adapters.py -f json -o /tmp/bandit_sync_v2_task2.json` -> 0 findings (`results: []`).

- Re-review fixes after `ecc068ab6`: added focused tests for no-envelope `duplicate_rename`, schema rejection of invalid duplicate rename requests, endpoint response identity/cursor serialization, and real-service persistence of inserted server envelope identity.
- `SyncConflictResolution` now rejects `duplicate_rename` without a `resolution_envelope` during Pydantic validation, while `SyncV2Service.resolve_conflict()` also fails closed for direct core callers and leaves the conflict unresolved.
- Conflict resolution with an inserted envelope now persists and returns the server-generated envelope ID (`srv_env_...`) plus the inserted envelope server cursor; the endpoint no longer passes a client envelope ID as the public resolved envelope identity.
- Red test evidence: the new focused tests initially failed because duplicate rename without an envelope validated/resolved, the endpoint passed `env-resolution` as `resolved_by_envelope_id`, and the service persisted `env-resolution-copy` instead of `srv_env_...`.
- Focused tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py::test_conflict_resolution_request_rejects_duplicate_rename_without_envelope tldw_Server_API/tests/Sync/test_sync_v2_models.py::test_conflict_batch_endpoint_resolves_locked_m1_request_shape tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_duplicate_rename_accepts_distinct_object_id tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_duplicate_rename_requires_resolution_envelope -q` -> 4 passed, 5 warnings.
- Updated targeted/conflict suite: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_object_state.py tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_duplicate_rename_accepts_distinct_object_id tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_rejects_expected_dataset_mismatch_without_mutation -q` -> 70 passed, 5 warnings.
- Updated Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/DB_Management/Sync_DB.py -f json -o /tmp/bandit_sync_v2_task2.json` -> 0 findings (`results: []`).

- Final code-quality fix after `0a066f6e`: added focused schema and real-service no-mutation tests for `skip`/service `dismiss` with a resolution envelope.
- `SyncConflictResolution` now rejects `skip` with `resolution_envelope`; `SyncV2Service.resolve_conflict()` rejects direct `dismiss` calls that include a resolution envelope before envelope evaluation/insertion.
- Red test evidence: the new focused tests initially failed because `skip` with `resolution_envelope` validated and service `dismiss` with an envelope did not raise. After the service guard was added, the no-mutation assertion was narrowed to verify the resolution envelope was not inserted while preserving existing conflict logging behavior.
- Focused tests: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py::test_conflict_resolution_request_rejects_skip_with_resolution_envelope tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_dismiss_rejects_resolution_envelope_without_mutation -q` -> 2 passed, 5 warnings.
- Updated targeted/conflict suite: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_object_state.py tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_duplicate_rename_accepts_distinct_object_id tldw_Server_API/tests/Sync/test_sync_v2_service.py::test_resolve_conflict_rejects_expected_dataset_mismatch_without_mutation -q` -> 71 passed, 5 warnings.
- Updated Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/core/Sync/v2/service.py -f json -o /tmp/bandit_sync_v2_task2.json` -> 0 findings (`results: []`).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 review fixes are incorporated. The M1 model/store path now fails closed for direct core callers, old SQLite envelope stores migrate before M1 indexes are created, idempotent retries ignore mutable apply status, conflict-resolution schemas and endpoint expose the locked M1 batch shape, malformed object-map inputs produce Pydantic validation errors, and the push mapper strips server-only API fields before persistence. Targeted Task 2 tests and Bandit pass; the broader Sync suite remains red only in later-task legacy service/adapter/endpoint areas that still assume the pre-M1 contract.

Re-review conflict-resolution fixes now support locked M1 `duplicate_rename` semantics through the real service path, scope conflict mutation by dataset, and expose `envelope_id` in the public batch response. Focused conflict tests, the Task 2 targeted suite, and Bandit pass.

Final conflict-resolution review fixes now reject no-envelope `duplicate_rename` requests at both schema and service boundaries, and resolution-created responses/persistence use the inserted server envelope ID and cursor rather than client envelope IDs or original conflict cursors.

Final code-quality blocker is fixed: M1 `skip`/service `dismiss` now fail closed when a resolution envelope is supplied, preventing accidental envelope insertion or conflict mutation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
