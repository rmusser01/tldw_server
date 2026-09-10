# Moodboard and Studio Sync Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate production synchronization for moodboards, manual placements, and accepted Studio documents with fail-closed capture, lifecycle integration, readiness-gated capabilities, portable Notes time projection, hard-delete protection, and verified rollback constraints.

**Architecture:** Reuse the dormant contracts, stores, adapters, materializers, bootstraps, and planners from TASK-13007.1–.4. One activation policy evaluates a persisted operator fleet-compatibility attestation, per-graph scope authority, bootstrap/readiness, dependency, capture, materializer, repair, RLS, and portable-time health. Moodboard and placement activate as one coupled unit; Studio activates independently on top of `notes.note`. All production writes append a durable Sync plan before product materialization, and note/Studio lifecycle commands expand into ordered groups. Capability advertisement is derived from the same predicates as write admission so the public contract cannot claim a path that capture would reject.

**Tech Stack:** Python 3.11, Sync v2, FastAPI, SQLite, PostgreSQL forced RLS, Pydantic, pytest, Ruff, Bandit, OpenAPI exporter, openapi-typescript.

**Design:** `Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md`

**Backlog task:** `TASK-13007.5`

**ADR required:** no
**ADR path:** `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
**Reason:** ADR-040 already fixes the activation boundary, coupled/independent readiness, scope authority, lifecycle groups, portable Notes time, hard-delete behavior, and rollback policy.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_activation.py` — shared activation predicates, coupled-domain rules, server-issued fleet-compatibility attestation validation, and fail-closed reason codes.
- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_lifecycle.py` — ordered note-plus-Studio tombstone/restore planning and capability-mismatch admission.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_lifecycle.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_end_to_end.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_postgres_activation.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_portable_time_activation.py`
- `tldw_Server_API/tests/Admin/test_admin_sync_scope_authority_attestation.py` — platform-admin issuance, inspection, rotation, revocation, expiry, audit, and OpenAPI tests.
- `Docs/Operations/Notes_Moodboard_Studio_Sync_Rollback.md` — rollout prerequisites, fleet attestation, pre/post-history rollback, maintenance mode, and recovery evidence.

**Modify**

- `tldw_Server_API/app/core/Sync/v2/models.py` — public domain versions, dataset activation/readiness metadata, portable note-time compatibility marker, and closed reason codes.
- `tldw_Server_API/app/core/Sync/v2/factory.py` — move the three domains from dormant internal wiring into the production registry without changing their adapters.
- `tldw_Server_API/app/core/Sync/v2/profile.py` — explicit enrollment/bootstrap orchestration, owner-only diagnostics, and coupled/independent readiness publication.
- `tldw_Server_API/app/core/Sync/v2/store.py` — activation-attestation, readiness, authority, and portable-note-head verification facades.
- `tldw_Server_API/app/core/Sync/v2/service.py` — enrollment gates, device capability checks, writable capabilities, lifecycle expansion, push admission, and repair routing.
- `tldw_Server_API/app/core/Sync/v2/server_origin.py` — production append-before-materialize capture for the three new domains and grouped note/Studio lifecycle operations.
- `tldw_Server_API/app/core/Sync/v2/restore.py` — active note-plus-Studio restore planning and retained-child checks.
- `tldw_Server_API/app/core/Sync/v2/replay.py` — active repair and predecessor-barrier dispatch.
- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_coordinator.py` — enable production board/placement capture only through activation policy.
- `tldw_Server_API/app/core/Sync/v2/notes_studio_coordinator.py` — enable accepted Studio singleton/compound capture only through activation policy.
- `tldw_Server_API/app/core/Notes/organization_capture.py` — permanently stamp/project portable `notes.note` modification time and preserve old-envelope fallback.
- `tldw_Server_API/app/core/Notes/studio_service.py` — route accepted persistence and note/Studio lifecycle through the active coordinator.
- `tldw_Server_API/app/core/DB_Management/Sync_DB.py` — persist privileged fleet-compatibility attestations and activation state; expose bounded readiness/history/hard-delete predicates.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` — activate portable note-time projection and gate moodboard/placement/note-with-Studio hard delete.
- `tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py` — active capture preconditions and bounded retained-placement checks.
- `tldw_Server_API/app/core/DB_Management/chacha/note_store.py` — active Studio lifecycle and bounded retained-sidecar checks.
- `tldw_Server_API/app/api/v1/endpoints/notes.py` — route board, placement, accepted Studio, note delete, and note restore through fail-closed active capture.
- `tldw_Server_API/app/api/v1/endpoints/sync.py` — enrollment, readiness diagnostics, capability, and activation error mapping.
- `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py` — public supported/writable-domain and activation/readiness response contracts.
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_system.py` — platform-admin fleet-attestation issue/read/revoke endpoints.
- `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` — closed fleet-attestation request/response models.
- `tldw_Server_API/app/services/admin_system_service.py` — authenticated issuance, rotation, revocation, and admin audit orchestration.
- `tldw_Server_API/app/services/admin_data_subject_requests_service.py` — replace raw Notes deletion with the existing DSR workflow upgraded to understand active canonical Sync history.
- `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_factory.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- `tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py`
- `tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py`
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py`
- `tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py`
- `tldw_Server_API/tests/Admin/test_data_subject_requests_api.py`
- `tldw_Server_API/tests/Services/test_openapi_contracts.py`
- `Docs/API/sync-v2.md` — advertise versions, readiness rules, lifecycle groups, portable note-time behavior, and failures.
- `Docs/Notes/Moodboards.md` — active sync behavior, coupled readiness, manual-only placement authority, and lifecycle.
- `Docs/Notes/Studio.md` — active accepted-persistence and note/Studio lifecycle behavior.
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

**Generate for verification; do not commit**

- `apps/tldw-frontend/lib/api/generated/openapi.json`
- `apps/tldw-frontend/lib/api/generated/schema.d.ts`

### Task 0: Start the activation child only after all dormant foundations land

- [ ] **Step 1: Verify dependencies and attach this plan**

```bash
for task_id in 13007.1 13007.2 13007.3 13007.4; do
  backlog task "$task_id" --plain
done
backlog task edit 13007.5 -s "In Progress" \
  --doc Docs/superpowers/plans/2026-08-25-moodboard-studio-sync-activation-implementation-plan.md \
  --plan $'1. Implement fail-closed fleet and per-domain activation predicates.\n2. Permanently activate portable Notes canonical-time stamping and repair.\n3. Wire production enrollment, bootstrap, capture, repair, and capabilities.\n4. Integrate note-plus-Studio lifecycle and hard-delete gates.\n5. Prove two-client SQLite/live PostgreSQL convergence and rollback constraints.\n6. Update OpenAPI/operator docs and close TASK-13007.5 plus its parent.\n\nADR required: no\nADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md\nReason: ADR-040 already governs activation and rollback.'
```

Expected: TASK-13007.1–.4 are Done; TASK-13007.5 becomes In Progress. Stop if any dependency is incomplete or its public capabilities are already advertising these domains.

### Task 1: Build one fail-closed activation policy

- [ ] **Step 1: Write fleet and shared-authority RED tests**

Cover missing/expired/wrong-contract privileged fleet attestation, non-platform-admin issuance/revocation, compatible issue/inspect/rotate/revoke flows, immutable audit evidence, conflicting owner authority, wrong dataset, `local-unbound`, false graph flag, empty verified graph, interleaved first-enrollment races, and independent graph failures. The persisted attestation must identify the deployment and assert that every write-serving instance uses per-graph authority semantics; dataset enrollment requests cannot self-assert it.

- [ ] **Step 2: Write coupled/independent readiness RED tests**

Cover:

- moodboard ready plus placement blocked advertises neither;
- placement ready plus moodboard blocked advertises neither;
- Studio ready while moodboards are blocked advertises Studio only;
- moodboards ready while Studio is blocked advertises the pair only;
- `notes.note`, organization-group, portable-time, source-domain, RLS, capture, materializer, repair, bootstrap, or drift failure removes only the dependent unit; and
- canonical history prevents silent unenrollment or direct-write fallback.

```python
def test_moodboard_pair_is_never_partially_writable() -> None:
    readiness = readiness_fixture(moodboard="ready", placement="blocked", studio="ready")
    writable = evaluate_writable_domains(readiness)
    assert "notes.moodboard" not in writable
    assert "notes.moodboard_note" not in writable
    assert "notes.studio_document" in writable
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py \
  tldw_Server_API/tests/Admin/test_admin_sync_scope_authority_attestation.py
```

Expected: import failure because the activation module does not exist.

- [ ] **Step 4: Implement the explicit platform-admin attestation surface**

Add closed request/response models and three authenticated endpoints under `admin_system.py`:

- `POST /api/v1/admin/system/sync/scope-authority-attestation` issues or rotates the current record;
- `GET /api/v1/admin/system/sync/scope-authority-attestation` returns redacted status; and
- `DELETE /api/v1/admin/system/sync/scope-authority-attestation` revokes it.

Every mutation explicitly requires `admin_scope_service.is_platform_admin(principal)`, records the actor/deployment/contract/issued/expiry/revocation facts through the existing admin audit service, and delegates to `SyncV2Store`; ordinary admin, dataset, and Sync client credentials are denied. Persist deployment identifier, contract identifier `per_graph_scope_authority_v1`, operator actor, issued/expiry times, and audit timestamp. Enrollment reads a current record and never accepts a client-supplied boolean. Rotation, revocation, and expiry fail closed without deleting canonical history. The operator guide requires fleet drain/verification before POST and records the deployment evidence digest, but never asks operators to edit the database directly.

- [ ] **Step 5: Implement the pure activation evaluator**

`notes_moodboard_studio_activation.py` receives already-authorized snapshots and returns immutable per-unit decisions plus privacy-safe reason codes. Keep it free of API/database reads. Use one evaluator for enrollment, write admission, and capability advertisement so they cannot drift.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Admin/test_admin_sync_scope_authority_attestation.py
git add tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_activation.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_system.py \
  tldw_Server_API/app/api/v1/schemas/admin_schemas.py \
  tldw_Server_API/app/services/admin_system_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Admin/test_admin_sync_scope_authority_attestation.py
git commit -m "feat(sync): define moodboard Studio activation policy"
```

### Task 2: Permanently activate portable Notes modification time

- [ ] **Step 1: Write RED tests for new and old envelopes**

Cover server-origin note create/update stamping `canonical_modified_at`, client-origin writes receiving the server acceptance time, explicit rejection of any client-supplied `canonical_modified_at`, missing portable time on old accepted envelopes, invalid/ambiguous server-bound timestamps, equal-timestamp replay, clock skew, conflict resolution, restore, and product projection. New accepted writes must always persist the server-selected portable value; old accepted heads retain a deterministic fallback that remains readable after rollback to a compatibility build.

- [ ] **Step 2: Write existing-head verification/repair RED tests**

Cover bounded keyset scanning, already-valid heads, reparable legacy heads, product/source drift, pending/failed predecessor barrier, malformed state, resumable cursor, idempotent retry, and final aggregate verification. Moodboard activation remains blocked until all in-scope note heads are verified.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_portable_time_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  -k 'canonical_modified_at or portable_time'
```

- [ ] **Step 4: Activate stamping, projection, verification, and fallback**

Enable the dormant TASK-13007.2 parser/projector in every production Notes capture path. Preserve the old-envelope fallback permanently. Verification uses keyset pages and existing durable-plan repair; it never mass-updates product state outside canonical capture.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_portable_time_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py
git add tldw_Server_API/app/core/Notes/organization_capture.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_portable_time_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py
git commit -m "feat(notes): activate portable canonical modification time"
```

### Task 3: Activate enrollment, bootstrap, capture, repair, and capabilities

- [ ] **Step 1: Write enrollment and bootstrap RED tests**

Cover default-personal/server-trusted-only policy, fleet attestation, shared authority, independent binding transactions, capture-enabled server REST before bootstrap, retryable external-push denial until ready, resumable source scans, already-deleted bootstrap, source drift correction, partial-domain retry, malformed/oversized blockers, live history, and irreversible active enrollment.

- [ ] **Step 2: Write production REST capture RED tests**

For board CRUD, manual placement pin/update/unpin/restore, accepted Studio singleton saves, and note-plus-Studio saves, prove:

1. active preconditions run before identifier allocation/provider work/product mutation;
2. canonical intent and durable plan append before materialization;
3. exact replay returns the stored result;
4. append-success/materialization-failure leaves repairable debt;
5. product-success/status-failure repairs without duplicate history; and
6. `not_enrolled` preserves the documented legacy product-only path without claiming history; capture-enabled `enrolling`/`bootstrapping` server REST writes append and materialize the canonical plan; external push during those phases returns retryable `domain_bootstrap_incomplete`; and blocked, unhealthy, or history-bearing capture-disabled state rejects without direct fallback.

Make these origin/phase cases separate tests so a generic readiness guard cannot accidentally reject prebootstrap REST capture or allow external device push.

- [ ] **Step 3: Write capability/device mismatch RED tests**

Cover supported adapter version maps, dataset writable maps, old devices, partial moodboard adapter claims, Studio-only devices, missing Studio adapter during retained-sidecar lifecycle, pull stream negotiation, and no advertisement when any activation predicate fails.

- [ ] **Step 4: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py
```

- [ ] **Step 5: Wire production activation**

Register the three strict adapters in the public factory. Extend explicit enrollment/profile bootstrap to invoke the dormant bind/bootstrap/verify machinery under the dataset fence. Gate server-origin and client push through the shared evaluator. Keep the moodboard pair coupled and Studio independent. If canonical history exists, a failed health check blocks writes but does not erase enrollment or re-enable legacy direct mutation.

- [ ] **Step 6: Derive capabilities from the same decision**

Advertise supported adapter versions server-wide after the implementation is deployed. For an authorized dataset, publish writable moodboard domains together or neither, and publish Studio independently. Device pull/push admission intersects server support, dataset writability, enrollment version, and the device's negotiated adapter versions.

- [ ] **Step 7: Run GREEN, regenerate OpenAPI, and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Admin/test_admin_sync_scope_authority_attestation.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_data_subject_requests_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
PYTHON=../../.venv/bin/python bun --cwd apps/tldw-frontend run generate:api-types
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
git add tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_coordinator.py \
  tldw_Server_API/app/core/Sync/v2/notes_studio_coordinator.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat(sync): activate moodboard and Studio domains"
```

### Task 4: Integrate note-plus-Studio lifecycle and hard-delete gates

- [ ] **Step 1: Write the complete lifecycle RED matrix**

For note tombstone and restore with and without a retained Studio sidecar, cover unbound, capture-enabled bootstrap, ready, unhealthy, predecessor-blocked, already-deleted bootstrap, and device-capability mismatch states. Expected rules:

- before enrollment, existing product behavior retains/hides the local sidecar without claiming Sync;
- during capture-enabled bootstrap, server REST writes append the full ordered group and external push waits;
- ready state expands note first, Studio second;
- a device without Studio support cannot tombstone/restore a note with a retained active Studio sidecar;
- unhealthy active Studio capture blocks lifecycle mutation;
- a pure Studio-readiness degradation may not block an ordinary note upsert when Notes capture and its predecessor chain remain healthy; and
- pending, failed, or conflicting accepted predecessors trigger the ADR-034 dataset barrier.

- [ ] **Step 2: Write hard-delete RED tests**

Cover active/inactive moodboards, placements, Studio sidecars, and notes with retained Studio state; bounded existence queries; unauthorized direct hard delete; normal retention without history truncation; and the existing DSR Notes erasure route in `admin_data_subject_requests_service.py`. Prove the DSR raw-SQL helper cannot delete product rows when active history exists unless the platform-admin DSR request first obtains a durable history-aware erasure authorization under the dataset fence. Sync-store unavailability, authorization/audit failure, and mid-erasure crashes fail closed and leave resumable state. Public delete remains a whole-object tombstone with explicit restore.

For category-only erasure where the user remains, assert the exact affected product/canonical sets are both empty, stale device epochs cannot replay erased state, server REST cannot write during erasure, and successful completion returns the same immutable dataset/authority to a freshly verified empty synchronized graph rather than legacy direct mode.

- [ ] **Step 3: Run RED**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_lifecycle.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_postgres_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_data_subject_requests_api.py
```

- [ ] **Step 4: Implement ordered lifecycle planning**

`notes_moodboard_studio_lifecycle.py` builds deterministic note-plus-Studio tombstone/restore commands using retained canonical heads and the TASK-13007.4 compound-intent rules. Clients never supply group metadata. Server REST, client commands, replay, and restore preview share the same planner.

- [ ] **Step 5: Add bounded hard-delete admission and upgrade the existing DSR erasure path**

At each destructive boundary, check activation/history and retained-child existence using indexed scoped predicates. Ordinary direct deletion fails when canonical history or retained children exist.

Replace `_erase_notes` raw-SQL admission in `admin_data_subject_requests_service.py` with the existing DSR workflow calling a new Sync-store `prepare_dataset_erasure` gate. Under the dataset materialization fence it verifies/reconciles pending plans, disables capture and writable capabilities, records request ID/actor/domain/history cursors in a durable audit-only erasure record, and returns one-use authorizations to the bounded product and canonical deletion helpers.

Define one exact Notes-category erasure catalog and test it for drift. Its product side is `notes`, `note_edges`, `note_wikilink_edges`, `note_studio_documents`, `moodboard_notes`, `moodboards`, the existing note-task rows/projections/events/read/reconciliation state selected through the erased notes, plus the owner/dataset smart-projection generations/matches/dirty state. Its canonical side is every envelope/head/conflict/bootstrap/readiness/private-intent row for `notes.note`, `notes.task`, `notes.task_activity`, `notes.moodboard`, `notes.moodboard_note`, and `notes.studio_document`. Delete both sides in bounded FK-safe pages; a cascade may assist but never substitutes for zero-row verification of every catalog entry. A catalog test compares this mapping to the registered Notes hard-child domains so a later domain cannot inherit unsafe erasure by omission. Retain only the privacy-safe DSR audit record with counts/digests required to prove erasure.

The resumable phases are `prepare_fence -> erase_canonical -> verify_canonical_empty -> erase_product -> verify_product_empty -> rotate_epoch -> verify_empty_graph -> complete`. A crash resumes from the durable phase/cursor without re-enabling capture. On `rotate_epoch`, increment a dataset `notes_erasure_epoch`, revoke affected-domain device cursors/negotiations, and require profile bootstrap to acknowledge the new epoch before external push/pull. Keep the immutable dataset ID, sole scope-authority row, and true graph-binding flags. After both sides verify empty, rebuild an empty derived projection, transition readiness through `verifying` to `ready`, and only then re-enable capture/writable capabilities as a fresh empty synchronized graph. Server REST is blocked until that transition completes; old devices receive stable `dataset_erasure_epoch_mismatch`; no phase can fall back to legacy direct writes. The raw SQL is unreachable without its one-use authorization, and a missing/unavailable Sync store fails the category rather than silently deleting Notes. Do not add a second erasure workflow.

Run the DSR service/API matrix on SQLite and the history/fence predicates in `test_sync_v2_notes_moodboard_studio_postgres_activation.py` with PostgreSQL required.

- [ ] **Step 6: Run GREEN and commit**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_lifecycle.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_postgres_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_data_subject_requests_api.py
git add tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_lifecycle.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Notes/studio_service.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/services/admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_lifecycle.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_postgres_activation.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py \
  tldw_Server_API/tests/Admin/test_data_subject_requests_api.py
git commit -m "feat(sync): coordinate note Studio lifecycle"
```

### Task 5: Prove two-client convergence and live PostgreSQL activation

- [ ] **Step 1: Add SQLite end-to-end scenarios**

Use two independent clients and one server dataset to prove board create/update/delete/restore, concurrent manual layout, pin/unpin/restore, smart-match exclusion, accepted Studio save, concurrent Studio revisions, note-plus-Studio save/delete/restore, pull/apply, exact retry, changed retry, reviewable conflicts, keyset pagination, crash repair, and restart continuation. Assert computed smart matches never appear as placement envelopes.

- [ ] **Step 2: Add required-live PostgreSQL scenarios**

Repeat the activation, enrollment race, scope/RLS isolation, capture, conflict, repair, lifecycle, hard-delete, and pagination paths against live PostgreSQL with `TLDW_TEST_POSTGRES_REQUIRED=1`. Include cross-owner/cross-dataset probes and query-plan assertions for readiness, retained-child, and keyset indexes.

- [ ] **Step 3: Run the focused end-to-end matrix**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_end_to_end.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_postgres_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_lifecycle.py
```

Expected: PostgreSQL tests execute rather than skip; two clients converge or receive stable reviewable conflicts, repair clears injected crash debt, RLS denies cross-tenant access, and all bounded paths use expected indexes.

- [ ] **Step 4: Run the broad Notes/Sync regression matrix**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync \
  tldw_Server_API/tests/ChaChaNotesDB \
  tldw_Server_API/tests/Notes_NEW/unit \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
```

- [ ] **Step 5: Commit test evidence**

```bash
git add tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_end_to_end.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_postgres_activation.py
git commit -m "test(sync): prove moodboard Studio convergence"
```

### Task 6: Document rollout/rollback, run final gates, and close the work stream

- [ ] **Step 1: Write operator and API documentation**

Document the platform-admin POST/GET/DELETE fleet-attestation workflow, required fleet drain/verification evidence, rotation before expiry, emergency revocation, dependency order, per-dataset enrollment, capture-before-bootstrap, coupled moodboard pair, independent Studio activation, readiness diagnostics, canonical-history point of no return, portable note-time compatibility, DSR history-aware erasure, hard-delete restrictions, maintenance mode, schema-compatible pre-activation rollback, and post-activation recovery from a pre-activation database state. Explicitly forbid direct database attestation edits and a pre-TASK-13007 write-serving binary against an activated dataset.

- [ ] **Step 2: Regenerate and verify public contracts**

```bash
PYTHON=../../.venv/bin/python bun --cwd apps/tldw-frontend run generate:api-types
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
```

- [ ] **Step 3: Run static, security, compilation, and diff gates**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_activation.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_lifecycle.py
  tldw_Server_API/app/core/Sync/v2/models.py
  tldw_Server_API/app/core/Sync/v2/factory.py
  tldw_Server_API/app/core/Sync/v2/profile.py
  tldw_Server_API/app/core/Sync/v2/store.py
  tldw_Server_API/app/core/Sync/v2/service.py
  tldw_Server_API/app/core/Sync/v2/server_origin.py
  tldw_Server_API/app/core/Sync/v2/restore.py
  tldw_Server_API/app/core/Sync/v2/replay.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_coordinator.py
  tldw_Server_API/app/core/Sync/v2/notes_studio_coordinator.py
  tldw_Server_API/app/core/Notes/organization_capture.py
  tldw_Server_API/app/core/Notes/studio_service.py
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
  tldw_Server_API/app/api/v1/endpoints/sync.py
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
  tldw_Server_API/app/api/v1/endpoints/admin/admin_system.py
  tldw_Server_API/app/api/v1/schemas/admin_schemas.py
  tldw_Server_API/app/services/admin_system_service.py
  tldw_Server_API/app/services/admin_data_subject_requests_service.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13007-5-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
git diff --check
```

- [ ] **Step 4: Perform the final security and rollback self-review**

Confirm capabilities and capture use the same predicate; no client can mint fleet compatibility or server provenance; moodboard domains never partially advertise; Studio is independently gated but always depends on Notes; no active path falls back to direct product mutation; smart matches never synchronize; lifecycle groups are note-first; hard-delete checks are scoped and bounded; old-envelope fallback is retained; pre-activation deactivation cannot erase schema requirements; and post-history rollback retains write gates or requires maintenance plus database restoration.

- [ ] **Step 5: Commit docs and close TASK-13007.5**

```bash
git add Docs/API/sync-v2.md Docs/Notes/Moodboards.md Docs/Notes/Studio.md \
  Docs/Operations/Notes_Moodboard_Studio_Sync_Rollback.md \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "docs(sync): document moodboard Studio activation"
```

Record exact SQLite/live-PostgreSQL/static/security/OpenAPI evidence, check all TASK-13007.5 AC and DoD boxes, add concise implementation notes and touched files, record only genuinely generalizable lessons, and set TASK-13007.5 Done. Then verify all five children and the parent acceptance criteria, add the required human-authored Change summary for the eventual PR, and close TASK-13007 without bypassing hooks.
