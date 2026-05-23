# Chatbook Sync v2 M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Milestone 1 of Sync v2: manual reliable personal Notes and Chat sync for tldw_chatbook server-connected modes, with an append-only envelope log, materialized server Notes/Chat projections, explicit conflicts, restore preview, tombstones, and server-trusted at-rest encryption posture.

**Architecture:** Retrofit the existing Sync v2 scaffolding under `/api/v1/sync` and `app/core/Sync/v2` into the approved M1 contract. The per-user Sync DB remains the envelope/audit/cursor store; accepted Notes and Chat envelopes are materialized into the user's normal ChaChaNotes DB through DB_Management-owned sync apply methods. The server uses explicit profile bootstrap to create or find the default personal dataset, records per-envelope projection status, and supports replay/repair from accepted envelopes.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL DB_Management backends, per-user Sync v2 DB, per-user ChaChaNotes DB, pytest, httpx/TestClient, Bandit.

---

## Scope Check

M1 implements the server-side Sync v2 contract for server-connected Chatbook modes only:

- `server_frontend`: Chatbook uses normal server APIs against materialized Notes/Chat state.
- `offline_sync`: Chatbook owns a local profile and uses Sync v2 push, pull, restore preview, and conflict resolution.

M1 must not add a server dependency to Chatbook `local_only` mode.

The branch already contains broad Sync v2 scaffolding:

- `/api/v1/sync/capabilities`, `/devices/register`, `/datasets/enroll`, `/push`, `/pull`, `/restore-manifest`, `/conflicts`, `/attachments`, and `/keys/recovery-bundle`.
- `SyncDatabase`, `SyncV2Store`, `SyncV2Service`, `SyncV2Envelope`, domain adapters, conflicts, key records, and attachment rows.

Implementation should align this scaffolding to the approved M1 contract instead of creating a parallel sync engine.

M1 excludes workspace datasets, media/library/source cache domains, background sync, binary/blob transfer, client-only encryption, key rotation, retention/GC, and Chatbook client implementation. Chatbook-side integration belongs in a separate repo/worktree after server contracts stabilize; this server plan stops at stable server contracts and verification fixtures that a Chatbook client can consume.

## Planning Gate Decisions

These decisions are locked for M1 implementation unless a blocker is found and recorded in Backlog before code changes continue.

1. Sync table location:
   - Keep Sync v2 metadata and append-only envelopes in the per-user Sync DB at `Databases/user_databases/<user_id>/Sync_v2.db`, implemented by `tldw_Server_API/app/core/DB_Management/Sync_DB.py`.
   - Materialized live Notes/Chat state remains in the per-user ChaChaNotes DB at `Databases/user_databases/<user_id>/ChaChaNotes.db`.
   - Do not put Sync v2 personal envelope logs in the AuthNZ DB. Cross-user isolation is simpler and matches existing per-user content storage.

2. Profile and device identity:
   - Add explicit `POST /api/v1/sync/profile/bootstrap`.
   - Bootstrap registers or refreshes a stable client-supplied `device_id` and optional `client_profile_id`.
   - Bootstrap creates or returns one active default personal dataset for the authenticated user. The dataset is marked by metadata `{"default_personal": true, "client_family": "chatbook"}` and M1 domains only.
   - If `device_id` is omitted, the server may generate one, but Chatbook should persist the returned value before pushing.

3. M1 at-rest encryption primitive:
   - Use `server_trusted_v1` as the M1 encryption policy name.
   - Normal authenticated server access unlocks data for trusted/self-hosted deployments.
   - M1's at-rest boundary is deployment-level encrypted storage for the user database directory, declared through a new Sync v2 security setting and exposed in capabilities/status. This is the only M1 mechanism broad enough to cover both `Sync_v2.db` and materialized `ChaChaNotes.db` without introducing SQLCipher or field-encryption rewrites.
   - Add code and docs that make this explicit: Sync v2 can advertise M1-ready `server_trusted_v1` only when the configured deployment mode attests encrypted volume/managed storage coverage. Later milestones can add passphrase/device-key and client-only modes.

4. Bootstrap contract:
   - `GET /api/v1/sync/profile` is read-only and never creates durable sync state.
   - `POST /api/v1/sync/profile/bootstrap` is the only endpoint that implicitly creates the default personal dataset.
   - Existing `/devices/register` and `/datasets/enroll` can remain internal-compatible helpers during the transition, but Chatbook M1 should use bootstrap.

5. M1 public domains:
   - Replace advertised public domains with `notes.note`, `chat.conversation`, `chat.message`, and `attachment.ref`.
   - Existing `workspaces`, `source_cache`, and `media` adapters are out of the M1 default registry. Keep their files only if tests require them as dormant future adapters.

## File Map

Server API and schemas:

- Modify `tldw_Server_API/app/api/v1/endpoints/sync.py`
  - Add profile, bootstrap, restore preview, and PRD-aligned conflict resolution endpoints.
  - Remove or disable legacy `/send` and `/get` media-shaped behavior.
  - Reject M1 blob upload paths with clear `sync_blob_transfer_not_supported` responses.
- Modify `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
  - Align public domains, operations, envelope fields, profile/bootstrap/status, restore preview, conflicts, and server-trusted encryption schemas.
- Leave `tldw_Server_API/app/api/v1/schemas/sync_server_models.py` only for deleted legacy tests or remove references once `/send` and `/get` are removed.

Sync core:

- Modify `tldw_Server_API/app/core/Sync/v2/models.py`
  - Add M1 domain/operation literals, base-state metadata, object revisions, apply status, profile/bootstrap models, restore preview models, and conflict actions.
- Modify `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
  - Add schema columns/tables for client sequence, base state, object state, apply status, encryption metadata, profile metadata, restore preview aggregates, and replay tracking.
- Modify `tldw_Server_API/app/core/Sync/v2/store.py`
  - Add facade methods for object state, apply status, profile bootstrap lookup, conflict lookup, and replay scans.
- Modify `tldw_Server_API/app/core/Sync/v2/service.py`
  - Add bootstrap/profile/status, server-trusted policy handling, push materialization, base-state conflict detection, restore preview, conflict resolution, and replay/repair orchestration.
- Modify `tldw_Server_API/app/core/Sync/v2/factory.py`
  - Wire M1 default adapter/materializer registry and ChaChaNotes DB access.
- Modify `tldw_Server_API/app/core/Sync/v2/security.py`
  - Add server-trusted at-rest configuration checks and log redaction for server-trusted payloads.
- Modify `tldw_Server_API/app/core/Sync/v2/adapters.py`
  - Replace known public domains with M1 domains and keep any future adapters out of default capabilities.
- Modify `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py`
- Modify `tldw_Server_API/app/core/Sync/v2/domain_adapters/chat.py`
  - Align validation with `notes.note`, `chat.conversation`, and `chat.message`.

New Sync core files:

- Create `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- Create `tldw_Server_API/app/core/Sync/v2/materializers/base.py`
- Create `tldw_Server_API/app/core/Sync/v2/materializers/notes.py`
- Create `tldw_Server_API/app/core/Sync/v2/materializers/chat.py`
- Create `tldw_Server_API/app/core/Sync/v2/materializers/attachment_refs.py`
- Create `tldw_Server_API/app/core/Sync/v2/profile.py`
- Create `tldw_Server_API/app/core/Sync/v2/restore.py`
- Create `tldw_Server_API/app/core/Sync/v2/replay.py`
- Create `tldw_Server_API/app/core/Sync/v2/server_origin.py`

ChaChaNotes projection APIs:

- Modify `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py` only if a reusable user-id resolver is needed by the Sync factory.
- Modify `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
  - Add sync-owned upsert/tombstone helpers that can preserve stable IDs and set sync metadata safely.
- Modify `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
  - Add sync-owned upsert/tombstone helpers for conversation metadata.
- Modify `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
  - Add sync-owned append/dedupe/tombstone helpers for stable message IDs.
- Modify `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Delegate new store helpers if needed.

Docs:

- Create `Docs/Design/Sync_V2_M1_Implementation_Decisions.md`
- Create or update `Docs/API/Sync_V2_M1.md`
- Update `Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md` only if implementation discovers a spec correction.

Tests:

- Modify existing Sync tests under `tldw_Server_API/tests/Sync/`.
- Create `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_object_state.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py`
- Create `tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py`
- Modify `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`
- Add focused ChaChaNotes store coverage under `tldw_Server_API/tests/ChaChaNotesDB/` if sync helper methods are added there.

Backlog:

- Keep `TASK-490` as the parent planning task.
- Use the M1 child Backlog tasks created from this plan for implementation PRs.

---

### Task 1: Lock M1 Decisions And Contract Docs

**Files:**
- Create: `Docs/Design/Sync_V2_M1_Implementation_Decisions.md`
- Create or modify: `Docs/API/Sync_V2_M1.md`
- Modify: `backlog/tasks/task-490 - Plan-Sync-v2-completion-roadmap-for-Chatbook-clients.md`

- [ ] **Step 1: Write the implementation decisions doc**

Document the five planning gate decisions from this plan:

- per-user `Sync_v2.db` for envelopes/state
- per-user `ChaChaNotes.db` for materialized projections
- explicit bootstrap endpoint
- `server_trusted_v1` deployment-level at-rest encryption attestation
- M1 domains only

- [ ] **Step 2: Write the API contract doc**

Document M1 request/response shapes for:

- `GET /api/v1/sync/profile`
- `POST /api/v1/sync/profile/bootstrap`
- `POST /api/v1/sync/push`
- `GET /api/v1/sync/pull`
- `POST /api/v1/sync/restore/preview`
- `POST /api/v1/sync/conflicts/resolve`

Include envelope examples for `notes.note`, `chat.conversation`, `chat.message`, `attachment.ref`, and tombstones.

- [ ] **Step 3: Run docs checks**

Run:

```bash
git diff --check
rg -n "T[B]D|T[O]DO|FIX[M]E|client_private_v1.*M1|workspaces|source_cache|media" Docs/Design/Sync_V2_M1_Implementation_Decisions.md Docs/API/Sync_V2_M1.md
```

Expected: no stale placeholders; any mentions of future domains are clearly marked M2/M3.

- [ ] **Step 4: Commit**

```bash
git add Docs/Design/Sync_V2_M1_Implementation_Decisions.md Docs/API/Sync_V2_M1.md backlog/tasks
git commit -m "docs: lock sync v2 m1 implementation decisions"
```

---

### Task 2: Align Sync v2 Models, Schemas, And Storage

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_object_state.py`

- [ ] **Step 1: Write failing schema tests**

Assert:

- capabilities advertise only `notes.note`, `chat.conversation`, `chat.message`, `attachment.ref`
- default encryption policy is `server_trusted_v1`
- envelopes accept `base_server_cursor`, `base_object_revision`, `base_object_hash`, `object_revision`, `client_sequence`, `parent_id`, `schema_version`, `payload`, `payload_hash`, `created_at_client`, `received_at_server`, `deleted`, and `encryption_metadata`
- whole-object domains require base metadata for updates/tombstones
- `chat.message` duplicate handling requires stable message ID plus `payload_hash`
- new `client_private_v1` envelopes are not accepted as M1 defaults unless explicitly marked future/disabled

- [ ] **Step 2: Write failing store tests**

Assert the Sync DB can persist and read:

- envelope base-state fields
- `payload_hash` as a first-class, indexed contract field, not only as derived payload metadata
- server cursor/object revision aliases
- apply status fields: `pending`, `applied`, `failed`, `conflict`
- object state rows keyed by dataset, domain, object id
- idempotency by dataset plus client envelope id and by dataset/device/client sequence
- default personal dataset lookup by user

- [ ] **Step 3: Update models and schemas**

Use API names from the PRD. Internally, keep compatibility aliases only where they reduce migration risk:

- expose `server_cursor` while mapping to `server_sequence`
- expose `payload` while reading old `payload_clear` in tests only if needed
- expose `object_id` while mapping to existing `entity_id` only during transition

Do not advertise out-of-M1 domains in capabilities.

- [ ] **Step 4: Update Sync DB schema**

Add or migrate:

- `client_sequence`
- `base_server_cursor`
- `base_object_revision`
- `base_object_hash`
- `object_revision`
- `parent_id`
- `schema_version`
- `payload_json`
- `payload_hash`
- `created_at_client`
- `received_at_server`
- `deleted`
- `encryption_metadata_json`
- `apply_status`
- `apply_error_code`
- `apply_error_message`
- `applied_at`
- `client_profile_id`

Add `sync_object_state`:

- `dataset_id`
- `domain`
- `object_id`
- `object_revision`
- `object_hash`
- `latest_server_cursor`
- `deleted`
- `updated_at`

Add indexes for dataset/cursor, dataset/domain/object, dataset/device/client sequence, and failed apply status.

- [ ] **Step 5: Update store facade**

Add methods:

- `get_or_create_default_personal_dataset(...)`
- `get_object_state(...)`
- `upsert_object_state(...)`
- `mark_envelope_apply_status(...)`
- `list_failed_applies(...)`
- `list_accepted_envelopes_for_replay(...)`

- [ ] **Step 6: Run storage tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_object_state.py \
  -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_object_state.py
git commit -m "feat: align sync v2 m1 envelope storage"
```

---

### Task 3: Add Profile Bootstrap And Status

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/security.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py`

- [ ] **Step 1: Write failing bootstrap/status tests**

Assert:

- `GET /profile` is read-only and returns no dataset when none exists
- `POST /profile/bootstrap` creates the default personal dataset and device
- repeated bootstrap is idempotent
- bootstrap supports `server_frontend` and `offline_sync` modes
- profile status returns profile-level summary and per-domain details
- production/multi-user Sync v2 refuses to advertise `server_trusted_v1` unless at-rest storage coverage is configured

- [ ] **Step 2: Implement security settings**

Add a small explicit config surface, for example:

- `SYNC_V2_AT_REST_ENCRYPTION_MODE=encrypted_volume|managed_storage|development_unencrypted`
- `SYNC_V2_SERVER_TRUSTED_ENABLED=true|false`

Tests should avoid requiring host-specific disk encryption detection. The server must be honest in capabilities/status about the configured mode.

- [ ] **Step 3: Implement service profile methods**

Add:

- `profile(user_id, device_id=None)`
- `bootstrap_profile(user_id, mode, device_id=None, client_profile_id=None, ...)`
- `profile_status(user_id, dataset_id, device_id=None)`

Return:

- active dataset id
- server protocol version and minimum supported version
- current server cursor
- supported domains
- encryption policy and at-rest status
- registered device status
- per-domain envelope counts, failed apply counts, conflicts, and last apply result

- [ ] **Step 4: Wire endpoints**

Add:

- `GET /api/v1/sync/profile`
- `POST /api/v1/sync/profile/bootstrap`

Keep `/devices/register` and `/datasets/enroll` only if current tests or internal callers still need them, but mark them lower-level in docs.

- [ ] **Step 5: Run profile tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py \
  -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/security.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_trusted_encryption.py
git commit -m "feat: add sync v2 profile bootstrap"
```

---

### Task 4: Implement Notes Materialization

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/base.py`
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/notes.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`

- [ ] **Step 1: Write failing Notes materializer tests**

Cover:

- clean `notes.note` upsert creates a normal note visible through `get_note_by_id`
- update with matching base state updates the note and object state
- update with stale `base_object_revision` or `base_object_hash` creates a whole-object conflict and does not overwrite the projection
- tombstone soft-deletes the note and updates object state
- tombstoned notes are not resurrected by stale upserts
- apply failure marks the accepted envelope as failed and leaves replay possible

- [ ] **Step 2: Add DB_Management sync helpers**

Add sync-specific note helpers in `note_store.py` rather than raw SQL in Sync core:

- `upsert_note_from_sync(...)`
- `tombstone_note_from_sync(...)`

Inputs should include stable note id, title, content, optional conversation/message refs, sync client id, object revision, object hash, and deletion state.

- [ ] **Step 3: Implement Notes materializer**

Validate payload shape and apply:

- upsert to ChaChaNotes
- tombstone to ChaChaNotes
- object state update after successful projection
- apply status update in Sync DB

Do not log title/content payloads.

- [ ] **Step 4: Run Notes tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/materializers \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py
git commit -m "feat: materialize sync v2 notes"
```

---

### Task 5: Implement Chat Materialization

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/chat.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/domain_adapters/chat.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`

- [ ] **Step 1: Write failing Chat materializer tests**

Cover:

- `chat.conversation` upsert creates metadata sufficient for server-front-end mode
- conversation metadata update with stale base state creates a whole-object conflict
- conversation tombstone soft-deletes the conversation
- `chat.message` append creates messages under an existing conversation
- duplicate stable message id with same hash is idempotent
- duplicate stable message id with different hash preserves both versions and creates a conflict record for review
- message tombstone soft-deletes the message and does not delete the conversation unless the conversation has its own tombstone

- [ ] **Step 2: Add DB_Management sync helpers**

Add sync-specific helpers:

- `upsert_conversation_from_sync(...)`
- `tombstone_conversation_from_sync(...)`
- `append_message_from_sync(...)`
- `tombstone_message_from_sync(...)`
- a message fetch helper that can include deleted rows

These helpers should preserve stable IDs and avoid bypassing store invariants.

- [ ] **Step 3: Implement Chat materializer**

Apply:

- `chat.conversation` whole-object upsert/tombstone
- `chat.message` append/dedupe/tombstone
- object state updates for both domains
- apply status updates

The message materializer must not require field-level merge UI in M1.

- [ ] **Step 4: Run Chat tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/materializers/chat.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters/chat.py \
  tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py
git commit -m "feat: materialize sync v2 chat"
```

---

### Task 6: Implement Attachment Ref Metadata Domain

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/attachment_refs.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/adapters.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py`

- [ ] **Step 1: Write failing attachment-ref tests**

Cover:

- `attachment.ref` envelopes validate required metadata: `attachment_id`, owning `parent_domain`, `parent_object_id`, `content_type`, `size_bytes`, `payload_hash`, and `availability`
- attachment refs are accepted, stored as envelopes, and returned by pull like other M1 domains
- duplicate attachment refs with the same `payload_hash` are idempotent
- duplicate attachment refs with different `payload_hash` create a conflict or rejection instead of overwriting history
- blob upload/download remains unsupported in M1 and returns `sync_blob_transfer_not_supported`
- restore preview reports refs as missing blobs unless availability says the server has the blob in a later mode

- [ ] **Step 2: Implement attachment-ref validation**

Add a small domain adapter or materializer validation path for `attachment.ref`. It should not write binary blobs. It should ensure attachment refs are linked to a known M1 parent domain when possible and preserve enough metadata for restore warnings.

- [ ] **Step 3: Wire attachment refs into service and schemas**

Ensure:

- capabilities advertise `attachment.ref`
- push accepts `attachment.ref`
- pull domain filters can include `attachment.ref`
- restore preview consumes attachment ref envelopes
- `/attachments` blob upload path is disabled or explicitly marked unsupported for M1

- [ ] **Step 4: Run attachment-ref tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/materializers/attachment_refs.py \
  tldw_Server_API/app/core/Sync/v2/adapters.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py
git commit -m "feat: sync attachment references metadata"
```

---

### Task 7: Wire Push, Pull, Conflicts, And Legacy Endpoint Replacement

**Files:**
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_error_mapping.py`

- [ ] **Step 1: Write failing service/API tests**

Assert:

- push accepts and materializes envelopes in order
- accepted push response includes server cursor and object revision
- projection failures return explicit apply errors without losing accepted envelopes
- pull returns accepted envelopes in deterministic cursor order with domain filters, pagination, `has_more`, next cursor, echo suppression by default, and opt-in same-device echoes for repair/debug
- conflicts use M1 actions: `keep_local`, `use_server`, `duplicate_rename`, `skip`
- conflict resolution persists a durable resolution record and, for `use_server` or `duplicate_rename`, creates a resolution envelope or accepted duplicate envelope instead of mutating historical envelopes
- cross-user access is blocked for datasets, pulls, pushes, conflicts, conflict resolution, and envelope ranges
- legacy `/send` and `/get` are removed or return a clear gone/replaced response
- attachment/blob upload endpoint returns `sync_blob_transfer_not_supported` in M1

- [ ] **Step 2: Add materializer registry**

Wire materializers into `SyncV2Service` through `factory.py`. Keep adapter validation separate from projection application:

- schema/domain validation
- base-state conflict detection
- append accepted envelope
- materialize
- update apply status and object state

If the same DB transaction cannot cover both Sync DB and ChaChaNotes DB, accepted envelopes must survive and failed projections must be replayable.

- [ ] **Step 3: Implement conflict resolution behavior**

Implement M1 actions:

- `keep_local`: record the user decision and leave the server projection unchanged; optionally mark the client conflict as resolved for that device/profile.
- `use_server`: record the decision and return the server object/envelope range the client should apply locally.
- `duplicate_rename`: create a new accepted envelope/object with a distinct object id or title/name suffix where the domain supports it.
- `skip`: dismiss the conflict without applying either side.

Resolution must append records/envelopes and never rewrite historical envelopes.

- [ ] **Step 4: Update endpoint contract**

Replace public behavior with M1 endpoints. Remove media-shaped request dependencies from the public path.

- [ ] **Step 5: Run service/API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_error_mapping.py \
  -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_error_mapping.py
git commit -m "feat: wire sync v2 m1 push pull conflicts"
```

---

### Task 8: Route Server-Origin Notes And Chat Changes Through Sync

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/server_origin.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`

- [ ] **Step 1: Write failing server-origin capture tests**

Assert:

- a note created or edited through normal server Notes APIs records a `notes.note` server-origin envelope before the ChaChaNotes projection can exist
- a conversation created or edited through normal Chat APIs records a `chat.conversation` server-origin envelope before the ChaChaNotes projection can exist
- a message created through normal Chat message APIs records a `chat.message` server-origin envelope before the ChaChaNotes projection can exist
- tombstones created through normal delete APIs record tombstone envelopes before projection deletion can exist
- a second offline-sync device can pull server-origin envelopes by cursor/domain filter
- server-origin envelopes are tagged with a server device/source id and do not require a Chatbook local `device_id`
- server-origin capture is idempotent when an API retries after a successful write
- envelope append failures fail or roll back the normal API mutation so no server projection exists without a log entry
- materialization failures leave an accepted/pending server-origin envelope with `apply_status=failed`, return an API error, and are replayable
- successful server-origin envelopes update sync object state and apply status
- failures are visible in profile status and do not silently claim Sync v2 is healthy

- [ ] **Step 2: Implement server-origin mutation service**

Add a small service that turns normal server mutations into Sync v2 envelopes and applies them through the same materializer path used by client push:

- derive domain and object id from the authorized mutation request plus current object state
- compute `payload_hash` and object hash from canonical payload data before append, then verify against the materialized row after apply
- set `base_server_cursor` and base object metadata from Sync object state before the mutation when available
- set `created_at_client` to the server mutation timestamp and `received_at_server` to append time
- use `server_trusted_v1`
- mark source as `server_frontend` or `server_api`
- append the envelope before projection writes happen
- materialize through the Notes/Chat materializers
- update object state and envelope apply status after materialization

The service must not bypass normal Notes/Chat authorization. The endpoint authorizes first, then calls this service instead of doing a direct ChaChaNotes mutation for M1 personal sync objects.

- [ ] **Step 3: Wire synced server API mutations through the service**

Wire M1 personal-scope server APIs that mutate synced objects through the server-origin mutation service:

- Notes create/update/delete in `notes.py`
- Chat conversation/session create/update/delete in `character_chat_sessions.py`
- Chat message create/delete in `character_messages.py`

If a user has not bootstrapped Sync v2, these endpoints keep their existing direct-write behavior. If the mutation is workspace-scoped, keep the existing direct-write behavior because workspace sync is an M3 domain. If the mutation is personal and Sync v2 is active, do not direct-write ChaChaNotes first.

- [ ] **Step 4: Expose capture health**

Profile/status responses must show pending or failed server-origin materialization. Server-front-end mode must not be able to create durable personal Notes/Chat state outside the envelope log once Sync v2 is active for that user.

- [ ] **Step 5: Run server-origin tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/server_origin.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
git commit -m "feat: route server-origin sync mutations"
```

---

### Task 9: Implement Restore Preview And Conflict Review Data

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/restore.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- Modify: `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`

- [ ] **Step 1: Write failing restore preview tests**

Cover:

- empty inventory returns clean restore plan and envelope ranges
- non-empty inventory with matching revisions returns safe applies
- same note id with different local hash returns whole-object conflict
- same conversation id with different local hash returns whole-object conflict
- tombstones appear as delete/hide actions
- attachment refs are included in preview and report missing blobs in M1
- cross-user restore access is blocked for dataset ids, envelope ranges, object summaries, conflicts, and attachment refs

- [ ] **Step 2: Implement restore preview service**

Add request model with local inventory:

- `domain`
- `object_id`
- `object_revision`
- `object_hash`
- `deleted`
- optional attachment availability

Return:

- available datasets/domains
- per-domain and total counts
- latest cursor per domain
- safe applies
- object conflicts
- tombstones
- missing blobs
- attachment ref summaries with parent object references
- envelope ranges needed for local apply
- encryption/key status

- [ ] **Step 3: Wire endpoint**

Add:

- `POST /api/v1/sync/restore/preview`

Keep `GET /restore-manifest` only if it remains useful as a metadata-only helper; the Chatbook M1 flow should use restore preview.

- [ ] **Step 4: Run restore tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py
git commit -m "feat: add sync v2 restore preview"
```

---

### Task 10: Add Replay And Repair

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/replay.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`

- [ ] **Step 1: Write failing replay tests**

Assert:

- replay can rebuild a note projection from accepted envelopes
- replay can rebuild a chat conversation plus messages
- failed apply envelopes can be retried after fixing the projection issue
- replay preserves tombstones
- replay never replays conflict envelopes as accepted changes

- [ ] **Step 2: Implement replay service**

Add repair path that scans accepted envelopes by dataset/domain/cursor and applies them through the same materializers used by push.

Keep replay internal/admin-safe for M1 unless product explicitly needs a public endpoint. If exposed, require authenticated user scope and explicit dataset id.

- [ ] **Step 3: Add status integration**

Profile status should show failed apply counts and latest repair result.

- [ ] **Step 4: Run replay tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py
git commit -m "feat: add sync v2 replay repair"
```

---

### Task 11: End-To-End Verification And Hardening

**Files:**
- Modify: `tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py`
- Modify: `Docs/API/Sync_V2_M1.md`
- Modify: `Docs/Design/Sync_V2_M1_Implementation_Decisions.md`
- Modify: `backlog/tasks/...` child implementation tasks

- [ ] **Step 1: Add final E2E scenario matrix**

Update `test_chatbook_sync_v2_restore.py` or split it into focused tests that cover:

- device A bootstraps, pushes a note, conversation, messages, tombstones, and attachment refs
- device B bootstraps, pulls with domain filters and pagination, and sees no same-device echoes unless requested
- server-front-end API writes route through Sync v2 and create server-origin envelopes that device B can pull
- clean profile restore preview returns safe applies and missing blob warnings
- non-empty restore preview returns note and conversation conflicts
- duplicate chat message IDs dedupe by stable ID plus `payload_hash`
- same message ID with a different `payload_hash` creates a conflict
- tombstones prevent deleted notes/messages from reappearing
- user B cannot access user A datasets, pulls, envelope ranges, conflicts, conflict resolutions, restore previews, or attachment refs

- [ ] **Step 2: Run targeted Sync and ChaChaNotes tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Sync \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py \
  tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  -q
```

Expected: pass.

- [ ] **Step 3: Run broader relevant API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB \
  tldw_Server_API/tests/e2e/test_chats_and_characters.py \
  tldw_Server_API/tests/e2e/test_workspace_chat_scope.py \
  -q
```

Expected: pass or document unrelated pre-existing failures.

- [ ] **Step 4: Run Bandit on touched production scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2 \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
  -f json -o /tmp/bandit_sync_v2_m1.json
```

Expected: no new findings in touched code.

- [ ] **Step 5: Run final diff checks**

Run:

```bash
git diff --check
rg -n "T[O]DO|FIX[M]E|client_private_v1.*default|source_cache|workspaces|media" \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/app/core/Sync/v2 \
  Docs/API/Sync_V2_M1.md \
  Docs/Design/Sync_V2_M1_Implementation_Decisions.md
```

Expected: no unresolved M1 contradictions.

- [ ] **Step 6: Update Backlog tasks**

For each child implementation task:

- record touched files
- record verification commands/results
- record Bandit result or skip rationale
- mark complete only after tests pass

- [ ] **Step 7: Final commit**

```bash
git add tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py \
  Docs/API/Sync_V2_M1.md \
  Docs/Design/Sync_V2_M1_Implementation_Decisions.md \
  backlog/tasks
git commit -m "test: verify sync v2 m1 end to end"
```

## Implementation Notes

- Use `superpowers:test-driven-development` before code changes in each task.
- Use `superpowers:systematic-debugging` for any failing or flaky tests.
- Use `superpowers:verification-before-completion` before claiming a task or PR is complete.
- Keep changes in the dedicated Sync worktree. Do not touch unrelated dirty files in the main checkout.
- Do not silently preserve legacy `/sync/send` and `/sync/get` semantics. The user explicitly approved replacing `/api/v1/sync` in place.
- Avoid raw SQL from Sync materializers into ChaChaNotes tables. Add DB_Management helper methods where projection behavior needs direct persistence.
- Do not claim server-trusted M1 encryption if only envelopes are covered. Status/capabilities must make the configured at-rest boundary explicit.
- Chatbook implementation is intentionally not in this server plan. Create a separate Chatbook plan after the server contract lands.
