# Character Conversation Behavior Snapshot Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make new server-owned character conversations resumable with their creation-time behavior by storing an immutable snapshot and exposing a capability-gated, idempotent, pre- and post-generation fenced completion contract.

**Architecture:** Add one pure snapshot module, one shared atomic character-conversation factory, one focused ChaCha persistence/CAS store, and typed API additions around the existing character-session routes. TASK-13134 allocates the next free schema version for snapshot, settings, and complete-history fencing; TASK-13135 then adds prior-tail-CAS idempotent append, snapshot-only prompt preparation, generation-fence persistence, and exact capability advertisement without changing legacy completion behavior.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, SQLite and PostgreSQL ChaCha backends, TestClient, pytest, Loguru, Bandit.

**Required implementation skills:** @superpowers:test-driven-development for every behavior change, @superpowers:systematic-debugging for unexpected failures, @superpowers:verification-before-completion before each PR, and @superpowers:requesting-code-review before merge.

**Approved downstream design:** `rmusser01/tldw_chatbook/Docs/superpowers/specs/2026-08-27-server-backed-roleplay-conversation-resume-design.md`

**Backlog tasks:** TASK-13134 (snapshot foundation), TASK-13135 (resume completion contract)

**ADR required:** yes

**ADR path:** `backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md`

**Reason:** This changes persistent schema, historical-data policy, authority, and the character completion/persistence interface.

---

## Global constraints

- Implement and merge TASK-13134 before starting TASK-13135. Keep each task to one reviewable PR.
- Do not backfill legacy conversations from current mutable sources. Their snapshot status is `missing`.
- Do not change legacy `/complete-v2` semantics unless the request selects contract version 1.
- Contract completion may read only the immutable snapshot, versioned conversation state, exact fenced history, and deployment/runtime inputs. It must not call current card, preset, exemplar, lore/world-book, or memory loaders.
- Never persist credentials, provider secrets, portrait/attachment binaries, raw auth context, or live tool/retrieval output in the snapshot.
- Non-stream assistant persistence is required. Stream persistence is advertised only after its CAS path is complete.
- Post-generation conflict never creates/selects a branch. Return generated text with `saved=false` and `code="generation_fence_changed"`.
- Before TASK-13134 implementation, rebase on current `origin/dev`, inspect the
  actual ChaCha schema head, and allocate the next free version for both SQLite and
  PostgreSQL. The plan-time head is v63 and examples call the candidate v64; do not
  hard-code v64 if the upstream head has advanced.
- Route every production character-conversation creator through the shared atomic
  snapshot factory. An intentionally unsupported creator, including the active-Sync
  path until its server-origin mutation is atomic with snapshot creation, must create
  explicit `missing`/non-resumable readiness and fail the contract append before it
  writes a user row.
- Fence the complete authoritative history, not only its tail: every add, update,
  delete, restore, branch/tail selection, and assistant insert advances a monotonic
  conversation `history_version` transactionally.
- Materialize the effective provider, model, and closed set of explicit sampling
  values into settings v1 during creation. A conversation without a complete valid
  effective-settings projection is non-resumable; later deployment-default changes
  never substitute for that missing or stored state.
- Stream persistence accepts only a short-lived, domain-separated HMAC grant emitted
  by the server after generation and bound to authenticated owner, scope,
  conversation, input/parent, stable assistant ID, full generation fence, and final
  content digest. Signing key material is dedicated and server-only, never the
  client-known single-user API key. Identical replay is idempotent; tamper or
  cross-context reuse fails.
- Do not advertise the contract or a feature until all behavior and tests for it exist.

## File map

### Create

- `tldw_Server_API/app/core/Character_Chat/character_behavior_snapshot.py` — schema-v1 payload, canonical JSON/digest, size/secret validation, and materialization helpers.
- `tldw_Server_API/app/core/Character_Chat/character_conversation_factory.py` — one atomic character-conversation creation seam shared by API and helper/workflow callers.
- `tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py` — contract constants, snapshot-only prompt preparation, generation-fence types, signed streamed-persist grants, and structured outcomes.
- `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py` — coherent snapshot/settings/history/tail reads, CAS-idempotent append, and fenced assistant commit.
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py`
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_generation_fence.py`
- `tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py`

### Modify

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` — next-version snapshot/history migrations, store construction/delegation, catalog verification.
- `tldw_Server_API/app/core/DB_Management/chacha/__init__.py` — lazy store export.
- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` — caller-owned transaction seam and settings version.
- `tldw_Server_API/app/core/DB_Management/chacha/message_store.py` — caller-owned transaction seam.
- `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py` — readiness, idempotency, completion-fence, response, and persist models.
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` — atomic snapshot creation, enriched reads/settings, contract completion, fenced persist.
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py` — caller-selected user IDs and conflict mapping.
- `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py` — delegate `start_new_chat_session` to the shared factory or explicitly return non-resumable readiness.
- `tldw_Server_API/app/api/v1/endpoints/config_info.py` — exact capability version/features.
- `tldw_Server_API/app/core/config.py` and `tldw_Server_API/Config_Files/{config.txt,README.md}` — dedicated server-only Roleplay persist signing secret and optional secondary rotation key; never expose their values.
- `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`
- `tldw_Server_API/tests/Characters/test_character_chat_lib.py`
- `backlog/tasks/task-13134 - Persist-immutable-character-conversation-behavior-snapshots.md`
- `backlog/tasks/task-13135 - Enforce-versioned-Roleplay-resume-completion-contract.md`

### Read-only reference

- `tldw_Server_API/app/core/Character_Chat/modules/character_prompt_presets.py`
- `tldw_Server_API/app/core/Character_Chat/modules/character_generation_presets.py`
- `tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py`
- `tldw_Server_API/app/core/Persona/exemplar_prompt_assembly.py`

## Delivery stages

| Stage | Owner | Outcome | Merge gate |
| --- | --- | --- | --- |
| 1 | TASK-13134 | Freeze snapshot v1 and input classification | Pure tests green |
| 2 | TASK-13134 | Persist atomically in the next schema and expose readiness across every creator | Migration/create tests green |
| 3 | TASK-13134 | Materialize/version settings plus complete history fences | PR 1 reviewed and merged |
| 4 | TASK-13135 | CAS idempotent user append and pre-dispatch snapshot/history fences | No-dispatch tests green |
| 5 | TASK-13135 | Generation CAS, optional stream persist, capabilities | PR 2 reviewed and merged |

## Stage 1 — TASK-13134: Freeze snapshot v1

### Task 1: Define the canonical behavior snapshot

**Files:**
- Create: `tldw_Server_API/app/core/Character_Chat/character_behavior_snapshot.py`
- Create: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py`

- [ ] **Step 1: Write failing canonicalization and mutation-isolation tests**

Build a two-participant source with shuffled mapping order. Assert deterministic digest, LF normalization, deep-copy isolation, participant coverage, and the following explicit top-level shape:

```python
{
    "schema_version": 1,
    "participants": [{
        "source": {"kind": "character", "id": "7", "version": 3},
        "identity": {"name": "Ari", "aliases": ["Ari"]},
        "prompt": {
            "system_prompt": "...",
            "description": "...",
            "personality": "...",
            "scenario": "...",
            "message_example": "...",
            "post_history_instructions": "...",
            "prompt_relevant_extensions": {},
        },
        "greeting": {"content": "...", "source": "default", "source_index": 0},
        "generation_defaults": {},
        "exemplars": [],
        "world_books": [],
        "default_memory": None,
    }],
    "routing_defaults": {"turn_taking_mode": "single"},
}
```

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement minimal canonicalization**

Deep-copy JSON-compatible values; normalize line endings; serialize with `ensure_ascii=False`, `sort_keys=True`, and compact separators; digest canonical bytes with SHA-256 and prefix `sha256:`. Return a frozen boundary object containing schema version, payload, canonical bytes, digest, and size.

- [ ] **Step 4: Add closed-schema/type/size fail-closed tests**

Prove the closed snapshot schema/classified source allowlist has no credential-bearing
fields. Reject explicitly credential-named keys only inside deliberately extensible
maps, plus binary values, non-finite floats, unsupported objects, missing participants,
duplicate participant identity, and configured-size overflow. Do not recursively
reject generic words such as `token` inside legitimate prompt-extension content; key
name scanning is not a substitute for the closed schema.

- [ ] **Step 5: Run GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py
git add tldw_Server_API/app/core/Character_Chat/character_behavior_snapshot.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py
git diff --cached --check
git commit -m "feat(character-chat): define behavior snapshot contract"
```

Expected: PASS; this commit has no API or DB changes.

## Stage 2 — TASK-13134: Store snapshots atomically

### Task 2: Allocate the next ChaCha schema version and add the resume store

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`
- Create: `tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`

- [ ] **Step 1: Rebase, recheck the schema head, then write migration tests**

Rebase on current `origin/dev`, record the actual schema head, and select exactly the
next free version (`NEXT`; v64 only if the head is still v63). Create a head-version
database with a legacy character conversation/settings row. Assert `NEXT` has a
one-to-one `conversation_behavior_snapshots` table with status, schema version,
canonical JSON, digest, size, and timestamp; `conversation_settings.settings_version`
starts at 1; conversations expose a monotonic `history_version`; legacy snapshot reads
as `missing` with no current-source body. Inject a migration checkpoint failure and
prove total rollback to the prior head.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py
```

Expected: FAIL because the selected next-version migration and resume-store methods are absent.

- [ ] **Step 3: Implement the next SQLite and PostgreSQL migration**

Update both current schema constants and migration ladders using the versions found in
Step 1. Enforce statuses `valid|missing|invalid`; valid rows require
version/digest/body/size, non-valid rows contain no snapshot body. Add
`history_version >= 1` to the conversation resume state. Verify catalog, FK,
uniqueness, digest check, indexes, and exact one-version advance for both backends.

- [ ] **Step 4: Implement the focused store and transaction seams**

Add:

```python
def put_behavior_snapshot(self, conversation_id: str, snapshot: BehaviorSnapshotV1, *, conn) -> None: ...
def get_conversation_behavior_snapshot(self, conversation_id: str) -> dict[str, Any]: ...
def get_roleplay_resume_state(self, conversation_id: str, *, conn=None) -> dict[str, Any]: ...
```

Allow existing conversation/message inserts to accept optional caller-owned `conn`;
use `nullcontext(conn)` when supplied and preserve legacy transaction behavior
otherwise. Centralize `history_version` advancement in the existing message mutation
primitives so add/edit/delete/restore/branch/tail paths cannot forget it. Register and
delegate the new store following current ChaCha patterns.

- [ ] **Step 5: Run backend tests and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_postgres_backends.py
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/__init__.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py
git diff --cached --check
git commit -m "feat(character-chat): persist behavior snapshot storage"
```

Expected: both backend contracts pass.

### Task 3: Snapshot conversation creation and expose readiness

**Files:**
- Create: `tldw_Server_API/app/core/Character_Chat/character_conversation_factory.py`
- Create: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`
- Modify: `tldw_Server_API/tests/Characters/test_character_chat_lib.py`

- [ ] **Step 1: Write failing creation/rollback/source-mutation tests**

Create a character using custom prompt/generation presets, greeting, exemplar, world
book, memory, effective provider/model, explicit sampling values, and a second
participant. Exercise the API endpoint,
`start_new_chat_session`, and the import/workflow caller; assert each either uses the
same atomic factory and receives a valid snapshot covering both participants, or is
explicitly classified non-resumable without advertising readiness. Mutate/delete each
source and change deployment provider/model/sampling defaults; assert stored
snapshot/settings bytes and effective completion inputs are unchanged. When a creator
cannot resolve a valid effective provider/model plus the closed sampling schema,
assert it creates explicit incomplete-settings/non-resumable readiness and contract
append rejects before writing. Patch snapshot persistence to
raise and prove conversation/settings/snapshot all roll back. Exercise active Sync and
prove its server-origin mutation path is either atomic with the snapshot or yields
`missing`/`resume_eligible=false` before any contract append can write.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  -k "creation or rollback or source_mutation or legacy"
```

Expected: FAIL because creation does not capture readiness.

- [ ] **Step 3: Add bounded readiness schemas**

```python
class BehaviorSnapshotStatus(BaseModel):
    status: Literal["valid", "missing", "invalid"]
    schema_version: int | None = None
    digest: str | None = None

class ConversationTailFence(BaseModel):
    message_id: str | None
    message_version: int | None
```

Extend session detail with `behavior_snapshot`, `resume_eligible`,
`resume_ineligible_reason`, `settings_version`, `history_version`, `message_count`,
and `tail`. Ordinary reads expose metadata, never snapshot body. Capability remains a
server-wide implementation claim; per-conversation readiness is authoritative for
whether append/completion is allowed.

- [ ] **Step 4: Implement one atomic creation factory**

Resolve every classified source with source version/provenance, materialize the
snapshot, then revalidate all source versions in the create transaction. Retry
materialization once on drift or fail. Before insertion, resolve explicit
creation-time provider/model or the configured effective defaults once, and normalize
the exact closed sampling fields consumed by completion. Insert those materialized
values as settings v1 together with conversation, history version, initial messages,
and snapshot in the same transaction through `character_conversation_factory.py`.
Never persist credentials. If effective provider/model/sampling cannot be validated,
store an explicit `resume_eligible=false` reason and make contract append reject
before insertion instead of consulting later deployment defaults. Make the endpoint and
`modules/character_chat.py::start_new_chat_session` delegate to this seam. Plain
non-character chats and intentionally unsupported/active-Sync creators remain
explicit snapshot-missing, `resume_eligible=false` records.

- [ ] **Step 5: Add invalid/legacy/size/auth/multiparticipant coverage and run GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py \
  tldw_Server_API/tests/Characters/test_character_chat_lib.py \
  -k "chat or snapshot or session or sync"
```

Expected: PASS; malformed storage reads `invalid`, legacy reads `missing`, oversize rolls back, and cross-user reads reveal nothing.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Character_Chat/character_conversation_factory.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_chat.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  tldw_Server_API/tests/Characters/test_character_chat_lib.py
git diff --cached --check
git commit -m "feat(character-chat): snapshot behavior at conversation creation"
```

## Stage 3 — TASK-13134: Version conversation behavior settings

### Task 4: Materialize settings and publish coherent fences

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py`
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py`

- [ ] **Step 1: Write failing materialization/history-version tests**

Start at version 1 with a complete materialized effective provider/model and explicit
sampling projection. Change deployment defaults and prove resume still uses the v1
values. Apply provider/model/sampling, preset/world-book/overlay/participant/memory
settings and assert resolved content/digest is stored, not just mutable IDs; version
becomes 2. A successful no-op behavior mutation still advances the fence.
Unknown/unauthorized references reject the entire write without version change. Later
source mutation cannot affect stored settings.

From a coherent resume-state read, add/edit/delete/restore an ancestor message and
change branch/tail selection; assert each successful mutation increments
`history_version` exactly once in its transaction even when the final tail row itself
is unchanged. Failed/rolled-back mutations do not advance it.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py \
  -k "settings_version or history_version or materialized or reference"
```

Expected: FAIL because settings currently use timestamp/conversation-version behavior only.

- [ ] **Step 3: Implement transactional materialization and versioning**

Resolve known behavior references before storage, canonicalize embedded materialized
values, validate provider/model plus the closed sampling schema, and update JSON plus
`settings_version + 1` in one transaction. Contract prompt construction consumes only
the stored effective provider/model/sampling values; credentials and current
availability remain runtime state. The contract path consumes an explicit classified
allowlist and rejects unclassified behavior input; preserve legacy unknown-key
compatibility outside contract completion.

- [ ] **Step 4: Make resume-state reads coherent**

Return snapshot metadata, materialized settings/version, `history_version`, message
count, and tail message ID/version from one transaction. Every central history
mutation advances `history_version`; retain tail/message versions for exact-row
identity but never treat them as a complete history fence. Do not overload settings
version for transcript mutations.

- [ ] **Step 5: Verify TASK-13134**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py \
  tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Character_Chat/character_behavior_snapshot.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  -f json -o /tmp/bandit_task_13134.json
git diff --check
```

Expected: targeted tests and Bandit pass.

- [ ] **Step 6: Close and merge TASK-13134**

Use Backlog CLI/MCP to link ADR-002/this plan, record exact evidence, complete AC/DoD, add implementation notes, and mark Done. Commit the task update, request code review, and merge PR 1 before TASK-13135 starts.

## Stage 4 — TASK-13135: Idempotent append and pre-dispatch fences

### Task 5: Add caller-ID append and exact request validation

**Files:**
- Create: `tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py`
- Create: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py`
- Create: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`

- [ ] **Step 1: Write failing request validation**

Add a contract block with version 1, expected snapshot digest/settings version,
expected `history_version`, exact input user ID/version, and expected tail ID/version.
When present, require `append_user_message` absent/false and reject `save_to_db=False`,
current-card context, directed speaker, provider/model/sampling, prompt preset,
steering, tools, and other behavior overrides. `stream` is transport only.

- [ ] **Step 2: Write failing append idempotency and prior-tail CAS tests**

Caller-selected user ID: identical retry returns the same authoritative ID/version
and no extra row; changed content/role/parent/conversation/image returns structured
`idempotency_conflict`; cross-user probing is concealed. A new append carries expected
prior tail ID/version and `history_version`; race a concurrent append, ancestor edit,
delete, or branch/tail change and assert `history_fence_changed`, no caller row, and no
implicit branch. Missing/invalid snapshot, active-Sync/non-resumable policy, or changed
settings rejects before insertion. Idempotent replay of an already-accepted caller ID
returns its authoritative result even though the conversation fence has since advanced.

- [ ] **Step 3: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py \
  -k "request or idempotent or conflicting_reuse or prior_tail or history_fence or resume_eligible"
```

Expected: FAIL because the contract block and caller-ID append do not exist.

- [ ] **Step 4: Implement minimal models and append CAS transaction**

Add a dedicated `RoleplayResumeUserAppendRequest`/contract block containing caller
`message_id`, expected snapshot digest/settings version, prior tail ID/version,
expected `history_version`, and contract version; do not make arbitrary caller IDs a
generic legacy `MessageCreate` capability. Route only contract user messages through
`append_idempotent_user_message()`: first resolve an identical existing caller ID;
otherwise atomically validate ownership/global scope, snapshot/resume eligibility,
settings/history/prior-tail fences, and then insert exactly one user row while
advancing `history_version`. Return authoritative ID/version/parent plus coherent
snapshot/settings/history/tail fences. Leave generated-ID legacy sends unchanged.

- [ ] **Step 5: Run GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py \
  -k "request or idempotent or conflicting_reuse or prior_tail or history_fence or resume_eligible"
git add tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py
git diff --cached --check
git commit -m "feat(character-chat): add idempotent resume input contract"
```

### Task 6: Freeze snapshot-only prompt inputs before dispatch

**Files:**
- Modify: `tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/Config_Files/README.md`
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py`
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py`

- [ ] **Step 1: Write failing no-dispatch fence tests**

Patch provider dispatch. Parameterize changed snapshot digest, settings version,
`history_version`, input ID/version, tail ID/version, deleted/non-user input,
ancestor edit/delete, missing/invalid snapshot, and Sync-origin restriction. Assert
structured conflict/policy result, provider not awaited, and no assistant insert.

- [ ] **Step 2: Write current-source isolation tests**

After creation, mutate/delete every classified source. Patch current-source loaders to raise if called. Capture provider input and assert effective prompt/settings derive only from snapshot/materialized settings/history.

- [ ] **Step 3: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py \
  -k "fence or no_dispatch or source_isolation"
```

Expected: FAIL before the contract path exists.

- [ ] **Step 4: Implement one pre-dispatch transaction**

`prepare_roleplay_resume_generation()` verifies ownership/scope/policy, parses
snapshot, compares all fences, loads exact undeleted authoritative branch history
ending at the acknowledged user, freezes snapshot/settings/history, and returns a
generation fence covering conversation, snapshot digest, settings version,
`history_version`, input ID/version, and tail ID/version. Resolve
credentials/availability/safety after the fence as runtime state; missing saved
provider/model fails honestly.

- [ ] **Step 5: Return effective fences and a signed streamed-persist grant**

Add effective snapshot/settings/history/input/tail fields to the non-stream response
and first SSE metadata event. For streaming, require the client to supply its stable
assistant ID before dispatch. After generation, the terminal SSE event returns final
content digest plus a short-lived opaque persist grant: canonical JSON authenticated
with HMAC-SHA256 using dedicated server-only
`ROLEPLAY_RESUME_SIGNING_SECRET` material, optional
`ROLEPLAY_RESUME_SECONDARY_SIGNING_SECRET` rotation material, and
a Roleplay-specific domain separator. The secret resolver rejects missing, short,
placeholder, public, or client-known `SINGLE_USER_API_KEY` material; it may use the
existing `derive_hmac_key_from_source()` helper only after selecting the dedicated
server-only secret. Bind contract version, authenticated owner,
scope, conversation, exact input/parent ID and version, stable assistant ID, complete
generation fence, content digest, issued-at, and expiry. Never expose snapshot body,
token payload fields separately as authority, or signing material. If production key
material cannot be derived, fail closed and do not advertise stream persistence.

- [ ] **Step 6: Run GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py
git add tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/Config_Files/README.md \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py
git diff --cached --check
git commit -m "feat(character-chat): fence snapshot resume before generation"
```

## Stage 5 — TASK-13135: Fence assistant commit and advertise support

### Task 7: Compare-and-swap non-stream and stream persistence

**Files:**
- Create: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_generation_fence.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`

- [ ] **Step 1: Write failing non-stream concurrency tests**

Block provider execution; from a second connection mutate each fenced value,
including editing/deleting a non-tail ancestor while preserving the same tail row.
Release provider and assert HTTP success with generated content, `saved=false`,
`generation_fence_changed`, no assistant, and no branch. Unchanged state inserts
exactly one assistant parented to the exact user and advances `history_version` in the
same transaction.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_generation_fence.py \
  -k nonstream
```

Expected: FAIL because current completion inserts without post-generation CAS.

- [ ] **Step 3: Implement `commit_fenced_assistant()`**

Re-read snapshot/settings/`history_version`/input/tail components and insert
assistant/metadata only if still equal, in one transaction. Mismatch returns typed
unsaved output. Post-commit optional validation degradation may return
`saved=true, validation_degraded=true`; never deny a committed assistant.

- [ ] **Step 4: Add stream stable-ID/grant/CAS tests**

Identical stable-ID plus identical content/grant replay returns the same assistant;
changed content/parent/fence conflicts. Flip every signed payload/signature field;
substitute owner, scope, conversation, input/parent, assistant ID, content, or an
expired grant; and assert constant-time verification rejects before CAS without
revealing cross-user existence. Verify current and configured secondary key material
during rotation. Prove `SINGLE_USER_API_KEY`, public keys, known placeholders, and
missing/short secrets cannot sign or enable the stream feature. Mutation before persist yields `generation_fence_changed` with no
assistant/branch; simulated response loss reconciles or idempotently retries with the
same stable ID and grant.

- [ ] **Step 5: Implement authenticated fenced stream persist**

Add an optional v1 persist block requiring stable assistant ID, exact authoritative
user parent, generated content, digest, and opaque persist grant. Verify the
domain-separated HMAC with `hmac.compare_digest()` against current/secondary keys
derived only from the dedicated server-only Roleplay signing secrets, then compare
every authenticated payload field to current auth, route, and body
values and recompute the content digest before calling the same CAS commit helper.
The signed grant is stateless and replayable only for the identical idempotent
operation; assistant-ID/content conflict handling prevents duplicate or altered
commit. Preserve the current legacy fingerprint/speaker path when the v1 block is
absent, but never let it advertise or satisfy `stream_assistant_persist`.

- [ ] **Step 6: Run GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_generation_fence.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py
git add tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_generation_fence.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py
git diff --cached --check
git commit -m "feat(character-chat): fence resumed assistant persistence"
```

### Task 8: Advertise exact capability and close TASK-13135

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/config_info.py`
- Modify: `tldw_Server_API/tests/Config/test_docs_info_capabilities.py`
- Modify: `backlog/tasks/task-13135 - Enforce-versioned-Roleplay-resume-completion-contract.md`

- [ ] **Step 1: Write failing capability tests**

Assert both capability maps expose version 1 and base features `snapshot_completion`, `fenced_completion`, `idempotent_user_append`, `nonstream_assistant_persist`, plus `stream_assistant_persist` only because Task 7 is complete and production HMAC key material is derivable. Disabled character routes, disabled Resume, or unavailable signing material report no affirmative stream feature; no endpoint advertises a grant it cannot authenticate.

- [ ] **Step 2: Run RED, implement, run GREEN**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py -k roleplay_resume
```

Expected before: FAIL for missing keys. Source safe constants from `roleplay_resume_contract.py`; advertise only when full route/feature implementation is enabled. Re-run; expected PASS and equal capability maps.

- [ ] **Step 3: Run the complete targeted matrix**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_behavior_snapshot.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_roleplay_resume_contract.py \
  tldw_Server_API/tests/DB_Management/test_character_behavior_snapshot_migration.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_completion.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_roleplay_resume_generation_fence.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_postgres_backends.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_conversation_context_primitives.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py \
  tldw_Server_API/tests/Characters/test_character_chat_lib.py \
  tldw_Server_API/tests/Config/test_docs_info_capabilities.py
```

Expected: PASS; record exact counts/skips. Do not run the repository-wide suite unless requested.

- [ ] **Step 4: Run targeted Ruff, Bandit, and diff checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/app/core/Character_Chat/character_behavior_snapshot.py \
  tldw_Server_API/app/core/Character_Chat/character_conversation_factory.py \
  tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_chat.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/__init__.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/app/core/Character_Chat/character_behavior_snapshot.py \
  tldw_Server_API/app/core/Character_Chat/character_conversation_factory.py \
  tldw_Server_API/app/core/Character_Chat/roleplay_resume_contract.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_chat.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/__init__.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_resume_store.py \
  tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/api/v1/endpoints/config_info.py \
  -f json -o /tmp/bandit_task_13135.json
git diff --check
```

Expected: both tools exit 0 with no new findings.

- [ ] **Step 5: Record authenticated smoke when available**

Create a fresh conversation; confirm snapshot valid; mutate its card; caller-ID append; non-stream contract completion; confirm same conversation, unchanged digest, saved assistant. Then induce tail drift during generation and confirm `saved=false` with no assistant. Record sanitized capability/version/request IDs and outcomes, never credentials or snapshot bodies.

- [ ] **Step 6: Close, review, rebase, and merge**

Use Backlog CLI/MCP to link ADR-002/this plan, record evidence, complete TASK-13135 AC/DoD, add notes, and mark Done. Commit capability/task docs. Request code review. Rebase on current `origin/dev`, rerun targeted tests/Bandit, resolve every review thread with evidence, and merge only with green required checks. Chatbook TASK-23089 remains blocked until the capability-bearing server work is merged and available to its test server.
