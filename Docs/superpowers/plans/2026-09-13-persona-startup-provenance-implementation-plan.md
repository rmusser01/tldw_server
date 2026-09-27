# Workspace Persona Startup Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record how a new personal Workspace conversation selected its assistant, with shared default resolution and trustworthy local lifecycle behavior.

**Architecture:** Extend the existing Workspace resolver and conversation store. Persist one bounded, reference-only provenance object in the same insertion as assistant identity; invalidate it only on actual normalized identity changes. Keep transport inputs untrusted and project origin references through current Workspace visibility.

**Tech Stack:** Existing Python/FastAPI/Pydantic, ChaChaNotes SQLite/PostgreSQL, Sync v2 materializers, pytest/Hypothesis, Loguru and Bandit. No new dependencies.

**Spec:** [Approved Stage 2 choice/provenance design](../../Design/2026-09-13-persona-workspace-choice-provenance-design.md), specifically slice 2B. The design file retains its original proposal header; approval and Stage 2A delivery are recorded in the [parent plan](2026-09-13-persona-workspace-parity-implementation-plan.md). This focused plan does not authorize the deferred 2C protocol.

**Tracking:** [#2950](https://github.com/rmusser01/tldw_server/issues/2950), TASK-13245; this planning unit is TASK-13245.3. Based on [2A PR #2959](https://github.com/rmusser01/tldw_server/pull/2959), commit `8d5e6a5be37653134f7249a9699070940a61cbe3`. Create a separate execution child task before runtime edits; search first to avoid duplicates.

**Delivery status:** Reviewed plan published in [draft PR #2961](https://github.com/rmusser01/tldw_server/pull/2961). Execution started on 2026-09-14 under TASK-13245.4, branch `codex/persona-startup-provenance`. Stages 1-4 are complete and independently approved at `6a8da29037107da8f901c7095841b9a70712a8ec`; Stage 5 is in progress with individually recorded gates below. All five stages remain one activation unit; no partial runtime rollout. Final whole-branch review, publication and human-owned Change summary are not implied by stage completion.

## Global Constraints

- Persona/backend work only. No Buddy, animations, VN, UI, design-system backlog changes, Research RAG adoption, provisioning, or MCP profile activation.
- Preserve existing chat identity and omission/null precedence. No strict startup route, versioned selector, receipts, replay guarantee, send-time admission expansion, or Workspace Sync rollout; those belong to 2C or later stages.
- All five stages below are one activation unit in one implementation PR. Intermediate commits may be tested independently but must not be deployed/merged separately. If split into multiple PRs, keep provenance writes and public projection inactive until every lifecycle/transport protection lands.
- No Persona names, prompts, assistant IDs, memory modes, policy snapshots, or raw requests in provenance. Existing Character behavior snapshots are a different feature and stay unchanged.
- Source values are exactly `workspace_default`, `explicit`, `explicit_none`, `system_fallback`, `fork`, `unknown`. JSON cap: 1 KiB (1024 encoded UTF-8 bytes); schema version exactly integer 1; fixed keys only.
- `workspace_default` requires an originating Workspace ID and positive strict-integer version. `system_fallback` has both references or neither. Other sources have null references. NULL/corrupt storage projects `unknown`, never inferred identity/default equality.
- SQL stays within DB_Management. Existing authenticated per-user DB/RLS and visibility rules remain authoritative. Never rely on a caller-supplied user ID to select another user's DB.
- Drain old API, background and direct DB writers before schema migration; compatible binaries only afterward. No rolling upgrade or data-preserving old-binary downgrade. Retain the [2A maintenance procedure](../../Code_Documentation/Workspace_Persona_Defaults.md).
- Use existing SQLite and `pg_database_config` fixtures. Require live PostgreSQL verification for concurrency/migration acceptance, not skipped or mocked SQL cases. No custom database provisioning.
- Activate the shared repository `.venv` before Python/pytest/Bandit. Do not install or alter dependencies for this slice. Every new/modified module, class and function needs an appropriate docstring.
- Preserve the required human-written `Change summary` merge gate. This plan and an AI draft summary do not satisfy that policy.

## Baseline And Parity

On 2026-09-13, fetched server dev is `ebdeeac384c58559fa90fd3a5f79f5262ae190d5`; Chatbook dev is `4631b60f8dd9623fc55bf16f4a37e29fcb1240c7`. Scoped diffs from the preceding `beac8e9449b0e2fa90bdab89cf8cbf7905d2b915` / `fbf374c9d32144d5dd23fd44c9adc0e77f00d58f` baselines show no changes to relevant server DB/Chat/Character_Chat/Workspaces/Sync and endpoint code, or Chatbook `Workspaces/` and `Chat/console_assistant_defaults.py`. Read committed Chatbook files; leave its local checkout untouched.

Server dev remains schema 67; the stacked 2A branch is 68. At this baseline 2B uses registered v68-to-v69 migrations on both backends. Recheck dev and the migration registry before implementation/rebase: if 69 has been allocated, use the next version and adjust tests, never reuse another feature's version.

Fresh planning baseline: **71 passed, six warnings** across `tests/Workspaces/test_workspace_assistant_creation.py`, `test_workspace_assistant_defaults_api.py`, and `tests/Chat/unit/test_chat_conversations_api.py`. Log: `/tmp/persona-provenance-plan-baseline.log`. 2A separately passed 114 SQLite/live-PostgreSQL tests without skips. Neither run tests new provenance behavior.

**Final execution refresh (2026-09-14):** Server dev `1e0bb6feddab5a9e4be794ea44ea21b9ea29bf30` adds only VZ drill tooling/docs since `ebdeeac384c58559fa90fd3a5f79f5262ae190d5`; Chatbook dev `87a3de4493d7c78d0263c184ccaf2f590c9c277e` changes Notes/import/sync, Library/console-access UI, first-run and release files since `4631b60f8dd9623fc55bf16f4a37e29fcb1240c7`. The compared Chat/Persona/Workspace contract paths are unchanged. Server dev still uses v67; the stack has registered v68-to-v69 migrations for 2B. No Chatbook checkout changes or rebase were made. Planning PR #2961 was rechecked open/draft at unchanged `8876d2d187`; implementation [draft PR #2963](https://github.com/rmusser01/tldw_server/pull/2963) targets that planning branch.

**Dev integration refresh (2026-09-25):** The complete prerequisite stack was replayed locally onto server dev `a2f5e1b816cfe189db7f553a1ccf8d481dc2edbe`. Dev now owns SQLite v68 and PostgreSQL v72. Stage 2A opt-out uses SQLite v69 and PostgreSQL v73; Stage 2B provenance uses SQLite v70 and PostgreSQL v74. The 2026-09-14 validation numbers and version descriptions below remain historical evidence from the earlier base; the current integration gates are tracked in `IMPLEMENTATION_PLAN_pr2963_dev_rebase_20260925.md` and TASK-13245.5.

Chatbook's `resolve_new_console_assistant` still bypasses defaults for supplied custom prompts and degrades unavailable defaults to an untracked assistant with a notice. Preserve the approved server difference: invalid/disabled configured defaults fail closed. No copying local IDs or prompt snapshots. Source comparison is not a Chatbook runtime test or full parity certification.

## File Responsibilities

All `app/` and `tests/` paths below are relative to `tldw_Server_API/`. Line references identify the baseline, not permanent insertion offsets.

| File | Responsibility |
| --- | --- |
| New `app/core/Chat/assistant_startup.py` | Dependency-light validated provenance value, bounded serializer/parser; no DB or HTTP imports. |
| Existing `app/core/Workspaces/assistant_defaults.py` | Shared effective-state rules, creation selection and visibility-safe provenance projection; no new parallel resolver. |
| `app/core/DB_Management/ChaChaNotes_DB.py` | Registered migration, connection-aware locking reads, existing initializer repair safeguards and façade signatures. |
| `app/core/DB_Management/chacha/persona_state_store.py` | Optional connection-aware/locking profile lookup with current owner predicate. |
| `app/core/DB_Management/chacha/conversation_store.py` | Trusted insertion keyword, normalization, atomic identity invalidation and Sync upsert race handling. |
| `app/api/v1/endpoints/workspaces.py` | Thin management projection adapter, preserving existing output and mapped DB errors. |
| `app/api/v1/endpoints/character_chat_sessions.py` | New Workspace chat orchestration, caller-input rejection integration, detail/list conversion; preserve global Sync branch. |
| `app/core/Character_Chat/character_conversation_factory.py` | Narrow internal provenance keyword forwarded to its existing atomic INSERT; no behavior-snapshot redesign. |
| `app/api/v1/schemas/chat_session_schemas.py`, `chat_conversation_schemas.py` | Read-only public value on existing models; targeted rejection in create/update models without globally forbidding legacy extras. |
| `app/api/v1/endpoints/chat.py` | Conversation list/tree/update projection integration. |
| Existing Sync and transport files in the inventory below | Keep explicit input/output field lists; exercise local authority and privacy boundaries. |

## Confirmed Mutation Inventory

| Path | Current boundary | Required behavior |
| --- | --- | --- |
| `character_chat_sessions.py:4505` | `create_chat_session` resolves before rate/quota checks and later inserts at 4726; Character branch calls its factory. | Keep async checks outside transactions. Resolve inherited Workspace identity/version and insert provenance in one synchronous transaction; do not reuse the earlier request copy as authority. |
| `conversation_store.py:387` | `add_conversation(..., conn=...)`, single INSERT. | Add a separate typed internal keyword. Arbitrary `conv_data` cannot supply public/raw provenance. Legacy/direct insertion with no trusted value starts unknown. |
| `character_conversation_factory.py:2328` | Factory prepares WorldBook/schema services before its own transaction at 2390; inserts at 2484. | Forward explicit/fork provenance into that INSERT. Do not wrap its preflight in a new outer transaction: legacy schema helpers can commit implicitly. |
| `conversation_store.py:1020` | Ordinary update reads identity, normalizes merged fields and performs CAS. | Lock/read and compare normalized old/new identity; reset provenance in the same UPDATE only on actual identity/memory changes. |
| `conversation_store.py:513` | Sync INSERT/ON CONFLICT independently replaces identity, scope, version and deletion status. | Metadata-equivalent binding preserves provenance; identity change clears it. New row is unknown. Handle a concurrent insert winner without overwriting its provenance from stale state. |
| `Sync/v2/materializers/chat.py:63,466` | Explicit argument allowlist, whole-object payload defaults (including a `sync-v2` Persona fallback). | Never pass incoming provenance. A partial title-only envelope is not necessarily metadata-only under current semantics; compare the actual resulting binding without changing the Sync protocol. |
| `ChaChaNotes_DB.py:17763,17934` | SQLite/PostgreSQL initialization repairs missing Character kind/ID outside normal updates. | On schemas with provenance, equivalent normalization preserves it; repair of an invalid/different binding clears it. Older schemas must never reference the new column. |
| `conversation_store.py:194`; `message_store.py:65` | Settings/history/version changes do not change assistant identity. | Preserve provenance. Do not install a blanket UPDATE/version-change invalidation trigger. |
| `conversation_store.py:1248,1318,1389,622` | Soft delete, restore, hard delete and Sync tombstone. | Soft deletion/restore/tombstone preserve stored origin; hard deletion removes it with the row. No receipt table in 2B. |
| `ChaChaNotes_DB.py:27652,27688` | Workspace deletion iterates conversation deletion helpers. | Cover behavior without redesigning cascade atomicity. No local move API exists; only existing scope mutation paths are in scope. |
| `services/admin_data_subject_requests_service.py:458` | Account erasure removes conversation rows directly. | Row deletion removes provenance; do not add separate retention. No change required unless tests expose a new export leak. |

Remaining named callers found by repository search delegate to these protected boundaries: `Chat/chat_helpers.py:329` (ordinary chat creation), `Chat/chat_history.py` and `Chatbooks/chatbook_service.py` (transport creation), `Chat/conversation_enrichment.py:222` and `endpoints/character_messages.py:1097,1233` (metadata/touches), `DB_Management/async_db_wrapper.py:49` and `transaction_utils.py:216,235,273,293` (wrappers), and `chacha/shared_workspace_chat_store.py:161` (shared-recipient thread creation). They gain no trusted-origin keyword automatically: legacy/direct/import/shared creation remains unknown, metadata delegates preserve origin, and caller-supplied dictionaries remain untrusted. Do not add Persona defaults to the shared-recipient or ordinary completion helper paths. Keep unrelated wrapper behavior out of this slice.

The SQLite conversation sync triggers at `ChaChaNotes_DB.py:1472-1522` enumerate fields explicitly. Leave provenance out of those payloads and the existing Sync v2 outgoing payload builder. Add regression assertions for both; do not widen transport schemas or turn an internal column into a Sync field.

## Stage 1: Value Contract And Shared Effective Resolver

**Goal:** Define a single validated local origin value and make management/startup consult identical effective-default rules.
**Success Criteria:** All source/reference relations and byte bounds are enforced; management wire behavior is unchanged; implicit startup cannot treat corrupt storage as unset.
**Tests:** New `tests/Chat/test_assistant_startup.py`; extend existing Workspace default API and creation tests.
**Status:** Complete.

**Interfaces:** Create `AssistantStartup` in the new core value module (Pydantic, frozen, `extra="forbid"`). Add `encode_assistant_startup(value: AssistantStartup) -> str` and `decode_assistant_startup(raw: object) -> AssistantStartup`. Encoding rejects non-model inputs, invalid Unicode and >1024-byte canonical JSON. Decoding NULL/invalid/oversized persisted data returns a fresh unknown value; catch only validation/decoding errors, not arbitrary DB exceptions. Do not log the raw object or validation error text containing input.

```python
class AssistantStartup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    source: Literal[
        "workspace_default", "explicit", "explicit_none",
        "system_fallback", "fork", "unknown",
    ] = "unknown"
    workspace_id: str | None = None
    workspace_version: Annotated[StrictInt, Field(gt=0)] | None = None
```

Add before/after validators: schema version must have `type(value) is int` and value 1 (reject true); ID must be a nonempty string when present; pair/source rules match Global Constraints. Validate serialized byte size on model construction as well as encoding so response-only construction cannot bypass the cap. Use `json.dumps(model.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")`; convert Unicode errors to bounded validation messages. There is no existing global 128-character Workspace-ID contract: do not invent one or truncate IDs. A legacy ID whose complete origin exceeds the byte cap fails before new-chat insertion with a bounded input error; existing Workspace/chat access stays unchanged.

- [x] Add model/parser tests and observe RED. Include exact keys, each valid source, missing/partial/forbidden references, true/string/zero versions, non-string/empty IDs, escaped/control text, invalid Unicode, extra snapshot fields, arrays and >1024-byte JSON. Hypothesis round trips must retain values and prove the encoded cap.

```python
def test_origin_is_reference_only_and_round_trips():
    origin = AssistantStartup(
        source="workspace_default", workspace_id="ws", workspace_version=2,
    )
    encoded = encode_assistant_startup(origin)
    assert decode_assistant_startup(encoded) == origin
    assert set(json.loads(encoded)) == {
        "schema_version", "source", "workspace_id", "workspace_version",
    }
    assert len(encoded.encode("utf-8")) <= 1024

@pytest.mark.parametrize("raw", [None, "broken", "[]", '{"schema_version":true}'])
def test_legacy_or_invalid_storage_has_unknown_origin(raw):
    assert decode_assistant_startup(raw) == AssistantStartup()
```

- [x] Move parsing/profile-cache/effective-state logic from `workspaces.py:174-305` into existing `Workspaces/assistant_defaults.py`, keeping thin adapters where existing callers/tests need them. Expose `resolve_effective_workspace_assistant_default(db, *, workspace: Mapping[str, Any], user_id: str, persona_profile_cache=None, conn=None) -> WorkspaceEffectiveAssistantDefault`. Preserve management-authorized stored references and permission-safe effective output, include-deleted behavior, per-request profile caching, and bounded logging.
- [x] Use `_assistant_defaults_invalid` as well as schema validation. A non-null default plus opt-out true stays unavailable. Consult `core.feature_flags.is_persona_enabled` before profile lookup. Let DB errors propagate to existing HTTP mappers at endpoint boundaries; no generic successful fallback.
- [x] Retain `resolve_new_conversation_assistant(db, *, user_id, request) -> ChatSessionCreate` as a compatibility wrapper for direct callers. Introduce `ResolvedConversationAssistant(request: ChatSessionCreate, startup: AssistantStartup, display_name: str)` and `resolve_workspace_assistant_startup(db, *, user_id: str, request: ChatSessionCreate, conn=None) -> ResolvedConversationAssistant` in that same module. Its selection matrix is Stage 3; the wrapper returns `.request`. Do not persist the display name in provenance.
- [x] Extend resolver tests: malformed raw JSON hidden by DB normalization, inconsistent opt-out pair, inactive/deleted/hidden Persona, disabled configured Persona, unavailable Workspace and DB failure. Startup maps configured disabled to 503 and other unavailable defaults to 409 without identity/name; explicit None bypasses Persona lookup. Management status semantics stay unchanged. Capture logs and assert private markers are absent.
- [x] Run RED then GREEN for the new value tests and both existing Workspace suites. Add docstrings and commit only this stage's files/tests plus the execution task and updated plan: `refactor(persona): share workspace default resolution`.

**Execution evidence (2026-09-14):** `e709ade99d`; 158 passed, five warnings, no skips; 98% core coverage. Production/test Bandit clean, no new Ruff findings. Independent stage review approved spec and quality with no actionable findings. Storage/public activation remains contingent on Stages 2-5.

## Stage 2: Storage And Mutation Protections

**Goal:** Store local provenance without permitting arbitrary DB dictionaries, Sync envelopes or initializer repairs to forge or stale it.
**Success Criteria:** Fresh/upgrade/rollback pass on both backends; every existing identity writer invalidates atomically; no-op/metadata/scope-only writes preserve it.
**Tests:** New `tests/DB_Management/test_conversation_assistant_startup.py`; extend `tests/ChaChaNotesDB/test_conversation_assistant_identity_db.py`, `tests/Sync/test_sync_v2_chat_materializer.py`, and `test_sync_v2_replay_repair.py`. Run existing `tests/ChaChaNotesDB/test_chacha_conversation_store.py` and `test_conversation_scope_db.py` as regressions; their new provenance lifecycle cases belong in the two-backend suite.
**Status:** Complete.

**Interfaces:** Add nullable `conversations.assistant_startup_json TEXT` through the next registered migration. SQLite CHECK uses byte length via `length(CAST(assistant_startup_json AS BLOB))`; PostgreSQL uses `octet_length`, both allow NULL and cap 1024. No default backfill, foreign key, snapshot table, lookup index or source inferred from legacy identity. Update historical/latest schema assertions precisely, never lower a modern DB marker to simulate an old schema.

Extend `add_conversation(conv_data, *, conn=None, assistant_startup: AssistantStartup | None = None)` in store/façade. Stage 3 adds and forwards the Character factory's trusted keyword alongside creation orchestration; that requirement is unchanged, but is not a Stage 2 deliverable. A separate typed argument is a trusted internal boundary, not public authorization. Reject `assistant_startup` and `assistant_startup_json` keys in `conv_data` or ordinary `update_data`, including null; imports must strip those fields at their explicit mapping boundary. New callers with no trusted keyword store NULL/unknown. Do not add a provenance keyword to Sync upsert.

- [x] Create a two-backend fixture using the pattern in `test_workspace_persona_optout_v69.py` (`pg_database_config` via `request.getfixturevalue`, fresh handles, close all connections). Write genuine pre-migration upgrade, fresh schema, oversized direct storage, failed migration rollback, restart and unknown legacy-row tests. Observe RED before adding the migration.
- [x] Register both migrations and update the SQLite post-v67 tail without moving legacy `executescript` helpers after the new transaction. Re-run 2A late-failure/rollback/interleaving tests. Stored history must survive Workspace deletion, hence no Workspace FK on the origin.
- [x] Add typed insertion and ensure the encoded origin is inserted once with identity. Test outer-transaction rollback leaves no conversation or origin. Decode only through the bounded parser on reads; public/raw projection handling is Stage 4.

```python
def test_identity_change_clears_origin_in_the_same_write(db_factory):
    db = db_factory()
    origin = AssistantStartup(source="explicit")
    cid = db.add_conversation(
        {"assistant_kind": "persona", "assistant_id": "persona-a",
         "persona_memory_mode": "read_only", "client_id": db.client_id},
        assistant_startup=origin,
    )
    before = db.get_conversation_by_id(cid)
    db.update_conversation(cid, {"title": "Renamed"}, before["version"])
    renamed = db.get_conversation_by_id(cid)
    assert decode_assistant_startup(renamed["assistant_startup_json"]) == origin
    db.update_conversation(cid, {"persona_memory_mode": "read_write"}, renamed["version"])
    changed = db.get_conversation_by_id(cid)
    assert decode_assistant_startup(changed["assistant_startup_json"]) == AssistantStartup()
```

- [x] In `update_conversation`, lock the PostgreSQL row (`FOR UPDATE`) before reading/normalizing its current identity; SQLite uses the existing outer `BEGIN IMMEDIATE`. Compare both normalized four-tuples using `_normalize_conversation_assistant_identity`. Invalidation is part of the same existing version-checked UPDATE. Preserve empty-update version bumps, metadata/settings/history updates and normalized no-ops; version conflicts and failed writes change neither identity nor origin. Changing back never restores origin.
- [x] Fix Sync replacement atomically, including absent-row races. Inside one transaction attempt INSERT with `ON CONFLICT(id) DO NOTHING`; when it loses, select the existing row with PostgreSQL `FOR UPDATE`, normalize/compare the current binding, then apply the existing replacement fields plus either retained origin or NULL. Do not catch a uniqueness error inside an aborted PostgreSQL transaction and keep using it. If the conflicting row disappears before it can be locked, raise a bounded retryable DB conflict; do not guess a prior origin. Preserve existing `created_at`, root, revision, deleted and scope replacement semantics. No mutation to the Sync wire contract or its separate apply-state transaction.
- [x] Exercise two-connection insertion-versus-Sync and local-update-versus-Sync races on PostgreSQL with events/barriers, bounded joins and fixture-owned databases. Never use timing sleeps as proof. A same-revision Sync replacement must not evade the local comparison. Retry after product commit but before Sync bookkeeping must not recreate origin.
- [x] Keep `ChatConversationMaterializer`'s explicit allowlist. Test both forged keys on new/existing envelopes, complete metadata-only envelopes with unchanged identity, and partial envelopes whose current defaults legitimately change identity. New remote rows remain unknown; no Workspace Sync activation.
- [x] Protect the narrow initializer repair predicates after the provenance column exists. Compare pre/post normalized bindings for affected rows; equivalent Character backfill retains provenance, invalid/different pre-repair binding clears it. Keep pre-69 SQL unchanged and add reopen tests. Do not disable historical repair or relax source-catalog checks.
- [x] Add a stateful property test over title/empty/no-op/identity/memory/scope/tombstone/restore operations: origin remains the initial value until the first actual identity change, then remains unknown. Hard delete removes the row. Scope-only Sync moves preserve historical origin. Test Workspace cascade without redesigning it.
- [x] Run focused store/Sync suites plus live backend cases and commit: `feat(persona): persist lifecycle-safe local startup provenance`. Public activation still requires Stages 3-5 in this same PR.

**Execution evidence (2026-09-14):** `d3dc4fdcfc` plus scope clarification `287df1a116`; 155 passed, five warnings, zero skips, including seven live PostgreSQL races. Production/test Bandit clean; no new Ruff findings. Independent storage review and scoped allocation reconciliation approved. Character factory forwarding remains a Stage 3 requirement.

## Stage 3: Atomic Workspace Creation

**Goal:** Record the actual selection made at creation without changing existing explicit, global, Character or fork semantics.
**Success Criteria:** Every supported new Workspace API choice has honest origin and one identity/provenance write; concurrent default edits cannot mix an old identity with a new Workspace version.
**Tests:** Existing `tests/Workspaces/test_workspace_assistant_creation.py` regressions and new `tests/Workspaces/test_workspace_assistant_provenance.py`; new `tests/DB_Management/test_workspace_assistant_creation_atomic.py` reuses Stage 2's real two-backend fixture for transaction and concurrency cases.
**Status:** Complete.

| Validated request | Source and references | Identity behavior |
| --- | --- | --- |
| Global scope, including existing global sync | unknown/null | Unchanged. |
| Workspace with validated parent lineage | fork/null | No new parent assistant inheritance or copying parent origin. Existing cross-Character behavior remains. |
| Workspace explicit tracked Persona/Character | explicit/null | Explicit normalized choice wins, even when it equals today's default. |
| Workspace untracked with any supplied identity-null field | explicit_none/null | Suppresses inheritance, even when Persona support is disabled. |
| Workspace identity omitted; valid configured default | workspace_default/actual ID and version | Persist saved Persona and memory mode. |
| Workspace identity omitted; unset or cleared default | system_fallback/examined ID and version | Untracked identity; no provisioning. Clear is not a request-level explicit choice. |
| Workspace identity omitted; unavailable configured default | no row | 503 disabled, otherwise 409 with bounded reason; missing/inaccessible Workspace 404. |

Precedence is based on the original request's `model_fields_set` and normalized identity, not a resolved `model_copy` or equality comparison. Existing schema normalization already adds Character fields; a non-null tracked identity wins over a simultaneous nullable compatibility field. Parent classification is allowed only after the existing owner/scope/message lineage checks succeed. Merely supplying a parent string is not authority for `fork`.

- [x] Write HTTP tests for all matrix rows and each explicit-null field. Include saved confirmed read-write versus read-only defaults, explicit memory mode, archive legacy behavior (unchanged), invalid/disabled defaults, parent/global bypass, and explicit Persona equal to the default. Observe missing-origin RED failures.
- [x] Extend DB `get_workspace` and `get_persona_profile` with optional keyword-only `conn=None, for_update=False`. `for_update=True` requires an existing transaction connection. Retain deletion/staged visibility and Persona owner predicates; PostgreSQL adds `FOR UPDATE`, SQLite relies on its outer write transaction. Never perform locking SQL in the core resolver. Management calls retain unlocked defaults.
- [x] Add `create_workspace_persona_conversation(db, *, user_id: str, request: ChatSessionCreate, conversation_data: Mapping[str, Any], title_timestamp: str) -> str` to the existing resolver module. It handles only non-Character Workspace requests after scope, quota and parent validation. Inside `with db.transaction() as conn`, call `resolve_workspace_assistant_startup(..., conn=conn)`, then build the final identity/title and invoke `db.add_conversation(..., conn=conn, assistant_startup=resolved.startup)`. Keep caller-supplied title; derive default Persona title only from the transaction's selected profile. Do not create a second INSERT implementation.

```python
with db.transaction() as conn:
    resolved = resolve_workspace_assistant_startup(
        db, user_id=user_id, request=request, conn=conn,
    )
    payload = dict(conversation_data)
    payload.update(
        assistant_kind=resolved.request.assistant_kind,
        assistant_id=resolved.request.assistant_id,
        character_id=resolved.request.character_id,
        persona_memory_mode=resolved.request.persona_memory_mode,
    )
    payload["title"] = request.title or (
        f"{resolved.display_name} Chat ({title_timestamp})"
        if resolved.request.assistant_kind in {"persona", "character"}
        else f"Chat ({title_timestamp})"
    )
    return db.add_conversation(payload, conn=conn, assistant_startup=resolved.startup)
```

The title step deliberately replaces any preflight-derived title when the caller did not supply one. Validate the supplied internal payload's scope/owner/lineage against the already validated request; never let an arbitrary mapping change those after resolution. Apply the effective resolver to locked Workspace then selected Persona rows in that order. No await, rate-limit service, LLM/network call or schema initializer inside this transaction.

- [x] Move inheritance out of the endpoint's early preflight, but retain the existing Workspace existence/visibility check for every Workspace request, including explicit Characters and Character forks. Use the scoped `db.get_workspace` result and the existing bounded 404 when missing/deleted/staged; `_resolve_chat_scope` checks syntax only and the FK is not a visibility check. Do not consult default Persona availability for explicit choices. Preserve legacy archived-Workspace behavior. The non-Character helper must still reread/lock the Workspace inside its transaction; a preflight read is not its selection authority.

```python
if scope.scope_type == "workspace":
    workspace = db.get_workspace(scope.workspace_id)
    if workspace is None or workspace.get("deleted"):
        raise HTTPException(status_code=404, detail="Workspace not found")
```

- [x] Keep asynchronous rate/quota checks and parent validation before the synchronous helper. Explicit Character creation uses the existing factory and receives a trusted `explicit` or `fork` value via its new keyword; no extra outer transaction around WorldBook preflight. Explicit global/Sync calls omit the keyword. Preserve existing accepted Character options on their existing branch. Add HTTP cases for missing/deleted/staged Workspaces with explicit Character and Character-fork payloads: 404 with no conversation/settings/messages inserted. Add the available archived-Workspace case to prove its legacy behavior is unchanged.
- [x] Keep trusted provenance out of arbitrary `conversation_data`; pass it separately into the Character factory's existing `db.add_conversation` call. Other factory callers/importers receive default unknown. Regression tests must preserve greetings, participants, preset/provider settings, behavior snapshot, root and fork-message lineage.
- [x] Reject public `assistant_startup` and internal `assistant_startup_json` at every create/update request model, including null, with targeted before-validators. Preserve unrelated legacy extra handling. Do not add/accept strict 2C selector fields here. Add no-write 422 tests; reserve a bounded mapped InputError for an internally unrepresentable origin, without echoing the offending ID.
- [x] Add two-backend failure injection after selection but before insertion/commit and prove there is no partial row. On PostgreSQL use two connections to race default clear/rebind and Persona deactivate with inherited create: if mutation commits first use its new state; if create holds locks first, persist the earlier consistent identity/version and let the mutation affect later creates. No stale-version conflict or retry-safety promise for this legacy route.
- [x] Run creation, store and existing Character factory regressions; commit: `feat(persona): record workspace assistant origin at creation`.

**Execution evidence (2026-09-14):** `f3d9697c78`; 363 passed across three disjoint focused gates, zero skips, including six new PostgreSQL creation/mutation races. Production/test Bandit clean; no new Ruff findings. Independent spec and quality review approved. Existing PostgreSQL Character WorldBook preflight limitation was reproduced at the base; no general PostgreSQL Character startup claim is made.

## Stage 4: Safe Reads And Transport Boundaries

**Goal:** Expose identical bounded provenance across owned read surfaces without leaking inaccessible origin references or allowing imported authority.
**Success Criteria:** Detail/list/tree/update/restore metadata agrees; origin access loss yields whole-object unknown while retaining storage; every import starts unknown; outgoing unsupported formats omit provenance.
**Tests:** Existing chat/conversation tests plus Character facade/export and Chatbook import/export tests listed below.
**Status:** Complete.

**Interfaces:** Add `assistant_startup: AssistantStartup = Field(default_factory=AssistantStartup, json_schema_extra={"readOnly": True})` to `ChatSessionListItem`, `ConversationListItem`, and `ConversationMetadata` (detail/tree models inherit them where applicable). Create `project_assistant_startup(db, *, raw: object, user_id: str, workspace_visibility_cache: dict[str, bool] | None = None) -> AssistantStartup` in `Workspaces/assistant_defaults.py`. It decodes before looking up anything, returns non-reference values directly, and resolves origin visibility through the authenticated DB's existing `get_workspace` boundary. Missing/deleted/hidden staged origin returns `AssistantStartup()`; visible archived origin remains history, not a new creation decision. DB errors propagate, not converted into successful unknown. The cache is request-local, bounded by returned distinct origins, and never shared between users or calls. Do not authorize an origin merely because the current conversation is readable.

| Surface | Existing mapping and exact disposition |
| --- | --- |
| `character_chat_sessions.py:1771,1810,1842` | Shared fields, detail and list builders. Supply only a visibility-checked public object, not raw column or parsed-but-unchecked value. Restore paths at 7905/7926 use the same projector. |
| `chat.py:6285,6944,7055,7183,7380` | Identity helper plus conversation list/detail/update/tree builders. Extend shared projection; origin authorization is separate from existing conversation scope checks. |
| `chat_session_schemas.py:268`, `chat_conversation_schemas.py:26,114` | Response models carry the same value. Provenance is not roleplay resume eligibility or behavior-snapshot authority. |
| `Character_Chat/modules/character_chat.py:929,966,1071` | Public library list/metadata/search currently return raw rows. Remove only `assistant_startup_json` from a copied result and add the bounded visibility-checked public object using the scoped DB owner; preserve existing metadata keys and caller isolation. Do not mutate DB row objects in place or add a broad new export of every field. |
| `character_chat_sessions.py:8088`; `Chat/chat_history.py:461` | JSON/history exports use explicit fields. Keep provenance out of unsupported formats and test bytes. |
| `Chatbooks/chatbook_service.py:6228,6660` | Chatbook conversation collector/importer use explicit maps. Export omits both startup keys; import remains unknown, including archive restore. |
| `Chatbooks/chatbook_service.py:4323,4537,4737` | OpenWebUI source metadata/settings and JSON/DB imports. Preserve existing source metadata as data; never promote a nested startup value to authority. New conversation remains unknown. |
| `Character_Chat/modules/character_io.py:1276` | Legacy import calls the existing creator with explicit arguments. Forged fields are ignored/stripped before the trusted creator; no origin keyword. |
| `Sharing/clone_models.py:126` | Shared Workspace snapshots contain no conversations. No new copying behavior or shared-recipient adoption. |

- [x] Add a projection matrix extending `tests/Chat/unit/test_chat_conversations_api.py` and the new `tests/Workspaces/test_workspace_assistant_provenance.py`: visible origin, moved conversation, old origin deleted/hidden, archived origin, legacy/corrupt JSON, and DB failure. Cover chat detail/list/restore and conversation detail/list/update/tree. Observe missing/unsafe projection RED before wiring builders.

```python
def test_projection_redacts_hidden_origin_without_mutating_storage(db, monkeypatch):
    origin = AssistantStartup(
        source="workspace_default", workspace_id="private-origin", workspace_version=2,
    )
    raw = encode_assistant_startup(origin)
    monkeypatch.setattr(db, "get_workspace", lambda *args, **kwargs: None)
    projected = project_assistant_startup(db, raw=raw, user_id=str(db.client_id))
    assert projected.model_dump() == {
        "schema_version": 1, "source": "unknown",
        "workspace_id": None, "workspace_version": None,
    }
    assert decode_assistant_startup(raw) == origin
```

The HTTP version must seed a real conversation then assert its stored origin is byte-for-byte unchanged after all read calls. Include `system_fallback` with inaccessible origin; do not leave the source set while redacting required references. Repeated list origins should use at most one visibility lookup per origin per request. A subsequent request must observe access loss even after a cached earlier response.

- [x] Thread the authenticated DB/user and request-local visibility cache through existing response construction or a shared pre-projection helper. Never let the default unknown value mask a builder accidentally forgetting the actual stored provenance. Assert an available non-unknown fixture across every builder, not just schema defaults.
- [x] Extend `tests/Character_Chat_NEW/unit/test_chat_settings_merge.py` conversion tests and `tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py` list-versus-resume/restore tests. Startup source must not change snapshot status, resume eligibility, settings/history versions, or greeting behavior. Extend `tests/Characters/test_character_chat_lib.py` for raw-column removal and unchanged preexisting metadata.
- [x] Extend `tests/Chatbooks/test_chatbook_service.py` collector/import tests, `tests/Chatbooks/test_openwebui_import_service.py` JSON and DB fixtures, and `tests/Character_Chat_NEW/integration/test_character_api.py` export tests. Seed a unique private-origin marker and both forged keys. Assert outgoing bytes lack the internal/public keys and reference marker, and imported DB rows decode to unknown. Metadata-only fake dictionaries are insufficient for final acceptance: exercise actual collector/import pipelines and existing factory calls.
- [x] Check legacy history import/export and Sync outgoing payloads with real stored origin fixtures. If their existing allowlists already pass, change tests only. Incoming provenance nested in source settings may remain uninterpreted source metadata; it must not affect selection, authoritative output, or stored origin. Cross-server verified transport remains deferred.
- [x] Run focused API, facade, Character and transport suites; commit: `feat(persona): expose privacy-safe local startup metadata`.

**Execution evidence (2026-09-14):** `6a8da29037107da8f901c7095841b9a70712a8ec`; final core/facade/schema gate 479 passed, 25 warnings, zero failures/skips (`/tmp/persona-stage4-final-core-facade.log`); focused facade/transport gate 123 passed, 20 warnings (`/tmp/persona-stage4-facade-transport.log`). These selections overlap and are not additive. Stage 4 spec and quality independently approved. The parent closed the remaining no-change snapshot-shape audit: `Sharing/clone_models.py:126` contains Workspace metadata/memberships/sources/notes/artifacts, no conversations; `clone_models.py` and `clone_service.py` are unchanged from execution base. Exact RED/GREEN, seven real transport boundaries, security comparisons and warning limits are in the Stage 4 report; Stage 5 below owns the broader final gates.

## Stage 5: Activation Review And Operational Verification

**Goal:** Prove the complete local provenance slice is safe before opening it for runtime review/merge.
**Success Criteria:** Every mutation/projection/transport inventory row has a test or a verified no-change disposition; live migrations/concurrency pass; maintenance instructions and parent tracker reflect only 2B completion.
**Tests:** Full focused matrix below, migration and real-connection concurrency; no live providers or browser work.
**Status:** Complete. Runtime, documentation, final acceptance/security checks, independent review and task-container cleanup are complete. [Draft PR #2963](https://github.com/rmusser01/tldw_server/pull/2963) delivers this one activation unit on planning PR #2961; TASK-13245.4 is complete. Human-written Change summary, stack merge and offline deployment remain separate gates. TASK-13245/#2950 and later parity slices remain open.

- [x] Run the following from the isolated worktree, after activating `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`. The final run includes all 12 original paths plus six newly relevant paths; no empty selections or collection errors count as verification.

```bash
python -m pytest \
  tldw_Server_API/tests/Chat/test_assistant_startup.py \
  tldw_Server_API/tests/DB_Management/test_conversation_assistant_startup.py \
  tldw_Server_API/tests/DB_Management/test_workspace_persona_optout_v69.py \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_creation.py \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_defaults_api.py \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_provenance.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_conversation_assistant_identity_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_conversation_scope_db.py \
  tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py \
  tldw_Server_API/tests/DB_Management/test_osce_migration_v67.py \
  tldw_Server_API/tests/DB_Management/test_workspace_assistant_creation_atomic.py \
  tldw_Server_API/tests/Workspaces/test_assistant_startup_projection.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py \
  tldw_Server_API/tests/API_Deps/test_chacha_startup_owner.py \
  -n 4 --benchmark-disable --basetemp="${TMPDIR%/}/persona-stage5-core-postfix" \
  -q -rs --tb=short
```

**Final core result:** 697 passed, 26 warnings, zero failures/skips in 203.96s on runtime `4f1af51224`; `/tmp/persona-stage5-core-postfix.log`. Parent used `env -u JOBS_DB_URL`, the task's `POSTGRES_TEST_DSN` on `127.0.0.1:55461/postgres`, `TLDW_TEST_PG_CONTAINER_NAME=tldw_persona_13245_4_postgres`, `TLDW_TEST_NO_DOCKER=1`, and `TLDW_TEST_POSTGRES_REQUIRED=1`. The official fixture creates/drops per-test databases; server version is PostgreSQL 18.6 (Debian 18.6-1.pgdg13+2). `--benchmark-disable` disables unused benchmark instrumentation, not ordinary tests. Basetemp uses actual macOS TMPDIR; application path policy is unchanged.

- [x] Run the Character/library/import/export suite separately so failures are attributable. Preserve existing fixtures and mock providers; do not fabricate successful runtime tests with snapshot-loader patches applied only in memory.

```bash
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_behavior_snapshot_api.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py \
  tldw_Server_API/tests/Characters/test_character_chat_lib.py \
  tldw_Server_API/tests/Chatbooks/test_chatbook_service.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py \
  -n 4 --benchmark-disable --basetemp="${TMPDIR%/}/persona-stage5-character-transport-owner-final" \
  -q -rs --tb=short
```

**Final Character result:** 394 passed, three pre-existing skips, 25 warnings, zero failures in 463.54s, exit 0; `/tmp/persona-stage5-character-transport-owner-final.log`. The same three baseline skips are `test_character_api.py:854` (Resource Governor limits), `:761` (V3 format fixture), and `:637` (removed streaming route). The two disjoint final gates total **1,091 passed, three skips, zero failures**; do not add overlapping intermediate-stage runs. All test processes exited; the task-specific PostgreSQL container and temporary volume were removed.

- [x] Run migration/concurrency tests using the official live PostgreSQL fixture with `TLDW_TEST_POSTGRES_REQUIRED=1`. Supply fixture configuration through its documented environment variables; `JOBS_DB_URL` takes precedence over `POSTGRES_TEST_DSN`, so unset it for a dedicated Persona test DSN. Use a task-specific container name/available port if fixture-managed Docker is needed; never remove another agent's/shared container. Record server version, exact command, failures/skips and counts (final core gate above).
- [x] Removed only `tldw_persona_13245_4_postgres` and its temporary volume after all test/review processes exited; shared containers and unrelated temporary folders were untouched.
- [x] Cover both PostgreSQL blocking orders for create/default changes and local/Sync writes, genuine v68-to-v69 upgrade, failed migration rollback, reopen/repair, cached old-writer hazard and ordinary global regression. Final core gate includes the seven storage races and six creation races. Historical migration assertions retain their exact stage; the full ordinary Character regression passed separately above, not a PostgreSQL Character startup certification.
- [x] Audit every current named writer again with `rg -n '\b(add_conversation|update_conversation|upsert_conversation_from_sync)\(' tldw_Server_API/app`. Also inspect `UPDATE conversations`/`INSERT INTO conversations` bypasses and response row spreads. Parent's final audit found only the planned Workspace helper added; prior dispositions stand. Historical pre-v69 repair stays unchanged, modern repair uses protected comparisons, message/settings/Buddy lock-only writes preserve binding, fixture INSERTs default NULL, and erasure deletes the row. No unrelated Buddy/animation edits.
- [x] Run Ruff on the exact changed Python files, compile them, run `git diff --check`, and run Bandit on all changed production files plus a separate test scan excluding only B101 assertions. Parent completed the 29-file scope and base comparison below; no new security suppression.
- [x] Update `Docs/Code_Documentation/Workspace_Persona_Defaults.md` and the parent plan with the real schema version, creation/read contract, unsupported transports, mutation semantics and the continued offline-upgrade requirement. Record refreshed parent-fetched server/Chatbook dev SHAs and scoped differences. Keep 2C/2D, send-time admission, profile/provisioning, Research RAG and full parity open.
- [x] Independent whole-branch spec/quality review checked import authority, PostgreSQL races, initializer repair and hidden-origin reads. Its one P2 finding was fixed in `4f1af51224` and independently re-reviewed with no remaining findings. Stage 5 documentation and subsequent fixture alignment also passed scoped review.
- [x] Prepare and verify documentation and execution-task evidence after the final Character result, included in this scoped commit: `docs(persona): document local startup provenance guarantees`. Still-pending parent gates are explicit; publication is not part of this documentation commit.
- [x] Opened [draft PR #2963](https://github.com/rmusser01/tldw_server/pull/2963) on `codex/persona-startup-provenance-plan` after rechecking #2961. Links TASK-13245 and #2950 without closing either. Human-written Change summary remains pending; no strict-route or legacy replay-safety claim.

### Final Static And Security Evidence

The final 34 changed Python files were scanned as 33 post-fix paths plus the final shared fixture: 13 production and 21 test files. Compilation and whitespace checks passed. Ruff found 60 issues versus 66 at execution base, with no new `(file, code, message)` multiset entries: 42/48 in `/tmp/persona-stage5-postfix-{ruff,base-ruff}.json` and 18/18 in `/tmp/persona-fixture-owner-{ruff,base-ruff}.json`.

Production Bandit: zero findings/errors. Test Bandit, excluding only B101 assertions: eight findings, zero scan errors, all findings reproduced at base (nine findings there). Seven are existing B105 fixture literals; one is the existing B311 library test random-data generator. Artifacts: `/tmp/persona-stage5-postfix-{bandit-production,bandit-tests,base-bandit-production,base-bandit-tests}.json`; the additional fixture scan and baseline in `/tmp/persona-fixture-owner-{bandit,base-bandit}.json` both have zero findings/errors. Base files/config were extracted with `git archive` into task-owned temporary storage; no checkout/rebase or new suppressions.

### Final Review Correction

The per-user cache retains caller attribution such as `voice_assistant` or worker aliases. Whole-branch review reproduced legitimate REST startup/read rejection when that alias was incorrectly equated with ownership. Construction now accepts a separate trusted, read-only `owner_user_id`, supplied by the existing authenticated dependency; standalone callers default to their initial client ID. Creation, projection and library reads use the canonical owner without rewriting client attribution or weakening wrong-owner checks. Nineteen real-cache/SQLite cases cover alias-first REST use, isolation, denial, reopen and initialization waiters. The combined dependency/voice-auth/seed-fixture gate passed 68 tests, zero skips; `/tmp/persona-final-fix-targeted.log`.

The first post-fix Character run had one failure (393 passes, three existing skips): its shared fixture constructed owner `test_client` and later changed only the client label to authenticated user `1`. The fixture now supplies that existing owner at construction. The unchanged snapshot visibility/resume/restore/greeting assertions plus the 19 owner tests passed 20 tests; `/tmp/persona-fixture-owner-green.log`. Full affected acceptance then passed as recorded above. No production change followed `4f1af51224`. The earlier 1,072-pass run is pre-review evidence, not the final corrected acceptance count.

**Limits:** The unchanged PostgreSQL Character WorldBook preflight failure was reproduced at the Stage 3 base before the factory transaction/INSERT; it is disclosed in the runbook, not fixed or counted as PostgreSQL Character startup coverage. No live-provider, Chatbook-runtime, browser, repository-wide or full-parity certification. TASK-13245/#2950 stay open, and the strict 2C route/receipts/replay/send-time contract, 2D and later stages remain deferred.

## Historical Plan Validation Record

The following is the original 2026-09-13 TASK-13245.3 planning record, not the current TASK-13245.4 execution status. Its documentation-only/Bandit-not-applicable language and unperformed gates refer only to that planning commit. Current execution evidence and pending gates are above; the 71-pass baseline must not be substituted for them.

This task changes documentation and Backlog only. Fresh baseline: 71 tests passed, six warnings, zero failures. Two read-only inventories checked DB mutation/Sync paths and read/transport paths. The plan incorporates their findings: independent Sync replacement and absent-row races, initialization repair bypasses, shared detail/list builders, raw library row exposure, unsupported import/export transport and existing Character factory preflight.

Self-review maps approved 2B requirements to Stages 1-4 and cross-cutting acceptance to Stage 5. It deliberately keeps receipts/current-send admission in 2C, preserves custom-prompt and fallback differences from Chatbook, and keeps legacy/global/direct imports unknown rather than inventing historical origin. The size-cap compatibility case is explicit: reject a new unrepresentable origin before insertion, never truncate or silently relabel it. No production code is changed, so Bandit is not applicable to this planning commit; it is mandatory for execution.

Not yet performed by this planning task: provenance RED/GREEN tests, new live PostgreSQL concurrency/migration tests, full Character/import/export suite, Chatbook runtime testing, browser UAT or repository-wide tests. Stage 2A PostgreSQL evidence is not substituted for those future gates.

Independent plan review found one P2 orchestration omission: moving the early resolver could remove Workspace visibility validation from the Character/fork branch, whose factory is outside the new Persona helper. Verified against the current resolver and `get_workspace` staged/deleted filtering. Stage 3 now explicitly retains that preflight plus no-write HTTP regressions, while keeping transaction-time revalidation for non-Character selection. No runtime regression was introduced because this task changes no runtime code. The reviewer reported no other actionable findings.
