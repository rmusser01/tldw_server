# Workspace Persona Strict Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver Stage 2C strict versioned Workspace startup with atomic bounded retry receipts, lifecycle protection and current Persona generation admission.

**Architecture:** A closed FastAPI request calls a small Workspace orchestrator backed by one operation-specific ChaCha receipt store. Share current resolver and conversation insertion; install transaction-local mutation hooks and one shared Persona admission helper before enabling the route. Preserve native fork semantics and all legacy creation behavior.

**Tech Stack:** Python/Pydantic/FastAPI, ChaCha SQLite/PostgreSQL backend abstraction, pytest and official per-test PostgreSQL fixtures, existing OpenAPI exporter and CI shard tooling.

**Spec:** [Current-dev refresh](../../Design/2026-09-27-persona-workspace-strict-startup-refresh.md) and [reviewed original contract](../../Design/2026-09-13-persona-workspace-choice-provenance-design.md).

**Tracking:** Planning TASK-13245.6; parent TASK-13245; issue #2950. Create/search a separate execution task before runtime edits. This document has not activated Stage 2C or completed Stage 2D.

## Global Constraints

- Work only in the isolated checkout; preserve shared dirty checkout, containers, recovery refs and stash backups.
- Recheck server dev, migration registry and all affected writers at execution start. Planning base `9668e1454b0b28b7a4de13e1a35496fa0b368c42`; SQLite 72/PostgreSQL 76. Candidate 73/77 is not a reservation.
- Chatbook ref `74965f694f435c590366ec66d671d3506d440ebc` was checked, not audited. No frontend adoption/parity certification in this backend slice.
- No new dependencies, generic receipt framework, native fork operation kind or duplicate conversation INSERT. All SQL remains in DB_Management.
- Preserve legacy omission/null, explicit Character/Persona, parent/global behavior, Workspace Sync exclusion and server-owned provenance import stripping.
- Required header 1-128 ASCII `[A-Za-z0-9][A-Za-z0-9._:-]*`; reject duplicates. Closed body, no query options; `inherit` requires positive strict whole-Workspace version, `none` forbids supplied version.
- Store only bounded receipt hashes/references/timestamps; no raw key/request/response/profile/content snapshots. Unique `owner_user_id`/key across Workspaces. Forced PostgreSQL owner-only RLS must retain tombstone visibility independently of live parents. Lifetime count defaults to 10000, includes every tombstone; no TTL/recycling.
- First acceptance 201; replay 200 with `Idempotency-Replayed: true`. No silent downgrade/fallback. Deleted target 410; actual binding/scope change permanently invalidates.
- Acquire owner admission before Workspace/Persona; reread receipts after blocking acquisitions. Replay uses Workspace FOR NO KEY UPDATE, then conversation before receipt; never FOR UPDATE against Sync's implicit FK KEY SHARE. Receipt-bound restore/Sync resurrection locks destination Workspace before conversation. Writer hooks must not introduce an inverse lock order.
- Native admission closure/system-operation state blocks strict startup; archived Workspace blocks new startup only. Existing native identity/Sync/projection guards remain intact.
- No network/LLM/credential calls in creation transaction. Persona generation admission happens before side effects; do not hold DB locks across generation.
- Offline drain/migrate/restart only, no mixed writers or old-binary rollback. Activate route only with lifecycle/capacity/current-admission safeguards.
- Every increment follows RED/GREEN, scoped Ruff/compile/Bandit, independent review and hooks-enabled commit. Use official SQLite/live-PG fixtures; do not suppress failures or add unrelated CI/MCP/Chunking/Buddy work.
- Resolve the separately tracked native Workspace cascade retry transaction leak (TASK-13245.7) before lifecycle qualification. Preserve the outermost/caller-transaction guard; never hide the leak with a test-only commit. Recheck latest dev to avoid duplicating an upstream repair.
- Required current-head hosted checks, normal up-to-date merge and requester-owned human Change summary remain gates. Historical green heads do not count.

## File Map And Interfaces

All paths in this section are relative to `tldw_Server_API/` unless stated otherwise. Proposed names below are implementation contracts, not existing functions.

| File | Responsibility |
| --- | --- |
| `app/api/v1/schemas/workspace_chat_startup_schemas.py` (new) | Closed request, selection/version validation, pure canonical fingerprint. |
| `app/core/DB_Management/chacha/workspace_chat_startup_schema.py` (new) | Backend-specific receipt table/index DDL called only by registered migrations. |
| `app/core/DB_Management/chacha/workspace_chat_startup_store.py` (new) | Owner serialization, receipt/conversation consistent reads, count/insert, mutation invalidation/hard-delete tombstones, bounded domain errors. |
| `app/core/DB_Management/ChaChaNotes_DB.py` | Migration registry, initialization of `workspace_chat_startups` store; existing Workspace cascade compatibility. |
| `app/core/DB_Management/backends/pg_rls_policies.py` | Guarded new receipt RLS builder and current policy-set integration, migration-safe when table absent. |
| `app/core/Workspaces/assistant_defaults.py` | Connection-taking shared insertion helper preserving existing resolver/title behavior. |
| `app/core/Workspaces/chat_startup.py` (new) | Strict transaction and replay orchestration, no SQL. |
| `app/core/DB_Management/chacha/conversation_store.py` | Hooks in actual identity/scope/delete writers. |
| `app/core/Persona/conversation_admission.py` (new) | Current Persona feature/owner/activity admission, no prompt reconstruction. |
| `app/api/v1/endpoints/chat.py`, `app/core/Chat/chat_service.py`, `app/api/v1/endpoints/character_chat_sessions.py` | Ordinary admission before routing/credentials plus service recheck; session admission and static strict route. |
| `app/api/v1/schemas/chat_session_schemas.py` | Reject strict-only fields on legacy requests; keep unrelated extra-ignore semantics. |

Define these small interfaces in their owning files and use them consistently:

```python
# workspace_chat_startup_schemas.py
import hashlib
import json
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import _validate_conversation_state

class WorkspaceChatStartupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_type: Literal["workspace"]
    workspace_id: str = Field(min_length=1)
    workspace_assistant_selection: Literal["inherit", "none"]
    workspace_assistant_default_version: StrictInt | None = None
    title: str | None = None
    state: str | None = None
    topic_label: str | None = None
    cluster_id: str | None = None
    source: str | None = None
    external_ref: str | None = None

    @field_validator("workspace_id")
    @classmethod
    def validate_workspace_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workspace id is required")
        return value.strip()

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: str | None) -> str | None:
        return _validate_conversation_state(value)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        version = self.workspace_assistant_default_version
        if self.workspace_assistant_selection == "inherit":
            if version is None or version < 1:
                raise ValueError("Inherit requires a positive Workspace version")
        elif "workspace_assistant_default_version" in self.model_fields_set:
            raise ValueError("None does not accept a Workspace version")
        return self

def startup_request_fingerprint(request: WorkspaceChatStartupRequest) -> str:
    data = {"schema_version": 1,
            "body": request.model_dump(mode="json", exclude_unset=True)}
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()

# workspace_chat_startup_store.py
@dataclass(frozen=True)
class WorkspaceStartupResult:
    conversation: Mapping[str, Any]
    replayed: bool

class WorkspaceStartupError(InputError):
    def __init__(self, code: str, status_code: int, reason: str | None = None):
        super().__init__(code)
        self.code = code
        self.status_code = status_code
        self.reason = reason
```

The second module imports `dataclass`, `Mapping`, `Any` and existing `InputError`. Exact store signatures, installed on `db.workspace_chat_startups`, are:

- `lock_owner(owner_id: str, *, conn: Any) -> None`.
- `get_receipt(owner_id: str, key_digest: str, *, conn: Any, for_update: bool = False) -> dict[str, Any] | None`.
- `lock_conversation(conversation_id: str, *, conn: Any) -> dict[str, Any] | None`.
- `lock_receipt_workspace(workspace_id: str, *, conn: Any, for_create: bool = False) -> dict[str, Any] | None`: current authenticated owner/access predicates and Workspace representation including lifecycle fields, PostgreSQL FOR NO KEY UPDATE by default and FOR UPDATE for fresh creation; caller classifies deleted/closure/archive. Existing getter hides system-operation rows and must not substitute for this owned strict lookup.
- `has_receipt(conversation_id: str, *, conn: Any) -> bool`: nonlocking owner-scoped association query for mutation/restore preflight; associations survive invalidation.
- `count_receipts(owner_id: str, *, conn: Any) -> int`.
- `count_live_chats(owner_id: str, workspace_id: str, *, conn: Any) -> int`.
- `insert_receipt(owner_id: str, key_digest: str, request_fingerprint: str, binding_digest: str, workspace_id: str, conversation_id: str, *, conn: Any) -> None`.
- `invalidate_changed_binding(conversation_id: str, before: Mapping[str, Any], after: Mapping[str, Any], *, conn: Any) -> None`.
- `mark_hard_deleted(conversation_id: str, *, conn: Any) -> None`.

Other exact interfaces:

- Store module `startup_binding_digest(db: CharactersRAGDB, conversation: Mapping[str, Any]) -> str`: canonicalize identity through the current conversation normalization rules and `decode_assistant_startup`; hash only kind/id/character/memory/provenance/scope. Stage 3 hooks pass complete before/after rows.
- Defaults module `insert_resolved_workspace_conversation(db: CharactersRAGDB, *, resolved: ResolvedConversationAssistant, conversation_data: Mapping[str, Any], title_timestamp: str, conn: Any) -> str`: extracted validated insertion body; guards remain with callers.
- Orchestrator `start_workspace_chat(db: CharactersRAGDB, *, owner_id: str, request: WorkspaceChatStartupRequest, idempotency_key: str, receipt_limit: int, chat_limit: int | None, title_timestamp: str) -> WorkspaceStartupResult`: Stage 2 transaction algorithm plus Stage 4 admission.
- Admission `require_current_persona(db: CharactersRAGDB, *, owner_id: str, conversation: Mapping[str, Any], conn: Any = None) -> dict[str, Any] | None`: None only for non-Persona; failed Persona raises a bounded mapped error. Stage 4 defines checks/order.
- Policy builder `build_workspace_chat_startup_rls_sql() -> list[str]` in existing `pg_rls_policies.py`: explicit guarded table statements, ENABLE/FORCE and owner_user_id USING/WITH CHECK. `build_chacha_rls_sql` includes it; new migration applies it after table creation.

Use existing metadata normalization/limits. Domain errors map at the endpoint to a detail dictionary with `code` and optional `reason`; no new public receipt model. For every code step below, first run its named RED test, then implement the specified interface and run GREEN. The route must remain absent until Stage 4 safety tests pass.

The fixed binding field set is implemented using the current store, not a parallel normalizer:

```python
def startup_binding_digest(db: CharactersRAGDB, conversation: Mapping[str, Any]) -> str:
    kind, assistant_id, character_id, memory = (
        db.conversation_store._normalize_conversation_assistant_identity(
            character_id=conversation.get("character_id"),
            assistant_kind=conversation.get("assistant_kind"),
            assistant_id=conversation.get("assistant_id"),
            persona_memory_mode=conversation.get("persona_memory_mode"),
        )
    )
    scope_type, workspace_id = db.conversation_store._normalize_scope(
        conversation.get("scope_type"), conversation.get("workspace_id"))
    data = {
        "schema_version": 1, "assistant_kind": kind, "assistant_id": assistant_id,
        "character_id": character_id, "persona_memory_mode": memory,
        "assistant_startup": decode_assistant_startup(
            conversation.get("assistant_startup_json")).model_dump(mode="json"),
        "scope_type": scope_type, "workspace_id": workspace_id,
    }
    return hashlib.sha256(json.dumps(
        data, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("ascii")).hexdigest()
```

Import `decode_assistant_startup` from `app/core/Chat/assistant_startup.py`. On replay, malformed stored identity/scope is a bounded changed-binding error, not a successful fallback. Schema/request test files import the new request/hash; receipt/lifecycle tests import new store/result/hash and `start_workspace_chat`; admission tests import `require_current_persona`. Wherever the snippets use `creation_db`, re-export the current fixture via `from tldw_Server_API.tests.DB_Management.test_workspace_assistant_creation_atomic import creation_db as creation_db` and its required `db_factory` fixture from `test_conversation_assistant_startup`. Preserve real backends and feature-flag setup.

## Stage 1: Closed Request And Inactive Receipt Storage

**Goal:** Establish independently testable syntax, hashes and registered storage without an active route.
**Success Criteria:** Strict model rejects ambiguous input; real SQLite/PG upgrades create bounded indexed storage; no receipt is exported; legacy runtime unchanged.
**Tests:** New `tests/DB_Management/test_workspace_chat_startup_migration.py`, `test_workspace_chat_startup_receipts.py`; schema cases in new `tests/Workspaces/test_workspace_assistant_startup.py`.
**Status:** Not Started.

- [ ] Search/create execution Backlog task and read both linked specs. Fetch/reinspect latest dev and current migrations/native writers. Record actual new migration numbers in that task before editing.
- [ ] Add parametrized schema RED cases: missing selection/scope/id, global scope, inherit missing/null/zero/negative/bool/string version, none supplied version including null, identity/provider/fork/default-valued extras. Add valid None/inherit and metadata-presence hash tests.

```python
@pytest.mark.parametrize("version", [None, 0, -1, True, "2"])
def test_inherit_rejects_nonpositive_or_noninteger_version(version):
    with pytest.raises(ValidationError):
        WorkspaceChatStartupRequest.model_validate({
            "scope_type": "workspace", "workspace_id": "ws",
            "workspace_assistant_selection": "inherit",
            "workspace_assistant_default_version": version,
        })
```

- [ ] Run `python -m pytest -q tldw_Server_API/tests/Workspaces/test_workspace_assistant_startup.py`; require the intended missing/new-contract failure, not an unrelated collection failure. Implement the model and fingerprint, preserving supplied fields; add before/after validators for version intent and existing state/metadata limits.
- [ ] Add property coverage for canonical request ordering using existing Hypothesis (import `given` and `strategies as st`), alongside explicit omission/null cases:

```python
@given(st.text(max_size=128))
def test_fingerprint_ignores_input_dictionary_order(title):
    payload = {"scope_type": "workspace", "workspace_id": "ws",
               "workspace_assistant_selection": "none", "title": title}
    reordered = dict(reversed(list(payload.items())))
    assert startup_request_fingerprint(WorkspaceChatStartupRequest.model_validate(payload)) == (
        startup_request_fingerprint(WorkspaceChatStartupRequest.model_validate(reordered)))
```
- [ ] Add migration RED tests using actual historical SQLite/PG schema fixtures: upgrade, failed migration rollback, reopen/idempotent application, unique owner/key, non-null hashes, conversation deletion sets reference null, Workspace deletion keeps receipt. Reject raw snapshot columns and any `ON DELETE CASCADE` from Workspace. Add new `tests/DB_Management/test_workspace_chat_startup_rls.py`: official `pg_temp_db` plus unique per-test NOSUPERUSER NOBYPASSRLS role, grants and cleanup; two tenant scopes and absent scope reject cross-owner SELECT/INSERT/UPDATE/DELETE/count/replay. Exercise orphan tombstones after parent deletion. Admin-only fixture passes are not RLS evidence.
- [ ] Create DB-only DDL/store and currently next migration on both backends, using recognized tenant column `owner_user_id`. Add/apply guarded `build_workspace_chat_startup_rls_sql` through current policy-set and migration; ensure older migration calls tolerate missing table. Both USING/WITH CHECK compare to `current_setting('app.current_user_id', true)`, with no live-parent EXISTS. Run `tests/CI/test_rls_coverage_ratchet.py`, `tests/DB_Management/test_pg_rls_policies_contract.py` and new live non-bypass suite; do not baseline-exempt receipts. Add `db.workspace_chat_startups`; parameterized helpers take explicit connection, not nested commits or unlocked conversation getter. Hash keys using SHA-256; PostgreSQL owner lock namespace is stable `workspace_chat_startup_v1`, not Python's randomized hash.
- [ ] Exercise two distinct owner/key namespaces, all-tombstone counts and owner lock with actual backend handles. Adapt three existing patterns: native receipt uniqueness, `organization_sync_store` advisory locking and ordinary conversation transaction helpers. Do not import native leases/canonical request storage.
- [ ] Run new schema/migration/store suites with official required PG fixture; scoped Ruff, compile and Bandit. Review DDL/privacy and commit inactive foundation with the execution task. No strict route yet.

## Stage 2: Atomic Selection, Acceptance And Replay

**Goal:** Implement transaction orchestration with duplicate recovery and finite lifetime capacity, still without public activation.
**Success Criteria:** One committed chat/receipt per owner/key; correct version/default/replay outcomes; no partial records; identical replay works at cap/quota; no blocking-lock stale duplicate decisions.
**Tests:** Extend receipt tests; new strict creation cases in `tests/DB_Management/test_workspace_assistant_creation_atomic.py` and `tests/Workspaces/test_workspace_assistant_startup.py`.
**Status:** Not Started.

- [ ] Reuse the existing parametrized `creation_db` fixture from `test_workspace_assistant_creation_atomic.py` (backed by `db_factory` from `test_conversation_assistant_startup.py`). Add this RED case plus stale/default-change, cross-Workspace same-key conflict and requested-metadata conflict cases:

```python
def test_accepted_replay_at_capacity_keeps_original_chat(creation_db):
    request = WorkspaceChatStartupRequest.model_validate({
        "scope_type": "workspace", "workspace_id": "ws",
        "workspace_assistant_selection": "inherit",
        "workspace_assistant_default_version": 2,
    })
    first = start_workspace_chat(
        creation_db, owner_id="user-1", request=request,
        idempotency_key="accepted-1", receipt_limit=1,
        chat_limit=None, title_timestamp="first",
    )
    replay = start_workspace_chat(
        creation_db, owner_id="user-1", request=request,
        idempotency_key="accepted-1", receipt_limit=1,
        chat_limit=0, title_timestamp="different",
    )
    assert (replay.replayed, replay.conversation["id"]) == (
        True, first.conversation["id"])
```

- [ ] Run each named RED test individually with `TLDW_TEST_POSTGRES_REQUIRED=1`; preserve failure evidence. Extract `insert_resolved_workspace_conversation` from existing helper while retaining its authority/lineage guards and wrapper transaction. Run all old creation tests immediately to prove unchanged behavior.
- [ ] Implement `start_workspace_chat`: assert owner matches `db.owner_user_id`, validate key/budgets; owner lock; receipt recheck; fresh `lock_receipt_workspace(..., for_create=True)`; recheck; gate deleted/archive/native closure/system operation; validate whole version only for unseen inherit; call existing resolver on an adapted `ChatSessionCreate`. None adaptation explicitly supplies null identity, never resolves defaults. Inherit adaptation omits identity. Lock selected Persona and recheck receipt before new decisions. Wrong-owner Workspace must be 404 with no identity disclosure or writes; do not use id-only lookup or change the legacy getter.
- [ ] Count capacity and the existing scope's nondeleted live chats on passed connection, not an unlocked count method. Reject unseen at capacity/quota without writes. Generate server id/root and resolved title, call shared insertion with `conn`, read committed-bound row, insert digest receipt, return locked row. Receipt and conversation roll back together; no network calls.
- [ ] Implement replay: initial unlocked receipt identifies references; call `lock_receipt_workspace` (FOR NO KEY UPDATE on PostgreSQL), then Persona/conversation locks and final receipt reread; compare request before returning anything; current owner/access; deletion before binding invalidation; digest verification; original current Persona admission (Stage 4 helper supplies final implementation); current metadata response. Never use existing FOR UPDATE Workspace getter on replay, resolve today's default, regenerate titles, consume capacity, restore a deleted row or clear invalidation.
- [ ] Inject failure immediately before/after real conversation and receipt INSERT; test rollback and commit-then-response-loss retry. Use two DB handles/events plus existing `_wait_for_blocked_writer` utilities for same-key, different-request, cross-Workspace cap=1 and default-clear races. Open/initialize all observer handles before entering a timed critical barrier; fixture initialization latency must not consume an unrelated race assertion's deadline. Test winner receipt after every blocking lock and bounded uniqueness-conflict recovery in a fresh transaction. Assert one row, not only equal responses; no sleep-only races.
- [ ] Validate invalid/zero/negative/noninteger configuration fails closed; cap counts repeated create/delete cycles. Document legacy preflight-quota race limitation. Run creation/store suites on both backends plus scoped lint/compile/Bandit; independently review transaction/lock order and commit inactive orchestrator.

## Stage 3: Lifecycle And Native Compatibility

**Goal:** Make every binding/scope/delete mutation maintain accepted receipts atomically.
**Success Criteria:** Change-away-and-back cannot replay; metadata/no-op can; deleted targets never recreate; native identity/Sync guards and staged Workspace closure survive.
**Tests:** New `tests/DB_Management/test_workspace_chat_startup_lifecycle.py`; extend `tests/Sync/test_sync_v2_chat_materializer.py`; retain native fork transaction/lifecycle tests.
**Status:** Not Started.

- [ ] Add RED cases for actual kind/id/character/memory changes, normalized no-op, metadata settings/counters, Sync replacements/scope moves, revert-to-original identity, rollback of mutation and invalidation together, soft delete/restore, hard delete/id reuse and both Workspace cascades. Add blocked Sync transfer-back-to-origin versus replay and closure-versus-receipt-bound restore/Sync resurrection. Imports cannot supply receipt authority.

```python
def test_metadata_does_not_change_binding_digest(creation_db):
    row = {"assistant_kind": None, "assistant_id": None,
           "character_id": None, "persona_memory_mode": None,
           "assistant_startup_json": None,
           "scope_type": "workspace", "workspace_id": "ws"}
    changed = dict(row, title="Edited", state="archived", version=8)
    assert startup_binding_digest(creation_db, row) == startup_binding_digest(creation_db, changed)
```

- [ ] Implement `invalidate_changed_binding` after normalized mutation inside its existing transaction, using locked before/after rows and `COALESCE(invalidated_at, CURRENT_TIMESTAMP)`. Hook both `update_conversation` and `upsert_conversation_from_sync`; do not relax native guards or mistake client attribution for owner transfer. Hook hard-delete before removing the conversation, null reference permanently; FK protects direct hard delete too. Soft delete is detected from current row, restore leaves invalidation untouched.
- [ ] Inventory call sites and direct writes again with `rg 'upsert_conversation_from_sync|update_conversation|DELETE FROM conversations|UPDATE conversations'`; classify migrations/settings/counters and every runtime binding/scope writer in execution notes. Add any newly discovered hook before activation; no endpoint-only shortcut.
- [ ] Barrier-test replay versus concurrent mutation: returned identity must be wholly pre-mutation or reject, never validate old digest then return edited identity. Verify no receipt-first/conversation-second row-lock cycle and FOR NO KEY UPDATE compatibility with Sync's implicit Workspace-FK KEY SHARE, especially transfer back to origin. Mutation hooks acquire no owner/Workspace/Persona lock after conversation. Confirm rollback restores both mutation and invalidation.
- [ ] Use `has_receipt` plus unlocked row preflight to extend restore and Sync replacement/resurrection/re-entry admission for receipt-bound chats (including invalidated receipts). Acquire destination Workspace via `lock_receipt_workspace` before conversation, including before Sync's initial INSERT ON CONFLICT (it can implicitly lock an existing row); reject missing/deleted/closing/system-operation destination, then recheck scope and receipt association after conversation lock. If the destination/association changed or a receipt appeared after preflight, roll back for bounded retry; never acquire a newly required Workspace lock after conversation. Preserve native guards and unrestricted legacy non-receipt semantics. Extend `_finish_native_workspace_delete` residual predicate to owned receipt-bound chats as well as the unchanged native predicate.
- [ ] Race strict startup, receipt-bound restore and Sync resurrection/re-entry with H2 `_begin_native_workspace_delete`: admission-before-close is cascaded; close-before-admission gets bounded closure error, no orphan. Test final residual protection, archived-new versus archived-replay, system-operation closure, soft/hard Workspace removal/access precedence and receipt budget after Workspace deletion. Preserve outermost delete transaction requirements.
- [ ] Run new lifecycle/Sync tests plus `test_native_fork_transactions.py`, `test_native_fork_workspace_lifecycle.py`, `test_native_fork_migration.py` with live PG required. Inspect JSON/Chatbook/export/import/clone paths to prove receipts are absent. Scoped lint/compile/Bandit, lifecycle review, then commit. Route remains inactive until Stage 4 passes.

## Stage 4: Current Admission And Strict Route Activation

**Goal:** Activate the protocol only when generation and all receipt safety prerequisites are present.
**Success Criteria:** Unusable Persona fails before provider/credential/message effects on ordinary and session paths; strict HTTP contract holds; no legacy or native fallback regression.
**Tests:** New `tests/Chat/test_persona_conversation_admission.py`; strict HTTP suite; current `test_persona_prompt_assembly.py`, `Chat/integration/test_persona_backed_chat_conversations.py`, chat API/error-mapping suites.
**Status:** Not Started.

- [ ] Add admission RED cases with a real owner-scoped Persona row: active success, inactive 409, soft-deleted/missing/wrong-owner 404, feature disabled 503, malformed binding 409. None/Character bypass lookup. Spy provider/router, credential and message calls to prove all are untouched on rejection; test ordinary fixed-model and model=auto, streaming/nonstreaming, direct service and session preview/preparation/complete-v2. Extend existing `tests/Chat/integration/test_chat_endpoint_auto_routing.py` rather than inventing another router harness.

```python
def test_nonpersona_admission_does_not_lookup_profile(creation_db, monkeypatch):
    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("None must not inspect Persona availability")
    monkeypatch.setattr(creation_db, "get_persona_profile", unexpected_lookup)
    assert require_current_persona(
        creation_db, owner_id="user-1",
        conversation={"assistant_kind": None, "assistant_id": None},
    ) is None
```

- [ ] Implement helper using existing feature flag/owner-scoped getter; allow `conn`/`for_update` only inside a transaction. Map bounded errors; never return None for failed Persona. In `chat.py:create_chat_completion`, authenticate and validate stored conversation scope/access, then call guard before `_resolve_auto_chat_routing_decision`, credential runtime construction/use and message writes. Do not move authentication/budget policy behind the guard. Recheck in ordinary service context assembly and use returned profile in existing projection for direct callers/current revocation; endpoint approval is not a durable cache. Session consumers guard before current context/exemplars; do not replace prompt assembly or route native continuation through legacy path. Replays call same helper under existing Persona lock.
- [ ] Add strict HTTP RED cases using existing authenticated Workspace dependency overrides: required/duplicate/invalid header; all forbidden/default-valued fields and all query options 422 with zero writes; unknown old-server route/no automatic retry downgrade; fresh201/replay200 header; stale409; capacity409; deleted410; closure/archive; feature/unavailable mappings and privacy. Add legacy strict-selector rejection while preserving unrelated unknown field compatibility.
- [ ] Register static route before `/{chat_id}`, offload synchronous DB orchestration with `run_in_threadpool`, use existing create rate limiter/quota source and ownership dependency, map domain errors and convert response with existing conversation projection. Response contains no key/hash/receipt/internal invalidation fields. Do not enable Workspace Sync or greeting side effects.
- [ ] Run schema/HTTP, ordinary/session prompt-memory, legacy creation/Character/fork and native regressions. Confirm precedence/exemplars/read-only/read-write/custom prompts are unchanged for usable rows, current revocation cannot yield plain Assistant. Guard check is point-in-time, no network-long row locks. If session Persona behavior cannot safely generate, reject explicitly rather than misadvertise support.
- [ ] Independently review full activation diff against both specs; fix valid findings with RED/GREEN and scoped Bandit. Only then commit activation. No deployment before offline writer-quiescence rehearsal.

## Stage 5: Acceptance, Contracts And Delivery

**Goal:** Deliver exact-head evidence and an operator-safe migration without claiming broader parity.
**Success Criteria:** Complete affected suites pass both backends, no new security findings, current hosted gates/reviews/human summary satisfied, deployment runbook preserves receipts.
**Tests:** All newly added suites, existing Workspace/provenance/creation, Sync, ordinary/session/Character import/export, native history/fork suites and migration/CI contracts.
**Status:** Not Started.

- [ ] Add all new test files exactly once to appropriate dedicated matrices in `.github/workflows/ci.yml` and allowlists in `tldw_Server_API/tests/CI/test_required_workflow_contracts.py`. Check all matrix variants structurally and `Helper_Scripts/ci/check_shard_coverage.py`; do not hide an unrelated upstream baseline failure.
- [ ] Generate OpenAPI via `Helper_Scripts/export_openapi_schema.py`, using disposable exact-CI dependency overlay if still needed; preserve shared venv. Update tracked `apps/tldw-frontend/lib/api/openapi.fingerprint.json` and generated API types using existing frontend command. Inspect closed strict request, error/header contract and unchanged legacy request. No frontend workflow adoption here.
- [ ] Run required backend matrix after virtualenv activation, using official PostgreSQL fixtures:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
TLDW_TEST_POSTGRES_REQUIRED=1 TLDW_TEST_NO_DOCKER=1 python -m pytest -q \
  tldw_Server_API/tests/Workspaces/test_workspace_assistant_startup.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_receipts.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_migration.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_rls.py \
  tldw_Server_API/tests/DB_Management/test_workspace_chat_startup_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_workspace_assistant_creation_atomic.py \
  tldw_Server_API/tests/DB_Management/test_conversation_assistant_startup.py \
  tldw_Server_API/tests/DB_Management/test_native_fork_transactions.py \
  tldw_Server_API/tests/DB_Management/test_native_fork_workspace_lifecycle.py \
  tldw_Server_API/tests/DB_Management/test_native_fork_migration.py \
  tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py \
  tldw_Server_API/tests/Chat/test_persona_conversation_admission.py \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py \
  tldw_Server_API/tests/Chat/test_persona_prompt_assembly.py \
  tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py
```

- [ ] Run all existing Workspace suites plus affected Character/export/import/clone/chat API suites separately; record disjoint counts, skips, warnings and actual failures. Reproduce suspected upstream failures on pristine exact base, never call a partially failing matrix green. The planning baseline is historical, not implementation evidence.
- [ ] Measure new receipt/schema/orchestrator/admission modules with pytest-cov in the focused matrix; target above 80% and inspect uncovered error/concurrency branches, not just the percentage. Run Ruff, changed Python compilation, full touched-scope Bandit (production and tests separately with assert exclusion), whitespace and OpenAPI/shard guards. Compare any test-template false positives to pristine base. No new findings may be deferred silently.
- [ ] Document offline drain/migrate/restart, old cached-handle hazard, compatible-binary rollback boundary, permanent receipt privacy/backup retention, capacity increases and no key recycling. Rehearse upgrade/rollback/reopen and cached-writer limitations using official fixtures, not a version-rewound modern DB.
- [ ] Complete independent whole-slice spec/security review; update assessment and execution task. Publish reviewable PR to dev, attach it to this chat, require requester-owned Change summary explaining implementation choices, current-head CI and fresh review inspection. Recheck actual dev before normal merge; rebase legitimate changes and require fresh evidence if needed. No admin bypass.
- [ ] After merge update execution task/parent/#2950 and canonical plan. Mark only Stage 2C delivered; Stage 2D/tool-profile/provisioning/Research adoption remain open until their own gates. Remove only this task's completed plan if repository policy requires, retaining links to its Git history.

## Planning Review And Evidence

Source audit and the unmodified 10-file baseline are tracked under TASK-13245.6. Current fixture code already supplies the snapshot-loader credential override; do not edit it merely to replay a historical repair. Independent plan review must cover mutation completeness, PostgreSQL lock order, native cascade integration, privacy, strict validation and activation boundaries. Requester review of this refreshed design is required before runtime implementation.

The baseline completed with **490 passed, one failed, one skipped, six warnings** (2226.21 seconds), not a green matrix. The native PostgreSQL hard-delete cascade retry fails because its enumeration leaves an implicit read transaction open after the injected child-delete failure. Exact reproduction and a temporary owned-read diagnostic confirm the cause; no source/test repair was made. TASK-13245.7 records the prerequisite and caller-transaction constraints. The skip is the PostgreSQL driver-transaction case running on SQLite. Persona prompt/memory tests passed unmodified, so the historical credential-fixture blocker is no longer current. Local logs and the diagnostic limitation are recorded in the refresh.

Independent source-backed review corrected endpoint admission timing, forced PostgreSQL owner RLS, replay versus FK lock ordering, and receipt-bound restore/Sync closure protection. Focused re-review found no remaining material findings for those corrections. Markdown link checks, example AST parsing and an in-memory documented-model probe passed; these do not certify implemented concurrency or runtime behavior. Bandit is not applicable to this Markdown/Backlog-only planning change.
