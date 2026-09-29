# Workspace Persona Strict Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Deliver Stage 2C strict versioned Workspace startup with atomic bounded retry receipts, lifecycle protection and current Persona generation admission.

**Architecture:** A closed FastAPI request calls a small Workspace orchestrator backed by one operation-specific ChaCha receipt store. Share current resolver and conversation insertion; install transaction-local mutation hooks and one shared Persona admission helper before enabling the route. Preserve native fork semantics and all legacy creation behavior.

**Tech Stack:** Python/Pydantic/FastAPI, ChaCha SQLite/PostgreSQL backend abstraction, pytest and official per-test PostgreSQL fixtures, existing OpenAPI exporter and CI shard tooling.

**Spec:** [Current-dev refresh](../../Design/2026-09-27-persona-workspace-strict-startup-refresh.md) and [reviewed original contract](../../Design/2026-09-13-persona-workspace-choice-provenance-design.md).

**Tracking:** Planning and requester-approved review amendments TASK-13245.6; execution TASK-13245.8; parent TASK-13245; issue #2950. This document has not activated Stage 2C or completed Stage 2D.

**ADR check:** ADR required: yes. ADR path: `Docs/ADR/056-workspace-chat-strict-startup-receipts.md` (draft during execution; renumbered after current-dev integration). Permanent owner-bound retry receipts and lifecycle invalidation establish a durable persistence/security rule distinct from ADR050 native forks.

## Global Constraints

- Work only in the isolated checkout; preserve shared dirty checkout, containers, recovery refs and stash backups.
- Recheck server dev, migration registry and all affected writers at execution start. Planning base `9668e1454b0b28b7a4de13e1a35496fa0b368c42`; SQLite 72/PostgreSQL 76. Candidate 73/77 is not a reservation.
- Chatbook ref `74965f694f435c590366ec66d671d3506d440ebc` was checked, not audited. No frontend adoption/parity certification in this backend slice.
- No new dependencies, generic receipt framework, native fork operation kind or duplicate conversation INSERT. All SQL remains in DB_Management.
- Preserve legacy omission/null, explicit Character/Persona, parent/global behavior, Workspace Sync exclusion and server-owned provenance import stripping.
- Required header 1-128 ASCII `[A-Za-z0-9][A-Za-z0-9._:-]*`; reject duplicates. Closed body, no query options; `inherit` requires positive strict whole-Workspace version, `none` forbids supplied version.
- Strict-only text validation rejects NUL/surrogates before normalization and hashing: workspace_id 256 UTF-8 bytes; title/external_ref 4096 each; topic_label 1024; cluster_id/source 256 each; state 16; combined supplied strings 8192. Route-local streamed raw body ceiling 65536 bytes before parsing (413); field/aggregate errors 422. Legacy models keep their current limits.
- Store only bounded receipt hashes/references/timestamps; no raw key/request/response/profile/content snapshots. Unique `owner_user_id`/key across Workspaces. Forced PostgreSQL owner-only RLS must retain tombstone visibility independently of live parents. Lifetime count defaults to 10000, includes every tombstone; no TTL/recycling.
- First acceptance 201; replay 200 with `Idempotency-Replayed: true`. No silent downgrade/fallback. Deleted target 410; actual binding/scope change permanently invalidates.
- Acquire owner admission before Workspace/Persona; reread receipts after blocking acquisitions. Replay uses Workspace FOR NO KEY UPDATE, then conversation before receipt; never FOR UPDATE against Sync's implicit FK KEY SHARE. Receipt-bound restore/Sync resurrection locks destination Workspace before conversation. Writer hooks must not introduce an inverse lock order.
- Native admission closure/system-operation state blocks strict startup; archived Workspace blocks new startup only. Existing native identity/Sync/projection guards remain intact.
- No network/LLM/credential calls in creation transaction. Persona generation admission happens before side effects; do not hold DB locks across generation.
- Strict startup/replay owns an idle outermost transaction, rejecting caller-managed/driver-open transactions without committing or rolling them back. Return a result only after successful context exit/commit; uniqueness recovery starts after rollback in a fresh owned transaction.
- Persona admission authority is immutable `db.owner_user_id`, not writer/device `client_id`; supplied owner must match. Session preparation/preview/complete-v2 remain global-only, with explicit Workspace scope options rejected and no Workspace-session support claim.
- Offline drain/migrate/restart only, no mixed writers or old-binary rollback. Activate route only with lifecycle/capacity/current-admission safeguards.
- Every increment follows RED/GREEN, scoped Ruff/compile/Bandit, independent review and hooks-enabled commit. Use official SQLite/live-PG fixtures; do not suppress failures or add unrelated CI/MCP/Chunking/Buddy work.
- Resolve the separately tracked native Workspace cascade retry transaction leak (TASK-13245.7) before lifecycle qualification. Preserve the outermost/caller-transaction guard; never hide the leak with a test-only commit. Recheck latest dev to avoid duplicating an upstream repair.
- Required current-head hosted checks, normal up-to-date merge and requester-owned human Change summary remain gates. Historical green heads do not count.

## File Map And Interfaces

All paths in this section are relative to `tldw_Server_API/` unless stated otherwise. Proposed names below are implementation contracts, not existing functions.

| File | Responsibility |
| --- | --- |
| `app/core/Workspaces/chat_startup_schemas.py` (new) | Closed request, selection/version validation, shared state grammar and pure canonical fingerprint. |
| `app/api/v1/schemas/workspace_chat_startup_schemas.py` (new) | Public reexports of the core-owned strict request/hash, without a core-to-API import. |
| `app/api/v1/endpoints/workspace_chat_startup_transport.py` (new) | Strict-route-only APIRoute body-size guard and sanitized validation detail; no global middleware change. |
| `app/core/DB_Management/chacha/workspace_chat_startup_schema.py` (new) | Backend-specific receipt table/index DDL called only by registered migrations. |
| `app/core/DB_Management/chacha/workspace_chat_startup_store.py` (new) | Owner serialization, receipt/conversation consistent reads, count/insert, mutation invalidation/hard-delete tombstones, bounded domain errors. |
| `app/core/DB_Management/ChaChaNotes_DB.py` | Migration registry, initialization of `workspace_chat_startups` store; existing Workspace cascade compatibility. |
| `app/core/DB_Management/backends/pg_rls_policies.py` | Guarded new receipt RLS builder and current policy-set integration, migration-safe when table absent. |
| `app/core/Workspaces/assistant_defaults.py` | Typed unavailable exception preserving legacy HTTP status/detail; connection-taking shared insertion helper preserving resolver/title behavior. |
| `app/core/Workspaces/chat_startup.py` (new) | Strict transaction and replay orchestration, no SQL. |
| `app/core/DB_Management/chacha/conversation_store.py` | Hooks in actual identity/scope/delete writers. |
| `app/core/Persona/conversation_admission.py` (new) | Current Persona feature/owner/activity admission, no prompt reconstruction. |
| `app/api/v1/endpoints/chat.py`, `app/core/Chat/chat_service.py`, `app/api/v1/endpoints/character_chat_sessions.py` | Ordinary admission before routing/credentials plus service recheck; session admission and static strict route. |
| `app/api/v1/schemas/chat_session_schemas.py` | Reject strict-only fields on legacy requests; keep unrelated extra-ignore semantics. |

Define these small interfaces in their owning files and use them consistently:

```python
# core/Workspaces/chat_startup_schemas.py
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, Self
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

ALLOWED_CONVERSATION_STATES = ("in-progress", "resolved", "backlog", "non-viable")

def _validate_conversation_state(value: str | None) -> str | None:
    """Shared state grammar, also imported by legacy API models."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("state cannot be empty")
    if normalized not in ALLOWED_CONVERSATION_STATES:
        raise ValueError(f"Invalid state '{value}'. Allowed: {', '.join(ALLOWED_CONVERSATION_STATES)}")
    return normalized

STARTUP_TEXT_BYTE_LIMITS = {
    "workspace_id": 256, "title": 4096, "state": 16,
    "topic_label": 1024, "cluster_id": 256,
    "source": 256, "external_ref": 4096,
}
STARTUP_TEXT_BYTES_MAX = 8192
STARTUP_BODY_BYTES_MAX = 65536

def startup_text_size(value: str, field: str) -> int:
    """Validate bounded raw decoded text without exposing its contents."""
    limit = STARTUP_TEXT_BYTE_LIMITS[field]
    if len(value) > limit or "\x00" in value:
        raise ValueError("Startup text exceeds its limit or contains NUL")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError:
        raise ValueError("Startup text must contain valid Unicode") from None
    if size > limit:
        raise ValueError("Startup text exceeds its UTF-8 byte limit")
    return size

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

    @model_validator(mode="before")
    @classmethod
    def validate_supplied_text(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            total = sum(
                startup_text_size(data[field], field)
                for field in STARTUP_TEXT_BYTE_LIMITS
                if field in data and isinstance(data[field], str)
            )
            if total > STARTUP_TEXT_BYTES_MAX:
                raise ValueError("Startup text exceeds its combined byte limit")
        return data

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

- `require_outermost() -> None`: no SQL; reject SQLite in_transaction or nonzero PostgreSQL wrapper/backend depth or non-IDLE driver status with `WorkspaceStartupError("workspace_chat_startup_transaction_required", 409)`. Never end caller work. Borrow the checks from `_require_outermost_workspace_delete`, not its deletion-specific exception or a relaxed tx_depth-only approximation.
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
- Transport `WorkspaceStartupRoute(APIRoute)` in the new endpoint helper module: its wrapped route handler buffers at most 65536 streamed bytes before calling FastAPI's original handler on a new Request with a bounded replay receive callable. Reject on the first over-limit chunk with 413; strictly decode UTF-8 and validate the same public model with `model_validate_json` on text, not bytes. Reject nonempty raw query strings, missing/invalid/duplicate keys and non-JSON media types before side-effecting dependencies; use FastAPI's stdlib MIME parser rather than another MIME grammar. Convert only model/UTF-8 failures to sanitized RequestValidationError. Do not call request.body() first, trust Content-Length alone, mutate Request private caches or apply this route class to legacy endpoints. Preserve disconnect/cancellation behavior and the underlying receive callable after the buffered body is delivered, rather than synthesizing endless empty request events. Catch only RequestValidationError from prevalidation/the original handler and return 422 using strict_startup_validation_detail below. Register only the strict route with `route_class_override=WorkspaceStartupRoute`. The header schema uses the same full-match key pattern with explicit anchors.
- Policy builder `build_workspace_chat_startup_rls_sql() -> list[str]` in existing `pg_rls_policies.py`: explicit guarded table statements, ENABLE/FORCE and owner_user_id USING/WITH CHECK. `build_chacha_rls_sql` includes it; new migration applies it after table creation.

Use existing metadata normalization after the strict-only raw-text/byte checks; current legacy string fields do not supply these bounds. Domain errors map at the endpoint to a detail dictionary with `code` and optional `reason`; no new public receipt model. For every code step below, first run its named RED test, then implement the specified interface and run GREEN. The route must remain absent until Stage 4 safety tests pass.

Typed unavailable translation keeps the existing resolver authoritative and legacy wire behavior unchanged. In `assistant_defaults.py`, add this narrow subclass using the existing bounded reason type from workspace_schemas, and raise it only where the resolver currently raises unavailable-default HTTPException:

```python
class WorkspaceDefaultUnavailable(HTTPException):
    """Preserve legacy HTTP behavior while exposing a bounded typed reason."""
    def __init__(self, reason: WorkspaceAssistantDefaultDegradedReason):
        self.reason = reason
        super().__init__(
            status_code=503 if reason == "persona_feature_disabled" else 409,
            detail="Workspace default Persona is unavailable; choose an assistant explicitly",
        )

# chat_startup.py; catch only this type around the existing resolver call.
def map_workspace_default_unavailable(
    error: WorkspaceDefaultUnavailable,
) -> WorkspaceStartupError:
    """Translate unavailable defaults, never generic HTTP/database errors."""
    code = ("persona_feature_disabled" if error.reason == "persona_feature_disabled"
            else "workspace_assistant_unavailable")
    return WorkspaceStartupError(code, error.status_code, error.reason)
```

Keep missing/inaccessible Workspace and unexpected storage failures on their existing mappings. No message parsing, arbitrary reason text, generic status-based catches or second effective-default resolver. Import the typed class/map dependencies into their respective owning modules, not through endpoint code.

The transport imports RequestValidationError, Any and the schema's text-limit keys. Its validation detail is deliberately content-free, including errors from the existing state validator and unknown field names:

```python
def strict_startup_validation_detail(
    error: RequestValidationError,
) -> list[dict[str, Any]]:
    """Return fixed messages and bounded static locations, never raw input."""
    allowed = set(STARTUP_TEXT_BYTE_LIMITS) | {
        "body", "header", "query", "scope_type",
        "workspace_assistant_selection", "workspace_assistant_default_version",
        "idempotency-key", "Idempotency-Key",
    }
    details = []
    for item in error.errors()[:32]:
        location = [part for part in item.get("loc", ())
                    if isinstance(part, str) and part in allowed][:4]
        details.append({
            "loc": location or ["body"],
            "msg": "Invalid Workspace startup request",
            "type": "value_error",
        })
    return details
```

Do not copy original msg/type/unknown loc, input, ctx, body or exception text into responses/logs. The bounded raw-UTF-8 failure uses the same fixed detail shape. No change to legacy validation handlers.

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
**Status:** Complete.

- [x] Search/create execution Backlog task and read both linked specs. Fetch/reinspect latest dev and current migrations/native writers. Record actual new migration numbers in that task before editing. TASK-13245.8 records inspected dev f4bcc9bd70 and next SQLite73/PostgreSQL77; recheck before publication.
- [x] Add parametrized schema RED cases: missing selection/scope/id, global scope, inherit missing/null/zero/negative/bool/string version, none supplied version including null, identity/provider/fork/default-valued extras. Add valid None/inherit and metadata-presence hash tests.
- [x] Add strict text RED cases for every accepted string: NUL, lone high/low surrogate, ASCII byte limit and limit+1, multibyte boundary, valid paired Unicode/non-ASCII round trip, pre-trim Workspace/state limits and combined 8192-byte boundary. Null/omitted metadata stays valid and hashes distinctly. Implement the documented raw-text validator; do not change legacy fields. Reserve route-level 65536-byte streaming/413 tests for Stage 4.

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

- [x] Run `python -m pytest -q tldw_Server_API/tests/Workspaces/test_workspace_assistant_startup.py`; require the intended missing/new-contract failure, not an unrelated collection failure. Implement the model and fingerprint, preserving supplied fields; add before/after validators for version intent and existing state/metadata limits.
- [x] Add property coverage for canonical request ordering using existing Hypothesis (import `given` and `strategies as st`), alongside explicit omission/null cases:

```python
@given(st.text(max_size=128))
def test_fingerprint_ignores_input_dictionary_order(title):
    payload = {"scope_type": "workspace", "workspace_id": "ws",
               "workspace_assistant_selection": "none", "title": title}
    reordered = dict(reversed(list(payload.items())))
    assert startup_request_fingerprint(WorkspaceChatStartupRequest.model_validate(payload)) == (
        startup_request_fingerprint(WorkspaceChatStartupRequest.model_validate(reordered)))
```
- [x] Add migration RED tests using actual historical SQLite/PG schema fixtures: upgrade, failed migration rollback, reopen/idempotent application, unique owner/key, non-null hashes, conversation deletion sets reference null, Workspace deletion keeps receipt. Reject raw snapshot columns and any `ON DELETE CASCADE` from Workspace. Add new `tests/DB_Management/test_workspace_chat_startup_rls.py`: official `pg_temp_db` plus unique per-test NOSUPERUSER NOBYPASSRLS role, grants and cleanup; two tenant scopes and absent scope reject cross-owner SELECT/INSERT/UPDATE/DELETE/count/replay. Exercise orphan tombstones after parent deletion. Admin-only fixture passes are not RLS evidence.
- [x] Create DB-only DDL/store and currently next migration on both backends, using recognized tenant column `owner_user_id`. Add/apply guarded `build_workspace_chat_startup_rls_sql` through current policy-set and migration; ensure older migration calls tolerate missing table. Both USING/WITH CHECK compare to `current_setting('app.current_user_id', true)`, with no live-parent EXISTS. Run `tests/CI/test_rls_coverage_ratchet.py`, `tests/DB_Management/test_pg_rls_policies_contract.py` and new live non-bypass suite; do not baseline-exempt receipts. Add `db.workspace_chat_startups`; parameterized helpers take explicit connection, not nested commits or unlocked conversation getter. Hash keys using SHA-256; PostgreSQL owner lock namespace is stable `workspace_chat_startup_v1`, not Python's randomized hash.
- [x] Exercise two distinct owner/key namespaces, all-tombstone counts and owner lock with actual backend handles. Adapt three existing patterns: native receipt uniqueness, `organization_sync_store` advisory locking and ordinary conversation transaction helpers. Do not import native leases/canonical request storage.
- [x] Run new schema/migration/store suites with official required PG fixture; scoped Ruff, compile and Bandit. Review DDL/privacy and commit inactive foundation with the execution task. No strict route yet.

Stage 1 qualification: exact-source eight-file SQLite/live-PostgreSQL matrix **466 passed, 20 warnings, no failures/skips** in 175.76 seconds (`/private/tmp/persona-strict-foundation-qualified-integration.log`). Independent scoped review passes after F1-F5/F2-R repairs. All 13 changed Python files pass Ruff/compilation; production five-file Bandit has no findings/errors, and test eight-file scan has only unchanged pristine-dev-matched B105 `--cov`. Structured scheduling covers all four new files once in each of five matrices; shard ratchet has no new uncovered files. Historical native-policy static-scanner and Chunking shard-contract baseline failures remain separately documented, not fixed or called green. Latest audited dev `f4b69eabea` has no migration collision; 73/77 must still be rechecked before publishing. Orchestration, lifecycle hooks, current-admission consumers and strict route remain absent.

## Stage 2: Atomic Selection, Acceptance And Replay

**Goal:** Implement transaction orchestration with duplicate recovery and finite lifetime capacity, still without public activation.
**Success Criteria:** One committed chat/receipt per owner/key; correct version/default/replay outcomes; no partial records; identical replay works at cap/quota; no blocking-lock stale duplicate decisions.
**Tests:** Extend receipt tests; new strict creation cases in `tests/DB_Management/test_workspace_assistant_creation_atomic.py` and `tests/Workspaces/test_workspace_assistant_startup.py`.
**Status:** Complete (inactive implementation only; lifecycle and activation remain separate gates).

Stage 2 dependency work: shared insertion/typed default failure is a bounded sidecar. The shared current-Persona helper core is pulled forward from Stage 4 so replay will not use a permissive temporary admission stub; ordinary/session consumers and route activation remain in Stage 4. Initial helper REDs were five pure-owner/bypass and twelve real SQLite contract failures; initial GREEN was 17 passed with 100% coverage of its then-38 statements. That historical snapshot predates the normalization-before-bypass review correction; final helper/source qualification is recorded below.

Execution checkpoint: shared insertion and typed default adaptation passed independent bounded review and 124 disjoint helper/legacy cases. Current-Persona normalization-before-bypass passed re-review after six malformed-binding REDs. The initial three-file SQLite/live-PostgreSQL run passed 181 cases; the earlier 731-case matrix passed but predates the final replay-hint guard and 18 additional acceptance cases. Final exact-source 12-file verification passed 749 cases, 20 warnings, no failures or skips in 848.75s; official live PostgreSQL was required. Evidence is `/private/tmp/persona-strict-stage2-final-integration.log` and its JUnit XML. Current SQLite-only coverage qualification passed 395 cases with 93% statement coverage across all five new modules (request/schema 100%, receipt store 86%, admission 100%, orchestrator 94%); it is not PostgreSQL coverage or lifecycle/process-boundary certification. Independent source review passes the final private orchestrator and caller-write fixture. Final seven-file Ruff/compilation and production/test scoped Bandit (tests exclude assertions) pass with zero findings/errors; shard scheduling is exact-once in all five matrices, with no new uncovered paths. No public route or ordinary/session consumer is wired.

Two review corrections retain bounded retry/lock order: reread a winner before mapping typed default/admission failures; reject a deleted hint that becomes live without Persona admission. Their deterministic REDs and revised SQLite boundary 14-case GREEN are recorded in TASK-13245.8. The managed caller probe now mutates Workspace metadata and uses the existing getter, avoiding an unrelated conversation-FTS schema-cache setup failure. No caller work is committed or rolled back by strict startup. The legacy endpoint's unlocked preflight quota race is unchanged; only this private strict acceptance counts quota inside its owned transaction. Stage 3-5 and public activation remain pending.

- [x] Reuse the existing parametrized `creation_db` fixture from `test_workspace_assistant_creation_atomic.py` (backed by `db_factory` from `test_conversation_assistant_startup.py`). Add this RED case plus stale/default-change, cross-Workspace same-key conflict and requested-metadata conflict cases:

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

- [x] Add outermost-boundary tests for creation and replay on SQLite and PostgreSQL: enclosing managed transaction; driver-open transaction at wrapper depth zero; no partial writes; rejected call leaves caller sentinel writes and transaction state untouched so caller can later commit or roll back. Preserve the native deletion guard. Inject commit failure to prove the orchestrator never returns an accepted result; assert another preinitialized handle sees chat plus receipt when a successful call returns.
- [x] Preserve deterministic recorded REDs and qualify every case on both backends with `TLDW_TEST_POSTGRES_REQUIRED=1`. Execution ruling: the final full dual-backend matrix, not a claim of individual PostgreSQL RED history for every case, supplies backend qualification. Extract `insert_resolved_workspace_conversation` while retaining authority/lineage guards and the legacy transaction. Add typed `WorkspaceDefaultUnavailable` plus strict translation for all bounded reasons; legacy string/status unchanged, unrelated HTTP/database errors propagate. Old creation suites pass.
- [x] Implement `start_workspace_chat`: assert owner matches `db.owner_user_id`, validate key/budgets; call `require_outermost` before any SQL/read; enter an owned existing DB transaction; owner lock; receipt recheck; fresh `lock_receipt_workspace(..., for_create=True)`; recheck; gate deleted/archive/native closure/system operation; validate whole version only for unseen inherit; call existing resolver on an adapted `ChatSessionCreate`, translating only its typed unavailable failure. None adaptation explicitly supplies null identity, never resolves defaults. Inherit adaptation omits identity. Lock selected Persona and recheck receipt before new decisions. Wrong-owner Workspace must be 404 with no identity disclosure or writes; do not use id-only lookup or change the legacy getter.
- [x] Count capacity and the existing scope's nondeleted live chats on passed connection, not an unlocked count method. Reject unseen at capacity/quota without writes. Generate server id/root and resolved title, call shared insertion with `conn`, read the transaction-bound row, insert digest receipt and hold the result until successful context exit/commit. Do not return inside a nested caller transaction. Receipt and conversation roll back together; no network calls.
- [x] Implement replay: initial unlocked receipt identifies references; call `lock_receipt_workspace` (FOR NO KEY UPDATE on PostgreSQL), then Persona/conversation locks and final receipt reread; compare request before returning anything; current owner/access; deletion before binding invalidation; digest verification; original current Persona admission (core pulled forward from Stage 4); current metadata response. Never use existing FOR UPDATE Workspace getter on replay, resolve today's default, regenerate titles, consume capacity, restore a deleted row or clear invalidation.
- [x] Inject failure immediately before/after real conversation and receipt INSERT; test rollback and commit-then-response-loss retry. Use two DB handles/events plus existing `_wait_for_blocked_writer` utilities for same-key, different-request, cross-Workspace cap=1 and default-clear races. Open/initialize all observer handles before entering a timed critical barrier; fixture initialization latency must not consume an unrelated race assertion's deadline. Test winner receipt after every blocking lock and bounded uniqueness-conflict recovery only after the failed owned context has exited/rolled back; the next attempt starts a fresh outermost transaction, never a savepoint/aborted PG transaction. Assert one row, not only equal responses; no sleep-only races.
- [x] Validate invalid/zero/negative/noninteger configuration fails closed; cap counts repeated create/delete cycles. Document legacy preflight-quota race limitation. Run creation/store suites on both backends plus scoped lint/compile/Bandit; independently review transaction/lock order and commit inactive orchestrator.

## Stage 3: Lifecycle And Native Compatibility

**Goal:** Make every binding/scope/delete mutation maintain accepted receipts atomically.
**Success Criteria:** Change-away-and-back cannot replay; metadata/no-op can; deleted targets never recreate; native identity/Sync guards and staged Workspace closure survive.
**Tests:** New `tests/DB_Management/test_workspace_chat_startup_lifecycle.py`; extend `tests/Sync/test_sync_v2_chat_materializer.py`; retain native fork transaction/lifecycle tests.
**Status:** Complete (inactive lifecycle checkpoint; activation and delivery remain open).

- [x] Add RED cases for actual kind/id/character/memory changes, normalized no-op, metadata settings/counters, Sync replacements/scope moves, revert-to-original identity, rollback of mutation and invalidation together, soft delete/restore, hard delete/id reuse and both Workspace cascades. Add blocked Sync transfer-back-to-origin versus replay and closure-versus-receipt-bound restore/Sync resurrection. Imports cannot supply receipt authority.

Preliminary runtime-write audit (read-only, not lifecycle qualification): ordinary identity updates and whole-object Sync replacement need complete locked before/after rows and the DB-owned invalidation hook. Sync must gate a receipt-bound destination before its initial INSERT ON CONFLICT; that statement can itself lock the existing chat. Restore needs receipt association and destination preflight, including already-invalidated associations. Hard deletion must lock the chat before touching strict receipts; Workspace final residual checks must include owned receipt-bound chats without relaxing native client predicates. Historical migrations precede receipts, but current-schema `_repair_conversation_assistant_identity` also runs on reopen/bootstrap and must be classified and tested before activation. Settings/history counters and Buddy's id=id CAS do not alter binding. The DSR direct SQLite delete uses `configure_sqlite_connection` with foreign keys enabled, so its receipt association is protected by SET NULL; privacy retention and this exact path still need qualification. Chatbooks imports create ordinary conversations from allowlisted fields rather than receipt authority. The initial five SQLite invalidation REDs reproduce missing permanent invalidation, including successful replay after Sync scope change-away-and-back; production hooks have not been installed yet.

First increment, after inactive Stage 2 commit `a8232b9075`: local and Sync writers now invoke the existing receipt hook on complete locked before/after rows inside their mutation transaction. Ten intended SQLite REDs preceded a 15-case SQLite GREEN; first eight-file matrix completed 472 passed, one failed, four known SQLite-only PostgreSQL-driver skips, 20 warnings. The sole failure was the existing Sync barrier spy requiring an explicit column token after SELECT changed to `*`. Its one-line correction retains the actual SELECT/FOR UPDATE, blocker and result assertions; the exact PostgreSQL retest passed. Fresh final-source qualification passed 473 tests, four known SQLite-only driver skips, 20 warnings, no failures in 962.68s; all 30 new lifecycle cases passed with official live PostgreSQL required. Evidence is `/private/tmp/persona-strict-stage3-invalidation-qualified.log` and its JUnit XML. Production/test Bandit is clean, new/modified tests pass Ruff and compilation, and the whole conversation store retains four unchanged TRY203 findings verified against a physical `a8232b9075` archive. Bounded source and spy-only re-review pass; neither is whole-lifecycle approval. Source hashes match the reviews and fresh execution; the invalidation increment is qualified, while admission/cascade/privacy gates below remain open.

Read-only inventory repeated across `app/`: settings update only version/time, message writes advance history/time, and Buddy CAS writes `id = id`; no binding hook is needed for these. Sync materialization passes explicit allowlisted arguments to the DB writer; Chatbooks import creates ordinary conversations without receipt fields. Public export/import privacy still requires behavioral qualification. Runtime Character repair is not dismissed as migration-only. For the next restore/Sync admission increment, add caller-owned transaction tests: SQLite nested contexts have no independent rollback/savepoint, so a changed-association retry must not settle caller work or acquire a newly discovered Workspace lock after conversation. Owned rollback/retry and bounded caller conflict require separate qualification.

```python
def test_metadata_does_not_change_binding_digest(creation_db):
    row = {"assistant_kind": None, "assistant_id": None,
           "character_id": None, "persona_memory_mode": None,
           "assistant_startup_json": None,
           "scope_type": "workspace", "workspace_id": "ws"}
    changed = dict(row, title="Edited", state="archived", version=8)
    assert startup_binding_digest(creation_db, row) == startup_binding_digest(creation_db, changed)
```

- [x] Implement `invalidate_changed_binding` after normalized mutation inside its existing transaction, using locked before/after rows and `COALESCE(invalidated_at, ?)` with the existing DB UTC timestamp string. The receipt column is TEXT on both backends; PostgreSQL rejects COALESCE with a timestamp-typed argument. Hook both `update_conversation` and `upsert_conversation_from_sync`; do not relax native guards or mistake client attribution for owner transfer. Hook hard-delete before removing the conversation, null reference permanently; FK protects direct hard delete too. Soft delete is detected from current row, restore leaves invalidation untouched.
- [x] Inventory call sites and direct writes again with `rg 'upsert_conversation_from_sync|update_conversation|DELETE FROM conversations|UPDATE conversations'`; classify migrations/settings/counters and every runtime binding/scope writer in execution notes. Add any newly discovered hook before activation; no endpoint-only shortcut.
- [x] Barrier-test replay versus concurrent mutation: returned identity must be wholly pre-mutation or reject, never validate old digest then return edited identity. Verify no receipt-first/conversation-second row-lock cycle and FOR NO KEY UPDATE compatibility with Sync's implicit Workspace-FK KEY SHARE, especially transfer back to origin. Mutation hooks acquire no owner/Workspace/Persona lock after conversation. Confirm rollback restores both mutation and invalidation.
- [x] Use `has_receipt` plus unlocked row preflight to extend restore and Sync replacement/resurrection/re-entry admission for receipt-bound chats (including invalidated receipts). Acquire destination Workspace via `lock_receipt_workspace` before conversation, including before Sync's initial INSERT ON CONFLICT (it can implicitly lock an existing row); reject missing/deleted/closing/system-operation destination, then recheck scope and receipt association after conversation lock. If the destination/association changed or a receipt appeared after preflight, roll back for bounded retry; never acquire a newly required Workspace lock after conversation. Preserve native guards and unrestricted legacy non-receipt semantics. Extend `_finish_native_workspace_delete` residual predicate to owned receipt-bound chats as well as the unchanged native predicate.
- [x] Race strict startup, receipt-bound restore and Sync resurrection/re-entry with H2 `_begin_native_workspace_delete`: admission-before-close is cascaded; close-before-admission gets bounded closure error, no orphan. Test final residual protection, archived-new versus archived-replay, system-operation closure, soft/hard Workspace removal/access precedence and receipt budget after Workspace deletion. Preserve outermost delete transaction requirements.
- [x] Run new lifecycle/Sync tests plus `test_native_fork_transactions.py`, `test_native_fork_workspace_lifecycle.py`, `test_native_fork_migration.py` with live PG required. Inspect JSON/Chatbook/export/import/clone paths to prove receipts are absent. Scoped lint/compile/Bandit, lifecycle review, then commit. Route remains inactive until Stage 4 passes.

Stage 3 final closure/admission qualification: the exact DB-source 12-file matrix passed **618 tests, 10 intentional SQLite nodes for PostgreSQL-only driver/lock cases, four warnings, no failures** in 1557.99s (`/private/tmp/persona-strict-stage3-admission-final.log/xml`). Native operation-before-chat ordering and failed-rollback retry fencing have recorded RED/GREEN regressions. Admission retries once only after an owned rollback leaves an idle boundary; caller transactions remain untouched. Runtime reopen/Character-repair qualification passed **12 tests on SQLite/live PostgreSQL**, preserving strict Persona/None receipts and replay without an additional repair hook (`persona-strict-stage3-repair-final.log/xml`).

Privacy review exposed a real DSR device-attribution gap: count/erasure now also follows immutable receipt ownership, retaining pre-receipt file support and FK tombstones. The receipt-table probe reuses the existing count/error helper after a corrupt-file RED. Final scoped Chatbook collector/import, DSR service and API qualification passed **26 tests, two intentional PostgreSQL nodes for the SQLite-only eraser, four warnings**, no failures (`/private/tmp/persona-strict-stage3-dsr-qualified.log/xml`). Collector-only construction follows the existing test pattern: the unchanged full Chatbook PostgreSQL constructor fails while initializing job metadata columns, so this does not certify that constructor. Two unrelated AuthNZ DSR repository setup failures reproduce on a pristine `e605b81ddb` archive (`persona-strict-stage3-dsr-repo-baseline.log`); neither is fixed or called green here.

Bounded independent source reviews pass the rollback, native lock-order, DSR and repair corrections. Final three-file production and four-file test Bandit (test B101 excluded) have zero findings/errors; compilation, scoped Ruff, whitespace and shard ratchet pass (4824 files, new_uncovered=0). Full conversation-store Ruff retains four physically baseline-matched TRY203 findings; scoped run excludes only those. All four lifecycle/privacy/repair paths are assigned exactly once in each of five full-suite matrices. No public route, ordinary/session wiring or hosted delivery; Stage 4/5 and the Stage 5 process-boundary acceptance remain open.

## Stage 4: Current Admission And Strict Route Activation

**Goal:** Activate the protocol only when generation and all receipt safety prerequisites are present.
**Success Criteria:** Unusable Persona fails before provider/credential/message effects on ordinary and session paths; strict HTTP contract holds; no legacy or native fallback regression.
**Tests:** New `tests/Chat/test_persona_conversation_admission.py`; strict HTTP suite; current `test_persona_prompt_assembly.py`, `Chat/integration/test_persona_backed_chat_conversations.py`, chat API/error-mapping suites.
**Status:** Complete locally; consumers and strict HTTP activation qualified before current-dev integration. No deployment or hosted delivery.

- [x] Add admission RED cases with a real owner-scoped Persona row: active success, inactive 409, soft-deleted/missing/wrong-owner 404, feature disabled 503, malformed binding 409. None/Character bypass lookup. Spy provider/router, credential and message calls to prove all are untouched on rejection; test ordinary fixed-model and model=auto, streaming/nonstreaming, direct service and session preview/preparation/complete-v2. Extend existing `tests/Chat/integration/test_chat_endpoint_auto_routing.py` rather than inventing another router harness.
- [x] Add immutable-owner RED cases: scoped owner differs from writer/device client_id; matching owner Persona is admitted; supplied owner mismatch rejects before lookup; another owner's profile cannot be admitted by matching writer attribution; changing client_id does not change profile authority or receipt binding. Exercise direct service calls, not only endpoint dependencies.
- [x] Add session-boundary HTTP RED cases for preparation, preview and complete-v2: owned Workspace chat ids remain 404 with zero effects; unsupported Workspace, malformed/duplicate scope and any workspace_id query options are 422 rather than ignored. Omitted scope and a single scope_type=global preserve the existing client's global behavior and ownership/native/Persona guards. These three routes are intentionally global-only; do not plumb Workspace generation into them or claim Workspace admission coverage. Ordinary Workspace turns validate stored scope for id-only clients, enforce explicit scope matches and always recheck current parent access.

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

- [x] Implement helper using existing feature flag/owner-scoped getter: require owner_id == db.owner_user_id before profile lookup, even for a non-Persona bypass; never derive authority from client_id. Allow `conn`/`for_update` only inside a transaction. Map bounded errors; never return None for failed Persona. In `chat.py:create_chat_completion`, authenticate and validate stored conversation scope/access, then call guard before `_resolve_auto_chat_routing_decision`, credential runtime construction/use and message writes. Do not move authentication/budget policy behind the guard. Recheck in ordinary service context assembly with db.owner_user_id and replace its old client-id-based profile lookup with the returned profile for existing projection; endpoint approval is not a durable cache. Global session consumers guard after their existing ownership/scope checks and before current context/exemplars; reject unsupported scope query options without rejecting the client's explicit global form. Do not replace prompt assembly or route native continuation through legacy path. Replays call same helper under existing Persona lock.

Current-admission consumer checkpoint: the exact seven-file SQLite/live-PostgreSQL matrix passed **285 tests, eight warnings, no failures/skips** in 860.46s (`/private/tmp/persona-strict-stage4-consumers-qualified.log/xml`). Clean service/ordinary/session REDs preceded wiring; an additional transferred-Workspace RED exposed the missing parent-owner check and the narrow correction passes. Current Workspace access, archive compatibility and mid-request revocation controls pass. Existing history/streaming doubles now provide immutable owner and the real ConversationStore normalizer rather than relaxing production admission. Bounded independent consumer/correction source reviews pass. Final production/test Bandit, compilation, scoped Ruff and whitespace pass; scoped Ruff excludes only one physically baseline-matched session B904 and the existing unit-file UP006/UP035 findings. Strict route activation, transport qualification and Stage 5 remain open.
- [x] Add strict HTTP RED cases using existing authenticated Workspace dependency overrides: required/duplicate/invalid header; all forbidden/default-valued fields and all query options 422 with zero writes; unknown old-server route/no automatic retry downgrade; fresh201/replay200 header; stale409; capacity409; deleted410; closure/archive; feature/unavailable mappings and privacy. Add legacy strict-selector rejection while preserving unrelated unknown field compatibility.
- [x] Add strict transport RED cases: raw body 65536 bytes accepted when otherwise valid (JSON plus whitespace padding), 65537 rejected with 413; missing/false Content-Length and multiple streamed chunks cannot exceed the ceiling; malformed raw UTF-8 returns content-free 422; escaped JSON under the raw cap still obeys decoded string limits; disconnect/cancellation does not invoke model/DB logic. Responses use fixed message/type and allowlisted locations, with at most 32 details and no echoed body/input/ctx or raw msg/unknown key. Include invalid-state messages embedding private text, surrogate/NUL unknown field keys, malicious loc/msg/type and UTF-8 response serialization. Spy the model handler/DB/providers on rejection. Implement WorkspaceStartupRoute with bounded buffering before the original FastAPI handler; do not claim decoded-field checks bound raw network input.
- [x] Register static route before `/{chat_id}` using only its WorkspaceStartupRoute override, offload synchronous DB orchestration with `run_in_threadpool`, use existing create rate limiter/quota source and ownership dependency, map domain errors/typed resolver failures and convert response only after owned commit with existing conversation projection. Response contains no key/hash/receipt/internal invalidation fields. Do not enable Workspace Sync or greeting side effects.
- [x] Run schema/HTTP, ordinary/global-session prompt-memory, legacy creation/Character/fork and native regressions. Confirm precedence/exemplars/read-only/read-write/custom prompts are unchanged for usable rows, current revocation cannot yield plain Assistant. Guard check is point-in-time, no network-long row locks. If a global session Persona behavior cannot safely generate, reject explicitly rather than misadvertise support. Workspace session preview/completion remains excluded and requires a separately reviewed feature before any broader parity claim.
- [x] Independently review full activation diff against both specs; fix valid findings with RED/GREEN and scoped Bandit. Only then commit activation. No deployment before offline writer-quiescence rehearsal.

Activation qualification: exact 18-file SQLite/live-PostgreSQL matrix **1142 passed, nine warnings, no failures/skips** in 1168.84s (`/private/tmp/persona-strict-stage4-activation-qualified.log/xml`). Bounded activation source review passes, including final anchored header and raw query-string checks. Production five-file Bandit has zero findings/errors; changed tests have only the physical pristine-base-matched B105 `--cov` literal. Scoped Ruff/compilation, OpenAPI check/types generation and shard guard pass (4826 test files, new_uncovered=0). Structured inspection admits both new HTTP/transport paths exactly once in all five full matrices and the exact auth/db allowlist. Current-dev integration must regenerate artifacts and refresh behavioral evidence before publication.

## Stage 5: Acceptance, Contracts And Delivery

**Goal:** Deliver exact-head evidence and an operator-safe migration without claiming broader parity.
**Success Criteria:** Complete affected suites pass both backends, no new security findings, current hosted gates/reviews/human summary satisfied, deployment runbook preserves receipts.
**Tests:** All newly added suites, existing Workspace/provenance/creation, Sync, ordinary/session/Character import/export, native history/fork suites and migration/CI contracts.
**Status:** In Progress; local qualification and current-dev integration are recorded below. Requester summary supplied and verified; latest-dev local qualification passes, while fresh exact-head hosted review/check and normal-merge delivery gates remain open.

- [x] Add all new test files exactly once to appropriate dedicated matrices in `.github/workflows/ci.yml` and allowlists in `tldw_Server_API/tests/CI/test_required_workflow_contracts.py`. Check all matrix variants structurally and `Helper_Scripts/ci/check_shard_coverage.py`; do not hide an unrelated upstream baseline failure. Recheck after current-dev integration.
- [x] Generate OpenAPI via `Helper_Scripts/export_openapi_schema.py`, using disposable exact-CI dependency overlay if still needed; preserve shared venv. Update tracked `apps/tldw-frontend/lib/api/openapi.fingerprint.json` and generated API types using existing frontend command. Inspect closed strict request, error/header contract and unchanged legacy request. No frontend workflow adoption here. Exporter and installed TypeScript CLI were invoked directly because the wrapper drops the overlay; current-dev integration will require regeneration.
- [x] Run required backend matrix after virtualenv activation, using official PostgreSQL fixtures; retain its exact base and separate newer-base integration evidence below:

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

- [x] Run all existing Workspace suites plus affected Character/export/import/clone/chat API suites separately; record disjoint counts, skips, warnings and actual failures. Reproduce suspected upstream failures on pristine exact base, never call a partially failing matrix green. File-isolated compatibility is explicitly not combined-order certification. The planning baseline is historical, not implementation evidence.
- [x] Qualify independent spawned-process same-key, cross-Workspace owner capacity and same-Workspace quota contention, with positive lock evidence; prove abrupt exits before/after commit roll back both rows or retain both for replay. Retain thread-based tests as distinct evidence, not process certification. Recheck after current-dev integration.
- [x] Measure new receipt/schema/orchestrator/admission modules with pytest-cov in the focused matrix; target above 80% and inspect uncovered error/concurrency branches, not just the percentage. Run Ruff, changed Python compilation, full touched-scope Bandit (production and tests separately with assert exclusion), whitespace and OpenAPI/shard guards. Compare any test-template false positives to pristine base. No new touched-code findings may be deferred silently. The historical upstream RLS-classification blocker is resolved by upstream #3044 and the latest combined ratchet passes; no local exemption was added.
- [x] Document offline drain/migrate/restart, old cached-handle hazard, compatible-binary rollback boundary, permanent receipt privacy/backup retention, capacity increases and no key recycling. Rehearse upgrade/rollback/reopen and cached-writer limitations using official fixtures, not a version-rewound modern DB. PostgreSQL dump/restore evidence is logical, not physical/PITR or production certification.
- [ ] Complete independent whole-slice spec/security review; update assessment and execution task. Publish reviewable PR to dev, attach it to this chat, require requester-owned Change summary explaining implementation choices, current-head CI and fresh review inspection. Recheck actual dev before normal merge; rebase legitimate changes and require fresh evidence if needed. No admin bypass.
- [ ] After merge update execution task/parent/#2950 and canonical plan. Mark only Stage 2C delivered; Stage 2D/tool-profile/provisioning/Research adoption remain open until their own gates. Remove only this task's completed plan if repository policy requires, retaining links to its Git history.

Release-candidate operator procedure: `Docs/Operations/Workspace_Persona_Strict_Startup_Runbook_2026_09_27.md`. Process review found exception-unsafe child cleanup; exact second-launch RED preceded the stdlib ExitStack correction. The first corrected run exposed an existing concurrent schema-bootstrap trigger race, so initialization is serialized before testing overlapping strict acceptance. This does not certify concurrent bootstrap or old binaries. Final six SQLite process cases pass (four warnings, 28.68s); the official live-PG process/cached-writer subset passes seven cases (four warnings, 63.67s). Offline cached-writer evidence is deliberately not old-binary compatibility certification.

Backup review caught a false-positive rehearsal that restored over unchanged source. A controlled no-op reached restoration and passed that old test (clean RED); the correction restores into an absent target before all retained receipt/replay/tombstone/capacity assertions. The actual helper passes one corrected case (four warnings, 3.45s); the same controlled no-op is now rejected. No backup implementation changed. Focused SQLite coverage passes **625 tests, 265 deselected, four warnings**, 493.33s: 395/420 statements, **94%** across six new production modules; schema/transport/admission are 100%, store 86%, orchestrator 94%. Uncovered store branches are PostgreSQL/error paths that require separate backend evidence. This coverage precedes the backup-test correction but production source is identical; corrected backup evidence is separate. Full integrated dual-backend coverage/security and operator PostgreSQL physical-backup rehearsal remain open.

Publication preflight at `origin/dev` `3c9d97c56b29abc4c0396274b9560859aee06959`: upstream Persona Companion now owns SQLite73/PostgreSQL77, and upstream retry-backoff owns ADR051. Finish the frozen checkpoint, preserve a recovery ref, then rebase and assign receipts the next free migration numbers and ADR ID with all links/tests updated. Do not publish the colliding registry or treat pre-integration green evidence as current-head qualification.

Current-dev integration is now on `3c9d97c56b29abc4c0396274b9560859aee06959`, with receipts SQLite74/PostgreSQL78 and ADR056. The recovery ref retains `723faed87c`. The final registered migration/Companion/import-contract subset passes **46 tests, four warnings, no failures/skips**, 438.18s. Three earlier live Companion failures were test-harness pool closure, corrected by releasing only the cached connection before offline DDL; Companion behavior is unchanged. Mapping adaptation preserves legacy selection intent and uses the existing resolver, with the strict model/state grammar owned by core and public API reexports.

Fresh whole-branch source/spec/security review reported no actionable findings. A subsequent exported-contract probe found absent declarations for existing 409/410 outcomes: two RED tests preceded response-description-only repair and four GREEN controls (5.29s). The regenerated export has **2105 paths/3245 schemas**, fingerprint prefix `d02f5ee5adf4`; standard fingerprint check and ignored TypeScript regeneration pass. This bounded metadata correction is qualified separately from the earlier review.

The existing PostgreSQL logical dump/restore helper passes **one rehearsal, eight warnings**, 40.13s, using only the official isolated per-test database. The rehearsal deliberately removes fixture-owned receipts/live chat after the dump, then verifies exact restored receipt/policy catalog contents, ENABLE/FORCE RLS, live-key replay, permanent 410 and lifetime capacity. A controlled no-op restore fails the intended snapshot comparison (19.92s). This supersedes the earlier pending backup entry: it is not physical cluster/PITR, non-bypass post-restore behavioral or production deployment certification. Separate native RLS acceptance remains required.

The broad Workspace non-PG sweep completed **1249 passed, four failed, 25 deselected, eight warnings**, 906.00s. All four Activity index 500 failures independently reproduce on pristine exact dev (four failed/one passed, four warnings, 9.22s); no unrelated activity repair or green-sweep claim. Legacy Character completed **184 passed, three failed, five skipped, 19 warnings**, 956.66s; its three failure nodes independently match pristine dev. Current-source security/static guards pass without new findings: production Bandit is clean; test B105 and 19 Ruff findings match physical pristine base. The final 23-file required-PG acceptance/coverage and corrected compatibility runs remain in progress; Stage 5 delivery/AC2-5 stay open until final results, new human summary, PR/current-head CI and normal merge.

Bounded correction review verified two minor OpenAPI description errors: live-chat quota is 429 rather than 409, and 410 covers soft-deleted as well as hard-deleted targets. Three clean metadata REDs preceded the description-only repair and 429 declaration; five final-source metadata/route checks pass (four warnings, 3.64s), fingerprint prefix `6c384f248a590`, with standard check/types and closed request/header/status inspection passing. Source-only closure review found no remaining findings; runtime behavior did not change. The running 1395-case matrix predates only this metadata edit and must not be relabeled as a new 1396-case run.

Compatibility test doubles were minimally aligned with immutable ownership/real identity normalization (130-case session file passes) and the current owned active Persona lookup signature (56-case retry file passes; pristine exact dev also passes that file). All existing error/payload/order/image/no-write assertions remain. The first broad diagnostic used `/private/tmp` outside the default approved fixture DB roots and predates these corrections: **2679 passed, 176 failed, 136 errors, five skipped, 30 warnings**, 1939.14s, not qualification. The corrected 2996-case run uses the existing `$TMPDIR` root without changing path guards; it collected the retry double before its correction, whose final-source full-file gate is separate. Three unrelated World Book PostgreSQL stub failures and one legacy history-helper assertion reproduce on pristine exact dev (four failed/five passed, three warnings, 20.62s). Do not mistake invalid-harness errors or independently proven baseline failures for strict-startup defects or hide them with a green-suite claim.

Final required-backend acceptance on integrated base `3c9d97c56b`: **1383 passed, 12 intentional backend-specific skips, seven warnings, no failures/errors**, 6207.69s, across 23 files/1395 collected cases (`/private/tmp/persona-strict-final-dual-backend.log/xml`). Official live PostgreSQL was required and available. The skips are SQLite variants of PostgreSQL driver/row-lock tests and two PostgreSQL variants of SQLite-file DSR erasure, not unavailable-backend skips. Coverage is **419/428 statements, 97.90%** across six new modules; the nine uncovered orchestrator lines were inspected as defensive disappearance/mutation, exhausted-retry and integrity-error paths. This run predates only the final response-metadata correction, separately qualified above. It is not newer-dev hosted evidence.

The existing Workspace PostgreSQL complement passes **25 tests, 1256 deselected, five warnings, no failures/skips**, 185.73s (`/private/tmp/persona-strict-final-workspaces-postgres.log/xml`). It completes the PostgreSQL-selected nodes omitted by the non-PG sweep; overlapping counts must not be summed as unique tests. Shared temporary-directory cleanup warnings after the summaries are retained; no other agents' files were removed.

The corrected monolithic compatibility run was interrupted after profiling showed garbage collection dominated execution; it exited 143 without completed JUnit and remains partial diagnostic evidence only. Qualification now reruns **every file** in fresh serial pytest processes with the same supported `$TMPDIR`, unchanged guards/fixtures/timeouts, and an official 2996-node collection manifest. Per-file results under `/private/tmp/persona-strict-compat-files` require complete node-set reconciliation, fresh XML and matching process outcomes. This qualifies file-isolated compatibility, not combined-order/resource-lifecycle behavior. The one-off external runner is not shipped infrastructure and does not promise automatic parent-SIGTERM child cleanup.

Latest inspected dev `de7f453593` adds auth capability-disclosure (#3008) and encrypted scheduled-task message-store (#3039) changes. The tracked intersection with this slice is only the generated OpenAPI fingerprint; Persona runtime, receipt migration/RLS and lifecycle source are unchanged upstream. Finish frozen file-level qualification, retain a recovery ref, integrate the latest actual dev and refresh relevant Persona/auth/automation/OpenAPI gates before publishing. The logical backup qualification above supersedes the historical pending physical-backup entry; no physical/PITR or production certification is claimed.

File-isolated compatibility is complete: **2987 passed, four failed, five existing skips, no errors**, across **124 complete files/2996 unique nodes**. Independent reconciliation matches the exact official collection manifest without missing/duplicate nodes, verifies fresh JUnit timestamps and process outcomes, and matches the entire failure set to the pristine-base reproduction. The failures are one legacy history-helper assertion and three World Book PostgreSQL stubs, not new strict-startup failures. Skips retain the existing mock-roundtrip, heartbeat coordination, Resource Governor rate-limit, V3 import and removed-streaming-endpoint limitations. Every file was rerun; interrupted diagnostics were not used to omit nodes. Evidence is `/private/tmp/persona-strict-compat-files/{collection,results}.json` and the per-file log/XML pairs. This is not a green monolithic suite or combined-order/resource-lifecycle certification.

Final integration base is **`97da6c2dfec239b5739eed4116469f1fb5db7ed5`**, including #3008 auth, #3039 message store and #3028 VN UI recovery (no backend delta from the latter). Recovery ref `codex/persona-strict-startup-pre-auth-automation-rebase-20260927` retains qualified `b7d27a21ac`. Completed range-diff replays all 12 commits: ten patches identical, two changed only generated fingerprints. All **16 touched production files** remain byte-for-byte identical to the qualified checkpoint; no runtime, migration/RLS, workflow or UI adoption repair was added.

Fresh 13-file latest-base gate at `cad863a4e0`: **659 passed, one failed, three warnings, no skips/errors**, 617.56s (`/private/tmp/persona-strict-auth-automation-integration.log/xml`). All **570 Persona schema/HTTP/transport/admission/migration/forced-RLS/selected acceptance cases** pass with official live PostgreSQL required. Auth/import-boundary, embeddings and automation service/store cases also pass. The sole failure is `test_no_new_tenant_owned_table_lacks_an_rls_policy`, naming upstream `automation_messages`. A pristine exact-base archive independently reproduces the same node/table: **15 passed, one failed, four warnings**, 5.14s (`/private/tmp/persona-strict-pristine-dev-automation-rls.log/xml`). This remains an upstream policy-classification/coverage delivery blocker, not a green gate, a proven runtime data leak or permission to add an exemption in this Persona slice. No tests/policies were disabled or unrelated automation implementation changed.

Latest-source guards: production **16-file Bandit zero findings/errors**; **26-file test scan retains one physically verified baseline B105**; **19 existing Ruff findings, zero new** (only the existing B904 coordinates move after the added 429 metadata line). All 42 touched Python files compile; OpenAPI standard export/check and ignored TypeScript 7.13 regeneration pass. Combined contract is **2105 paths/3245 schemas**, fingerprint **`1e589f78cab60d13f993e96398757e01269a35c8094c902f95599eaef044604e`**. Closed request, three required fields, anchored required key and seven response statuses pass inspection. All 13 new test paths are assigned exactly once in five full matrices; shard guard **4875 files/new_uncovered=0** and actionlint pass (unavailable external ShellCheck/Pyflakes disabled). Bounded independent new-base source/artifact review has no actionable findings; it does not certify tests, CI or deployment. AC2/3 are locally qualified; AC4/5 and Stage 5 delivery remain open for the upstream gate, new requester-owned summary, draft PR/hosted checks/reviews and normal merge.

Reviewable **draft [PR #3041](https://github.com/rmusser01/tldw_server/pull/3041)** is published against `dev` and attached to this chat. Its body records both passing evidence and open baseline/harness limits. The new requester-owned Change summary was requested separately; PR #2963's summary is not reused. No merge, heartbeat, admin bypass or broader parity claim. Preserve this worktree for review; update parent/#2950 only after a normally gated merge.

Initial hosted `onboarding-docs-gate` at `935ccb3c87` failed two tests (210 passed): the new ADR056 lacked its tracked published mirror, and that same untracked path produced a revision-date warning in the strict build. The manifest assertion independently reproduces RED (one failure, six warnings, 3.04s). The existing `refresh_docs_published.sh` mechanically regenerates only the ADR056 mirror and published ADR index; both match their canonical sources byte-for-byte. With the mirror tracked, the complete unchanged Docs suite passes **212 tests, eight warnings, no failures/skips**, 62.60s (`/private/tmp/persona-strict-pr3041-docs-qualified.log/xml`), including its strict-build test. Public/private publication, onboarding command-boundary and endpoint-drift checks pass. No runtime, test, workflow, MkDocs warning or publication-policy changes; Bandit is not applicable to this Markdown/generated-artifact correction. Shared temporary-directory cleanup warnings are retained without deleting other agents' files. Fresh hosted checks on the correction remain required; upstream RLS, requester summary and normal merge gates remain open.

The publication repair is committed as `b4a9e14b86`. On that committed tree, the complete Docs suite again passes **212 tests, eight warnings, no failures/errors/skips**, 40.84s (`/private/tmp/persona-strict-pr3041-docs-committed.log/xml`). The standalone `mkdocs build --strict -f Docs/mkdocs.yml` passes in 10.80s with no warning/error records (`persona-strict-pr3041-docs-build-committed.log`); committing the mirror supplies its missing Git history without changing date-plugin settings. The earlier pre-commit standalone build exited zero but retained the new-path date warning and is not warning-free evidence. Actual dev remains `97da6c2dfe`; preceding hosted results are historical, and this documentation-only head still requires fresh hosted checks.

Requester supplied PR3041's new `Change summary` on 2026-09-28; it is saved verbatim and verified against the published body, with bot content preserved. It meets the canonical ownership gate by explaining retry-safe selection, durable owner-bound receipts against lost-response duplication, current access/lifecycle safeguards and dual-backend/offline/legacy/no-UI constraints. Published `87aeb7ef2a` has 70 successful checks and 26 skips; the cancelled license audit has a successful replacement, and the body update triggers a fresh queued audit. Actual `dev` is now `5f9815293bdd72c9b013aed80aad95daca04f6b2`; GitHub reports DIRTY, so previous base qualification is not latest-dev evidence. Integration/conflict resolution, RLS requalification and fresh reviews remain open. No runtime changes or merge; this tracking-only checkpoint will be published with the next qualified integration rather than resetting hosted CI for bookkeeping. Stage 2D and broader parity remain open.

Latest-dev integration on 2026-09-28 rebases onto actual **`3d102e0d31667d6d1fc2a74402dafc2416fed3d4`**, including release sync, route authentication, SSE helpers, history metadata handling, bootstrap validation and upstream RLS coverage repair. Recovery ref `codex/persona-strict-startup-pre-release-sync-rebase-20260928` preserves `884204ad3c`, including the requester-summary record. All 17 commits replay: 15 patches identical and two changed only generated fingerprints. All 16 Persona production files remain byte-identical to the qualified checkpoint. No implementation, migration, workflow-policy or UI change was added in resolving the two fingerprint conflicts.

Upstream PR #3044 classifies `automation_messages` as a separate per-owner SQLite store and tightens coverage to require FORCE RLS. This supersedes the earlier classification blocker; the current combined ratchet passes, and the Persona receipt table remains non-exempt. The old temporary schema overlay was unavailable. A fresh disposable Pydantic 2.13.5 overlay with its declared core 2.46.5 dependency matches the checked-in fingerprint on pristine exact dev, without changing shared dependencies. Standard combined export/check and ignored TypeScript 7.13 regeneration pass: 2105 paths/3245 schemas, fingerprint **`192ea22ffdd0aa4a6967b530e4f8975de5a9be352b25ded42e26dc2f0b97252f`**. The initial shared-Pydantic export and incompatible core-version probe are diagnostic only, not qualified artifacts.

At `5ad442bf64`, the native non-bypass receipt-RLS probe passes with the official isolated PostgreSQL fixture. The 16-file focused integration matrix completes with **929 passed, one failed, one intentional backend skip, 294 warnings, no errors**, 661.37s. All Persona acceptance/migration/admission/HTTP/transport and tenant ratchets pass. The skip tests SQLite connection lifetime on the PostgreSQL variant. The sole failure is the native history API stale-selection case, which passes on pristine exact dev: one passed, seven warnings, 9.93s. This is not an upstream baseline: the module-local `history_api` fixture constructs Workspace rows for writer `test_user` but requests as owner `1`, so the required current Workspace access guard correctly rejects before stale-selection validation. Independent review confirms the fixture mismatch. The minimal fixture correction uses authenticated `client_id="1"`, retains the general seed fixture and every runtime/status/no-write/admission assertion, and passes the exact node: one passed, seven warnings, 9.96s. The fresh full integration rerun also includes the 274 strict request/hash cases; earlier failing evidence is not relabeled green.

Fresh final-source 17-file integration qualification passes **1204 tests, one intentional backend skip, 295 warnings, no failures/errors**, 1155.69s, exit zero (`/private/tmp/persona-strict-release-sync-qualified-integration.log/xml`). JUnit reconciliation retains every original 931 node and adds exactly 274 strict request/hash cases, without duplicates; the full 24-case history API file passes with every original assertion. Official isolated live PostgreSQL is required and available. The single skip is the SQLite connection-lifetime case on its PostgreSQL variant, not unavailable-backend evidence. The complete Docs suite passes **212 tests, eight warnings**, 26.75s, and full changed shard/PostgreSQL-policy contracts pass **72 tests, four warnings**, 12.12s; both have zero failures/errors/skips. These runs do not relabel the earlier failing matrix or historical broader compatibility baselines green.

Production 16-file Bandit has zero findings/errors; the refreshed 27-file test scan has only the physically pristine-base-matched B105 `--cov` literal. Ruff has 19 current findings versus 23 on physical pristine dev, with zero new finding identities; all 43 touched Python files compile. The corrected fixture alone has clean Bandit/Ruff and independent bounded diff review with no actionable findings. All 13 new test paths are assigned exactly once across five full matrices, shard guard reports 4879 files/new_uncovered=0, and ADR mirrors remain byte-identical. One-off inspection probes initially assumed the wrong shard, key grammar and a nonexistent prior Ruff report filename; corrected inspection uses actual assignments, the declared anchored grammar and a fresh physical-base comparison, without changing source or CI. All 16 Persona production files still match the recovery checkpoint byte-for-byte. Actual dev remains `3d102e0d31` at pre-publication inspection. Fresh exact-head hosted delivery remains open; CodeRabbit's draft skip is not review evidence, the human summary stays verbatim, and Stage 2D/broader parity remain open.

## Post-Review Corrections

Qodo reviewed exact head `64936b4363` with three bug and seven rule findings (comment `5875731215`). Six valid findings are corrected: single explicit-global session compatibility; owner-verified stored scope for id-only Workspace turns, including native history; receipt-aware privacy SQL moved into DB_Management; content-free storage-error logging; the test helper return annotation; and endpoint parameter/result/error documentation. Workspace session support is not added. Explicit mismatches, current parent/Persona admission and stale-target fail-closed behavior remain guarded.

Four findings are dispositioned with source-backed rationale: the approved core-owned grammar/hash avoids an API dependency in private orchestration; the store-local InputError subclass avoids the existing central-exceptions import cycle; supplied missing/deleted targets must not silently create or downgrade a chat; and private-state/fail-fast assertions are necessary to prove no backend refresh or SQL at outer preflight. The legacy helper does not clear an unresolved id. No-id legacy creation remains supported.

Scope RED produced seven failures/11 passes, then 18 targeted cases passed on SQLite/live PostgreSQL. Logging RED/GREEN each exercised the missing safe event. Additional history RED produced one failure/one pass before reusing the already verified scope; all 25 history API cases then passed. The first six-file run completed 481 passed/two failed/16 warnings in 742.24s: both failures were new deleted-target setup missing the required optimistic deletion version, not admission failures. Corrected setup retains all 404/no-effect assertions; four target cases pass. Historical failures are not relabeled green.

Fresh final-source six-file qualification passes **484 tests, 16 warnings, no failures/errors/skips**, 994.89s, exit zero (`/private/tmp/persona-strict-pr3041-qodo-final-qualified.log/xml`), with official isolated live PostgreSQL required. Every original node remains except the single history test deliberately replaced by scoped/id-only variants, with no duplicate nodes. Final privacy/Admin three-file qualification passes **30 tests, three intentional SQLite-file-erasure-on-PostgreSQL skips, four warnings**, 73.68s, including corrupt-probe, missing-file and atomic rollback cases. The complete Docs suite passes **212 tests, eight warnings**, 58.82s. Separate history/target subsets overlap the full matrix and are not summed as unique coverage.

Production 17-file Bandit has zero findings/errors; the 27-file test scan retains only the physically pristine-base-matched B105 `--cov` literal. All 44 changed Python files compile; Ruff has 19 existing findings and no new physical-base identities. Standard schema export/check and ignored TypeScript regeneration pass at **2105 paths/3245 schemas**, fingerprint **`cb01670263838fb731e2272e0799a0276f574995f593f43ba4d1c1d98d21e155`**. Shard coverage remains 4879 files/new_uncovered=0; whitespace and bounded independent final-diff reviews pass. Actual dev remains `3d102e0d31`, requester summary is verbatim, and no workflow/policy/UI or shared dependency change is included. Publication must require fresh exact-head hosted CI/review and normal merge; Stage 5, AC4/5, Stage 2D and broader parity remain open.

After all 70 hosted checks passed on `b8e17aa4fa`, strict merge gating reported BEHIND. Actual `dev` advanced through AuthNZ cause-chain logging PR #3047, Chat test repair PR #3046 and backlog/test correction PR #3048 to **`414a9619cc8ae9446262ee54980017d874d2a739`**. Recovery ref `codex/persona-strict-startup-pre-authnz-chat-rebase-20260928` preserves that reviewed green head. All 19 commits rebased cleanly, and `git range-diff` confirms every patch identical; the six-file upstream delta has no overlap with Persona changes. Actual `dev` was rechecked unchanged after the integration run. The former green checks are historical, not current-head evidence.

Fresh nine-file required-PostgreSQL integration on the rebased source completed **497 passed, three intentional SQLite-file-eraser-on-PostgreSQL skips, 76 warnings, zero failures/errors**, 1505.96s, process exit zero (`/private/tmp/persona-pr3041-authnz-chat-rebase-qualified.log/xml`). It covers strict startup acceptance/API, privacy, Persona admission, ordinary routing/history and the newly merged AuthNZ sanitizer/bootstrap and Chat NetworkError regressions. JUnit confirms 500 unique cases and the three explicit skip reasons. The first attempt was deliberately interrupted after 113 passes/22 setup errors because the dedicated PostgreSQL container had stopped and sandboxed local TCP was unavailable; restarting only `tldw_persona_pr3041_20260928` and using normal local-test access made the official fixture pass. No product/test change or backend skip substituted for that failure.

Production 17-file Bandit again has zero findings/errors; changed Python compiles, shard coverage reports 4880 files/new_uncovered=0, whitespace passes. The complete unchanged Docs suite passes **212 tests, six warnings, no failures/errors/skips**, 93.40s (`/private/tmp/persona-pr3041-authnz-chat-rebase-docs.log/xml`). The initial OpenAPI check used an expired disposable overlay and reported a diagnostic mismatch from shared Pydantic 2.11.7; restoring only the temporary Pydantic 2.13.5/core 2.46.5 overlay yields a passing standard fingerprint check with unchanged tracked fingerprint `cb01670263838fb731e2272e0799a0276f574995f593f43ba4d1c1d98d21e155`. Shared dependencies remain unchanged. Publish only after recording this evidence, then require fresh exact-head hosted CI and review, a fresh actual-dev check, and normal merge. Stage 5/AC4/5 and Stage 2D/broader parity remain open.

All 70 hosted checks passed on the `9f4990f47f` head before upstream VZ boot-stall PR #3022 advanced actual `dev` to **`0f9e6917cef2deb5da36d6fc2f85b4457f0ce884`**. Its 17-file delta is confined to VZ tests/scripts, Sandbox docs and TASK-13243.10, without Persona/API/schema/workflow intersection. Recovery `codex/persona-strict-startup-pre-vz-rebase-20260928` preserves the prior green head. All 20 commits replayed without conflict or changed patches in `git range-diff`; no Persona implementation, test, contract or runbook content changed. Actual `dev` was rechecked unchanged before publication.

The preceding full 500-node integration remains valid for identical Persona/AuthNZ/Chat source, but is historical for the new base. Bounded current-base integration passes **10 tests, 12 warnings, no failures/skips** (official isolated live PostgreSQL acceptance plus merged AuthNZ/Chat regressions; `/private/tmp/persona-pr3041-vz-rebase-smoke.log/xml`). Complete Docs passes **212 tests, six warnings**, 55.55s. Production 17-file Bandit reports zero findings/errors; changed Python compiles, whitespace and shard coverage (4881 files/new_uncovered=0) pass. Qualified-overlay OpenAPI fingerprint check passes unchanged. The previous head's 70 hosted successes must not be treated as new-head checks; publish and require fresh CI/review and normal up-to-date merge. Stage 5/AC4/5 and Stage 2D/broader parity stay open.

Quick Ingest/web-extraction PR #3050 advanced actual `dev` to **`e5186a28d9b4f09af60bc8fd53073c1b6d5c0603`** while the `5ff89eec1a` head had 69 successful checks and only the container-build aggregate queued without a runner. That head was BEHIND regardless of the aggregate result. Recovery `codex/persona-strict-startup-pre-quick-ingest-rebase-20260928` preserves it. The seven-file upstream delta is confined to Quick Ingest UI and web-extraction logging/service/tests; all 21 Persona commits replayed without conflicts or changed patches in `git range-diff`. No Persona runtime, test, schema, workflow or runbook content changed. Current-base focused verification passes **135 tests, 11 warnings, no failures/skips** in 110.40s, including official isolated live-PostgreSQL startup and the new ingestion regressions (`/private/tmp/persona-pr3041-quick-ingest-rebase.log/xml`). Production 17-file Bandit has zero findings/errors, changed Python compiles, qualified-overlay OpenAPI fingerprint and whitespace pass, and shard coverage reports 4882 files/new_uncovered=0. The prior full 500-node matrix and hosted checks are historical; the rebased head still requires fresh exact-head review/CI and normal up-to-date merge. Stage 5/AC4/5 and Stage 2D/broader parity remain open.

Current `dev` advanced through backlog/MCP/Playground PR #3049 and task-closure PR #3052 to **`0da68530e80c713ed3a323a741998e1fed37e3e9`** after all 70 checks passed on historical `3527191ba6`. Recovery `codex/persona-strict-startup-pre-ratchets-rebase-20260929` preserves that head. The 21-file delta adds MCP verbatim-argument handling, a history-link repair and required CI ratchets; no Persona/schema intersection. All 22 prior patches replayed identically without conflicts. The new required CI contracts/source-pinning/coercion ratchets plus upstream MCP regressions pass **53 tests, six warnings**, 13.73s. The complete strict-startup API file passes **84 tests, four warnings, no failures/skips**, 52.83s, using the official isolated live-PostgreSQL fixture (`/private/tmp/persona-pr3041-ratchets-rebase-{gates,startup}.log/xml`). Production 17-file Bandit has zero findings/errors, changed Python compiles, OpenAPI fingerprint and whitespace pass, and shard coverage reports 4884 files/new_uncovered=0. Actual `dev` remains the tested base. Publish and require fresh checks/review for the new head and normal merge; Stage 5/AC4/5 and Stage 2D/broader parity remain open.

## Planning Review And Evidence

The entries below preserve planning-time evidence and approval history. Their Not Started statuses are historical; the current execution statuses are the five stage headings above.

Source audit and the unmodified 10-file baseline are tracked under TASK-13245.6. Current fixture code already supplies the snapshot-loader credential override; do not edit it merely to replay a historical repair. Independent plan review must cover mutation completeness, PostgreSQL lock order, native cascade integration, privacy, strict validation and activation boundaries. Requester review of this refreshed design is required before runtime implementation.

The baseline completed with **490 passed, one failed, one skipped, six warnings** (2226.21 seconds), not a green matrix. The native PostgreSQL hard-delete cascade retry fails because its enumeration leaves an implicit read transaction open after the injected child-delete failure. Exact reproduction and a temporary owned-read diagnostic confirm the cause; no source/test repair was made. TASK-13245.7 records the prerequisite and caller-transaction constraints. The skip is the PostgreSQL driver-transaction case running on SQLite. Persona prompt/memory tests passed unmodified, so the historical credential-fixture blocker is no longer current. Local logs and the diagnostic limitation are recorded in the refresh.

Independent source-backed review corrected endpoint admission timing, forced PostgreSQL owner RLS, replay versus FK lock ordering, and receipt-bound restore/Sync closure protection. Focused re-review found no remaining material findings for those corrections. Markdown link checks, example AST parsing and an in-memory documented-model probe passed; these do not certify implemented concurrency or runtime behavior. Bandit is not applicable to this Markdown/Backlog-only planning change.

Requester-approved follow-up amendments address five later findings: strict startup must reject caller transactions and commit before response; raw/decoded text needs strict-only Unicode/byte bounds; service admission must use immutable DB ownership; strict errors must translate typed legacy unavailable failures; session scope coverage must be explicit. The plan now retains global-only session preparation/preview/complete-v2 and tests deliberate rejection, with Workspace generation on ordinary chat. TASK-13245.6 records refreshed example probes and review evidence. All five runtime stages remain Not Started; TASK-13245.7 remains the separate native cascade prerequisite.

TASK-13245.7 repair verification: local branch `codex/persona-workspace-cascade-retry` settles only cascade-owned enumeration/message-page reads and preserves caller transactions and durable closure. Current 15-file affected verification is 714 passed, four SQLite-only driver-case skips, no failures; scoped lint/compile/Bandit and independent review are recorded in the task and refresh. This resolves the reproduced retry defect locally, not through a hosted merge. Refresh the latest execution base and integration evidence before lifecycle qualification; all five strict-startup stages remain Not Started.

Fresh documentation validation: 17 local links, 10 Python AST blocks, 32 accepted/60 rejected model cases, omission/null/hash invariants and six typed error mappings pass. Independent review found that copying validation msg/loc could echo state input or a surrogate unknown key. The documented sanitizer now uses fixed message/type and allowlisted bounded locations; 10 error-detail/privacy/UTF-8 probes pass. No other material findings were reported. No application/test/config/workflow edits or runtime test certification; raw-stream, commit, live-RLS/concurrency and admission gates remain in the five Not Started stages.
