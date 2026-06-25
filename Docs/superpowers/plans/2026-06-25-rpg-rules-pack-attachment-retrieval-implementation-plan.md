# RPG Rules-Pack Attachment Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let RPG campaigns and sessions attach user-owned media items or media collections as rules references, then use scoped retrieval to augment RPG rules lookup and session context with citations and optional grounded answers.

**Architecture:** Keep rules content in existing Media/RAG stores, store only normalized source references in RPG campaign/session JSON columns, validate every dereference through existing ownership/readability checks, and run lookup through injectable async retrieval and answer-generation adapters shared by REST and MCP.

**Tech Stack:** FastAPI, Pydantic v2, SQLite through `CharactersRAGDB`, existing Media DB and Collections DB dependencies, existing RAG retrieval executor surfaces, async chat generation through `perform_chat_api_call_async`, Loguru, pytest, Bandit, MCP Unified module APIs.

---

## Reference Inputs

- Approved design: `Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md`
- Existing RPG runtime plan: `Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md`
- Backlog task for design and handoff: `TASK-12029`
- Current repository storage: `tldw_Server_API/app/core/DB_Management/RPG_DB.py`
- Current RPG service: `tldw_Server_API/app/core/RPG/service.py`
- Current rules lookup: `tldw_Server_API/app/core/RPG/rules/lookup.py`
- Current rules content models: `tldw_Server_API/app/core/RPG/rules/content_packs.py`
- Current context builder: `tldw_Server_API/app/core/RPG/context.py`
- Current REST endpoint: `tldw_Server_API/app/api/v1/endpoints/rpg.py`
- Current REST schemas: `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py`
- Current MCP module: `tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py`
- RAG request schema reference: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- RAG scoped retrieval reference: `tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py`
- RAG collection readiness reference: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Chat generation reference: `tldw_Server_API/app/core/Chat/chat_service.py`

## Public Contract

Rules refs use this server-owned shape when serialized into `linked_rules_pack_refs_json` or `active_rules_pack_refs_json`:

```json
{
  "ref_id": "media_item:42",
  "source_type": "media_item",
  "source_id": 42,
  "display_name": "Player Handbook SRD Notes",
  "enabled": true,
  "created_at": "2026-06-25T00:00:00Z",
  "updated_at": "2026-06-25T00:00:00Z",
  "metadata": {
    "source_label": "user"
  }
}
```

Supported `source_type` values for this implementation are `media_item` and `media_collection`. A disabled ref stays attached and is skipped during retrieval. A readable collection with no ready media items remains a valid ref and produces diagnostics instead of a hard lookup failure.

Rules lookup response shape:

```json
{
  "query": "How does advantage work?",
  "mode": "lookup",
  "results": [
    {
      "origin": "user_provided",
      "text": "short retrieved snippet",
      "citation": {
        "source_type": "media_item",
        "source_id": 42,
        "source_title": "Player Handbook SRD Notes",
        "source_url": null,
        "license": null,
        "attribution": null,
        "trust_level": "user_provided",
        "content_hash": "sha256:abc123def456",
        "snippet_id": "media:42:chunk:7"
      },
      "score": 0.78
    }
  ],
  "answer": null,
  "answer_status": "not_requested",
  "answer_citation_ids": [],
  "diagnostics": {
    "linked_rules_pack_count": 2,
    "enabled_rules_pack_count": 1,
    "ready_media_item_count": 1,
    "retrieval_result_count": 1,
    "bundled_citation_count": 1,
    "skipped_refs": []
  }
}
```

`mode="answer"` runs the same lookup first. If no user-provided snippets are returned, answer generation is skipped and `answer_status` is `no_evidence`. If snippets exist, generation uses the existing async chat service and only cites `snippet_id` values returned by lookup.

## File Structure

Create:

- `tldw_Server_API/app/core/RPG/rules/refs.py`: typed ref normalization, server timestamp preservation, replacement-list validation, result dataclasses.
- `tldw_Server_API/app/core/RPG/rules/retrieval.py`: async protocols and concrete retrieval adapter for scoped media/collection lookups.
- `tldw_Server_API/app/core/RPG/rules/answering.py`: grounded answer generator over lookup snippets using `perform_chat_api_call_async`.
- `tldw_Server_API/tests/RPG/test_rpg_rules_refs.py`: ref normalization, timestamp, duplicate, and replacement semantics.
- `tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py`: retrieval adapter and source validation behavior.
- `tldw_Server_API/tests/RPG/test_rpg_rules_answering.py`: answer generation statuses, citation filtering, provider errors.

Modify:

- `tldw_Server_API/app/core/RPG/models.py`: add typed lookup item fields and serialize-friendly citation fields while preserving public dataclass compatibility.
- `tldw_Server_API/app/core/RPG/rules/content_packs.py`: extend `RuleLookupItem` and `RuleLookupResult` for `origin`, answer fields, and diagnostics.
- `tldw_Server_API/app/core/RPG/rules/lookup.py`: merge bundled citation-only entries with scoped retrieval results.
- `tldw_Server_API/app/core/RPG/context.py`: use lookup-mode retrieved snippets in bounded session context.
- `tldw_Server_API/app/core/RPG/service.py`: expose campaign/session ref list and replace operations, async lookup, and async context build.
- `tldw_Server_API/app/core/DB_Management/RPG_DB.py`: add public campaign getter and optimistic whole-list ref replacement methods.
- `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py`: add rules-pack ref request/response schemas and lookup `mode`.
- `tldw_Server_API/app/api/v1/endpoints/rpg.py`: add campaign/session ref endpoints, wire media dependencies, convert lookup/context handlers to async.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py`: add ref-management tools and update lookup schema.
- `tldw_Server_API/Config_Files/privilege_catalog.yaml`: add route and MCP privilege metadata for new endpoints/tools when route snapshots require it.
- `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`: regenerate if endpoint metadata changes.
- `tldw_Server_API/app/core/RPG/README.md`: document user-provided rules refs, lookup behavior, and licensing stance.

Update existing tests:

- `tldw_Server_API/tests/RPG/test_rpg_db.py`
- `tldw_Server_API/tests/RPG/test_rpg_service.py`
- `tldw_Server_API/tests/RPG/test_rpg_rules_context.py`
- `tldw_Server_API/tests/RPG/test_rpg_api.py`
- `tldw_Server_API/tests/RPG/test_rpg_mcp_module.py`

## Implementation Notes

- Use `source .venv/bin/activate` before Python, pytest, helper scripts, or Bandit.
- Use a separate Backlog.md implementation task before runtime code edits begin; link this plan and `TASK-12029`.
- Whole-list replacement is the only write form for rules refs in this implementation.
- Ref writes require `expected_version` matching `RPGCampaign.version` or `RPGSession.version`.
- Repository writes increment the campaign/session `version` and `updated_at` in the same transaction.
- Idempotency scopes:
  - Campaign refs: `campaign:{campaign_id}:rules_pack_refs`
  - Session refs: `session:{session_id}:rules_pack_refs`
- Idempotency replays return the stored response when request hash matches and raise `RPGConflictError("idempotency_key_payload_mismatch")` when it differs.
- Server code owns `created_at` and `updated_at`; client-supplied timestamp fields are ignored.
- Preserve `created_at` for refs whose identity and source ID match an existing ref during replacement.
- Do not copy extracted rules prose into RPG tables.
- Do not add broad RAG fallback, web fallback, or cross-user source discovery.
- REST auth must include both RPG permissions and `media.read` on endpoints that dereference or retrieve attached media sources.
- Built-in citations stay citation-only with score `0.0`; generated answers never cite bundled citation-only rows as evidence.

---

### Task 1: Add Rules-Pack Ref Model and Repository Replacement

**Files:**

- Create: `tldw_Server_API/app/core/RPG/rules/refs.py`
- Modify: `tldw_Server_API/app/core/DB_Management/RPG_DB.py`
- Modify: `tldw_Server_API/app/core/RPG/models.py`
- Create: `tldw_Server_API/tests/RPG/test_rpg_rules_refs.py`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_db.py`

- [ ] **Step 1: Write failing unit tests for ref normalization**

Add `test_rpg_rules_refs.py` with these cases:

- `normalize_rules_pack_refs_accepts_media_item_and_collection`
- `normalize_rules_pack_refs_rejects_unknown_source_type`
- `normalize_rules_pack_refs_rejects_non_positive_source_id`
- `normalize_rules_pack_refs_rejects_duplicate_ref_identity`
- `normalize_rules_pack_refs_ignores_client_timestamps`
- `normalize_rules_pack_refs_preserves_created_at_for_existing_ref`
- `normalize_rules_pack_refs_updates_updated_at_for_existing_ref`
- `normalize_rules_pack_refs_limits_metadata_to_json_object`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_refs.py -v
```

Confirm these tests fail because `refs.py` does not exist.

- [ ] **Step 2: Implement typed ref normalization**

Add these dataclasses and helpers in `rules/refs.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

RulesPackSourceType = Literal["media_item", "media_collection"]


@dataclass(frozen=True, slots=True)
class RulesPackRef:
    ref_id: str
    source_type: RulesPackSourceType
    source_id: int
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RulesPackRefReplacementResult:
    refs: list[RulesPackRef]
    version: int
    replayed: bool = False
```

Implementation requirements:

- `normalize_rules_pack_ref_payloads(payloads, existing_refs, now)` accepts a list of dictionaries and returns `list[RulesPackRef]`.
- `source_type` must be exactly `media_item` or `media_collection`.
- `source_id` must be an integer greater than zero.
- `ref_id` is server-derived as `f"{source_type}:{source_id}"`.
- Duplicate `ref_id` values raise `RPGValidationError("duplicate_rules_pack_ref")`.
- `display_name` defaults to `ref_id` when omitted or blank.
- `enabled` defaults to `True`.
- `metadata` defaults to `{}` and must be a JSON object.
- The helper must serialize refs through `rules_pack_ref_to_dict(ref)` using ISO-8601 UTC strings.
- The helper must deserialize existing dicts through `rules_pack_ref_from_dict(data)`.

- [ ] **Step 3: Write failing repository tests for whole-list replacement**

In `test_rpg_db.py`, add these cases:

- `test_get_campaign_returns_owner_scoped_campaign`
- `test_replace_campaign_rules_pack_refs_requires_expected_version`
- `test_replace_campaign_rules_pack_refs_increments_version`
- `test_replace_campaign_rules_pack_refs_replays_idempotency_key`
- `test_replace_campaign_rules_pack_refs_rejects_idempotency_payload_mismatch`
- `test_replace_session_rules_pack_refs_requires_expected_version`
- `test_replace_session_rules_pack_refs_increments_version`
- `test_replace_session_rules_pack_refs_replays_idempotency_key`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_db.py -v
```

Confirm the new tests fail because the repository methods are missing.

- [ ] **Step 4: Add repository methods**

Add public methods to `RPGRepository` with these signatures:

```text
get_campaign(owner_user_id: int, campaign_id: int) -> RPGCampaign
replace_campaign_rules_pack_refs(owner_user_id: int, campaign_id: int, rules_pack_refs: list[dict[str, Any]], expected_version: int, idempotency_key: str, request_payload_hash: str, source_type: str) -> RulesPackRefReplacementResult
replace_session_rules_pack_refs(owner_user_id: int, session_id: int, rules_pack_refs: list[dict[str, Any]], expected_version: int, idempotency_key: str, request_payload_hash: str, source_type: str) -> RulesPackRefReplacementResult
```

Behavior:

- Call `_validate_source_type(source_type)` before idempotency logic.
- Load the current campaign/session inside the same transaction.
- Compare `expected_version` to the row `version`.
- Normalize new refs with the current row refs and `now`.
- Update only the JSON ref column, `version = version + 1`, and `updated_at = now`.
- Use `WHERE owner_user_id = ? AND id = ? AND version = ?` and require `rowcount == 1`.
- Store response JSON with `refs` and `version`.
- Replay returns `RulesPackRefReplacementResult(refs=stored_refs, version=stored_version, replayed=True)`.
- Missing owner-scoped rows raise `RPGNotFoundError`.
- Stale versions raise `RPGConflictError("stale_rules_pack_ref_version")`.

- [ ] **Step 5: Run task verification**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_refs.py tldw_Server_API/tests/RPG/test_rpg_db.py -v
```

---

### Task 2: Add Service-Level Source Validation and Session Copy Semantics

**Files:**

- Modify: `tldw_Server_API/app/core/RPG/service.py`
- Modify: `tldw_Server_API/app/core/RPG/rules/refs.py`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_service.py`

- [ ] **Step 1: Write failing service tests**

Add these service tests with fake media and collection validators:

- `test_create_session_copies_campaign_rules_refs_when_request_omits_refs`
- `test_create_session_uses_explicit_empty_rules_refs`
- `test_replace_campaign_rules_pack_refs_validates_each_enabled_source`
- `test_replace_session_rules_pack_refs_validates_each_enabled_source`
- `test_replace_rules_pack_refs_allows_disabled_unreadable_source_without_dereference`
- `test_replace_rules_pack_refs_rejects_unreadable_media_item`
- `test_replace_rules_pack_refs_allows_empty_readable_collection`
- `test_list_campaign_rules_pack_refs_returns_current_version`
- `test_list_session_rules_pack_refs_returns_current_version`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v
```

Confirm the new tests fail because the service methods and validator protocol are missing.

- [ ] **Step 2: Add source validation protocols**

Add service-facing protocol/dataclasses in `rules/refs.py`:

```python
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RulesPackSourceValidation:
    ref_id: str
    readable: bool
    display_name: str | None
    ready_media_ids: list[int] = field(default_factory=list)


class RulesPackSourceValidator(Protocol):
    async def validate_media_item(self, owner_user_id: int, media_id: int) -> RulesPackSourceValidation:
        raise NotImplementedError

    async def validate_media_collection(self, owner_user_id: int, collection_id: int) -> RulesPackSourceValidation:
        raise NotImplementedError
```

Task 3 adds the REST validator adapter and Task 6 wires MCP construction. Unit tests can inject fakes.

- [ ] **Step 3: Add service methods**

Extend `RPGService` constructor to accept optional `rules_source_validator`. Add service methods with these signatures:

```text
list_campaign_rules_pack_refs(campaign_id: int) -> RulesPackRefReplacementResult
list_session_rules_pack_refs(session_id: int) -> RulesPackRefReplacementResult
replace_campaign_rules_pack_refs(campaign_id: int, refs: list[dict[str, Any]], expected_version: int, idempotency_key: str, source_type: str = "user") -> RulesPackRefReplacementResult
replace_session_rules_pack_refs(session_id: int, refs: list[dict[str, Any]], expected_version: int, idempotency_key: str, source_type: str = "user") -> RulesPackRefReplacementResult
```

Behavior:

- Normalize before validation to get stable `ref_id` values.
- Validate enabled refs only.
- Media item validation rejects unreadable, deleted, trash, or missing items through `RPGValidationError("rules_pack_source_unreadable")`.
- Media collection validation rejects unreadable or missing collections through the same domain error.
- Empty readable collections pass validation.
- Request hashes must include normalized refs and `expected_version`.
- Use repository replacement methods after validation passes.

- [ ] **Step 4: Update session creation copy behavior**

Change `RPGService.create_session()` so:

- If the request omits `active_rules_pack_refs`, it loads the campaign and copies `campaign.linked_rules_pack_refs`.
- If the request supplies `active_rules_pack_refs=[]`, the session starts with no refs.
- If the request supplies a non-empty list, the service validates and stores that list.
- Existing tests for session creation continue to pass.

- [ ] **Step 5: Run task verification**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_service.py -v
```

---

### Task 3: Add REST Schemas, Endpoints, and Authorization Contract

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/rpg_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rpg.py`
- Modify: `tldw_Server_API/Config_Files/privilege_catalog.yaml`
- Modify: `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_api.py`

- [ ] **Step 1: Write failing REST schema and endpoint tests**

Add or update API tests:

- `test_campaign_rules_pack_refs_get_requires_campaign_read_and_media_read`
- `test_campaign_rules_pack_refs_put_requires_campaign_manage_and_media_read`
- `test_session_rules_pack_refs_get_requires_session_read_and_media_read`
- `test_session_rules_pack_refs_put_requires_session_manage_and_media_read`
- `test_replace_campaign_rules_pack_refs_returns_version_and_refs`
- `test_replace_session_rules_pack_refs_returns_version_and_refs`
- `test_replace_rules_pack_refs_rejects_stale_version_with_409`
- `test_replace_rules_pack_refs_replays_idempotency_key`
- `test_rules_lookup_accepts_lookup_mode`
- `test_rules_lookup_accepts_answer_mode`
- `test_rules_lookup_rejects_unknown_mode`
- `test_endpoint_scope_catalog_includes_rules_pack_routes`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v
```

Confirm the new tests fail because schemas/routes/scope metadata are missing.

- [ ] **Step 2: Add Pydantic schemas**

Add:

```python
class RPGRulesPackRefInput(BaseModel):
    source_type: Literal["media_item", "media_collection"]
    source_id: int = Field(gt=0)
    display_name: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RPGRulesPackRefsReplaceRequest(BaseModel):
    refs: list[RPGRulesPackRefInput] = Field(default_factory=list, max_length=50)
    expected_version: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=128)


class RPGRulesPackRefResponse(BaseModel):
    ref_id: str
    source_type: Literal["media_item", "media_collection"]
    source_id: int
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class RPGRulesPackRefsResponse(BaseModel):
    refs: list[RPGRulesPackRefResponse]
    version: int
    replayed: bool = False
```

Update `RPGRulesLookupRequest`:

```python
class RPGRulesLookupRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: Literal["lookup", "answer"] = "lookup"
    provider: str | None = None
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=600, ge=64, le=2000)
```

- [ ] **Step 3: Add concrete media/collection validator for REST**

In `endpoints/rpg.py`, create a small adapter class that receives the authenticated user's media DB and collections DB:

- `validate_media_item()` calls `media_db.get_media_by_id(media_id, include_deleted=False, include_trash=False)`.
- `validate_media_collection()` loads the collection through the same read path used by collections endpoints and resolves ready media IDs using statuses `completed` and `skipped_existing`.
- The validator returns a display name when the source has a title/name.
- The validator must not return media IDs from unreadable, deleted, trash, or missing items.

Keep this adapter in the endpoint layer unless shared MCP construction needs the same concrete class; if both REST and MCP need it, move it to `tldw_Server_API/app/core/RPG/rules/retrieval.py`.

- [ ] **Step 4: Add endpoints**

Add constants:

```python
RPG_CAMPAIGNS_READ = "rpg.campaigns.read"
MEDIA_READ = "media.read"
```

Add routes:

- `GET /api/v1/rpg/campaigns/{campaign_id}/rules-packs`
- `PUT /api/v1/rpg/campaigns/{campaign_id}/rules-packs`
- `GET /api/v1/rpg/sessions/{session_id}/rules-packs`
- `PUT /api/v1/rpg/sessions/{session_id}/rules-packs`

Endpoint dependency requirements:

- Campaign GET: `RequirePermission(RPG_CAMPAIGNS_READ)` and `RequirePermission(MEDIA_READ)`
- Campaign PUT: `RequirePermission(RPG_CAMPAIGNS_MANAGE)` and `RequirePermission(MEDIA_READ)`
- Session GET: `RequirePermission(RPG_SESSIONS_READ)` and `RequirePermission(MEDIA_READ)`
- Session PUT: `RequirePermission(RPG_SESSIONS_MANAGE)` and `RequirePermission(MEDIA_READ)`
- Lookup endpoint: keep `RequirePermission(RPG_RULES_READ)` and add `RequirePermission(MEDIA_READ)`

Convert lookup and context endpoints that touch async service methods to `async def`.

- [ ] **Step 5: Update privilege catalog and snapshot**

Run the project helper after endpoint metadata is added:

```bash
source .venv/bin/activate && python Helper_Scripts/update_privilege_registry_snapshot.py
```

Then run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v
```

- [ ] **Step 6: Run task verification**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v
```

---

### Task 4: Add Scoped Retrieval-Backed Lookup

**Files:**

- Create: `tldw_Server_API/app/core/RPG/rules/retrieval.py`
- Modify: `tldw_Server_API/app/core/RPG/rules/content_packs.py`
- Modify: `tldw_Server_API/app/core/RPG/rules/lookup.py`
- Modify: `tldw_Server_API/app/core/RPG/service.py`
- Create: `tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_rules_context.py`

- [ ] **Step 1: Write failing lookup and retrieval tests**

Add retrieval tests:

- `test_retrieval_skips_disabled_refs`
- `test_retrieval_resolves_media_item_to_allowed_media_ids`
- `test_retrieval_resolves_collection_ready_items_only`
- `test_retrieval_reports_empty_collection_without_error`
- `test_retrieval_reports_no_ready_sources_without_broad_fallback`
- `test_retrieval_passes_allowed_media_ids_to_executor`
- `test_retrieval_maps_documents_to_user_provided_lookup_items`
- `test_retrieval_uses_stable_snippet_ids`

Update context/lookup tests:

- `test_lookup_returns_user_results_before_bundled_citations`
- `test_lookup_keeps_bundled_citations_score_zero`
- `test_lookup_returns_diagnostics_for_skipped_refs`
- `test_lookup_does_not_call_retriever_when_query_is_blank`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py -v
```

Confirm the new tests fail because retrieval-backed lookup is missing.

- [ ] **Step 2: Extend lookup dataclasses**

In `content_packs.py`, update or add serializable dataclasses:

```python
@dataclass(frozen=True, slots=True)
class RuleLookupCitation:
    source_type: str
    source_id: int | None
    source_title: str
    source_url: str | None
    license: str | None
    license_url: str | None
    attribution: str | None
    trust_level: str
    content_hash: str
    snippet_id: str


@dataclass(frozen=True, slots=True)
class RuleLookupItem:
    origin: Literal["user_provided", "bundled_citation"]
    text: str
    citation: RuleLookupCitation
    score: float


@dataclass(frozen=True, slots=True)
class RuleLookupResult:
    query: str
    mode: Literal["lookup", "answer"]
    results: list[RuleLookupItem]
    answer: str | None
    answer_status: str
    answer_citation_ids: list[str]
    diagnostics: dict[str, Any]
```

Update bundled citation creation to map existing `RuleCitation` into `RuleLookupCitation` with `source_type="bundled_rules_citation"` and `source_id=None`.

- [ ] **Step 3: Add retrieval protocols and concrete adapter**

Create:

```python
class RulesRetriever(Protocol):
    async def retrieve(
        self,
        *,
        owner_user_id: int,
        query: str,
        refs: list[RulesPackRef],
        max_results: int,
    ) -> RulesRetrievalResult:
        raise NotImplementedError
```

Concrete adapter requirements:

- Validate and resolve refs through `RulesPackSourceValidator`.
- Build a deduplicated list of ready media IDs.
- Return early with diagnostics when no media IDs are ready.
- Call the existing retrieval executor with `source="media_db"` semantics and `allowed_media_ids` set to resolved IDs.
- Do not pass note IDs, web sources, or provider-generated synthetic content.
- Map retrieved documents to `RuleLookupItem(origin="user_provided", text=chunk_text, score=score)`.
- Use `snippet_id` from the retriever when present; otherwise derive `f"media:{media_id}:chunk:{chunk_index}"`.
- Truncate single snippet text to a conservative bound, such as 1,500 characters, before returning to lookup.

- [ ] **Step 4: Update `RulesLookupService`**

Constructor:

```python
class RulesLookupService:
    def __init__(
        self,
        *,
        retriever: RulesRetriever | None = None,
        answer_generator: RulesAnswerGenerator | None = None,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator
```

Lookup method:

```python
async def lookup(
    self,
    *,
    owner_user_id: int,
    adapter_key: str,
    query: str,
    linked_rules_pack_refs: list[dict[str, Any]],
    mode: Literal["lookup", "answer"] = "lookup",
    answer_options: RulesAnswerOptions | None = None,
) -> RuleLookupResult:
    raise NotImplementedError
```

Behavior:

- Blank query raises `RPGValidationError("rules_query_required")`.
- Normalize refs and count linked/enabled refs.
- User-provided retrieval results come before bundled citations.
- Bundled citations are always included when an adapter has citation metadata.
- Retrieval failures produce `answer_status="retrieval_error"` only for answer mode; lookup mode returns bundled citations plus diagnostics unless the error is a validation/auth error.
- Validation/auth errors from unreadable refs propagate as domain errors.
- Diagnostics include linked count, enabled count, ready media count, retrieval result count, bundled citation count, and skipped refs.

- [ ] **Step 5: Update service async lookup path**

Change `RPGService.lookup_rules()` to `async def` and pass `owner_user_id`, `mode`, and answer options to `RulesLookupService`.

Update REST and MCP call sites after this change. Temporary test fakes can call `await service.lookup_rules(session_id=1, query="advantage", mode="lookup")`.

- [ ] **Step 6: Run task verification**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py tldw_Server_API/tests/RPG/test_rpg_service.py -v
```

---

### Task 5: Add Grounded Answer Mode and Async Context Building

**Files:**

- Create: `tldw_Server_API/app/core/RPG/rules/answering.py`
- Modify: `tldw_Server_API/app/core/RPG/rules/lookup.py`
- Modify: `tldw_Server_API/app/core/RPG/context.py`
- Modify: `tldw_Server_API/app/core/RPG/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rpg.py`
- Create: `tldw_Server_API/tests/RPG/test_rpg_rules_answering.py`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_rules_context.py`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_api.py`

- [ ] **Step 1: Write failing answer-mode tests**

Add:

- `test_answer_mode_returns_not_requested_for_lookup_mode`
- `test_answer_mode_returns_no_evidence_without_user_snippets`
- `test_answer_mode_calls_chat_service_with_grounded_prompt`
- `test_answer_mode_extracts_openai_content`
- `test_answer_mode_filters_unknown_citation_ids`
- `test_answer_mode_returns_generation_error_on_provider_failure`
- `test_answer_mode_uses_request_provider_and_model`
- `test_answer_mode_uses_default_temperature_and_token_bounds`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_answering.py -v
```

Confirm the tests fail because `answering.py` is missing.

- [ ] **Step 2: Implement answer generator**

Create:

```python
@dataclass(frozen=True, slots=True)
class RulesAnswerOptions:
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 600


@dataclass(frozen=True, slots=True)
class RulesAnswerResult:
    answer: str | None
    answer_status: str
    citation_ids: list[str]


class RulesAnswerGenerator:
    async def generate(
        self,
        *,
        query: str,
        evidence: list[RuleLookupItem],
        options: RulesAnswerOptions,
    ) -> RulesAnswerResult:
        raise NotImplementedError
```

Concrete generator requirements:

- Use `perform_chat_api_call_async`.
- Use `tldw_Server_API.app.core.Workflows.adapters._common.extract_openai_content`.
- Prompt the model to answer only from provided snippets and return JSON with `answer` and `citation_ids`.
- Pass:
  - `messages=[{"role": "user", "content": user_prompt}]`
  - `system_message=<grounding instructions>`
  - `api_provider=options.provider`
  - `model=options.model`
  - `temperature=options.temperature`
  - `max_tokens=options.max_tokens`
  - `stream=False`
- Parse JSON when returned; if plain text is returned, use the text as `answer` and cite all evidence IDs used in the prompt.
- Filter citation IDs to the evidence `snippet_id` set.
- Return `generation_error` when the chat service raises a `ChatAPIError` or `ChatProviderError`.

- [ ] **Step 3: Wire answer mode into lookup**

In `RulesLookupService.lookup()`:

- `mode="lookup"` sets `answer=None`, `answer_status="not_requested"`, and `answer_citation_ids=[]`.
- `mode="answer"` with no user-provided evidence sets `answer_status="no_evidence"`.
- `mode="answer"` with evidence calls the generator.
- Generator result statuses pass through as `answered` or `generation_error`.
- Generated answer text is not included in context builder output.

- [ ] **Step 4: Convert context builder to async evidence inclusion**

Change:

```python
async def build_context(self, session_id: int, query: str | None = None, max_chars: int = MAX_RPG_CONTEXT_CHARS) -> SessionContext
```

Context behavior:

- Call rules lookup with `mode="lookup"`.
- Include user-provided snippets within existing context bounds.
- Include bundled citations as citation lines only.
- Include diagnostics with retrieval result count and skipped refs.
- If lookup fails due to source validation, include a diagnostics entry and continue with session state context.
- Do not call answer generation.

- [ ] **Step 5: Update service and REST context endpoints**

Change `RPGService.build_context()` to async and update REST/MCP call sites to await it.

Add API tests:

- `test_context_endpoint_includes_retrieved_rules_snippets`
- `test_context_endpoint_does_not_generate_answer`
- `test_context_endpoint_reports_rules_lookup_diagnostics`

- [ ] **Step 6: Run task verification**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_answering.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py tldw_Server_API/tests/RPG/test_rpg_api.py -v
```

---

### Task 6: Add MCP Tooling, Documentation, and Final Verification

**Files:**

- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py`
- Modify: `tldw_Server_API/app/core/RPG/README.md`
- Modify: `tldw_Server_API/tests/RPG/test_rpg_mcp_module.py`
- Modify: `backlog/tasks/task-12029 - Design-RPG-rules-pack-attachment-and-retrieval-backed-lookup.md`
- Modify: implementation Backlog.md task created before code execution

- [ ] **Step 1: Write failing MCP tests**

Add:

- `test_rpg_mcp_tool_list_includes_rules_pack_ref_tools`
- `test_rpg_mcp_rules_pack_ref_tools_have_read_write_metadata`
- `test_rpg_mcp_rules_pack_ref_replace_validates_expected_version`
- `test_rpg_mcp_rules_lookup_accepts_answer_mode`
- `test_rpg_mcp_context_build_awaits_async_service`
- `test_rpg_mcp_read_tools_require_media_read_for_attached_refs`
- `test_rpg_mcp_write_tools_require_media_read_for_source_validation`

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_mcp_module.py -v
```

Confirm tests fail because MCP tools are missing.

- [ ] **Step 2: Add MCP tools**

Add tool names:

- `rpg.campaigns.rules_packs.get`
- `rpg.campaigns.rules_packs.replace`
- `rpg.sessions.rules_packs.get`
- `rpg.sessions.rules_packs.replace`

Tool metadata:

- GET tools are read tools with `rpg.campaigns.read` or `rpg.sessions.read` and `media.read`.
- Replace tools are write tools with `rpg.campaigns.manage` or `rpg.sessions.manage` and `media.read`.
- Replace schemas include `expected_version`, `idempotency_key`, and `refs`.
- Lookup schema includes `mode`, `provider`, `model`, `temperature`, and `max_tokens`.

Service construction:

- Use the same ChaChaNotes DB path handling as current RPG tools.
- Inject media/collection validation only when MCP context exposes user media and collection DB paths.
- If MCP context lacks media DB access, ref dereference tools return a structured error rather than silently treating refs as empty.

- [ ] **Step 3: Update README**

Add a concise section to `tldw_Server_API/app/core/RPG/README.md`:

- Ref source types and ownership behavior.
- Campaign-to-session copy behavior.
- Whole-list replacement and version/idempotency expectations.
- Lookup/answer mode behavior.
- Licensing/privacy statement: user-provided rules content stays in user media stores; bundled adapters remain mechanics metadata and citations.

- [ ] **Step 4: Run focused RPG suite**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG -v
```

- [ ] **Step 5: Run privilege catalog checks**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v
```

- [ ] **Step 6: Run Bandit on touched Python scope**

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/RPG tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py -f json -o /tmp/bandit_task_12029_rpg_rules_packs.json
```

Open the JSON report and fix any new finding in touched code before marking the implementation task complete.

- [ ] **Step 7: Run formatting and diff checks**

```bash
git diff --check
```

If the project environment has Ruff installed:

```bash
source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/RPG tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py tldw_Server_API/tests/RPG
```

- [ ] **Step 8: Update Backlog.md records**

Record in the implementation task:

- Plan link.
- Modified files.
- Focused pytest commands and results.
- Privilege catalog check result.
- Bandit report path and result.
- Known skips with reason.
- Final summary.

Record in `TASK-12029`:

- This plan path.
- The implementation task ID.
- Final summary that design and implementation planning are complete.

---

## Review Checklist Before Runtime Code Starts

- [ ] The implementation task exists in Backlog.md and links to this plan.
- [ ] Task 1 starts with failing tests before repository changes.
- [ ] Ref writes use existing `version` fields and do not add new DB tables.
- [ ] Media and collection refs are validated through user-scoped DB access.
- [ ] Lookup never falls back to unscoped RAG or web search.
- [ ] Answer mode uses existing chat provider governance and never fabricates citation IDs.
- [ ] Context builder uses lookup evidence only and does not call answer generation.
- [ ] MCP and REST surfaces expose the same semantics.
- [ ] Bandit scope is limited to touched Python files and every new finding is fixed.
