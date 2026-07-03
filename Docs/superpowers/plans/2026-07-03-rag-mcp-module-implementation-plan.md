# RAG MCP Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proper `rag.*` MCP module exposing curated RAG capabilities, source health, retrieval, and grounded answer generation without introducing a `research.*` layer.

**Architecture:** Extract the route-local RAG request/capability/source-health seams into transport-neutral service helpers, then add a focused MCP `RagModule` that maps strict MCP arguments into `UnifiedRAGRequest`, executes the shared RAG pipeline, and compacts responses into citation-aware MCP payloads. HTTP and MCP must share request resolution, source health, response mapping, query quota, and usage-accounting helpers.

**Tech Stack:** Python 3, FastAPI route helpers, Pydantic schemas, MCP Unified `BaseModule`, existing RAG service helpers, pytest, Bandit.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md`
- Design Backlog: `TASK-12118`
- Implementation Plan Backlog: `TASK-12119`

## File Structure

Create:

- `tldw_Server_API/app/core/RAG/rag_service/transport.py` - transport-neutral RAG helper seams shared by HTTP and MCP.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py` - MCP `RagModule` implementation and local contract helpers.
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py` - unit coverage for extracted RAG transport helpers.
- `tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py` - unit coverage for schemas, argument mapping, compaction, control calls, and domain errors.
- `tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py` - config/default-registration coverage for the new module.

Modify:

- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` - delegate route-local helper logic to `rag_service.transport`.
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py` - update imports/monkeypatch targets after helper extraction.
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py` - update helper monkeypatch targets after source-health extraction.
- `tldw_Server_API/Config_Files/mcp_modules.yaml` - add default-enabled `rag` module only after control parity is implemented.
- `tldw_Server_API/Config_Files/mcp_tool_categories.yaml` - add `rag.search: search` and `rag.answer: rag_generation`.
- `tldw_Server_API/Config_Files/resource_governor_policies.yaml` - add `mcp.search` and `mcp.rag_generation` policy mappings.
- `tldw_Server_API/app/core/MCP_unified/module_surface.py` - classify `rag` as read-only data access.
- `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py` - catalog discovery regression coverage for curated `rag.*` workflow catalogs.
- `tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py` - HTTP `/tools/execute` wrapper compatibility coverage for `rag.*`.
- `tldw_Server_API/tests/MCP_unified/test_mcp_knowledge_rbac.py` - executable regression proving `knowledge.search` remains FTS/source-module fan-out.
- `tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py` - source fan-out defaults coverage if needed for `knowledge.search` regression.
- `Docs/MCP/mcp_tool_catalogs.md` - document a curated RAG/library workflow catalog using the existing DB-backed catalog service.
- `Docs/MCP/Unified/User_Guide.md` - document `rag.*` tools and catalog/security boundaries.
- `Docs/MCP/Unified/Client_Snippets.md` - add JSON-RPC snippets for `rag.search` and `rag.answer`.

Do not modify in the first slice:

- `slides.generate.from_rag` behavior, except optional regression coverage if a shared helper change affects it.
- Any `research.*` module, facade, or workflow.
- Batch, streaming, feedback, note-writing, export, ingestion, web fallback, URL scraping, image search, or video search surfaces.

---

### Task 1: Extract Shared RAG Transport Helpers

**Files:**
- Create: `tldw_Server_API/app/core/RAG/rag_service/transport.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py`

- [x] **Step 1: Write failing helper extraction tests**

Create `tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py` with tests for:

```python
def test_build_unified_pipeline_kwargs_preserves_search_agent_defaults(monkeypatch):
    request = UnifiedRAGRequest(query="default behavior check")
    kwargs = build_unified_pipeline_kwargs(
        request=request,
        db_paths={"media_db_path": None, "notes_db_path": None, "character_db_path": None, "kanban_db_path": None},
        media_db=None,
        chacha_db=None,
        current_user=None,
        search_agent_setting_fn=lambda env_key, config_key: {"SEARCH_QUERY_CLASSIFICATION": "true"}.get(env_key),
    )
    assert kwargs["enable_query_classification"] is True
    assert kwargs["sources"] == ["media_db"]


def test_build_source_health_payload_uses_existing_paths_without_leaking_paths(monkeypatch):
    payload = build_source_health_payload(
        current_user=SimpleNamespace(id=1, id_int=1),
        existing_source_db_paths_fn=lambda *_args, **_kwargs: {"media_db": "/secret/media.db"},
        media_db_uses_non_file_storage_fn=lambda: False,
    )
    assert [entry.source_id for entry in payload.sources][:2] == ["media_db", "notes"]
    assert "/secret" not in str(payload)
```

Also update existing tests to import `build_unified_pipeline_kwargs`, `resolve_existing_source_db_paths`, and source-health helpers from `tldw_Server_API.app.core.RAG.rag_service.transport` instead of monkeypatching `rag_unified`.

- [x] **Step 2: Run tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py \
  -q
```

Expected: FAIL because `rag_service.transport` does not exist and existing tests still target endpoint-local helpers.

- [x] **Step 3: Add `rag_service.transport`**

Move or wrap these route-local helpers from `rag_unified.py` into `transport.py`:

```python
def search_agent_setting(env_key: str, config_key: str) -> str | None: ...
def build_unified_pipeline_kwargs(...) -> dict[str, Any]: ...
def build_standard_request_bundle(...) -> ResolvedRequestBundle: ...
def resolve_source_health_user_id(...) -> str | None: ...
def resolve_existing_source_db_paths(...) -> dict[str, str]: ...
def media_db_uses_non_file_storage() -> bool: ...
def build_source_health_source_sets(...) -> tuple[set[Any], set[Any]]: ...
def build_source_health_payload(...) -> KnowledgeSourceHealthResponse: ...
async def log_rag_queries_for_org_context(..., units: int = 1) -> None: ...
def build_rag_capabilities_payload() -> dict[str, Any]: ...
```

Implementation notes:

- Keep `UnifiedRAGRequest`, `ResolvedRequestBundle`, and `build_request_bundle` behavior unchanged.
- `build_rag_capabilities_payload()` should contain the existing `/capabilities` response construction moved out of the route without changing field names.
- `build_source_health_payload()` should call `build_source_health_entries()` and must not instantiate source databases or retrievers.
- `log_rag_queries_for_org_context()` should preserve best-effort behavior and never raise.
- `rag_unified.py` should import these helpers and expose thin route wrappers only.

- [x] **Step 4: Run extraction tests to verify pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/RAG/rag_service/transport.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py
git commit -m "refactor: share rag transport helpers"
```

---

### Task 2: Define Pure MCP RAG Contract Helpers

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py`
- Create/Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py`

- [x] **Step 1: Write failing contract tests**

Add tests that import private helpers from `rag_module.py` until the module API stabilizes:

```python
def test_mcp_sources_accept_aliases_but_return_canonical_ids():
    request, metadata = _build_mcp_rag_request(
        "rag.search",
        {"query": "q", "sources": ["media", "notes"]},
    )
    assert request.sources == ["media_db", "notes"]
    assert metadata["sources_explicit"] is True
    assert metadata["sources_requested"] == ["media_db", "notes"]


def test_mcp_sources_omitted_tracks_implicit_default():
    request, metadata = _build_mcp_rag_request("rag.search", {"query": "q"})
    assert request.sources == ["media_db"]
    assert metadata["sources_explicit"] is False


def test_advanced_is_rejected():
    with pytest.raises(ValueError, match="advanced"):
        _build_mcp_rag_request("rag.search", {"query": "q", "advanced": {"debug_mode": True}})


def test_sql_source_is_rejected_in_stage_one():
    with pytest.raises(ValueError, match="sql"):
        _build_mcp_rag_request("rag.search", {"query": "q", "sources": ["sql"]})


def test_unknown_and_internal_sources_are_rejected():
    for source in ("unknown", "claims"):
        with pytest.raises(ValueError, match=source):
            _build_mcp_rag_request("rag.search", {"query": "q", "sources": [source]})


def test_compact_response_truncates_documents_and_preserves_citations():
    response = UnifiedRAGResponse(
        query="q",
        documents=[{"id": "d1", "content": "abcdef", "metadata": {}, "score": 0.9}],
        citations=[{"id": "c1"}],
        chunk_citations=[{"id": "chunk-1"}],
        metadata={"hard_citations": {"coverage": 0.5}, "knowledge_trust": {"state": "grounded"}},
    )
    payload = _compact_rag_response(
        response,
        mode="search",
        request_metadata={"sources_requested": ["media_db"], "sources_explicit": True},
        max_documents=1,
        max_content_chars=3,
    )
    assert payload["documents"][0]["content"] == "abc"
    assert payload["documents"][0]["content_truncated"] is True
    assert payload["chunk_citations"] == [{"id": "chunk-1"}]
    assert payload["metadata"]["hard_citation_coverage"] == 0.5
    assert payload["metadata"]["sources_used"] == ["media_db"]
    assert payload["metadata"]["sources_unavailable"] == []
    assert payload["metadata"]["documents_truncated"] is False
    assert payload["metadata"]["max_documents"] == 1
    assert payload["metadata"]["max_content_chars"] == 3
```

- [x] **Step 2: Run tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py \
  -q
```

Expected: FAIL because `rag_module.py` and helpers do not exist.

- [x] **Step 3: Add contract helpers in `rag_module.py`**

Implement only pure helpers first:

```python
_CANONICAL_PUBLIC_SOURCES = (
    "media_db", "notes", "chats", "characters", "kanban", "prompts", "world_books", "dictionaries",
)
_DEFERRED_STAGE_ONE_SOURCES = {"sql"}
_SEARCH_MODES = ("hybrid", "vector", "fts")
_PROFILES = ("fast", "balanced", "accuracy")


def _sources_were_explicit(arguments: dict[str, Any]) -> bool:
    return "sources" in arguments and arguments.get("sources") is not None


def _build_mcp_rag_request(tool_name: str, arguments: dict[str, Any]) -> tuple[UnifiedRAGRequest, dict[str, Any]]:
    if "advanced" in arguments:
        raise ValueError("advanced is not supported by rag.* first slice")
    sources_explicit = _sources_were_explicit(arguments)
    sources = normalize_sources_public(arguments.get("sources"))
    deferred = sorted(set(sources) & _DEFERRED_STAGE_ONE_SOURCES)
    if deferred:
        raise ValueError(f"unsupported Stage 1 source: {deferred[0]}")
    payload = {
        "query": _required_string(arguments, "query", max_length=20000),
        "sources": sources,
        "search_mode": _enum(arguments.get("search_mode", "hybrid"), _SEARCH_MODES),
        "top_k": _bounded_int(arguments.get("top_k", 10), minimum=1, maximum=50),
        "min_score": _bounded_float(arguments.get("min_score", 0.0), minimum=0.0, maximum=1.0),
        "rag_profile": _optional_enum(arguments.get("rag_profile"), _PROFILES),
        "enable_generation": tool_name == "rag.answer",
        "enable_citations": True,
        "enable_chunk_citations": bool(arguments.get("include_chunk_citations", True)),
        "include_metadata": True,
        "include_sources": bool(arguments.get("include_documents", True)),
    }
    if tool_name == "rag.search":
        payload["enable_generation"] = False
    request = UnifiedRAGRequest(**payload)
    return request, {
        "sources_explicit": sources_explicit,
        "sources_requested": list(sources),
        "allow_partial": bool(arguments.get("allow_partial", False)),
        "max_documents": _bounded_int(arguments.get("max_documents", 6), minimum=0, maximum=20),
        "max_content_chars": _bounded_int(arguments.get("max_content_chars", 2000), minimum=0, maximum=8000),
    }
```

Add `_compact_rag_response()`, `_answer_status()`, `_domain_error_payload()`, `_mcp_safe_search_agent_overrides()`, and `_unsupported_scope_warnings_or_error()` as pure functions. Keep all payloads JSON-serializable and avoid provider secrets, paths, and prompts.

- [x] **Step 4: Run contract tests to verify pass**

Run the pytest command from Step 2.

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py
git commit -m "feat: add rag mcp contract helpers"
```

---

### Task 3: Implement `RagModule` Tools And Execution

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py`

- [ ] **Step 1: Write failing module tests**

Add async tests for:

```python
@pytest.mark.asyncio
async def test_rag_module_exposes_four_strict_tools():
    module = RagModule(ModuleConfig(name="rag"))
    tools = {tool["name"]: tool for tool in await module.get_tools()}
    assert set(tools) == {"rag.capabilities", "rag.source_health", "rag.search", "rag.answer"}
    for tool_name in ("rag.capabilities", "rag.source_health", "rag.search", "rag.answer"):
        assert tools[tool_name]["inputSchema"]["additionalProperties"] is False
    assert tools["rag.answer"]["metadata"]["category"] == "rag_generation"


@pytest.mark.asyncio
async def test_rag_search_executes_shared_pipeline_without_generation(monkeypatch):
    module = RagModule(ModuleConfig(name="rag"))
    calls = {}

    async def fake_pipeline(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            documents=[{"id": "d1", "content": "Evidence", "metadata": {}, "score": 0.9}],
            query="q",
            metadata={"hard_citations": {"coverage": 1.0}},
            timings={"retrieval": 1.2},
            citations=[],
            chunk_citations=[],
            generated_answer=None,
            errors=[],
        )

    monkeypatch.setattr(rag_module_impl, "unified_rag_pipeline", fake_pipeline)
    ctx = RequestContext(request_id="r1", user_id="1", db_paths={"media": "media.db", "chacha": "notes.db"})
    out = await module.execute_tool("rag.search", {"query": "q", "sources": ["media"]}, context=ctx)
    assert out["ok"] is True
    assert out["mode"] == "search"
    assert "answer" not in out
    assert calls["enable_generation"] is False
    assert calls["sources"] == ["media_db"]
```

Also add tests for:

- `rag.answer` includes `answer.status`.
- `rag.answer` maps sufficiently cited/grounded output to `answer.status="answered"`.
- `rag.answer` maps generated output with no citations to `partial` or `abstained`, never `answered`.
- `rag.answer` maps weak evidence, failed citation coverage, or low trust metadata to `partial` or `abstained`, never `answered`.
- `rag.source_health` returns safe canonical source entries and does not consume RAG query quota.
- `rag.source_health`, `rag.search`, and `rag.answer` authorize each normalized source independently instead of treating `media.read` as global source access.
- `tools/call` enforces `tools.execute:rag.capabilities`, `tools.execute:rag.source_health`, `tools.execute:rag.search`, and `tools.execute:rag.answer` independently.
- `rag.source_health`, `rag.search`, and `rag.answer` call a transport-neutral `rbac_rate_limit("rag.search")`-equivalent posture check before source access; `rag.capabilities` does not.
- `rag.source_health`, `rag.search`, and `rag.answer` enforce TokenScopeGuard/API-key-scope-equivalent read access using existing MCP scope normalization; a context with incompatible API-key/tool scopes fails closed before source access or pipeline execution.
- supported source scopes from `context.metadata` are applied before retrieval: `media_id`/`media_ids` into `include_media_ids`, `note_id`/`note_ids` into `include_note_ids`, and workspace/session metadata into `workspace_id`/safe request metadata where the RAG pipeline supports it. Tests must prove scoped contexts cannot run unscoped retrieval for `media_db` or `notes`.
- `rag.search` and `rag.answer` call the RAG query quota checker before pipeline execution and the best-effort usage logger after successful execution.
- `rag.capabilities` does not consume RAG query quota.
- `rag.answer` propagates only LLM-provider-safe request metadata into generation/pipeline kwargs and strips authorization, API keys, raw config paths, raw prompts, and debug payloads.
- explicit unavailable source returns `ok:false` unless `allow_partial=true`.
- `sources=["sql"]` returns a Stage 1 rejection rather than entering the pipeline.
- explicit requests with unsupported `conversation_id`, `character_id`, or `prompt_id` scopes fail closed with `unsupported_scope` or `source_unavailable`.
- implicit/default source selection filters sources whose scopes cannot be enforced and reports warnings instead of running unscoped retrieval.
- MCP execution forces out-of-scope Search-Agent and research defaults off even when config/profile defaults enable `enable_research_loop`, `search_url_scraping`, `enable_image_search`, `enable_video_search`, web fallback, or similar external provider behavior.
- RAG-domain pipeline exceptions become structured `ok:false` payloads.

- [ ] **Step 2: Run module tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py \
  -q
```

Expected: FAIL because `RagModule.get_tools()` and `execute_tool()` are not implemented.

- [ ] **Step 3: Implement `RagModule`**

Implement:

```python
class RagModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> dict[str, bool]: ...
    async def get_tools(self) -> list[dict[str, Any]]: ...
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any: ...
```

Tool metadata:

- `rag.capabilities`: `metadata={"category": "utility", "readOnlyHint": True}`
- `rag.source_health`: `metadata={"category": "search", "readOnlyHint": True}`
- `rag.search`: `metadata={"category": "search", "readOnlyHint": True}`
- `rag.answer`: `metadata={"category": "rag_generation", "readOnlyHint": True}`

Execution rules:

- `rag.capabilities` returns `build_rag_capabilities_payload()`.
- `rag.source_health` returns `build_source_health_payload()` using `context.user_id` and `context.db_paths` where possible.
- `rag.search` and `rag.answer` call `_build_mcp_rag_request()`, `build_standard_request_bundle()`, `unified_rag_pipeline(**bundle.pipeline_kwargs)`, `rag_result_to_response(rag_result_from_unified_search_result(result))`, then `_compact_rag_response()`.
- Add an injectable `_McpRagControls` seam, or equivalent small dependency object, so tests can assert control calls without invoking real AuthNZ/Billing services. It must cover protocol-category rate limiting, RBAC-rate-limit posture, TokenScopeGuard/API-key-scope posture, RAG query quota, and usage logging.
- `rag.source_health`, `rag.search`, and `rag.answer` call `_enforce_rag_rbac_rate_limit(context, resource="rag.search")` before source access. The default implementation must reuse the existing `auth_deps.enforce_rbac_rate_limit` SQL/policy selection path by extracting a transport-neutral helper if needed; do not duplicate the RBAC rate-limit SQL in `rag_module.py`.
- `rag.source_health`, `rag.search`, and `rag.answer` call `_require_mcp_rag_read_scope(context, tool_name)` before source access. The default implementation must reuse existing MCP scope helpers such as `ToolExecutionSecurity.api_key_allows(context, is_write=False)`, `ToolExecutionSecurity.scope_allows_tool_name()`, and protocol-provided tool authorization helpers when available. Do not invent new RAG-specific scope strings, and do not parse scope metadata ad hoc.
- `rag.source_health`, `rag.search`, and `rag.answer` call `_authorize_sources(context, sources, sources_explicit, allow_partial)` before source access. The helper should check module availability and tool/source permission per normalized source, using `tools.execute:<source tool>` where applicable and `media.read` only for `media_db`.
- Use this Stage 1 source authorization map:
  - `media_db`: media module enabled, `media.read` or wildcard entitlement, and permission for the relevant `media.*` read/search path.
  - `notes`: notes module enabled and permission for the relevant `notes.*` read/search path.
  - `chats`: chats module enabled and permission for the relevant `chats.*` read/search path; fail closed when requested scopes cannot be enforced.
  - `characters`: characters module enabled and permission for the relevant `characters.*` read/search path; fail closed when requested scopes cannot be enforced.
  - `kanban`: kanban module enabled and permission for the relevant `kanban.*` read/search path.
  - `prompts`: prompts module enabled and permission for the relevant `prompts.*` read/search path; fail closed when requested scopes cannot be enforced.
  - `world_books`: backing module/source entitlement available and enabled; explicit requests fail closed otherwise, implicit defaults filter with a warning.
  - `dictionaries`: backing module/source entitlement available and enabled; explicit requests fail closed otherwise, implicit defaults filter with a warning.
- `_authorize_sources()` returns the filtered canonical source list plus `sources_unavailable` and warnings. Explicitly denied or unavailable sources fail closed unless `allow_partial=true`; implicit/default sources are filtered with warnings.
- `_unsupported_scope_warnings_or_error()` must inspect `context.metadata`/persona scope for `conversation_id`, `character_id`, and `prompt_id`. When these scopes are supplied with explicit affected sources, return `ok:false` with `unsupported_scope` or `source_unavailable`; when sources are implicit/default, filter affected sources and add warnings.
- `_apply_supported_source_scopes()` must apply supported item/workspace scopes after argument validation and before `build_standard_request_bundle()`/pipeline execution. Normalize `context.metadata["media_id"]`, `context.metadata["media_ids"]`, `context.metadata["note_id"]`, and `context.metadata["note_ids"]` into `UnifiedRAGRequest.include_media_ids` and `UnifiedRAGRequest.include_note_ids`, intersecting with any explicit include lists rather than widening them. Normalize `context.metadata["workspace_id"]` and `context.session_id`/`context.metadata["session_id"]` into `workspace_id` and safe request metadata only where the current RAG schema/pipeline enforces them. Invalid scoped ids fail closed for explicit affected sources and filter/warn for implicit defaults.
- `_mcp_safe_search_agent_overrides()` must be merged into pipeline kwargs after `build_standard_request_bundle()` so profile/config defaults cannot enable out-of-scope external behavior. Force these keys false/disabled in Stage 1: `enable_research_loop`, `search_url_scraping`, `enable_image_search`, `enable_video_search`, `enable_discussion_search` when it would call external providers, web fallback flags, and any equivalent research/web provider toggles present in the payload.
- `_safe_generation_metadata(context, request_info)` must provide the only request metadata passed to generation/provider-facing kwargs for `rag.answer`. Keep bounded scalar identifiers such as request/session/correlation ids and canonical source ids; redact or omit authorization headers, API keys, raw config paths, raw prompts, full context metadata, and debug/provider payloads.
- `rag.search` and `rag.answer` call an injectable `_check_rag_query_quota(context, units=1)` before pipeline execution. The default implementation should use existing Billing enforcement (`LimitCategory.RAG_QUERIES_DAY`) when org context is available, and preserve single-user/orgless behavior where Billing is disabled or explicitly orgless.
- `rag.search` and `rag.answer` call `log_rag_queries_for_org_context()` after successful query execution. Ledger failures remain best-effort and must not change the tool result.
- `context.db_paths` keys should accept existing MCP conventions (`media`, `chacha`, `prompts`, `kanban`) and map to RAG pipeline keys (`media_db_path`, `notes_db_path`, `character_db_path`, `prompts_db_path`, `kanban_db_path`).
- Do not enable Search-Agent web/research loop flags, URL scraping, image search, or video search through MCP first-slice arguments.

- [ ] **Step 4: Run module tests to verify pass**

Run the pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py
git commit -m "feat: add rag mcp module"
```

---

### Task 4: Wire Config, Governance Categories, And Registration

**Files:**
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Modify: `tldw_Server_API/Config_Files/mcp_tool_categories.yaml`
- Modify: `tldw_Server_API/Config_Files/resource_governor_policies.yaml`
- Modify: `tldw_Server_API/app/core/MCP_unified/module_surface.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py` if category-map coverage needs extension.

- [ ] **Step 1: Write failing registration/config tests**

Add tests:

```python
def test_rag_module_is_in_default_module_config():
    config = yaml.safe_load(Path("tldw_Server_API/Config_Files/mcp_modules.yaml").read_text())
    rag = next(module for module in config["modules"] if module["id"] == "rag")
    assert rag["enabled"] is True
    assert rag["class"].endswith("rag_module:RagModule")


def test_rag_tool_category_config_separates_generation():
    mapping = yaml.safe_load(Path("tldw_Server_API/Config_Files/mcp_tool_categories.yaml").read_text())
    assert mapping["rag.search"] == "search"
    assert mapping["rag.answer"] == "rag_generation"


def test_rag_search_and_answer_have_concrete_mcp_policies():
    policies = yaml.safe_load(Path("tldw_Server_API/Config_Files/resource_governor_policies.yaml").read_text())
    assert "mcp.search" in policies["policies"]
    assert "mcp.rag_generation" in policies["policies"]


def test_rag_answer_missing_generation_category_is_guarded(monkeypatch):
    mapping = dict(MCP_TOOL_CATEGORY_MAP)
    mapping.pop("rag.answer", None)
    monkeypatch.setattr(protocol_module, "MCP_TOOL_CATEGORY_MAP", mapping)
    # tools/call must fail closed or produce an explicit guarded config error.
    # It must not rate-limit or execute rag.answer as read/default.
    assert call_rag_answer_with_fake_rate_limiter().reason_code == "rag_generation_category_required"


def test_module_surface_classifies_rag_as_read_only():
    assert MODULE_RISK_TIERS["rag"][0] == "read_only"
```

Add a policy test that `resource_governor_policies.yaml` contains concrete `mcp.search` and `mcp.rag_generation` policies, or the equivalent operation mapping used by MCP rate limiting. Do not leave `rag.search` mapped to `search` without a concrete `mcp.search` policy or deliberate tested alias to an existing category.

Add protocol-level category tests with a fake rate limiter proving `tools/call` for `rag.search` receives category `search` and `tools/call` for `rag.answer` receives category `rag_generation` via `MCP_TOOL_CATEGORY_MAP`/`mcp_tool_categories.yaml`, because the runtime only trusts a small built-in set of metadata categories before consulting the config map. Add a negative test that removes or hides the `rag.answer: rag_generation` mapping and asserts `rag.answer` fails closed with an explicit guarded config error instead of falling back to `read`, `search`, or `default`.

In `test_protocol_catalog_filter.py`, add a catalog regression using the existing `tool_catalog_provider` injection:

```python
@pytest.mark.asyncio
async def test_catalog_membership_does_not_grant_rag_execute_permission():
    provider = FakeCatalogProvider({"rag.search", "rag.answer", "knowledge.search"})
    proto = MCPProtocol(dependencies=_protocol_dependencies(tool_catalog_provider=provider))
    proto.module_registry = RegistryStub(
        {
            "rag": ModuleStub(["rag.search", "rag.answer"]),
            "knowledge": ModuleStub(["knowledge.search"]),
        }
    )
    proto._has_module_permission = always_allow_module
    proto._has_tool_permission = allow_all_except({"rag.answer"})
    ctx = RequestContext(request_id="rag-catalog", user_id="1", client_id="unit")

    listed = await proto._handle_tools_list({"catalog": "library-rag", "catalog_strict": True}, ctx)
    by_name = {tool["name"]: tool for tool in listed["tools"]}
    assert by_name["rag.search"]["canExecute"] is True
    assert by_name["rag.answer"]["canExecute"] is False

    denied = await proto.process_request(
        MCPRequest(method="tools/call", params={"name": "rag.answer", "arguments": {"query": "q"}}, id=1),
        ctx,
    )
    assert denied.error is not None
```

This test must prove catalogs reduce discovery noise only. They must not bypass `tools.execute:rag.*`, API-key scopes, MCP scopes, governance category checks, or module/source authorization.

- [ ] **Step 2: Run registration/config tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py \
  -q
```

Expected: FAIL because `rag` is not configured yet.

- [ ] **Step 3: Update config**

Add to `mcp_modules.yaml` after `knowledge`:

```yaml
  - id: rag
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.rag_module:RagModule
    enabled: true
    name: RAG
    version: "1.0.0"
    department: knowledge
    max_concurrent: 6
    settings:
      max_documents: 6
      max_content_chars: 2000
```

Add to `mcp_tool_categories.yaml`:

```yaml
rag.search: search
rag.answer: rag_generation
```

Add a resource-governor policy in `resource_governor_policies.yaml`:

```yaml
  mcp.search:
    <<: *mcp_base
    requests: { rpm: 60, burst: 1.0 }

  mcp.rag_generation:
    requests: { rpm: 30, burst: 1.0 }
    scopes: [user, api_key]
```

If the policy file has an MCP category map, map `mcp.search` and `mcp.rag_generation` there rather than creating dead policies.

Do not rely on tool metadata alone for MCP rate limiting. Keep metadata for governance/observability, but make runtime categories deterministic through `mcp_tool_categories.yaml`. `rag.answer` must require the configured `rag_generation` category; if that mapping or matching policy is missing, return a guarded configuration error and do not execute the tool under the runtime fallback category.

Add to `module_surface.py`:

```python
"rag": ("read_only", "Run grounded retrieval and answer generation over configured knowledge sources."),
```

- [ ] **Step 4: Run registration/config tests to verify pass**

Run the pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/Config_Files/mcp_modules.yaml \
  tldw_Server_API/Config_Files/mcp_tool_categories.yaml \
  tldw_Server_API/Config_Files/resource_governor_policies.yaml \
  tldw_Server_API/app/core/MCP_unified/module_surface.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py
git commit -m "config: register rag mcp module"
```

---

### Task 5: Documentation, Smoke Verification, And Final Hardening

**Files:**
- Modify: `Docs/MCP/mcp_tool_catalogs.md`
- Modify: `Docs/MCP/Unified/User_Guide.md`
- Modify: `Docs/MCP/Unified/Client_Snippets.md`
- Modify: `tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_mcp_knowledge_rbac.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py` if the `knowledge.search` regression fits better there.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py` only if adding `rag.*` to built-in profile metadata exposes a missing-prefix test.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py` only if discovery/canExecute fixtures need `rag.*` coverage.

- [ ] **Step 1: Write failing docs/discovery tests where applicable**

If built-in profile metadata is updated to include `rag.search` or `rag.answer`, add/update tests so `rag` is categorized and progressive disclosure limits still pass.

Add a lightweight docs regression, or extend an existing docs/catalog test, that confirms `Docs/MCP/mcp_tool_catalogs.md` includes a curated `library-rag` or updated `research-kit` catalog example containing `rag.search` and `rag.answer`. Do not add a new static catalog seed file unless the repository already has a first-class static seed mechanism for tool catalogs; the current catalog system is DB-backed through the existing catalog management APIs.

Add `/api/v1/mcp/tools/execute` compatibility coverage in `test_mcp_http_auth_paths.py` using the existing dummy-server pattern. The test should call `rag.search` and prove the HTTP facade preserves the current `ToolExecutionResponse` wrapper shape for JSON results instead of silently moving the inner `rag.*` payload to a different top-level contract.

Add an executable `knowledge.search` regression in `test_mcp_knowledge_rbac.py` or `test_knowledge_search_defaults.py` proving `KnowledgeModule.execute_tool("knowledge.search", ...)` still fans out to source search tools such as `notes.search`/`media.search`, preserves FTS-style result metadata like `score_type="fts"` when the source module returns it, and never invokes `rag.search`, `RagModule`, or `unified_rag_pipeline`.

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_knowledge_rbac.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py \
  -q
```

Expected: PASS if no profile metadata changes are needed; otherwise FAIL until category fixtures are updated.

- [ ] **Step 2: Update docs**

Add a concise `rag.*` section:

```markdown
### RAG tools

- `rag.capabilities` lists MCP-safe RAG capabilities and limits.
- `rag.source_health` reports safe, canonical source readiness.
- `rag.search` retrieves bounded, citation-aware evidence without answer generation.
- `rag.answer` generates a grounded answer over retrieved evidence and reports `answered`, `partial`, or `abstained`.

Accepted source aliases are normalized through the existing RAG source registry. Responses use canonical source ids such as `media_db` and `notes`. Catalogs can group `rag.*`, `knowledge.*`, `media.*`, and `notes.*` for retrieval workflows, but catalog membership does not grant execution rights.
```

Update `Docs/MCP/mcp_tool_catalogs.md` with a curated workflow catalog example. Prefer `library-rag` as a neutral catalog name; if retaining the existing `research-kit` example for compatibility, explicitly state that it is only a discovery catalog and not a `research.*` MCP layer. Suggested entries:

```markdown
Recommended Catalog: `library-rag`
- Use this catalog to keep existing-library retrieval workflows compact for autonomous clients.
- Suggested entries:
  - `rag.capabilities`
  - `rag.source_health`
  - `rag.search`
  - `rag.answer`
  - `knowledge.search`
  - `knowledge.get`
  - `media.search`
  - `notes.search`
```

Add JSON-RPC snippets for:

```json
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag.search","arguments":{"query":"What did I save about retrieval?", "sources":["media","notes"], "top_k":5}},"id":"rag-search-1"}
```

```json
{"jsonrpc":"2.0","method":"tools/call","params":{"name":"rag.answer","arguments":{"query":"Summarize the evidence on contextual retrieval.", "sources":["media_db"], "include_documents":true}},"id":"rag-answer-1"}
```

- [ ] **Step 3: Run targeted test suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_knowledge_rbac.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Run smoke through MCP protocol**

Add or run an automated smoke test that registers `RagModule` with a stub pipeline and calls:

```python
MCPRequest(method="tools/call", params={"name": "rag.search", "arguments": {"query": "q"}}, id="smoke-rag-search")
MCPRequest(method="tools/call", params={"name": "rag.answer", "arguments": {"query": "q"}}, id="smoke-rag-answer")
```

Expected:

- `resp.error is None`
- JSON content is under `resp.result["content"][0]["json"]`
- `rag.search` has no `answer`
- `rag.answer["answer"]["status"]` is one of `answered`, `partial`, `abstained`

Also add or run an automated HTTP facade smoke through `POST /api/v1/mcp/tools/execute` for `rag.search` with a stub server, asserting the existing wrapper contract remains intact.

- [ ] **Step 5: Run Bandit on touched code scopes**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py \
  tldw_Server_API/app/core/RAG/rag_service/transport.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  -f json -o /tmp/bandit_rag_mcp.json
```

Expected: exit 0 or no new findings in touched code. Fix new findings before continuing.

- [ ] **Step 6: Commit final docs and hardening**

```bash
git add \
  Docs/MCP/mcp_tool_catalogs.md \
  Docs/MCP/Unified/User_Guide.md \
  Docs/MCP/Unified/Client_Snippets.md \
  tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_knowledge_rbac.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py
git commit -m "docs: document rag mcp tools"
```

---

## Final Verification Checklist

- [ ] `rag.capabilities`, `rag.source_health`, `rag.search`, and `rag.answer` are discoverable.
- [ ] All four `rag.*` tool schemas use `additionalProperties=false`.
- [ ] `rag.search` returns bounded evidence and omits `answer`.
- [ ] `rag.answer` returns bounded evidence plus `answer.status`, and weak/uncited answers are `partial` or `abstained` rather than `answered`.
- [ ] Source aliases normalize to canonical ids, reject unknown/internal sources such as `claims`, and preserve `sources_explicit`.
- [ ] Response metadata includes `sources_requested`, `sources_used`, `sources_unavailable`, and truncation fields.
- [ ] `sql` is rejected/deferred in Stage 1 and is not advertised by `rag.source_health`.
- [ ] `rag.source_health`, `rag.search`, and `rag.answer` enforce `rbac_rate_limit("rag.search")`-equivalent posture; `rag.capabilities` does not.
- [ ] `rag.source_health`, `rag.search`, and `rag.answer` enforce TokenScopeGuard/API-key-scope-equivalent read access before source access.
- [ ] Supported `media_id`, `note_id`, workspace, and session scopes are applied before retrieval where current RAG paths can enforce them.
- [ ] Per-source authorization/module enablement is enforced independently; `media.read` only authorizes `media_db`.
- [ ] Explicit unavailable sources fail closed unless `allow_partial=true`.
- [ ] Unsupported `conversation_id`, `character_id`, and `prompt_id` scopes fail closed for explicit source requests and filter with warnings for implicit/default source selection.
- [ ] `advanced` is rejected.
- [ ] MCP-safe overrides force off external Search-Agent/research defaults, web fallback, URL scraping, image search, and video search in Stage 1.
- [ ] `rag.answer` only passes LLM-provider-safe request metadata to generation/provider-facing kwargs.
- [ ] Curated `library-rag`/catalog documentation exists and catalog filtering does not grant execute permission.
- [ ] `rag.source_health` and `rag.capabilities` do not consume `RAG_QUERIES_DAY`.
- [ ] `rag.search` and `rag.answer` enforce RAG query quota and usage accounting.
- [ ] `mcp.search` and `mcp.rag_generation` have concrete category/rate policies, and missing `rag_generation` config fails closed instead of falling back to `read`.
- [ ] `knowledge.search` remains FTS/source-module discovery, not a RAG wrapper.
- [ ] JSON-RPC `tools/call` and HTTP `/tools/execute` wrapper contracts are preserved.
- [ ] Targeted pytest suite passes.
- [ ] Bandit touched-scope scan is clean or any findings are fixed.
