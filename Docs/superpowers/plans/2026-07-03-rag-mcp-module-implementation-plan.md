# RAG MCP Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proper `rag.*` MCP module exposing curated RAG capabilities, source health, retrieval, and grounded answer generation without introducing a `research.*` layer.

**Architecture:** Extract the route-local RAG request/capability/source-health seams into transport-neutral service helpers, then add a focused MCP `RagModule` that maps strict MCP arguments into `UnifiedRAGRequest`, executes the shared RAG pipeline, and compacts responses into citation-aware MCP payloads. HTTP and MCP must share request resolution, source health, response mapping, query quota, and usage-accounting helpers.

**Tech Stack:** Python 3, FastAPI route helpers, Pydantic schemas, MCP Unified `BaseModule`, existing RAG service helpers, pytest, Bandit.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-07-03-rag-mcp-module-design.md`
- Backlog: `TASK-12119`

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
- `tldw_Server_API/Config_Files/resource_governor_policies.yaml` - add `mcp.rag_generation` policy mapping.
- `tldw_Server_API/app/core/MCP_unified/module_surface.py` - classify `rag` as read-only data access.
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

- [ ] **Step 1: Write failing helper extraction tests**

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

- [ ] **Step 2: Run tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_transport_helpers.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_source_health_endpoint.py \
  -q
```

Expected: FAIL because `rag_service.transport` does not exist and existing tests still target endpoint-local helpers.

- [ ] **Step 3: Add `rag_service.transport`**

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

- [ ] **Step 4: Run extraction tests to verify pass**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write failing contract tests**

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
```

- [ ] **Step 2: Run tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module.py \
  -q
```

Expected: FAIL because `rag_module.py` and helpers do not exist.

- [ ] **Step 3: Add contract helpers in `rag_module.py`**

Implement only pure helpers first:

```python
_CANONICAL_PUBLIC_SOURCES = (
    "media_db", "notes", "chats", "characters", "kanban", "prompts", "world_books", "dictionaries", "sql",
)
_SEARCH_MODES = ("hybrid", "vector", "fts")
_PROFILES = ("fast", "balanced", "accuracy")


def _sources_were_explicit(arguments: dict[str, Any]) -> bool:
    return "sources" in arguments and arguments.get("sources") is not None


def _build_mcp_rag_request(tool_name: str, arguments: dict[str, Any]) -> tuple[UnifiedRAGRequest, dict[str, Any]]:
    if "advanced" in arguments:
        raise ValueError("advanced is not supported by rag.* first slice")
    sources_explicit = _sources_were_explicit(arguments)
    sources = normalize_sources_public(arguments.get("sources"))
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

Add `_compact_rag_response()`, `_answer_status()`, and `_domain_error_payload()` as pure functions. Keep all payloads JSON-serializable and avoid provider secrets, paths, and prompts.

- [ ] **Step 4: Run contract tests to verify pass**

Run the pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

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
    assert tools["rag.search"]["inputSchema"]["additionalProperties"] is False
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
- `rag.source_health` returns safe canonical source entries and does not consume RAG query quota.
- `rag.source_health`, `rag.search`, and `rag.answer` fail closed without `media.read` or wildcard permission in `context.metadata["permissions"]`.
- `rag.search` and `rag.answer` call the RAG query quota checker before pipeline execution and the best-effort usage logger after successful execution.
- explicit unavailable source returns `ok:false` unless `allow_partial=true`.
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
- `rag.source_health`, `rag.search`, and `rag.answer` call a small `_require_media_read_permission(context)` helper before source access. The helper should accept `*` and `media.read` from `context.metadata["permissions"]` and fail closed when permissions are absent or malformed.
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


def test_module_surface_classifies_rag_as_read_only():
    assert MODULE_RISK_TIERS["rag"][0] == "read_only"
```

Add a policy test that `resource_governor_policies.yaml` contains `mcp.rag_generation` or the equivalent operation mapping used by MCP rate limiting.

Add a protocol-level category test with a fake rate limiter proving `tools/call` for `rag.answer` receives category `rag_generation` via `MCP_TOOL_CATEGORY_MAP`/`mcp_tool_categories.yaml`, because the runtime only trusts a small built-in set of metadata categories before consulting the config map.

- [ ] **Step 2: Run registration/config tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_rag_module_registration.py \
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
  mcp.rag_generation:
    requests: { rpm: 30, burst: 1.0 }
    scopes: [user, api_key]
```

If the policy file has an MCP category map, map `mcp.rag_generation` there rather than creating a dead policy.

Do not rely on `metadata={"category": "rag_generation"}` alone for MCP rate limiting. Keep the metadata for governance/observability, but make the runtime category deterministic through `mcp_tool_categories.yaml`.

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
  tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py
git commit -m "config: register rag mcp module"
```

---

### Task 5: Documentation, Smoke Verification, And Final Hardening

**Files:**
- Modify: `Docs/MCP/Unified/User_Guide.md`
- Modify: `Docs/MCP/Unified/Client_Snippets.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py` only if adding `rag.*` to built-in profile metadata exposes a missing-prefix test.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py` only if discovery/canExecute fixtures need `rag.*` coverage.

- [ ] **Step 1: Write failing docs/discovery tests where applicable**

If built-in profile metadata is updated to include `rag.search` or `rag.answer`, add/update tests so `rag` is categorized and progressive disclosure limits still pass.

Run:

```bash
source .venv/bin/activate && python -m pytest \
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

Accepted source aliases are normalized through the existing RAG source registry. Responses use canonical source ids such as `media_db` and `notes`. Catalogs can group `rag.*`, `knowledge.*`, `media.*`, and `notes.*` for research workflows, but catalog membership does not grant execution rights.
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
  Docs/MCP/Unified/User_Guide.md \
  Docs/MCP/Unified/Client_Snippets.md \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py
git commit -m "docs: document rag mcp tools"
```

---

## Final Verification Checklist

- [ ] `rag.capabilities`, `rag.source_health`, `rag.search`, and `rag.answer` are discoverable.
- [ ] `rag.search` returns bounded evidence and omits `answer`.
- [ ] `rag.answer` returns bounded evidence plus `answer.status`.
- [ ] Source aliases normalize to canonical ids and preserve `sources_explicit`.
- [ ] Explicit unavailable sources fail closed unless `allow_partial=true`.
- [ ] `advanced` is rejected.
- [ ] Catalog filtering does not grant execute permission.
- [ ] `rag.source_health` and `rag.capabilities` do not consume `RAG_QUERIES_DAY`.
- [ ] `rag.search` and `rag.answer` enforce RAG query quota and usage accounting.
- [ ] `knowledge.search` remains FTS discovery.
- [ ] Targeted pytest suite passes.
- [ ] Bandit touched-scope scan is clean or any findings are fixed.
