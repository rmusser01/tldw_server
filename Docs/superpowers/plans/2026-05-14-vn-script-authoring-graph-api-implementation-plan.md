# VN Script Authoring Graph API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-only computed VN script authoring graph API for stored drafts, supplied draft previews, and published versions.

**Architecture:** Implement a pure `authoring_graph.py` builder that accepts parsed script programs and emits outline, graph, graph diagnostics, deterministic IDs, bracket JSON paths, limits, and content hashes without DB access. Add service methods that own script lookup, ownership, source selection, non-mutating validation, and published-version snapshot context. Expose the result through VN Scripts endpoints and advertise `features.script_authoring_graph`.

**Tech Stack:** FastAPI, Pydantic v2, existing `VNScriptService`, existing VN script validator, SQLite-backed `VNScriptsRepository`, pytest, Bandit.

---

## Reference Documents

- Spec: `Docs/superpowers/specs/2026-05-14-vn-script-authoring-graph-design.md`
- API docs: `Docs/API/VN.md`
- Existing authoring catalog plan: `Docs/superpowers/plans/2026-05-12-vn-script-authoring-catalog.md`
- Existing backend tests:
  - `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py`
  - `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

## File Structure

- Create `tldw_Server_API/app/core/VN_Scripts/authoring_graph.py`
  - Pure graph builder.
  - Graph response constants and limits.
  - Percent-encoded stable ID helpers.
  - Bracket JSON path helpers.
  - Static edge extraction.
  - Reachability and terminal classification.
  - Graph diagnostics.
  - Canonical content hash helper.
- Modify `tldw_Server_API/app/core/VN_Scripts/validator.py`
  - Add public helper for graph-relevant static label edges or expose existing reachability behavior safely.
  - Keep validator diagnostics authoritative.
- Modify `tldw_Server_API/app/core/VN_Scripts/service.py`
  - Add `get_draft_graph()`, `preview_draft_graph()`, and `get_version_graph()`.
  - Use non-mutating validation for graph responses.
  - Use published-version pinned context for version graphs.
- Modify `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
  - Add graph request/response schemas with `extra="forbid"`.
- Modify `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
  - Add the three graph routes.
  - Map graph-specific value errors into existing VN error envelopes.
- Modify `tldw_Server_API/app/core/VN_Platform/capabilities.py`
  - Add `features.script_authoring_graph`.
- Modify `tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py`
  - Add the feature field if the schema currently enumerates capability feature keys.
- Modify `Docs/API/VN.md`
  - Document routes, response shape, source modes, diagnostics, limits, and custom frontend flow.
- Create `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py`
  - Pure builder and service tests.
- Modify `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`
  - Endpoint and capability tests.

## Task 1: Pure Authoring Graph Builder

**Files:**
- Create: `tldw_Server_API/app/core/VN_Scripts/authoring_graph.py`
- Modify: `tldw_Server_API/app/core/VN_Scripts/validator.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py`

- [ ] **Step 1: Write failing tests for stable IDs, paths, graph layers, and hash**

Add tests like:

```python
def test_build_script_authoring_graph_returns_outline_graph_and_hash() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "intro.scene",
        "labels": {
            "intro.scene": [
                {"op": "narrate", "text": "Opening."},
                {"op": "jump", "target": "end label"},
            ],
            "end label": [{"op": "end"}],
        },
    }

    result = build_script_authoring_graph(program)

    assert result["schema_version"] == "vn_script_authoring_graph.v1"
    assert result["graph_semantics_version"] == "vn_script_authoring_graph_edges.v1"
    assert result["content_hash"].startswith("sha256:")
    assert result["outline"]["entry_label"] == "intro.scene"
    assert result["outline"]["labels"][0]["id"] == "label:intro%2Escene"
    assert result["outline"]["labels"][0]["source_path"] == "$.labels['intro.scene']"
    assert result["graph"]["nodes"][1]["id"] == "op:intro%2Escene:0"
    assert result["graph"]["nodes"][2]["source_path"] == "$.labels['intro.scene'][1]"
    assert result["graph"]["edges"][0]["type"] == "jump"
    assert result["graph"]["edges"][0]["target_id"] == "label:end%20label"
```

Add a separate test for label separators:

```python
def test_label_ids_are_percent_encoded_but_display_label_remains_raw() -> None:
    result = build_script_authoring_graph({
        "schema_version": "vn_script_program.v1",
        "entry_label": "route:1/a",
        "labels": {"route:1/a": [{"op": "end"}]},
    })

    assert result["outline"]["labels"][0]["id"] == "label:route%3A1%2Fa"
    assert result["outline"]["labels"][0]["label"] == "route:1/a"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py -q
```

Expected: fail because `authoring_graph.py` does not exist.

- [ ] **Step 3: Implement graph constants, helpers, and basic builder**

In `authoring_graph.py`, add:

```python
SCHEMA_VERSION = "vn_script_authoring_graph.v1"
GRAPH_SEMANTICS_VERSION = "vn_script_authoring_graph_edges.v1"
PROGRAM_SCHEMA_VERSION = "vn_script_program.v1"
MAX_SUPPLIED_DRAFT_BYTES = 1_048_576
MAX_LABELS = 500
MAX_OPS = 5000
MAX_EDGES = 10000
MAX_SUMMARY_LENGTH = 240
```

Add helpers:

```python
def encoded_label_id(label: str) -> str:
    return "label:" + quote(label, safe="")

def operation_id(label: str, index: int) -> str:
    return f"op:{quote(label, safe='')}:{index}"

def bracket_label_path(label: str) -> str:
    escaped = label.replace("\\", "\\\\").replace("'", "\\'")
    return f"$.labels['{escaped}']"
```

Add `build_script_authoring_graph(program, *, source="stored_draft", script_id=None, base_revision=None, version_id=None, validation_diagnostics=None, validation_context_source="current_draft_context") -> dict[str, Any]`.

Build label nodes, operation nodes, an empty diagnostics envelope, deterministic limits, `truncated=False`, and `content_hash`.

- [ ] **Step 4: Run tests to verify GREEN for the basic graph**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py -q
```

Expected: initial graph shape tests pass.

- [ ] **Step 5: Add edge extraction tests**

Add tests for:

```python
def test_graph_extracts_jump_choice_generate_and_cancel_edges() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {
            "start": [
                {"op": "choice", "choices": [{"id": "left", "text": "Left", "target": "left"}]},
                {"op": "generate", "output_schema": "choice_set", "on_generated_choice": "generated", "on_cancel": "cancel"},
                {"op": "jump", "target": "done"},
            ],
            "left": [{"op": "end"}],
            "generated": [{"op": "end"}],
            "cancel": [{"op": "end"}],
            "done": [{"op": "end"}],
        },
    }

    result = build_script_authoring_graph(program)
    edge_types = [edge["type"] for edge in result["graph"]["edges"]]

    assert edge_types == ["choice", "generated_choice_handler", "generation_cancel", "jump"]
```

Also test missing target edges:

```python
def test_missing_targets_emit_edges_and_diagnostics() -> None:
    result = build_script_authoring_graph({
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {"start": [{"op": "jump", "target": "missing"}]},
    })

    assert result["graph"]["edges"][0]["target_id"] is None
    assert result["graph"]["edges"][0]["missing_target"] is True
    assert result["diagnostics"]["errors"][0]["code"] == "graph_target_missing"
```

- [ ] **Step 6: Implement static edge extraction**

Extract only:

- `jump.target`
- `choice.choices[].target`
- `generate.on_generated_choice`
- `generate.on_cancel`

Do not infer `random`, conditionals, model output, or fallthrough.

Use deterministic edge IDs:

```python
edge_id = f"edge:{source_id}:{edge_type}:{target_id or 'missing:' + encoded_missing_target}"
```

- [ ] **Step 7: Add reachability and terminal classification tests**

Add tests proving:

- reachable labels match graph edges.
- unreachable labels emit `graph_label_unreachable`.
- `end` produces `terminal`.
- labels with `random`, `return`, conditions, malformed bodies, or dynamic flow produce `unknown`.

Also compare with validator warnings:

```python
def test_graph_reachability_matches_validator_unreachable_warnings() -> None:
    program = program_with_unreachable_label()
    graph = build_script_authoring_graph(program)
    validation = validate_script_program(program, VNScriptValidationContext()).to_dict()

    graph_unreachable = {diag["details"]["label"] for diag in graph["diagnostics"]["warnings"] if diag["code"] == "graph_label_unreachable"}
    validator_unreachable = {diag["details"]["label"] for diag in validation["warnings"] if diag["code"] == "label_unreachable"}

    assert graph_unreachable == validator_unreachable
```

- [ ] **Step 8: Implement reachability and conservative terminal states**

Use graph edges for reachability. If needed, add a validator helper that both validator and graph tests can rely on, such as:

```python
def static_reachable_labels(entry_label: str, labels: Mapping[str, Any]) -> set[str]:
    ...
```

Keep behavior aligned with existing `_reachable_labels()` for `jump`, `choice`, `generate.on_cancel`, and `generate.on_generated_choice`.

Terminal classification:

- `terminal`: last meaningful op is `end`.
- `continues`: label has at least one static outgoing edge.
- `unknown`: malformed, dynamic, conditional, `random`, `return`, or ambiguous.

- [ ] **Step 9: Add malformed and limit tests**

Add tests for:

- non-list label body.
- non-object op.
- missing `labels`.
- invalid choice arrays.
- malformed generate handlers.
- `MAX_LABELS`, `MAX_OPS`, and `MAX_EDGES` truncation.
- canonical-equivalent programs produce the same content hash.

- [ ] **Step 10: Implement malformed tolerance, sorting, limits, and content hash**

Use canonical JSON:

```python
json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

Hash only source program, `PROGRAM_SCHEMA_VERSION`, and `GRAPH_SEMANTICS_VERSION`.

Set `truncated=True` and add `graph_node_limit_exceeded` or `graph_edge_limit_exceeded` diagnostics when graph output is partial.

- [ ] **Step 11: Run focused graph tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py -q
```

Expected: pass.

- [ ] **Step 12: Commit graph builder slice**

Run:

```bash
git add tldw_Server_API/app/core/VN_Scripts/authoring_graph.py tldw_Server_API/app/core/VN_Scripts/validator.py tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py
git commit -m "Add VN script authoring graph builder"
```

## Task 2: Service Graph Methods

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Scripts/service.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py`

- [ ] **Step 1: Write failing service tests**

Add tests using the existing `CharactersRAGDB` fixture pattern:

```python
def test_service_get_draft_graph_is_non_mutating(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=graph_program())
    before = service.get_draft(script["id"])

    result = service.get_draft_graph(script["id"])
    after = service.get_draft(script["id"])

    assert result["source"] == "stored_draft"
    assert result["base_revision"] == before["revision"]
    assert result["validation_context_source"] == "current_draft_context"
    assert after == before
```

```python
def test_service_preview_draft_graph_accepts_stale_revision_with_warning(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=graph_program())

    result = service.preview_draft_graph(script["id"], graph_program(), draft_revision=0)

    assert result["source"] == "supplied_draft"
    assert result["base_revision"] == 1
    assert any(diag["code"] == "graph_preview_revision_stale" for diag in result["diagnostics"]["warnings"])
```

```python
def test_service_version_graph_uses_published_version_snapshot_context(chacha_db: CharactersRAGDB) -> None:
    service = _service(chacha_db)
    script = _create_script(service, draft=graph_program())
    published = service.publish_script(script["id"], draft_revision=1, idempotency_key="publish-graph")

    result = service.get_version_graph(script["id"], published["version_id"])

    assert result["source"] == "published_version"
    assert result["version_id"] == published["version_id"]
    assert result["validation_context_source"] == "published_version_snapshot"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py -q
```

Expected: fail because service methods do not exist.

- [ ] **Step 3: Implement draft graph methods**

Add to `VNScriptService`:

```python
def get_draft_graph(self, script_id: int) -> dict[str, Any]:
    script = self._require_script(script_id)
    draft_row = self.get_draft(script_id)
    diagnostics = self.validate_draft_payload(script, draft_row["draft"])
    return build_script_authoring_graph(
        draft_row["draft"],
        source="stored_draft",
        script_id=script_id,
        base_revision=int(draft_row["revision"]),
        validation_diagnostics=diagnostics,
        validation_context_source="current_draft_context",
    )
```

```python
def preview_draft_graph(self, script_id: int, draft: Mapping[str, Any], *, draft_revision: int | None = None) -> dict[str, Any]:
    script = self._require_script(script_id)
    current_revision = int(self.get_draft(script_id)["revision"])
    diagnostics = self.validate_draft_payload(script, draft)
    result = build_script_authoring_graph(
        draft,
        source="supplied_draft",
        script_id=script_id,
        base_revision=current_revision,
        validation_diagnostics=diagnostics,
        validation_context_source="current_draft_context",
    )
    if draft_revision is not None and int(draft_revision) != current_revision:
        result["diagnostics"]["warnings"].append(...)
    return result
```

Do not call `repo.replace_draft()` or any method that stores diagnostics.

- [ ] **Step 4: Implement version graph method**

Use `self.get_version(script_id, version_id)` and any existing version snapshot helpers. Pass `version["program"]` to the graph builder.

Resolve validation context from pinned version data. If a current helper cannot validate a version without mutable context, add a small private helper in `service.py` such as `_validate_version_program_payload(version)` and keep it local to this service task.

- [ ] **Step 5: Add supplied draft size guard**

Before graphing supplied drafts, measure canonical JSON bytes. If greater than `MAX_SUPPLIED_DRAFT_BYTES`, raise `ValueError("supplied_draft_too_large")`.

If the draft is not a mapping, raise `ValueError("supplied_draft_invalid_shape")`.

- [ ] **Step 6: Run service graph tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py -q
```

Expected: pass.

- [ ] **Step 7: Commit service slice**

Run:

```bash
git add tldw_Server_API/app/core/VN_Scripts/service.py tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_graph.py
git commit -m "Add VN script authoring graph service"
```

## Task 3: API Schemas And Endpoints

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

- [ ] **Step 1: Write failing API tests**

Add endpoint tests:

```python
def test_draft_graph_endpoint_returns_stored_graph(client: TestClient, chacha_dbs: dict[int, CharactersRAGDB]) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft", json={"if_revision": 0, "draft": graph_program(asset_pack_id)})

    response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "stored_draft"
    assert payload["base_revision"] == 1
    assert payload["outline"]["labels"]
```

```python
def test_graph_preview_does_not_persist_supplied_draft(client: TestClient, chacha_dbs: dict[int, CharactersRAGDB]) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    stored = graph_program(asset_pack_id)
    supplied = graph_program(asset_pack_id)
    supplied["labels"]["start"][0]["text"] = "Unsaved."
    client.put(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft", json={"if_revision": 0, "draft": stored})

    response = client.post(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview", json={"draft": supplied, "draft_revision": 1})
    draft_after = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()

    assert response.status_code == 200
    assert response.json()["source"] == "supplied_draft"
    assert draft_after["draft"] == stored
```

Also add tests for:

- version graph endpoint.
- malformed supplied draft shape -> `400`.
- oversized supplied draft -> `413`.
- graph problems return `200` with diagnostics.
- no full op payloads or provider secrets in output.

- [ ] **Step 2: Run API tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py -q
```

Expected: fail because schemas/routes are missing.

- [ ] **Step 3: Add Pydantic schemas**

In `vn_script_schemas.py`, add models with `ConfigDict(extra="forbid")`:

- `VNScriptGraphPreviewRequest`
- `VNScriptGraphDiagnostic`
- `VNScriptGraphDiagnostics`
- `VNScriptGraphOutlineLabel`
- `VNScriptGraphOutline`
- `VNScriptGraphNode`
- `VNScriptGraphEdge`
- `VNScriptGraphBody`
- `VNScriptAuthoringGraphResponse`

Use `Literal` for stable enums where practical:

```python
source: Literal["stored_draft", "supplied_draft", "published_version"]
terminal: Literal["terminal", "continues", "unknown"]
```

Keep `draft: dict[str, Any]`.

- [ ] **Step 4: Add endpoints**

In `vn_scripts.py`, add:

```python
@router.get("/scripts/{script_id}/draft/graph", response_model=VNScriptAuthoringGraphResponse)
async def get_draft_graph(...):
    ...
```

```python
@router.post("/scripts/{script_id}/draft/graph-preview", response_model=VNScriptAuthoringGraphResponse)
async def preview_draft_graph(...):
    ...
```

```python
@router.get("/scripts/{script_id}/versions/{version_id}/graph", response_model=VNScriptAuthoringGraphResponse)
async def get_version_graph(...):
    ...
```

Map:

- `supplied_draft_invalid_shape` -> `400 ERROR_INVALID_REQUEST`
- `supplied_draft_too_large` -> `413 ERROR_INVALID_REQUEST`
- `script_version_not_found` -> existing not-found mapping
- `script_not_found` / `draft_not_found` -> existing not-found mapping

- [ ] **Step 5: Run API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py -q
```

Expected: pass.

- [ ] **Step 6: Commit API slice**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py
git commit -m "Expose VN script authoring graph API"
```

## Task 4: Capabilities And Documentation

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Platform/capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py`
- Modify: `Docs/API/VN.md`
- Test: `tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

- [ ] **Step 1: Write failing capability test**

Add or update:

```python
def test_vn_capabilities_advertises_script_authoring_graph(client: TestClient) -> None:
    response = client.get("/api/v1/vn/vn-capabilities")

    assert response.status_code == 200
    assert response.json()["features"]["script_authoring_graph"] is True
```

- [ ] **Step 2: Run capability tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py -q
```

Expected: fail until feature flag is added.

- [ ] **Step 3: Add capability flag**

In `capabilities.py`, set:

```python
script_authoring_graph_enabled = enabled_modules["scripts"]
```

Add to `features`:

```python
"script_authoring_graph": script_authoring_graph_enabled,
```

If `vn_capabilities_schemas.py` has a concrete features schema, add `script_authoring_graph: bool = False`.

- [ ] **Step 4: Update API docs**

In `Docs/API/VN.md`, document:

- the three graph endpoints.
- source modes.
- response envelope.
- outline and graph layers.
- diagnostics vs validation diagnostics.
- limits and truncation.
- content hash and graph semantics version.
- custom frontend usage.
- non-goals: no execution, model calls, mutation, persistence, or node editor.

- [ ] **Step 5: Run focused docs/capability checks**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py -q
```

Expected: pass.

- [ ] **Step 6: Commit docs/capability slice**

Run:

```bash
git add tldw_Server_API/app/core/VN_Platform/capabilities.py tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py Docs/API/VN.md tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py
git commit -m "Document VN script authoring graph capability"
```

## Task 5: Final Verification And PR Prep

**Files:**
- Modify: `backlog/tasks/<implementation-task>.md`
- Review all changed files.

- [ ] **Step 1: Run focused VN script tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts -q
```

Expected: pass.

- [ ] **Step 2: Run VN platform capability tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Platform/test_vn_capabilities_api.py -q
```

Expected: pass.

- [ ] **Step 3: Run compile verification**

Run:

```bash
source .venv/bin/activate && python -m compileall tldw_Server_API/app/core/VN_Scripts tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/tests/VN_Scripts
```

Expected: no compile failures.

- [ ] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/VN_Scripts tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/core/VN_Platform/capabilities.py -f json -o /tmp/bandit_vn_script_authoring_graph.json
```

Expected: no new findings in touched code.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Update Backlog task**

Record:

- implementation summary.
- focused test results.
- compile result.
- Bandit output path and result.
- known skips or blockers.
- PR URL once opened.

- [ ] **Step 7: Squash if requested**

If the user requests one commit, squash task commits into one after all verification passes.

- [ ] **Step 8: Create PR against `dev`**

Run:

```bash
git push -u origin codex/vn-script-authoring-graph-api
gh pr create --base dev --head codex/vn-script-authoring-graph-api --title "Add VN script authoring graph API" --body-file /tmp/vn_script_authoring_graph_pr.md
```

Expected: PR created with a human-editable `Change summary` section.
