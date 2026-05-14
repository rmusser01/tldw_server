# VN Script Authoring Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-owned VN script authoring catalog plus server-side snippet preview/apply APIs, then let the bundled WebUI consume that API without becoming a second VN rule engine.

**Architecture:** Implement static backend catalog/snippet definitions beside the existing VN script templates, then add a pure snippet patcher used by `VNScriptService` for preview/apply. Preview uses the side-effect-free validation helper and never mutates stored drafts or diagnostics; apply persists through the existing atomic draft replacement path with `if_revision`. The WebUI only renders catalog-driven forms, previews, and apply results; backend validation, manifests, policy, generation profiles, and publish authority remain authoritative.

**Tech Stack:** FastAPI, Pydantic v2, existing `VNScriptService`, existing VN script validator, SQLite-backed `VNScriptsRepository`, pytest, Next.js/React, Vitest.

---

## File Structure

- Create `tldw_Server_API/app/core/VN_Scripts/authoring_catalog.py` for catalog dataclasses/constants, operation metadata, snippet metadata, capability tokens, and safe JSON Schema payloads.
- Create `tldw_Server_API/app/core/VN_Scripts/snippet_patcher.py` for parsed-object snippet patching, anchor resolution, recursive parameter checks, collision handling, and patch summaries.
- Create `tldw_Server_API/app/core/VN_Scripts/authoring_errors.py` for typed snippet/catalog exceptions carrying stable `code`, HTTP `status_code`, and `details`.
- Modify `tldw_Server_API/app/core/VN_Scripts/validator.py` to expose known op/output-schema/routing-key constants through public read-only helpers used by catalog tests.
- Modify `tldw_Server_API/app/core/VN_Scripts/service.py` to add `get_authoring_catalog()`, `preview_snippet()`, and `apply_snippet()`.
- Modify `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py` to add authoring catalog, snippet preview/apply request, response, and summary schemas.
- Modify `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py` to add `GET /vn-authoring-catalog`, `POST /scripts/{script_id}/draft/snippet-preview`, and `POST /scripts/{script_id}/draft/snippet-apply`.
- Modify `tldw_Server_API/app/core/VN_Platform/capabilities.py` and `tldw_Server_API/app/api/v1/schemas/vn_capabilities_schemas.py` only if schema shape changes are needed for `script_authoring_catalog` and canonical authoring capability tokens.
- Test backend contract in `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py` and add focused pure tests in `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py`.
- Modify `apps/tldw-frontend/types/vn-scripts.ts` and `apps/tldw-frontend/lib/api/vnScripts.ts` for catalog/preview/apply client contracts.
- Modify `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx` to add a catalog-driven guided insert panel next to the existing JSON editor.
- Modify `apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts` and `apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx`.
- Update `Docs/API-related/VN_PLATFORM_API.md`.
- Update `backlog/tasks/task-302 - Write-VN-script-authoring-catalog-implementation-plan.md` during this planning work and create implementation task records before executing code tasks.

## Task 1: Backend Catalog Metadata

**Files:**
- Create: `tldw_Server_API/app/core/VN_Scripts/authoring_catalog.py`
- Modify: `tldw_Server_API/app/core/VN_Scripts/validator.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py`

- [ ] **Step 1: Write failing catalog tests**

Add tests proving:

```python
def test_authoring_catalog_lists_every_validator_opcode_once():
    payload = list_authoring_catalog()
    assert {item["op"] for item in payload["operations"]} == set(known_script_ops())
```

```python
def test_authoring_catalog_is_preview_safe_and_backend_owned():
    payload = list_authoring_catalog()
    assert "script_authoring_catalog" in payload["capability_tokens"]
    assert "scripted_generation.output_schema.choice_set" in payload["capability_tokens"]
    assert "validation_codes" not in json.dumps(payload)
    assert "api_key" not in json.dumps(payload)
    assert "provider_config" not in json.dumps(payload)
```

```python
def test_snippet_parameter_schemas_forbid_extra_object_fields():
    for snippet in list_authoring_catalog()["snippets"]:
        assert_all_objects_forbid_extra_fields(snippet["parameters_schema"])
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py -q
```

Expected: fail because the catalog module and public validator helpers do not exist.

- [ ] **Step 3: Expose validator constants safely**

In `tldw_Server_API/app/core/VN_Scripts/validator.py`, add helpers that return sorted tuples:

```python
def known_script_ops() -> tuple[str, ...]:
    """Return supported vn_script_program.v1 opcode names."""
    return tuple(sorted(_KNOWN_OPS))

def supported_generation_output_schemas() -> tuple[str, ...]:
    """Return supported generation output schema names."""
    return tuple(sorted(_SUPPORTED_OUTPUT_SCHEMAS))

def forbidden_generation_routing_keys() -> tuple[str, ...]:
    """Return raw generation routing keys that drafts must not contain."""
    return tuple(sorted(_RAW_GENERATION_ROUTING_KEYS))
```

- [ ] **Step 4: Implement catalog module**

Create immutable catalog metadata with:

- `SCHEMA_VERSION = "vn_script_authoring_catalog.v1"`
- `PROGRAM_SCHEMA_VERSION = "vn_script_program.v1"`
- canonical capability tokens:
  - `script_authoring_catalog`
  - `scripted_generation`
  - `scripted_generation.output_schema.choice_set`
  - `scripted_generation.output_schema.scene_update`
  - `scripted_generation.user_confirmation`
- operation categories: story, branching, visuals, audio, generation, state.
- operation entries for every `known_script_ops()` value.
- no `validation_codes` field.
- snippet entries at minimum:
  - `narration`
  - `dialogue`
  - `authored_choice`
  - `generated_choice_set`
  - `scene_update_generation`
  - `confirm_gated_generation`
  - `set_background`
  - `show_sprite`
  - `play_bgm`
  - `set_variable`
  - `ending`

Use `copy.deepcopy()` or equivalent when returning catalog payloads so tests cannot mutate module state.

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py -q
```

Expected: pass.

- [ ] **Step 6: Commit backend catalog slice**

Run:

```bash
git add tldw_Server_API/app/core/VN_Scripts/validator.py tldw_Server_API/app/core/VN_Scripts/authoring_catalog.py tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py
git commit -m "Add VN script authoring catalog metadata"
```

## Task 2: Pure Snippet Patcher

**Files:**
- Create: `tldw_Server_API/app/core/VN_Scripts/authoring_errors.py`
- Create: `tldw_Server_API/app/core/VN_Scripts/snippet_patcher.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py`

- [ ] **Step 1: Write failing patcher tests**

Add tests proving:

```python
def test_generated_choice_snippet_inserts_generate_and_handler_label():
    result = apply_snippet_patch(
        base_program(),
        snippet_id="generated_choice_set",
        anchor={"label": "start", "op_index": 0, "mode": "after"},
        parameters={"handler_label": "generated_choice", "max_choices": 2},
    )
    op = result.draft["labels"]["start"][1]
    assert op["op"] == "generate"
    assert op["output_schema"] == "choice_set"
    assert op["on_generated_choice"] == "generated_choice"
    assert result.draft["labels"]["generated_choice"][-1]["op"] == "end"
```

```python
def test_preview_patcher_rejects_nested_routing_keys():
    with pytest.raises(VNScriptAuthoringError) as exc_info:
        apply_snippet_patch(
            base_program(),
            snippet_id="authored_choice",
            anchor={"label": "start", "mode": "append"},
            parameters={"choices": [{"id": "x", "text": "X", "target_label": "x", "model": "bad"}]},
        )
    assert exc_info.value.code == "snippet_parameter_invalid"
    assert exc_info.value.details["field_path"] == "$.parameters.choices[0].model"
```

```python
def test_patcher_rejects_anchor_and_label_conflicts():
    with pytest.raises(VNScriptAuthoringError) as missing_anchor:
        apply_snippet_patch(base_program(), snippet_id="ending", anchor={"label": "missing", "mode": "append"}, parameters={})
    assert missing_anchor.value.code == "snippet_anchor_not_found"
    assert missing_anchor.value.details["anchor"]["label"] == "missing"

    with pytest.raises(VNScriptAuthoringError) as label_conflict:
        apply_snippet_patch(base_program(existing_label="generated_choice"), snippet_id="generated_choice_set", anchor={"label": "start", "mode": "append"}, parameters={"handler_label": "generated_choice"})
    assert label_conflict.value.code == "snippet_label_conflict"
    assert label_conflict.value.details["label"] == "generated_choice"
```

Also add security-limit tests:

```python
def test_patcher_rejects_excessive_parameter_depth_length_and_payload_size():
    for parameters in [
        {"text": "x" * (MAX_SNIPPET_PARAMETER_STRING_LENGTH + 1)},
        deeply_nested_parameters(MAX_SNIPPET_PARAMETER_MAX_DEPTH + 1),
        oversized_parameters(MAX_SNIPPET_PARAMETER_PAYLOAD_BYTES + 1),
    ]:
        with pytest.raises(VNScriptAuthoringError) as exc_info:
            apply_snippet_patch(base_program(), snippet_id="narration", anchor={"label": "start", "mode": "append"}, parameters=parameters)
        assert exc_info.value.code == "snippet_parameter_invalid"
        assert "field_path" in exc_info.value.details
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py -q
```

Expected: fail because snippet patching does not exist.

- [ ] **Step 3: Implement patcher data types and guards**

Create small types/functions:

- `VNScriptAuthoringError(code: str, message: str, status_code: int = 400, details: dict[str, Any] | None = None)`
- `SnippetPatchResult(draft: dict[str, Any], patch_summary: dict[str, Any])`
- `apply_snippet_patch(draft, snippet_id, anchor, parameters) -> SnippetPatchResult`
- `_validate_anchor(labels, anchor)`
- `_reject_forbidden_keys(value, path="$")`
- `_enforce_parameter_limits(value, path="$", depth=0)`
- `_assert_no_extra_fields(parameters, allowed_schema)`
- `_insert_ops(labels, anchor, ops)`
- `_create_label(labels, label, body)`

Use parsed dict/list mutation against a deep copy only. Never mutate the caller's draft.

Use explicit security limits:

- `MAX_SNIPPET_PARAMETER_DEPTH = 8`
- `MAX_SNIPPET_PARAMETER_STRING_LENGTH = 8000`
- `MAX_SNIPPET_PARAMETER_PAYLOAD_BYTES = 65536`

All patcher failures must raise `VNScriptAuthoringError` with stable `code` and structured `details`; do not rely on string-matched `ValueError`.

- [ ] **Step 4: Implement V1 snippets**

Start with deterministic snippets needed by WebUI:

- `narration`
- `authored_choice`
- `generated_choice_set`
- `confirm_gated_generation`
- `ending`

Then add remaining catalog snippets. Every snippet must return changed paths and counts:

```python
{
    "inserted_ops": 1,
    "created_labels": ["generated_choice"],
    "changed_paths": ["$.labels.start[1]", "$.labels.generated_choice"],
}
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py -q
```

Expected: pass.

- [ ] **Step 6: Commit patcher slice**

Run:

```bash
git add tldw_Server_API/app/core/VN_Scripts/authoring_errors.py tldw_Server_API/app/core/VN_Scripts/snippet_patcher.py tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py
git commit -m "Add VN script snippet patcher"
```

## Task 3: Service Preview And Apply

**Files:**
- Modify: `tldw_Server_API/app/core/VN_Scripts/service.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py`

- [ ] **Step 1: Write failing service behavior tests**

Add service-level tests proving:

- Stored-draft preview returns patched draft and leaves stored `revision`, `draft`, and `diagnostics` unchanged.
- Supplied-draft preview requires script ownership, returns `base_revision`, and does not persist the supplied draft.
- Apply increments revision, persists diagnostics, and returns the patched draft.
- Stale apply raises `ValueError("draft_revision_conflict")` or a typed conflict that endpoint mapping later converts to HTTP `409`.
- Duplicate apply against the same revision produces one success and one service-level conflict.

Construct `VNScriptService` directly with a test `CharactersRAGDB`, a deterministic manifest resolver, and an audio resolver that can inspect the patched draft. Do not use FastAPI routes in this task.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py -q
```

Expected: fail because service preview/apply methods do not exist.

- [ ] **Step 3: Add service methods**

Add a patch-building helper and preview/apply helpers. The helper split lets the service own the validation boundary while still allowing API code to pass already-resolved async resources into the service:

```python
def build_snippet_patch(self, script_id: int, *, snippet_id: str, anchor: Mapping[str, Any], parameters: Mapping[str, Any], draft: Mapping[str, Any] | None = None) -> dict[str, Any]:
    script = self._require_script(script_id)
    stored = self.get_draft(script_id)
    base_draft = dict(draft) if draft is not None else stored["draft"]
    patch = apply_snippet_patch(base_draft, snippet_id=snippet_id, anchor=anchor, parameters=parameters)
    return {"script": script, "base_revision": int(stored["revision"]), "patch": patch}
```

Add side-effect-free preview validation:

```python
def preview_snippet_patch(self, *, script: Mapping[str, Any], base_revision: int, snippet_id: str, patch: SnippetPatchResult, audio_refs: Mapping[str, Mapping[str, Any]] | None = None, ...) -> dict[str, Any]:
    validation = self.validate_draft_payload(script, patch.draft, audio_refs=audio_refs, ...)
    return {...}
```

Do not call `validate_draft()` from preview because it stores diagnostics.

Add apply persistence through the existing atomic draft write path:

```python
def apply_snippet_patch_result(self, script_id: int, *, if_revision: int, script: Mapping[str, Any], snippet_id: str, patch: SnippetPatchResult, audio_refs: Mapping[str, Mapping[str, Any]] | None = None, ...) -> dict[str, Any]:
    validation = self.validate_draft_payload(script, patch.draft, audio_refs=audio_refs, ...)
    draft_row = self.repo.replace_draft(... if_revision=if_revision, draft=patch.draft, diagnostics=validation)
    return {...}
```

Use the existing repository `revision = ?` update for atomic optimistic concurrency.

Also add `get_authoring_catalog()` as a thin wrapper around `list_authoring_catalog()`.

Do not add HTTP status handling in this task. Endpoint status mapping belongs to Task 4.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py -q
```

Expected: pass.

- [ ] **Step 5: Commit service slice**

Run:

```bash
git add tldw_Server_API/app/core/VN_Scripts/service.py tldw_Server_API/tests/VN_Scripts/test_vn_script_authoring_catalog.py
git commit -m "Add VN script snippet preview apply service"
```

## Task 4: API Schemas, Endpoints, Capabilities, Docs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
- Modify: `tldw_Server_API/app/core/VN_Platform/capabilities.py`
- Modify: `Docs/API-related/VN_PLATFORM_API.md`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

- [ ] **Step 1: Write failing endpoint contract tests**

Add tests proving:

- `GET /api/v1/vn/vn-scripts/vn-authoring-catalog` returns schema version, capability tokens, operations, snippets, and no secrets.
- `POST /scripts/{script_id}/draft/snippet-preview` accepts stored and supplied drafts.
- `POST /scripts/{script_id}/draft/snippet-apply` requires `if_revision`.
- Invalid snippet IDs return `404 snippet_not_found`.
- Invalid parameters return `400 snippet_parameter_invalid` with `field_path`.
- Oversized, overly deep, or overlong snippet parameters return `400 snippet_parameter_invalid` with `field_path`.
- Invalid anchors return `400 snippet_anchor_invalid` or `400 snippet_anchor_not_found` with anchor details.
- Stale revision returns `409 draft_revision_conflict` with `current_revision`.
- `GET /api/v1/vn/vn-capabilities` includes `features.script_authoring_catalog = true` when routes are present.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py -q
```

Expected: fail because endpoint schemas/routes are missing.

- [ ] **Step 3: Add Pydantic schemas**

Add models with `ConfigDict(extra="forbid")`:

- `VNScriptAuthoringCatalogResponse`
- `VNScriptAuthoringOperation`
- `VNScriptAuthoringSnippet`
- `VNScriptSnippetAnchor`
- `VNScriptSnippetPreviewRequest`
- `VNScriptSnippetApplyRequest`
- `VNScriptSnippetPatchSummary`
- `VNScriptSnippetPreviewResponse`
- `VNScriptSnippetApplyResponse`

Keep `parameters: dict[str, Any] = Field(default_factory=dict)` because snippet-specific schemas are catalog data, but endpoint code must run recursive extra/routing-key validation through the patcher.

- [ ] **Step 4: Add endpoints**

Add routes to `vn_scripts.py`:

- `@router.get("/vn-authoring-catalog", response_model=VNScriptAuthoringCatalogResponse)`
- `@router.post("/scripts/{script_id}/draft/snippet-preview", response_model=VNScriptSnippetPreviewResponse)`
- `@router.post("/scripts/{script_id}/draft/snippet-apply", response_model=VNScriptSnippetApplyResponse)`

Endpoints should authenticate, deserialize, call the service patch-building helper, resolve async resources required by the existing validation context, and delegate validation/persistence back to the service. They should not make independent policy, manifest, generation-profile, diagnostic, or publish decisions. Preview must use the supplied draft when present and never call `service.validate_draft()`.

When catching `VNScriptAuthoringError`, map its `code`, `status_code`, and `details` directly into the VN error envelope. When catching `ValueError("draft_revision_conflict")`, look up the current draft revision and return `409` with `{"current_revision": revision}`.

Update `_handle_value_error()` or local error mapping so new codes produce the documented statuses and details.

- [ ] **Step 5: Add capabilities/docs**

In `capabilities.py`, add `script_authoring_catalog` based on the scripts route being registered. Keep `scripted_generation` behavior as-is. If tests need token discovery, add a small `authoring_capability_tokens` list under `scripted_generation` or docs only; do not break the existing `VNCapabilitiesResponse` unless necessary.

Update `Docs/API-related/VN_PLATFORM_API.md` with:

- catalog endpoint
- preview/apply endpoint examples
- non-mutating preview warning
- `if_revision` apply behavior
- error code/status table
- custom frontend flow

- [ ] **Step 6: Run backend tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts -q
```

Expected: pass.

- [ ] **Step 7: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/VN_Scripts tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/core/VN_Platform/capabilities.py -f json -o /tmp/bandit_vn_authoring_catalog.json
```

Expected: no new findings in touched code.

- [ ] **Step 8: Commit backend API slice**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/core/VN_Platform/capabilities.py Docs/API-related/VN_PLATFORM_API.md tldw_Server_API/tests/VN_Scripts
git commit -m "Expose VN script authoring catalog API"
```

## Task 5: Frontend API Client And Types

**Files:**
- Modify: `apps/tldw-frontend/types/vn-scripts.ts`
- Modify: `apps/tldw-frontend/lib/api/vnScripts.ts`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts`

- [ ] **Step 1: Write failing Vitest API tests**

Add tests proving:

- `getVNScriptAuthoringCatalog()` calls `/vn/vn-scripts/vn-authoring-catalog`.
- `previewVNScriptSnippet(scriptId, request)` calls `/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview`.
- `applyVNScriptSnippet(scriptId, request)` calls `/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts
```

Expected: fail because helpers/types are missing.

- [ ] **Step 3: Add TypeScript types**

Add interfaces mirroring backend schemas:

- `VNScriptAuthoringCatalogResponse`
- `VNScriptAuthoringOperation`
- `VNScriptAuthoringSnippet`
- `VNScriptSnippetAnchor`
- `VNScriptSnippetPreviewRequest`
- `VNScriptSnippetApplyRequest`
- `VNScriptSnippetPatchSummary`
- `VNScriptSnippetPreviewResponse`
- `VNScriptSnippetApplyResponse`

- [ ] **Step 4: Add API helpers**

Add:

```ts
export function getVNScriptAuthoringCatalog(): Promise<VNScriptAuthoringCatalogResponse> {
  return apiClient.get(`${VN_SCRIPTS_BASE}/vn-authoring-catalog`);
}

export function previewVNScriptSnippet(scriptId: number, request: VNScriptSnippetPreviewRequest): Promise<VNScriptSnippetPreviewResponse> {
  return apiClient.post(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/draft/snippet-preview`, request);
}

export function applyVNScriptSnippet(scriptId: number, request: VNScriptSnippetApplyRequest): Promise<VNScriptSnippetApplyResponse> {
  return apiClient.post(`${VN_SCRIPTS_BASE}/scripts/${scriptId}/draft/snippet-apply`, request);
}
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts
```

Expected: pass.

- [ ] **Step 6: Commit frontend client slice**

Run:

```bash
git add apps/tldw-frontend/types/vn-scripts.ts apps/tldw-frontend/lib/api/vnScripts.ts apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts
git commit -m "Add VN script authoring catalog frontend client"
```

## Task 6: WebUI Guided Insert Panel

**Files:**
- Modify: `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx`

- [ ] **Step 1: Write failing component tests**

Add tests proving:

- Catalog loads after templates and scripts.
- When catalog load fails, the raw JSON editor remains usable and an inline non-blocking status appears.
- When `vn-capabilities` omits or disables `features.script_authoring_catalog`, the guided insert panel remains hidden/disabled and raw JSON editing remains available.
- Selecting a snippet renders form inputs from `parameters_schema` and defaults.
- Preview calls backend preview, displays diagnostics, and does not update stored draft revision.
- Apply sends current draft revision, updates draft text/revision/diagnostics from the backend response, and shows changed paths.
- A `draft_revision_conflict` response triggers a refetch-safe state/message without duplicating snippet content.

- [ ] **Step 2: Run component tests to verify RED**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx
```

Expected: fail because UI does not load or use authoring catalog.

- [ ] **Step 3: Implement guided insert state**

In `VNScriptsWorkbench.tsx`, add state for:

- `vnCapabilities`
- `authoringCatalog`
- `isLoadingAuthoringCatalog`
- `selectedSnippetId`
- `snippetParameters`
- `snippetAnchor`
- `snippetPreview`
- `isPreviewingSnippet`
- `isApplyingSnippet`
- `snippetError`

Load VN capabilities before or alongside the catalog. Only fetch or show the guided authoring catalog when `features.script_authoring_catalog === true`; otherwise keep raw JSON editing available and do not treat the missing catalog as an error. If capabilities loading fails, degrade to raw JSON editing.

- [ ] **Step 4: Implement compact panel**

Add a compact panel beside or below the JSON editor depending on existing layout. It should:

- group snippets by category
- hide snippets whose `required_capabilities` are unavailable from VN capabilities
- render only simple text/number/boolean/enum fields in V1
- leave advanced nested parameters editable as JSON if necessary
- call preview before apply
- show patch summary and diagnostics returned by backend

Do not implement frontend VN validation. Only render backend diagnostics.

- [ ] **Step 5: Run component tests to verify GREEN**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx
```

Expected: pass.

- [ ] **Step 6: Run frontend focused suite**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts
```

Expected: pass.

- [ ] **Step 7: Commit WebUI slice**

Run:

```bash
git add apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx
git commit -m "Add VN script guided snippet insert panel"
```

## Task 7: Final Verification And PR Prep

**Files:**
- Modify: `backlog/tasks/<implementation-task>.md`
- Review all files changed by Tasks 1-6.

- [ ] **Step 1: Run backend focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts -q
```

Expected: pass.

- [ ] **Step 2: Run frontend focused tests**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts
```

Expected: pass.

- [ ] **Step 3: Run Python compile/import verification**

Run:

```bash
source .venv/bin/activate && python -m compileall tldw_Server_API/app tldw_Server_API/tests/VN_Scripts
```

Expected: no compile failures.

- [ ] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/VN_Scripts tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/core/VN_Platform/capabilities.py -f json -o /tmp/bandit_vn_authoring_catalog.json
```

Expected: no new findings in touched code.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Update Backlog task**

Mark the implementation task acceptance criteria and Definition of Done complete. Record:

- focused backend test command/results
- focused frontend test command/results
- Bandit output path/result
- known skips or blockers
- PR URL once opened

- [ ] **Step 7: Final commit or squash**

If the implementation was done with multiple task commits and the user wants a single PR commit, squash them into one clean commit after verification. Otherwise keep the task commits if they are reviewable.

- [ ] **Step 8: Create PR against `dev`**

Run:

```bash
git push -u origin codex/vn-generated-choice-set
gh pr create --base dev --head codex/vn-generated-choice-set --title "Add VN script authoring catalog" --body-file /tmp/vn_authoring_catalog_pr.md
```

Expected: PR created. Include a human-editable `Change summary` section in the PR body per repo policy.
