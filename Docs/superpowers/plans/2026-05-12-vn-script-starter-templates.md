# VN Script Starter Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add backend-owned VN script starter templates and a guided WebUI creation flow for issue #1604.

**Architecture:** Keep templates as deterministic backend catalog data exposed by the existing `/api/v1/vn/vn-scripts` router. Template instantiation creates a normal script shell and replaces its draft through the same validation path used by custom clients, so publish/runtime semantics remain unchanged. The WebUI only lists templates and calls the backend create-from-template flow; it does not duplicate template content or validation logic.

**Tech Stack:** FastAPI, Pydantic, existing `VNScriptService`, existing VN script validator, pytest, Next.js/React, Vitest.

---

## File Structure

- Create `tldw_Server_API/app/core/VN_Scripts/templates.py` for built-in template definitions, sanitized catalog payloads, deterministic draft instantiation, and template lookup errors.
- Modify `tldw_Server_API/app/core/VN_Scripts/service.py` to expose `list_templates()` and `create_script_from_template()`.
- Modify `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py` to add template catalog and create-from-template request/response schemas.
- Modify `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py` to add `GET /templates` and `POST /templates/{template_id}/scripts`.
- Modify `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py` for endpoint coverage.
- Modify `tldw_Server_API/tests/VN_Scripts/test_vn_script_validator.py` or add a focused template test if service-level validation needs lower-level coverage.
- Modify `apps/tldw-frontend/types/vn-scripts.ts`, `apps/tldw-frontend/lib/api/vnScripts.ts`, and existing VN scripts tests for the client contract.
- Modify `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx` and its tests to add the guided template picker.
- Update `backlog/tasks/task-298 - Add-VN-script-starter-templates-and-guided-authoring.md` with completion notes and verification.

## Task 1: Backend Template Catalog And Instantiation

**Files:**
- Create: `tldw_Server_API/app/core/VN_Scripts/templates.py`
- Modify: `tldw_Server_API/app/core/VN_Scripts/service.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

- [x] **Step 1: Write failing API tests**

Add tests proving:

```python
def test_template_catalog_lists_preview_safe_starter_templates(client):
    response = client.get("/api/v1/vn/vn-scripts/templates")
    assert response.status_code == 200
    payload = response.json()
    assert {item["id"] for item in payload["items"]} >= {
        "linear_scene",
        "authored_choices",
        "generated_choice_set",
        "scene_update",
        "confirm_gated_generation",
    }
    assert "draft" not in payload["items"][0]
    assert "raw_prompt" not in payload["items"][0]
```

```python
def test_create_script_from_template_stores_valid_draft(client, chacha_dbs):
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    response = client.post(
        "/api/v1/vn/vn-scripts/templates/linear_scene/scripts",
        json={
            "title": "Template Route",
            "primary_asset_pack_id": asset_pack_id,
            "content_rating": "general",
        },
    )
    assert response.status_code == 201
    script_id = response.json()["script"]["id"]
    draft = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()
    assert draft["revision"] == 1
    assert draft["diagnostics"]["valid"] is True
    assert draft["draft"]["primary_asset_pack_id"] == asset_pack_id
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py::test_template_catalog_lists_preview_safe_starter_templates tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py::test_create_script_from_template_stores_valid_draft -q
```

Expected: fail because the template endpoint does not exist.

- [x] **Step 3: Implement catalog module**

Create immutable template definitions with:

- Stable IDs: `linear_scene`, `authored_choices`, `generated_choice_set`, `scene_update`, `confirm_gated_generation`.
- Catalog fields: `id`, `label`, `description`, `category`, `recommended_content_rating`, `required_capabilities`, `preview`, `default_title`, `default_description`.
- Draft factory that accepts script metadata and returns a valid `vn_script_program.v1` draft using only supported opcodes.
- Sanitized catalog output that never includes full draft JSON, raw prompts, internal comments, or policy overrides.

- [x] **Step 4: Implement service methods**

Add `VNScriptService.list_templates()` and `VNScriptService.create_script_from_template(...)`. The create method should:

- Reuse `create_script(...)`.
- Build the template draft with the created script metadata.
- Validate through `replace_draft(...)` using `if_revision=0`.
- Return both the script metadata and draft response.

- [x] **Step 5: Run tests to verify GREEN**

Run the same focused pytest command. Expected: pass.

## Task 2: Backend API Schemas And Safety Cases

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
- Test: `tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py`

- [x] **Step 1: Write failing safety tests**

Add tests for:

- Unknown template ID returns 404 with `template_not_found`.
- Create-from-template validates unknown profile IDs the same as normal create.
- Template-created generated-choice drafts validate and publish through existing paths.
- Catalog entries do not contain `policy_profile_id`, `generation_profile_id`, or hidden draft bodies.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts/test_vn_scripts_api.py -q
```

Expected: new tests fail before endpoint/schema work is complete.

- [x] **Step 3: Implement schemas and endpoints**

Add:

- `VNScriptTemplateSummary`
- `VNScriptTemplateListResponse`
- `VNScriptCreateFromTemplateRequest`
- `VNScriptCreateFromTemplateResponse`

Endpoint shape:

- `GET /api/v1/vn/vn-scripts/templates`
- `POST /api/v1/vn/vn-scripts/templates/{template_id}/scripts`

The POST endpoint should resolve profiles before script creation, resolve audio refs if a template ever includes media refs, and map `template_not_found` to 404.

- [x] **Step 4: Run backend tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts -q
```

Expected: pass.

## Task 3: Frontend API Client And Types

**Files:**
- Modify: `apps/tldw-frontend/types/vn-scripts.ts`
- Modify: `apps/tldw-frontend/lib/api/vnScripts.ts`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts`

- [x] **Step 1: Write failing Vitest client tests**

Add tests proving:

- `listVNScriptTemplates()` calls `/vn/vn-scripts/templates`.
- `createVNScriptFromTemplate(templateId, request)` calls `/vn/vn-scripts/templates/{templateId}/scripts` with the request payload.

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts
```

Expected: fail because functions/types are missing.

- [x] **Step 3: Implement TypeScript contract**

Add interfaces mirroring backend schemas and two API helpers:

- `listVNScriptTemplates(): Promise<VNScriptTemplateListResponse>`
- `createVNScriptFromTemplate(templateId: string, request: VNScriptCreateFromTemplateRequest): Promise<VNScriptCreateFromTemplateResponse>`

- [x] **Step 4: Run tests to verify GREEN**

Run the same Vitest command. Expected: pass.

## Task 4: WebUI Template Picker

**Files:**
- Modify: `apps/tldw-frontend/components/vn-scripts/VNScriptsWorkbench.tsx`
- Test: `apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx`

- [x] **Step 1: Write failing component tests**

Add tests proving:

- Templates load on mount and render as selectable starter options.
- Selecting a template and submitting calls `createVNScriptFromTemplate` rather than raw `createVNScript`.
- Created template scripts are prepended, selected, and their returned draft is shown without waiting for a stale detail reload.
- The plain shell path still works when the user selects a blank/custom option.

- [x] **Step 2: Run component test to verify RED**

Run:

```bash
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx
```

Expected: fail because template helpers/UI do not exist.

- [x] **Step 3: Implement WebUI flow**

Add a compact template picker in the create form. Keep the current raw shell creation as “Blank/custom JSON”. When a template is selected:

- Submit through `createVNScriptFromTemplate`.
- Use returned `script` and `draft` to update selected state immediately.
- Still expose the JSON editor, validation, diagnostics, save, and publish controls.
- Keep visible text concise and operational.

- [x] **Step 4: Run component test to verify GREEN**

Run the same Vitest command. Expected: pass.

## Task 5: Documentation, Verification, And Closeout

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-12-vn-script-starter-templates.md`
- Modify: `backlog/tasks/task-298 - Add-VN-script-starter-templates-and-guided-authoring.md`
- Possibly modify API docs if the repo has a VN scripts doc page.

- [x] **Step 1: Document custom frontend contract**

Add concise implementation notes that describe the two endpoints, stable template IDs, sanitized catalog payload, and create-from-template behavior.

- [x] **Step 2: Run verification**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts -q
bunx vitest run apps/tldw-frontend/__tests__/vn-scripts/vnScriptsApi.test.ts apps/tldw-frontend/__tests__/vn-scripts/VNScriptsWorkbench.test.tsx
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/core/VN_Scripts -f json -o /tmp/bandit_vn_script_templates.json
git diff --check
```

- [x] **Step 3: Update Backlog task**

Check completed acceptance criteria, record verification output, and add a final summary.

- [x] **Step 4: Commit**

Commit all task files, backend, frontend, tests, and docs in one focused commit for #1604.
