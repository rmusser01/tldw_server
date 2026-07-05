# First-Run MCP Tool Packs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the optional first-run MCP Tool Packs step that seeds a visible MCP Hub default profile, validates safe tool execution directly, and lets setup continue without requiring a chat/LLM tool call.

**Architecture:** Add a thin setup-specific catalog/policy service that generates MCP Hub permission profiles from static v1 pack metadata, then expose it through three first-run setup endpoints. The WebUI adds one optional wizard step between `optional_advanced` and `first_chat`, using the backend as the source of truth for packs, add-ons, validation state, and effective tool summaries. Existing MCP Hub profiles, assignments, tool registry metadata, and first-run state storage remain authoritative; packs are not a second permission system.

**Tech Stack:** FastAPI, Pydantic, existing MCP Hub service/repo, existing MCP Hub tool registry, existing MCP Unified server/module execution, pytest, Next.js/React/TypeScript, Vitest/Testing Library, Tailwind utility classes, lucide-react.

---

Backlog: `TASK-12132`

Spec: `Docs/superpowers/specs/2026-07-04-first-run-mcp-tool-packs-design.md`

## Scope Decisions

- Use the existing TLDW MCP Hub DB-backed API. `profile_id` and `assignment_id` are numeric IDs in first-run setup responses.
- Store first-run provenance under `policy_document.first_run_mcp_tools`; MCP permission profiles do not have a standalone metadata column.
- Use `target_type: "default"`, `target_id: null`, `owner_scope_type: "global"`, and `owner_scope_id: null` for the single-user v1 default assignment.
- Keep `mcp_tools` optional. Do not add it to `REQUIRED_FIRST_RUN_STEPS`.
- Default generated policy uses explicit `allowed_tools`; do not grant module-wide patterns for mixed read/write modules.
- Do not add a separate pack database, migrations, governance-pack import, or manual external-server builder.
- Built-in sample validation should first use existing `mcp.tools.list`. Add a tiny diagnostic tool only if tests prove `mcp.tools.list` cannot be executed safely in setup.

## File Structure

Backend:

- Create `tldw_Server_API/app/core/Setup/first_run_mcp_tools.py`
  - Static v1 pack/add-on catalog.
  - Validation state constants.
  - Pack selection normalization.
  - Generated policy creation and hash calculation.
  - Safe persisted-state builder.
- Create `tldw_Server_API/app/services/setup_mcp_tools_service.py`
  - Orchestrates catalog, apply, validation, MCP Hub profile/assignment upsert, and manual-edit conflict handling.
  - Accepts injectable `McpHubService`, `McpHubToolRegistryService`, and MCP tool executor dependencies for tests.
- Modify `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
  - Add request/response models for catalog/apply/validate.
- Modify `tldw_Server_API/app/api/v1/endpoints/setup.py`
  - Add `mcp_tools` to `_FIRST_RUN_STEP_DATA_ALLOWED_KEYS`.
  - Add first-run MCP tools routes.
  - Wire service dependency factories.
- Modify `tldw_Server_API/app/services/mcp_hub_service.py` only if implementation needs a tiny helper to avoid duplicate profile/assignment lookup code.

Backend tests:

- Create `tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py`
- Create `tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py`
- Create or extend `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`
- Extend `tldw_Server_API/tests/Setup/test_first_run_state.py` only if optional-step behavior needs direct store coverage.

Frontend:

- Modify `apps/packages/ui/src/types/setup-onboarding.ts`
- Modify `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify `apps/packages/ui/src/hooks/useSetupOnboarding.ts`
- Modify `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Create `apps/packages/ui/src/components/Option/Onboarding/steps/McpToolsStep.tsx`
- Modify `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- Modify or create a tiny MCP Hub first-run status helper if the page is too large to edit directly.

Frontend tests:

- Extend `apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts`
- Extend `apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx`
- Extend `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`
- Create `apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx`

## Generated Policy Shape

Default profile policy document:

```json
{
  "allowed_tools": [
    "knowledge.search",
    "knowledge.get",
    "media.search",
    "media.get",
    "prompts.search",
    "prompts.get",
    "mcp.catalogs.list",
    "mcp.modules.list",
    "mcp.tools.list"
  ],
  "first_run_mcp_tools": {
    "setup_origin": "first_run_mcp_tools",
    "setup_instance_id": "first_run:2026-07-04T00:00:00+00:00",
    "catalog_version": "2026-07-04.v1",
    "selected_pack_ids": ["research"],
    "selected_addon_ids": [],
    "generated_policy_hash": "sha256:...",
    "last_generated_hash": "sha256:..."
  }
}
```

The hash input must include only:

```json
{
  "allowed_tools": ["..."],
  "selected_pack_ids": ["..."],
  "selected_addon_ids": ["..."],
  "catalog_version": "2026-07-04.v1"
}
```

Do not include display name, description, timestamps, profile DB IDs, assignment DB IDs, or user-edited fields outside `first_run_mcp_tools`.

## Task 1: Backend Catalog And Policy Generator

**Files:**

- Create: `tldw_Server_API/app/core/Setup/first_run_mcp_tools.py`
- Test: `tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py`

- [ ] **Step 1: Write failing catalog tests**

Add tests covering the contract before implementation:

```python
def test_default_catalog_exposes_five_selected_packs():
    catalog = build_mcp_tools_catalog(tool_entries=[])
    assert [pack["pack_id"] for pack in catalog["packs"]] == [
        "research",
        "learning",
        "writing",
        "media_library",
        "personal_knowledge",
    ]
    assert all(pack["default_selected"] is True for pack in catalog["packs"])


def test_default_policy_uses_explicit_allowed_tools_not_module_patterns():
    policy = generate_first_run_policy(
        selected_pack_ids=["research", "writing"],
        selected_addon_ids=[],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {"tool_name": "notes.search", "module": "notes", "risk_class": "low", "mutates_state": False},
            {"tool_name": "notes.create", "module": "notes", "risk_class": "high", "mutates_state": True},
        ],
    )
    assert "notes.search" in policy["allowed_tools"]
    assert "notes.create" not in policy["allowed_tools"]
    assert "module_patterns" not in policy
```

Also add tests for:

- unknown saved pack IDs are returned as unavailable legacy choices instead of dropped;
- strong add-ons require both `selected_addon_ids` and `confirmed_addon_ids` for the current confirmation version;
- default local file, external network, write, destructive, and process add-ons are off.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v
```

Expected: FAIL because `first_run_mcp_tools` does not exist.

- [ ] **Step 3: Implement static catalog constants**

Implement:

```python
CATALOG_VERSION = "2026-07-04.v1"
CONFIRMATION_VERSION = "2026-07-04.v1"
SETUP_ORIGIN = "first_run_mcp_tools"
PROFILE_DISPLAY_NAME = "First-run default"
VALIDATION_STATES = {
    "not_run",
    "built_in_passed",
    "external_discovered",
    "external_tool_passed",
    "no_safe_external_tool",
    "external_discovery_incomplete",
    "failed",
    "skipped",
}
```

Use the pack/tool lists from the spec verbatim. Build policies by intersecting selected pack tool names with currently registered low-risk read-only tool entries when entries are supplied. Keep the spec tool names available even when the current registry is unavailable, but mark missing tools as unavailable in the catalog response.

- [ ] **Step 4: Implement strict policy generation**

Policy generation rules:

- normalize unknown pack IDs into a legacy/unavailable bucket;
- include only explicit `allowed_tools`;
- never include `module_patterns` for first-run defaults;
- reject strong add-ons with a stale or missing confirmation version;
- include `network.external` and explicit low-risk external read tool names only when `external_network_read` is selected; do not include external write/process/filesystem tools;
- include local filesystem read tools only when `local_file_read` is selected, only if the tool registry marks them low-risk/read-only/path-boundable, and add `filesystem.read`;
- include writable TLDW tools only when `workspace_write` is selected and present in `confirmed_addon_ids` for the current `CONFIRMATION_VERSION`; enumerate tool names from registry metadata, do not add broad write module patterns;
- include destructive/delete tools only when `destructive_actions` is selected and confirmed; enumerate exact delete/destructive tool names and add `filesystem.delete` only when a selected destructive filesystem tool requires it;
- include process/run-command tools only when `process_run_command` is selected and confirmed; enumerate exact process tools and add `process.execute`;
- never include `filesystem.read`, `filesystem.write`, `filesystem.delete`, `network.external`, or `process.execute` in the default policy.

Add-on-to-policy tests should include one fake registry row per add-on:

```python
def test_local_file_read_addon_adds_only_safe_read_file_tools():
    policy = generate_first_run_policy(
        selected_pack_ids=["research"],
        selected_addon_ids=["local_file_read"],
        confirmed_addon_ids=[],
        confirmation_version=None,
        setup_instance_id="first_run:test",
        tool_entries=[
            {
                "tool_name": "fs.read_text",
                "module": "filesystem",
                "risk_class": "low",
                "mutates_state": False,
                "uses_filesystem": True,
                "path_boundable": True,
            },
            {
                "tool_name": "fs.write_text",
                "module": "filesystem",
                "risk_class": "high",
                "mutates_state": True,
                "uses_filesystem": True,
            },
        ],
    )
    assert "fs.read_text" in policy["allowed_tools"]
    assert "fs.write_text" not in policy["allowed_tools"]
```

- [ ] **Step 5: Run catalog tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit catalog slice**

```bash
git add tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py
git commit -m "feat: add first-run MCP tool pack catalog"
```

## Task 2: Backend MCP Hub Apply Service

**Files:**

- Create: `tldw_Server_API/app/services/setup_mcp_tools_service.py`
- Modify: `tldw_Server_API/app/core/Setup/first_run_mcp_tools.py`
- Test: `tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py`

- [ ] **Step 1: Write failing service tests with fakes**

Use fake MCP Hub and tool registry services. Required test cases:

```python
@pytest.mark.asyncio
async def test_apply_creates_profile_and_default_assignment():
    service = SetupMcpToolsService(hub=fake_hub, tool_registry=fake_registry)
    result = await service.apply_selection(
        state=first_run_state,
        request=McpToolsApplyRequest(selected_pack_ids=["research"], selected_addon_ids=[]),
    )
    assert result.profile_id == 1
    assert result.assignment_id == 10
    assert fake_hub.created_profiles[0]["name"] == "First-run default"
    assert fake_hub.created_assignments[0]["target_type"] == "default"
    assert fake_hub.created_assignments[0]["profile_id"] == 1
```

Also test:

- existing profile is found by `policy_document.first_run_mcp_tools.setup_origin` and `setup_instance_id`, not display name alone;
- repeated apply updates the generated policy when `last_generated_hash` matches;
- manual edit conflict returns a structured conflict without overwriting;
- `keep_existing` records profile/assignment IDs and current effective tool count without profile update;
- `replace_existing` overwrites only after explicit request;
- service returns a safe first-run step payload containing only allowlisted `mcp_tools` fields.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v
```

Expected: FAIL because the service does not exist.

- [ ] **Step 3: Implement service apply flow**

Implement `SetupMcpToolsService.apply_selection(...)`:

1. Load current tool entries from `McpHubToolRegistryService.list_entries()`.
2. Generate policy from pack/add-on selection.
3. Compute `setup_instance_id` from first-run state, using `first_run:{state.created_at.isoformat()}`.
4. Find profile by scanning `hub.list_permission_profiles(owner_scope_type="global", owner_scope_id=None)` for `policy_document.first_run_mcp_tools.setup_origin == "first_run_mcp_tools"` and matching `setup_instance_id`.
5. Create if missing:
   - name `First-run default`
   - owner scope global/null
   - mode `custom`
   - path scope null
   - generated policy document
6. Update if found and `last_generated_hash` matches.
7. Return `409`-style conflict data from the service if found and hash does not match.
8. Ensure a global default assignment exists by listing assignments with `target_type="default"`, `target_id=None`, owner global/null, then creating/updating one to point at the profile.

Service result should include:

```python
{
    "status": "applied",
    "profile_id": 1,
    "assignment_id": 10,
    "catalog_version": CATALOG_VERSION,
    "selected_pack_ids": [...],
    "selected_addon_ids": [...],
    "effective_tool_count": 12,
    "effective_tools": [...],
    "disabled_addons": [...],
    "validation_state": "not_run",
}
```

- [ ] **Step 4: Implement manual-edit conflict handling**

Conflict response shape:

```python
{
    "status": "conflict",
    "conflict": {
        "reason": "profile_manually_changed",
        "profile_id": 1,
        "current_hash": "sha256:...",
        "expected_hash": "sha256:...",
    },
}
```

`keep_existing` should not mutate the profile. `replace_existing` should overwrite the generated policy and write a new `last_generated_hash`.

- [ ] **Step 5: Run service tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit service slice**

```bash
git add tldw_Server_API/app/services/setup_mcp_tools_service.py tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py
git commit -m "feat: apply first-run MCP tool packs to MCP Hub"
```

## Task 3: Backend Setup Schemas And Endpoints

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py`
- Test: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`
- Test: `tldw_Server_API/tests/Setup/test_first_run_state.py`

- [ ] **Step 1: Write failing endpoint tests**

Add endpoint tests:

```python
def test_first_run_mcp_tools_catalog_returns_defaults(setup_client, monkeypatch, tmp_path):
    response = setup_client.get("/api/v1/setup/first-run/mcp-tools/catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["catalog_version"] == "2026-07-04.v1"
    assert {pack["pack_id"] for pack in body["packs"]} >= {"research", "learning", "writing"}


def test_first_run_mcp_tools_state_rejects_raw_external_config(setup_client, monkeypatch, tmp_path):
    response = setup_client.post(
        "/api/v1/setup/first-run/state",
        json={"step": "mcp_tools", "data": {"endpoint_config": {"url": "https://example.test"}}},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "unsupported_first_run_step_data"
```

Also test:

- `POST /api/v1/setup/first-run/mcp-tools/apply` persists `mcp_tools` step data with numeric `profile_id` and `assignment_id`;
- `POST /api/v1/setup/first-run/mcp-tools/apply` returns `409` for profile conflict;
- `POST /api/v1/setup/first-run/mcp-tools/validate` rejects validation before packs are saved;
- first-run completion is still allowed when `mcp_tools.validation_state` is `not_run`, `failed`, `skipped`, or absent, provided required steps and first chat are complete;
- public first-run state includes only allowlisted `mcp_tools` keys.

- [ ] **Step 2: Run endpoint tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_first_run_state.py -k "mcp_tools or complete_requires" -v
```

Expected: FAIL for missing endpoints/step allowlist.

- [ ] **Step 3: Add setup schemas**

Add Pydantic models:

```python
class McpToolsValidationState(str, Enum):
    NOT_RUN = "not_run"
    BUILT_IN_PASSED = "built_in_passed"
    EXTERNAL_DISCOVERED = "external_discovered"
    EXTERNAL_TOOL_PASSED = "external_tool_passed"
    NO_SAFE_EXTERNAL_TOOL = "no_safe_external_tool"
    EXTERNAL_DISCOVERY_INCOMPLETE = "external_discovery_incomplete"
    FAILED = "failed"
    SKIPPED = "skipped"
```

Add `McpToolsCatalogResponse`, `McpToolsApplyRequest`, `McpToolsApplyResponse`, `McpToolsValidateRequest`, and `McpToolsValidateResponse`. Keep response fields explicit; do not use `dict[str, Any]` for request bodies except for backend-owned summary sections.

- [ ] **Step 4: Add `mcp_tools` state allowlist**

In `setup.py`, add:

```python
"mcp_tools": frozenset(
    {
        "acknowledged",
        "selected_pack_ids",
        "selected_addon_ids",
        "confirmed_addon_ids",
        "confirmation_version",
        "validation_state",
        "profile_id",
        "assignment_id",
        "catalog_version",
        "effective_tool_count",
        "validated_at",
        "validation_message",
        "last_validation_run_id",
    }
),
```

Do not add `mcp_tools` to `REQUIRED_FIRST_RUN_STEPS`.

- [ ] **Step 5: Add route dependency factories**

Add factories near setup route helpers:

```python
async def get_setup_mcp_tools_service() -> SetupMcpToolsService:
    return SetupMcpToolsService(
        hub=await get_mcp_hub_service(),
        tool_registry=McpHubToolRegistryService(),
    )
```

Use a new local helper dependency for apply/validate that chooses the trust boundary by setup status:

```python
async def _require_mcp_tools_setup_or_admin_access(request: Request) -> None:
    status_snapshot = setup_manager.get_status_snapshot()
    if not bool(status_snapshot.get("setup_completed")):
        await _require_first_run_write_access(request)
        return

    # Do not resolve the auth principal before this point. Fresh first-run setup
    # intentionally permits local unauthenticated setup writes.
    principal = await get_auth_principal(request)
    permissions = set(principal.permissions) if principal is not None else set()
    if principal is None or not (principal.is_admin or SYSTEM_CONFIGURE in permissions or "*" in permissions):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="system_configure_required")
```

Use `_require_mcp_tools_setup_or_admin_access` for apply and validate. Use `require_local_setup_access` for the first-run catalog while setup is incomplete; after setup completion, catalog access must use normal auth or be served through a separate MCP Hub admin/recovery surface.

- [ ] **Step 6: Add routes**

Routes:

- `GET /api/v1/setup/first-run/mcp-tools/catalog`
- `POST /api/v1/setup/first-run/mcp-tools/apply`
- `POST /api/v1/setup/first-run/mcp-tools/validate`

After successful apply/validate, call `FirstRunStateStore.update_step("mcp_tools", safe_payload)`.

Also add admin-gated recovery aliases under setup admin paths so MCP Hub can run recovery after first-run setup is complete without using unauthenticated first-run trust:

- `GET /api/v1/setup/admin/mcp-tools/status`
- `POST /api/v1/setup/admin/mcp-tools/validate`

These admin paths should reuse `SetupMcpToolsService`, require `SYSTEM_CONFIGURE`, and return the same safe status/validation response shape. They should not reopen the first-run wizard or mark required first-run steps.

- [ ] **Step 7: Run backend endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_first_run_state.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit endpoint slice**

```bash
git add tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_first_run_state.py
git commit -m "feat: expose first-run MCP tool setup endpoints"
```

## Task 4: Safe Built-In And External Validation

**Files:**

- Modify: `tldw_Server_API/app/services/setup_mcp_tools_service.py`
- Modify: `tldw_Server_API/app/core/Setup/first_run_mcp_tools.py`
- Test: `tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py`
- Test: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`

- [ ] **Step 1: Write failing validation tests**

Required cases:

```python
@pytest.mark.asyncio
async def test_validate_runs_builtin_sample_when_no_external_servers():
    result = await service.validate_selection(saved_state=saved_mcp_state)
    assert result.validation_state == "built_in_passed"
    assert result.sample_tool_name == "mcp.tools.list"


@pytest.mark.asyncio
async def test_validate_no_safe_external_tool_is_not_error():
    fake_hub.external_servers = [{"id": "docs", "enabled": True}]
    fake_registry.entries = [{"tool_name": "external.docs.write", "module": "external_federation", "risk_class": "high"}]
    result = await service.validate_selection(saved_state=saved_mcp_state)
    assert result.validation_state == "no_safe_external_tool"
```

Also test:

- built-in sample is denied if removed from generated `allowed_tools`;
- external discovery failure returns `external_discovery_incomplete` and a redacted message;
- eligible external tool must have explicit trusted metadata, low risk, `mutates_state is False`, no filesystem/process/destructive capability, and no required input arguments;
- an eligible external no-arg read tool returns `external_tool_passed`;
- admin recovery validation after setup completion requires `SYSTEM_CONFIGURE`;
- unauthenticated first-run validation after setup completion returns `403`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py -k validate -v
```

Expected: FAIL because validation is not implemented.

- [ ] **Step 3: Implement built-in validation**

Implement built-in validation using existing `mcp.tools.list`:

1. Load saved/generated policy from current first-run `mcp_tools` state or profile.
2. Confirm `mcp.tools.list` is allowed by the generated policy. Use existing policy-pattern matching if practical; otherwise use exact match against generated `allowed_tools` in v1 and cover this with tests.
3. Execute `mcp.tools.list` through the MCP module execution path or `MCPProtocol._handle_tools_list` with a setup-scoped `RequestContext`.
4. Treat any returned `tools` list, including empty, as success.
5. Do not query media/notes/knowledge data for validation.

- [ ] **Step 4: Implement external discovery readiness**

Flow:

1. List enabled external servers from `McpHubService.list_external_servers()`.
2. If none exist, return built-in result with external status `not_configured`.
3. For configured servers, refresh discovery via the existing `external.tools.refresh` tool.
4. If refresh fails for every server, return `external_discovery_incomplete`.
5. If refresh succeeds but no eligible no-arg safe read-only tool exists, return `no_safe_external_tool`.
6. If an eligible tool exists, execute one with `{}` and return `external_tool_passed`.

Eligibility must fail closed:

```python
def is_safe_external_validation_candidate(entry, tool_def):
    return (
        entry.get("module") == "external_federation"
        and entry.get("metadata_source") == "explicit"
        and entry.get("risk_class") == "low"
        and entry.get("mutates_state") is False
        and entry.get("uses_filesystem") is False
        and entry.get("uses_processes") is False
        and "filesystem.write" not in entry.get("capabilities", [])
        and "filesystem.delete" not in entry.get("capabilities", [])
        and "process.execute" not in entry.get("capabilities", [])
        and not tool_def.get("inputSchema", {}).get("required")
    )
```

- [ ] **Step 5: Sanitize validation messages**

Validation messages must not include raw URLs with credentials, filesystem paths, tokens, command strings, stack traces, or exception reprs. Reuse the public setup sanitization pattern from `setup.py` or implement a local helper returning fixed messages such as:

- `Built-in MCP tool check passed.`
- `External discovery did not complete.`
- `No safe no-argument external read-only tool was available.`

- [ ] **Step 6: Run validation tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -k "mcp_tools or validate" -v
```

Expected: PASS.

- [ ] **Step 7: Commit validation slice**

```bash
git add tldw_Server_API/app/services/setup_mcp_tools_service.py tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
git commit -m "feat: validate first-run MCP tool execution"
```

## Task 5: Frontend API Types, Service, And Hook

**Files:**

- Modify: `apps/packages/ui/src/types/setup-onboarding.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/hooks/useSetupOnboarding.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts`
- Test: `apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx`

- [ ] **Step 1: Write failing frontend service tests**

Add tests for:

```ts
it("fetches the first-run MCP tools catalog without configured auth", async () => {
  vi.mocked(bgRequest).mockResolvedValueOnce({ catalog_version: "2026-07-04.v1", packs: [], addons: [] })

  await setupOnboardingMethods.getMcpToolsCatalog.call({})

  expect(bgRequest).toHaveBeenCalledWith({
    path: "/api/v1/setup/first-run/mcp-tools/catalog",
    method: "GET",
    noAuth: true,
  })
})
```

Also cover apply and validate POST bodies. Include a conflict test proving
`applyMcpTools` passes `expectedStatuses: [409]` to `bgRequest` and returns the
typed conflict body instead of throwing when the server reports an existing
non-generated MCP profile.

- [ ] **Step 2: Write failing hook tests**

Add `useSetupOnboarding` tests proving:

- `loadMcpToolsCatalog` stores response in hook state;
- `applyMcpTools` refreshes first-run state after success;
- `validateMcpTools` refreshes first-run state after success;
- errors bubble to callers so the step can show retry UI.

- [ ] **Step 3: Run frontend service/hook tests to verify failure**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx
```

Expected: FAIL for missing methods/types.

- [ ] **Step 4: Add TypeScript contract types**

Add:

```ts
export type McpToolsValidationState =
  | "not_run"
  | "built_in_passed"
  | "external_discovered"
  | "external_tool_passed"
  | "no_safe_external_tool"
  | "external_discovery_incomplete"
  | "failed"
  | "skipped"
```

Use `number | null` for `profile_id` and `assignment_id`.

- [ ] **Step 5: Add API methods and OpenAPI guard entries**

Add client paths:

- `/api/v1/setup/first-run/mcp-tools/catalog`
- `/api/v1/setup/first-run/mcp-tools/apply`
- `/api/v1/setup/first-run/mcp-tools/validate`
- `/api/v1/setup/admin/mcp-tools/status`
- `/api/v1/setup/admin/mcp-tools/validate`

Add service methods:

- `getMcpToolsCatalog`
- `applyMcpTools`
- `validateMcpTools`

Implement `applyMcpTools` with the existing `bgRequest` `expectedStatuses`
option:

```ts
return bgRequest<McpToolsApplyResponse>({
  path: "/api/v1/setup/first-run/mcp-tools/apply",
  method: "POST",
  body: payload,
  noAuth: true,
  expectedStatuses: [409],
})
```

Do not suppress any other non-OK status. The `409` response is part of the
wizard contract because the user must choose between keeping the existing MCP
Hub profile and replacing it with the generated first-run profile.

- [ ] **Step 6: Add hook state and methods**

Add hook state:

```ts
const [mcpToolsCatalog, setMcpToolsCatalog] =
  React.useState<McpToolsCatalogResponse | null>(null)
```

Return:

- `mcpToolsCatalog`
- `loadMcpToolsCatalog`
- `applyMcpTools`
- `validateMcpTools`

- [ ] **Step 7: Run frontend service/hook tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit frontend API slice**

```bash
git add apps/packages/ui/src/types/setup-onboarding.ts apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/hooks/useSetupOnboarding.ts apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx
git commit -m "feat: add first-run MCP tool setup client"
```

## Task 6: Frontend MCP Tools Wizard Step

**Files:**

- Create: `apps/packages/ui/src/components/Option/Onboarding/steps/McpToolsStep.tsx`
- Modify: `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx`

- [ ] **Step 1: Write failing step component tests**

Cover:

- default selected packs render checked;
- risky add-ons render collapsed/off by default;
- strong add-on selection requires confirmation UI before apply;
- `Save packs` calls `applyMcpTools`;
- backend conflict response shows `Keep existing` and `Replace generated profile`;
- `Keep existing` re-calls apply with conflict mode `keep_existing`;
- `Replace generated profile` re-calls apply with conflict mode `replace_existing`;
- `Run sample tool` calls `validateMcpTools` after save;
- `Continue` records `not_run` if packs were saved but validation was not run;
- `Skip MCP tools` records `skipped`;
- summary shows enabled packs, effective tool count, available tools, and `Open MCP Hub`.

- [ ] **Step 2: Write failing wizard flow tests**

Extend `UnifiedSetupWizard.test.tsx`:

- after `optional_advanced`, wizard goes to `mcp_tools`;
- if initial state has all required steps complete and no `mcp_tools` state, wizard starts at `mcp_tools`, not `first_chat`;
- if `mcp_tools` is completed/skipped/not_run, wizard starts at `first_chat`;
- first-run completion still only depends on first chat and required backend steps.

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: FAIL for missing component/wizard step.

- [ ] **Step 4: Implement `McpToolsStep`**

Use existing onboarding styling from `OptionalAdvancedStep` and `ProviderSetupStep`. Use lucide icons where useful, for example `Blocks`, `ShieldCheck`, `Play`, and `ExternalLink`.

Props:

```ts
type McpToolsStepProps = {
  catalog: McpToolsCatalogResponse | null
  loadCatalog: () => Promise<McpToolsCatalogResponse>
  applyMcpTools: (payload: McpToolsApplyRequest) => Promise<McpToolsApplyResponse>
  validateMcpTools: (payload: McpToolsValidateRequest) => Promise<McpToolsValidateResponse>
  onContinue: () => void
  onBack: () => void
  onSkip: () => void
}
```

UI behavior:

1. Load catalog on mount if missing.
2. Initialize selected packs from saved state, otherwise backend defaults.
3. Render pack cards as compact checkboxes with short purpose text.
4. Render add-ons inside collapsed sections; local file read is visible but off.
5. Disable `Run sample tool` until `Save packs` succeeds.
6. Show validation progress and final state.
7. On `status: "conflict"`, render a compact conflict panel with:
   - `Keep existing`, which applies `{ conflict_resolution: "keep_existing", profile_id }`;
   - `Replace generated profile`, which applies `{ conflict_resolution: "replace_existing", profile_id }`;
   - `Open MCP Hub`, linked to `/mcp-hub?source=first-run&profile_id=<id>`.
8. Link to `/mcp-hub?source=first-run`.
9. Do not include in-app tutorial copy about keyboard shortcuts or implementation internals.

- [ ] **Step 5: Wire wizard step**

Modify `WizardStep` union to include `"mcp_tools"`.

Update `stepFromState`:

```ts
if (!completed.has("optional_advanced")) return "optional_advanced"
const mcpToolsData = state?.step_data?.mcp_tools
const mcpToolsDone =
  mcpToolsData?.acknowledged === true ||
  mcpToolsData?.validation_state === "skipped"
if (!mcpToolsDone) return "mcp_tools"
return "first_chat"
```

After `OptionalAdvancedStep.onContinue`, navigate to `mcp_tools`. Back from first chat should return to `mcp_tools` when provider selection already exists.

- [ ] **Step 6: Run wizard tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit wizard slice**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/steps/McpToolsStep.tsx apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
git commit -m "feat: add MCP tool packs onboarding step"
```

## Task 7: MCP Hub Follow-Up Status, Polish, And Verification

**Files:**

- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- Modify: `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts`
- Modify: `apps/packages/ui/src/types/setup-onboarding.ts`
- Test: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.first-run-status.test.tsx` or the closest existing MCP Hub page test.
- Modify: `backlog/tasks/task-12132 - Plan-first-run-MCP-tool-packs-implementation.md` during task finalization.

- [ ] **Step 1: Write failing MCP Hub follow-up tests**

Cover required follow-up states:

- `validated during setup` for `built_in_passed` or `external_tool_passed`;
- `not validated during setup` for `not_run`;
- `validation failed` for `failed`;
- `external discovery incomplete`;
- `profile manually changed` when admin status detects generated hash mismatch;
- recovery button calls admin validate endpoint for skipped/failed/not-run states.

- [ ] **Step 2: Add MCP Hub status/recovery API client**

Add admin-authenticated methods to the setup onboarding domain or a small MCP Hub recovery domain:

- `getMcpToolsRecoveryStatus`: `GET /api/v1/setup/admin/mcp-tools/status`
- `validateMcpToolsRecovery`: `POST /api/v1/setup/admin/mcp-tools/validate`

These must not set `noAuth: true`.

- [ ] **Step 3: Add MCP Hub follow-up panel**

Add a compact panel near the profiles/assignments overview when a first-run profile exists or the URL has `source=first-run`.

The panel should:

- show the status labels from Step 1;
- link directly to the `First-run default` profile when `profile_id` is available;
- offer `Run validation` for skipped/failed/not-run/discovery-incomplete states;
- offer `Review profile` for `profile manually changed`;
- avoid blocking normal MCP Hub management if the status endpoint fails.

- [ ] **Step 4: Run MCP Hub follow-up tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.first-run-status.test.tsx apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run focused backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_first_run_state.py -v
```

Expected: PASS.

- [ ] **Step 6: Run focused frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Verify OpenAPI guard**

Run from `apps/packages/ui`:

```bash
bun run verify:openapi
```

Expected: PASS. If it fails because the backend-generated OpenAPI spec differs, reconcile the new first-run MCP paths in the guard or backend schema.

- [ ] **Step 8: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py -f json -o /tmp/bandit_first_run_mcp_tools.json
```

Expected: no new findings in touched code. If there are findings, fix them before continuing.

- [ ] **Step 9: Run diff checks**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 10: Update Backlog task**

Use Backlog.md MCP/CLI:

```bash
backlog task edit TASK-12132 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "Implemented first-run MCP tool packs setup plan and recorded verification results." --plain
```

If implementation work uses a separate task, update that implementation task instead and leave `TASK-12132` as the planning record.

- [ ] **Step 11: Final commit**

```bash
git status --short
git add tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_first_run_state.py apps/packages/ui/src/types/setup-onboarding.ts apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/hooks/useSetupOnboarding.ts apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx apps/packages/ui/src/components/Option/Onboarding/steps/McpToolsStep.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.first-run-status.test.tsx "backlog/tasks/task-12132 - Plan-first-run-MCP-tool-packs-implementation.md"
git commit -m "feat: add first-run MCP tool packs setup"
```

## Risk Checks Before Implementation

- Profile lookup by display name alone is a bug. Use `policy_document.first_run_mcp_tools.setup_origin` and `setup_instance_id`.
- `mcp_tools` becoming required is a bug. Keep it out of `REQUIRED_FIRST_RUN_STEPS`.
- Default policy containing `module_patterns` for `notes`, `quizzes`, `flashcards`, or `slides` is a bug.
- External validation must fail closed when safety metadata is heuristic, missing, or conflicting.
- Validation failure must not block setup completion.
- First-run setup endpoints must reject raw external server config, credentials, URLs with secrets, filesystem paths, command strings, and arbitrary nested tool config.
- Frontend must use numeric `profile_id`/`assignment_id`.
- Local file read must remain off by default.

## Final Verification Command Set

Run before claiming implementation complete:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_mcp_tools_catalog.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/Setup/test_first_run_state.py -v
```

```bash
bunx vitest run apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts apps/packages/ui/src/hooks/__tests__/useSetupOnboarding.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/McpToolsStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
```

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Setup/first_run_mcp_tools.py tldw_Server_API/app/services/setup_mcp_tools_service.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py -f json -o /tmp/bandit_first_run_mcp_tools.json
```

```bash
git diff --check
```
