# llama.cpp Managed Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the backend-owned llama.cpp managed runtime so self-hosted admins can safely run multiple durable llama.cpp profiles, recover supervised services, preserve V1 compatibility, and use profile capability metadata from the WebUI.

**Architecture:** Build on the runtime already present on `origin/dev`: `llamacpp_profile_store`, `llamacpp_supervisor_service`, `llamacpp_process_runner`, profile/instance endpoints, asset inventory, mmproj launch resolution, and Admin profiles/runtime panels. This plan hardens supervision, lifecycle reconciliation, validation, provider routing, and final WebUI verification without creating a second process manager or frontend-owned runtime. Remote downloads and catalogs remain deferred to the acquisition/import workflow plan.

**Tech Stack:** FastAPI, Pydantic v2, JSON-backed profile store, asyncio process supervision, existing llama.cpp Admin endpoints/schemas, pytest/TestClient, React/Ant Design shared UI, Vitest/testing-library, Playwright E2E, Bandit.

---

## References

- Roadmap design: `Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md`
- Stage 1 plan already landed: `Docs/superpowers/plans/2026-05-16-llamacpp-managed-runtime-stage1-implementation-plan.md`
- Asset Inventory V2 plan already landed: `Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md`
- Model-family/mmproj plan already landed: `Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md`
- Saved profile editor plan already landed: `Docs/superpowers/plans/2026-05-16-llamacpp-admin-profile-editor-plan.md`
- Local acquisition/import plan: `Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md`
- Current endpoint module: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Current runtime models: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
- Current profile store: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py`
- Current supervisor: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Current process runner: `tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py`
- Current profile capability resolver: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py`
- Current Admin page: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Current Admin panels: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`, `LlamacppProfilesPanel.tsx`, `LlamacppRuntimePanel.tsx`
- Tracking task: `TASK-418`

## Current Baseline

Already present on current `origin/dev`:

- Durable JSON profile store in `llamacpp_profile_store.py`.
- Profile/runtime Pydantic models in `llamacpp_runtime_models.py`.
- Multi-profile manual supervisor in `llamacpp_supervisor_service.py`.
- One-process runner in `llamacpp_process_runner.py`.
- Asset inventory with GGUF/mmproj/folder/unknown support in `llamacpp_inventory_service.py`.
- Profile capability and mmproj launch resolution in `llamacpp_profile_capabilities.py`.
- Admin endpoints for assets, profiles, lifecycle actions, instances, profile log tails, V1 wrappers, and hardware.
- WebUI client/types and Admin profiles/runtime/assets panels.
- Tests covering profile store, process runner, supervisor basics, runtime API, inventory, provider/log APIs, profile capabilities, and model metadata.

Remaining work should harden and connect these pieces instead of rebuilding them. Treat this document as a consolidation and closeout plan for the managed-runtime sprint, not a replacement for the already-landed stage plans.

## Scope Guardrails

- Do not reintroduce a frontend-owned process manager.
- Do not replace the current V1 endpoints; keep them as wrappers around the default profile.
- Do not silently wire profiles into Chat on start. `use-in-chat` remains explicit.
- Do not add remote model downloads, curated catalogs, Hugging Face auth, or marketplace UX in this plan.
- Do not weaken allowlist checks, symlink resolution, or raw-argument validation.
- Do not hard-block advisory hardware warnings when path, args, and ports are otherwise safe.

## Command Note

Run Python verification commands after activating the project virtual environment from the repository root, for example `source .venv/bin/activate`. Frontend commands should run from `apps/packages/ui` unless noted otherwise.

## Task 1: Supervision Reconciliation And Durable Failure State

**Goal:** Add startup/shutdown reconciliation for autostart profiles plus bounded crash-restart behavior and durable last-failure state.

**Files:**
- Create: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_reconciler.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/LLM_Inference_Manager.py`
- Modify: `tldw_Server_API/app/services/lifespan_startup_sequence.py` or the narrow startup helper it delegates to, matching the current lifecycle pattern
- Modify: `tldw_Server_API/app/services/lifespan_shutdown_sequence.py` or the narrow shutdown helper it delegates to, matching the current lifecycle pattern
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_reconciler.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py`
- Test: existing lifecycle tests only if startup/shutdown registration requires it

- [ ] **Step 1: Write failing reconciler tests**

Add tests covering:

```python
async def test_reconciler_autostarts_enabled_profiles_with_autostart(tmp_path):
    # store has enabled autostart profile
    # reconciler calls supervisor.start_profile(profile_id)
    # non-autostart profiles are skipped
```

Also cover:
- disabled profiles do not autostart;
- failed start records durable `last_error`, `exit_code` when available, `restart_count`, and `state="failed"`;
- restart policy stops retrying after `max_restarts`;
- `pause_profile()` suppresses reconciliation until resume;
- shutdown stops supervisor-owned runners without trying to adopt arbitrary old PIDs.

- [ ] **Step 2: Run reconciler tests to verify failure**

```bash
python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_reconciler.py -v
```

Expected: FAIL because `llamacpp_runtime_reconciler.py` does not exist.

- [ ] **Step 3: Add reconciler service**

Implement a small async service with methods:

```python
class LlamaCppRuntimeReconciler:
    def __init__(self, supervisor: LlamaCppSupervisor, *, sleep: Callable = asyncio.sleep): ...
    async def reconcile_startup(self) -> list[LlamaCppRuntime]: ...
    async def reconcile_once(self) -> list[LlamaCppRuntime]: ...
    async def shutdown(self) -> None: ...
```

Keep it thin: the reconciler decides which profiles need action; the supervisor still owns profile locks, runner state, and process operations.

- [ ] **Step 4: Persist last failure metadata**

Extend the store/profile model conservatively. Prefer a dedicated optional field such as:

```python
last_runtime_failure: dict[str, object] = Field(default_factory=dict)
```

Do not store unbounded logs or full raw environment. Persist only bounded diagnostic fields needed after restart.

- [ ] **Step 5: Register lifecycle hooks**

Attach the reconciler to app startup only when `llm_manager.llamacpp_supervisor` exists. Store the handle on `app.state` so shutdown can call it. Follow existing lifecycle helper style; avoid adding direct heavy imports to `main.py`.

- [ ] **Step 6: Run focused backend tests**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_reconciler.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_reconciler.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
  tldw_Server_API/app/core/Local_LLM/LLM_Inference_Manager.py \
  tldw_Server_API/app/services \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_reconciler.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py
git commit -m "feat: reconcile llama.cpp runtime profiles"
```

## Task 2: Profile Validation And Launch Policy Hardening

**Goal:** Make profile creation/update fail closed for unsafe or internally conflicting launch definitions while preserving warnings for advisory risks.

**Files:**
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py`

- [x] **Step 1: Write failing validation tests**

Cover:
- duplicate explicit host/port conflicts across enabled profiles, including wildcard host conflicts;
- disabled profiles do not block explicit ports;
- `vision` profiles require a valid mmproj asset;
- `server_args["mmproj"]` and `mmproj_model_id` must resolve to the same file if both are present;
- raw args cannot override reserved structured flags such as host, port, model, and mmproj unless the backend explicitly allows it;
- path-like args such as `grammar_file`, `chat_template_file`, `prompt_cache`, and `lora_base` are resolved through the same allowlist policy or rejected.

- [x] **Step 2: Run validation tests to verify failure**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py \
  -k "profile or mmproj or reserved or allowlist or port" -v
```

Expected: FAIL for the new cases.

- [x] **Step 3: Centralize validation**

Add helper functions near the runtime/profile modules rather than duplicating validation across endpoints and UI. The supervisor should call the helper before persisting and before starting, because persisted state and current asset state can diverge.

- [x] **Step 4: Preserve advisory warnings**

Keep hardware/resource fit warnings in response/runtime warning lists. Only hard-fail unsafe paths, invalid ports, reserved flag conflicts, missing required assets, or incompatible mode/asset combinations.

- [x] **Step 5: Run focused backend tests**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py
git commit -m "fix: harden llama.cpp profile launch validation"
```

## Task 3: V1 Compatibility And Multi-Profile API Contract

**Goal:** Prove the legacy one-server API still targets only the reserved default profile while new profile/instance APIs expose all managed runtimes.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py`
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py`
- Test: `tldw_Server_API/tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py`

- [ ] **Step 1: Write failing compatibility tests**

Add TestClient coverage for:
- `POST /api/v1/llamacpp/start_server` updates/starts only the default profile;
- `POST /api/v1/llamacpp/start-by-model` updates/starts only the default profile;
- `GET /api/v1/llamacpp/status` keeps the legacy response shape and does not return every profile;
- `GET /api/v1/llamacpp/instances` returns both default and non-default profiles;
- `GET /api/v1/llamacpp/logs/tail` maps default-profile stopped/not-running state to HTTP 409;
- all new profile/runtime routes require admin permissions.

- [ ] **Step 2: Run API tests to verify failure**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -v
```

Expected: FAIL only for newly added coverage.

- [ ] **Step 3: Patch endpoint/schema gaps**

Keep response mappings narrow:
- 400 for validation/path/config mistakes;
- 404 for missing profile IDs;
- 409 for stopped/not-running state conflicts;
- 500 only for unexpected provider write failures or internal errors.

- [ ] **Step 4: Run API tests**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py \
  tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_lifecycle_api_contract.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
git commit -m "test: lock llama.cpp runtime API compatibility"
```

## Task 4: Provider Metadata And Routing Follow-Through

**Goal:** Make managed profiles visible and usable through existing model/provider metadata without hardcoded frontend assumptions.

**Files:**
- Modify: `tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
- Modify: `tldw_Server_API/app/core/Usage/pricing_catalog.py` only if the existing provider metadata source requires a catalog hook
- Test: `tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py`
- Test: relevant `llm_providers` metadata tests under `tldw_Server_API/tests/unit`

- [ ] **Step 1: Write failing metadata tests**

Cover:
- chat profile appears as chat-capable;
- vision profile with valid mmproj advertises text+image input;
- embedding profile advertises embedding output and is not treated as chat-only;
- rerank profile advertises score output;
- invalid/stale profile assets return bounded capability warnings instead of breaking the whole metadata endpoint;
- disabled profiles are clearly marked not configured or omitted according to the existing metadata convention.

- [ ] **Step 2: Run metadata tests to verify failure**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py \
  tldw_Server_API/tests/unit/test_llm_providers_error_mapping.py -v
```

Expected: FAIL for newly added cases if the metadata endpoint does not yet use the managed profile contract fully.

- [ ] **Step 3: Patch metadata assembly**

Use `managed_profile_model_metadata()` as the single translation point for managed profile metadata. Avoid adding frontend-only capability inference.

- [ ] **Step 4: Run metadata tests**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py \
  tldw_Server_API/tests/unit/test_llm_providers_error_mapping.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py \
  tldw_Server_API/app/api/v1/endpoints/llm_providers.py \
  tldw_Server_API/app/core/Usage/pricing_catalog.py \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py \
  tldw_Server_API/tests/unit/test_llm_providers_error_mapping.py
git commit -m "feat: expose llama.cpp managed profile metadata"
```

## Task 5: Admin WebUI Runtime Console Hardening

**Goal:** Make the `/admin/llamacpp` console reliable for repeated self-hosted operations: assets, profiles, runtime state, explicit Chat wiring, and warnings-first guidance.

**Files:**
- Modify: `apps/packages/ui/src/types/llamacpp-admin.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx`

- [ ] **Step 1: Write failing UI tests**

Cover:
- profile form can select a GGUF asset and optional mmproj asset;
- vision mode requires projector selection in the UI before submit when no manual mmproj arg is present;
- runtime panel shows running, stopped, failed, and paused states with endpoint and warnings;
- `Use in Chat` is shown only for running profiles;
- duplicate/edited profile forms preserve `provider_alias`, tags, mode, and autostart fields;
- asset warnings render without blocking profile creation unless the backend rejects the request.

- [ ] **Step 2: Run UI tests to verify failure**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: FAIL for newly added cases.

- [ ] **Step 3: Patch UI/client gaps**

Keep the page operations-console style. Do not create landing-page copy or a separate runtime wizard. Prefer compact forms, explicit action buttons, warnings, tags, and status rows.

- [ ] **Step 4: Run UI tests**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add \
  apps/packages/ui/src/types/llamacpp-admin.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx \
  apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx \
  apps/packages/ui/src/components/Option/Admin/__tests__
git commit -m "feat: harden llama.cpp runtime admin console"
```

## Task 6: Real-Server Verification, Docs, And Rollout Closeout

**Goal:** Add final documentation and smoke coverage so the managed runtime can be merged with clear operational expectations.

**Files:**
- Modify: `Docs/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md`
- Modify: `Docs/Published/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md`
- Modify: `Docs/API-related/llamacpp_integration_modes.md`
- Modify: `Docs/Published/API-related/llamacpp_integration_modes.md`
- Create or modify: `apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts`
- Modify: `backlog/tasks/task-418 - Plan-llama.cpp-managed-runtime-closeout-implementation.md` only for task closeout if this implementation happens in the same branch family
- Test: focused backend, frontend, E2E, Bandit, diff checks

- [ ] **Step 1: Write E2E smoke coverage**

Use mocked backend responses unless the existing E2E server fixture can safely expose llama.cpp admin stubs. Cover:
- assets load;
- profile list loads;
- runtime list loads;
- failed/stopped/running states render;
- `Use in Chat` is not offered for stopped profiles;
- backend warnings display as warnings, not hard-blocking full page load.

- [ ] **Step 2: Update docs**

Document:
- profile store location and purpose;
- default profile compatibility behavior;
- local import/register semantics;
- mmproj pairing expectations;
- autostart/restart policy limitations;
- remote downloads/catalogs are deferred to the acquisition workflow.

- [ ] **Step 3: Run backend verification**

```bash
python -m pytest \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_process_runner.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py \
  tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_capabilities.py \
  tldw_Server_API/tests/LLM_Local/test_llm_models_metadata_llamacpp_profiles.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -v
```

Expected: PASS.

- [ ] **Step 4: Run frontend verification**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx \
  src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run E2E smoke if environment is available**

```bash
# Set TLDW_E2E_API_KEY in your shell from your local test configuration first.
TLDW_E2E_SERVER_URL=127.0.0.1:8000 \
bunx playwright test apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts --reporter=line
```

Expected: PASS, or document the missing server/browser environment blocker.

- [ ] **Step 6: Run Bandit on touched backend paths**

```bash
python -m bandit \
  -r tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_reconciler.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_process_runner.py \
     tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py \
     tldw_Server_API/app/api/v1/endpoints/llamacpp.py \
  -f json -o /tmp/bandit_llamacpp_runtime.json
```

Expected: no high/medium findings in touched code. Fix new findings before closeout.

- [ ] **Step 7: Run diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only expected files are modified.

- [ ] **Step 8: Commit Task 6**

```bash
git add \
  Docs \
  apps/tldw-frontend/e2e/workflows/llamacpp-runtime-admin.spec.ts \
  backlog/tasks/task-418\ -\ Plan-llama.cpp-managed-runtime-closeout-implementation.md
git commit -m "docs: close out llama.cpp managed runtime rollout"
```

## Final PR Checklist

- [ ] Backend focused tests pass.
- [ ] Frontend focused tests pass.
- [ ] E2E smoke passes or blocker is documented.
- [ ] Bandit runs on touched backend Python paths.
- [ ] `git diff --check` passes.
- [ ] Backlog child tasks are current.
- [ ] PR body includes a human-owned `Change summary` placeholder for maintainer completion.
- [ ] Remote downloads/catalogs remain deferred and are not partially implemented in this runtime PR.
