# MCP Hub Live Discovery and Chat Payload Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship PR 1 of the MCP Hub walkthrough remediation: adding or changing an external MCP server in MCP Hub must update the live MCP runtime without a backend restart, the WebUI must invalidate stale MCP tool state after refresh, chat request payloads must include MCP tools only when the selected chat model and MCP state make them valid, and a degraded API health response must not block the app shell.

**Architecture:** Add an explicit backend refresh/reconcile command at the MCP Hub external-server boundary, backed by the live `get_mcp_server()` singleton and the external federation module's `ExternalServerManager`. Reuse the existing MCP tool discovery query/store and shared chat-tool normalization utilities so MCP Hub, raw preview, and `pageAssistModel` stay aligned. Keep readiness handling in the frontend networking gate as a narrow degraded-health relaxation.

**Tech Stack:** FastAPI, Pydantic, MCP Unified module registry, pytest, Next.js/React, Ant Design, TanStack Query, Zustand, Vitest, Testing Library.

---

## Stage 1: Backend Runtime Refresh/Reconcile

**Goal:** Provide a mutation endpoint that reconciles MCP Hub managed external-server storage with the already-running MCP Unified external federation module.

**Success Criteria:**
- MCP Hub exposes `POST /api/v1/mcp/hub/external-servers/refresh-discovery` with optional `server_id`.
- The endpoint requires the same mutation permission path as create/update/delete external-server endpoints.
- The endpoint resolves the live `get_mcp_server()` singleton, initializes it if needed, locates the external federation module by preferred id `external_federation` with a class-name fallback, and returns a typed response.
- New, updated, disabled, and deleted managed servers are reconciled without backend restart.
- Module capability caches and module registry tool mappings are refreshed after reconciliation so `/api/v1/mcp/tools`, the MCP Hub Tool Catalog, and chat execution see the same tools.

**Implementation Tasks:**
- [ ] Add `ExternalServerDiscoveryRefreshResponse` to `tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py` with fields like `ok`, `server_id`, `reconciled_servers`, `refreshed_servers`, `total_servers`, `virtual_tools`, and `errors`.
- [ ] Add `ExternalServerManager.reconcile_servers(server_id: str | None = None)` in `tldw_Server_API/app/core/MCP_unified/external_servers/manager.py`.
- [ ] In `reconcile_servers`, reload the configured server list through `_server_loader` when present, fall back to `load_external_server_registry`, filter enabled servers, close and remove adapters for deleted/disabled servers, replace adapters for materially changed server config, create adapters for new servers, connect/discover refreshed targets, and clear tools for failures without clearing unrelated servers.
- [ ] Keep `refresh_discovery()` focused on existing adapters; route the Hub endpoint through `reconcile_servers()` so storage changes are included.
- [ ] Add a public registry method in `tldw_Server_API/app/core/MCP_unified/modules/registry.py`, for example `refresh_module_registries(module_id: str)`, that removes old mappings for the module and re-runs `_update_registries()` on the current module instance.
- [ ] Add a helper in `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py` that resolves the external federation module from the live MCP server, calls `reconcile_servers()`, invalidates the module's capability caches, refreshes the module registry mapping, and raises a clear 503 if the module is unavailable.
- [ ] Add the endpoint on the non-conflicting collection path `/external-servers/refresh-discovery`, placed before nearby parameterized external-server routes for readability; keep `/external-servers/{server_id}` paths unchanged.
- [ ] Add `ExternalFederationModule.validate_tool_arguments()` so write-capable federation calls satisfy protocol pre-exec validation. Validate `external.tools.refresh` as `{ server_id?: string }`, allow external virtual tool argument dictionaries through after sanitization, and validate `__confirm_write` as boolean when present so upstream write confirmation is not silently malformed.

**Tests:**
- [ ] Add manager tests in `tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py` for new server discovery after initialization, disabled/deleted server removal, changed server replacement, and partial reconciliation failure isolation.
- [ ] Add MCP Hub endpoint tests in `tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py` for permission enforcement, successful live refresh, and external federation unavailable response.
- [ ] Add protocol/module coverage in `tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py` or adjacent tests for `external.tools.refresh` validation and external write-tool validator behavior.

**Status:** Complete

**Verification Notes:** Committed in `118004a1d`. Focused MCP backend pytest passed 61 tests; Bandit over touched backend production files reported 0 findings; `git diff --check` passed.

---

## Stage 2: Frontend Refresh Hooks and Query Invalidation

**Goal:** Make MCP Hub external-server mutations and Tool Catalog refreshes update frontend-visible MCP tools immediately after the backend runtime refresh completes.

**Success Criteria:**
- Creating an enabled managed server calls single-server refresh after save and invalidates MCP tool queries.
- Updating an enabled server calls single-server refresh; updating a server to disabled calls collection refresh/reconcile.
- Importing a legacy server calls single-server refresh for the imported managed server.
- Deleting a managed server calls collection refresh/reconcile so stale virtual tools are removed.
- Tool Catalog includes an explicit refresh action that runs collection refresh and reloads registry summary.
- Save/import/delete success messages distinguish “saved and refreshed” from “saved but refresh failed”.

**Implementation Tasks:**
- [ ] Add `refreshExternalServerDiscovery(serverId?: string)` to `apps/packages/ui/src/services/tldw/mcp-hub.ts`, targeting `POST /api/v1/mcp/hub/external-servers/refresh-discovery`.
- [ ] In `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`, add refresh state plus a small helper that calls the service and invalidates `["mcp-tools"]`, `["mcp-tool-catalogs"]`, `["mcp-tool-modules"]`, and `["mcp-health"]` queries via TanStack Query.
- [ ] Wire post-mutation refresh into `handleSaveServer`, `handleImport`, and `handleDeleteServer` with the create/update/delete rules above.
- [ ] Add visible transient refresh feedback without blocking the existing form flow; warnings should preserve the successful persistence result when runtime refresh fails.
- [ ] In `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx`, add a refresh button/action near the registry intro, reuse the refresh service, reload `getToolRegistrySummary()`, and surface refresh errors.

**Tests:**
- [ ] Extend `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx` for create/import/update/delete refresh calls and refresh-failure warning copy.
- [ ] Extend `apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx` for the explicit refresh action and registry reload.

**Status:** Complete

**Verification Notes:** Committed in `3a2b3bb75`. Package-local Vitest for `ExternalServersTab` and `ToolCatalogsTab` passed 13 tests; `git diff --check` passed. Bandit was not applicable to frontend-only TypeScript changes.

---

## Stage 3: Shared Chat Tool Eligibility and Omission Reasons

**Goal:** Ensure MCP tools enter `/api/v1/chat/completions` payloads only when valid, and make omission reasons inspectable in the raw preview/control state instead of silently dropping tools.

**Success Criteria:**
- `pageAssistModel` and normal/comparison raw preview use the same utility for MCP tool eligibility.
- Tools are omitted when the model lacks `tools`, MCP is unavailable/unhealthy, tool choice is `none`, all tools are disabled/colliding/unexecutable, or no normalized request tools remain.
- When tools are included, `tool_choice` is included only for valid supported values and the existing loop-compat header behavior is preserved.
- Raw preview for normal and comparison mode shows the wire payload accurately and exposes the omission reason outside the wire body.
- Character `complete-v2`, image generation, and specialized flows remain out of scope unless they already use `pageAssistModel`.

**Implementation Tasks:**
- [x] Extend `apps/packages/ui/src/utils/chat-tools.ts` with a shared resolver, for example `resolveChatToolRequest({ tools, toolChoice, modelSupportsTools, mcpHealthState, hasMcp })`, returning `tools`, `toolChoice`, `omittedReason`, and filter counts.
- [x] Update `apps/packages/ui/src/models/index.ts` to use the shared resolver, preserving current model capability lookup and `X-TLDW-Loop-Compat` header behavior only when effective tools exist.
- [x] Update `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts` to use the same resolver for normal and comparison previews.
- [x] Extend `ChatRequestDebugSnapshot` in `apps/packages/ui/src/services/tldw/chat-request-debug.ts` with optional metadata/debug fields for `toolOmissionReason` so the captured debug snapshot can explain omitted tools without polluting the request body.
- [x] Thread the resolver status into `useMcpToolsControl` or the raw request modal state only where the control already has enough information; keep copy concise and avoid adding a broad new diagnostics panel in PR 1.

**Tests:**
- [x] Extend `apps/packages/ui/src/utils/__tests__/chat-tools.test.ts` for each omission reason and one successful inclusion case.
- [x] Extend `apps/packages/ui/src/models/__tests__/pageAssistModel.mcp-tools.test.ts` for model-without-tools, unhealthy MCP, `none` tool choice, collision-only tools, and successful inclusion.
- [x] Extend `apps/packages/ui/src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx` for omission reasons and comparison mode parity.

**Status:** Complete

**Verification Notes:** Initial Stage 3 focused Vitest passed 26 tests and `git diff --check` passed. Quality follow-up added RED tests for live debug metadata forwarding; RED failed on missing lower-level client options, then package-local Vitest passed 42 tests across `chat-tools`, `pageAssistModel`, raw preview, and `TldwChatService` coverage. `git diff --check` passed. Focused quality re-review approved the fix. PR review follow-up cleaned the Stage 3 task/status bookkeeping.

---

## Stage 4: Degraded Health Readiness Gate

**Goal:** Allow the WebUI app shell to load when the API health endpoint reports degraded service status rather than a total outage.

**Success Criteria:**
- `ServerReadinessGate` treats HTTP 200/206 with `status: degraded`, `status: healthy`, or `status: ok` as app-enterable.
- Unreachable, malformed, and explicitly unhealthy responses still show the existing waiting/timeout behavior.
- A degraded response can surface a lightweight non-blocking diagnostic indicator without replacing the app shell.

**Implementation Tasks:**
- [ ] Change `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx` from boolean `checkHealth()` to a structured readiness result.
- [ ] Treat `res.status === 206` and degraded health body as enterable; preserve retry behavior for network failures and unhealthy statuses.
- [ ] Add a minimal degraded banner/state hook only in this gate if practical; otherwise limit PR 1 to unblocking degraded entry and leave richer diagnostics to PR 2.

**Tests:**
- [ ] Add readiness tests in `apps/tldw-frontend/__tests__/app/app-networking-guard.test.tsx` or a focused gate test to cover HTTP 206 degraded entry, healthy entry, and unhealthy retry/timeout behavior.

**Status:** Complete

**Verification Notes:** Stage 4 RED focused Vitest failed as expected on HTTP 206 degraded and HTTP 200 degraded health responses staying in the retrying gate. Implemented structured readiness parsing that accepts HTTP 200/206 with `status: degraded`, `status: healthy`, or `status: ok`. Focused readiness Vitest passed 6 tests.

---

## Stage 5: Verification and PR Packaging

**Goal:** Verify the PR 1 slice with focused backend/frontend tests, security scan touched backend files, and record results in Backlog.

**Success Criteria:**
- Focused tests pass locally.
- Bandit runs on touched backend production files with no new findings.
- Backlog `TASK-223.1` records implementation notes, verification output, touched files, and final summary.
- The final PR notes call out why refresh is explicit and why chat eligibility is shared.

**Verification Commands:**
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py tldw_Server_API/tests/MCP_unified/test_mcp_protocol_external_federation.py -q`
- [x] `cd apps/packages/ui && bun run test src/utils/__tests__/chat-tools.test.ts src/models/__tests__/pageAssistModel.mcp-tools.test.ts src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx src/services/__tests__/tldw-chat.message-sanitization.test.ts src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx`
- [x] `cd apps/tldw-frontend && bun run test:run components/networking/__tests__/ServerReadinessGate.test.tsx`
- [x] `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/registry.py tldw_Server_API/app/core/MCP_unified/modules/implementations/external_federation_module.py -f json -o /tmp/bandit_mcp_hub_pr1.json`
- [x] `git diff --check`

**Status:** Complete

**Verification Notes:** Backend focused pytest passed 61 tests with 5 warnings. UI package focused Vitest passed 55 tests across six files. Frontend readiness Vitest passed 6 tests. Bandit reported 0 findings in `/tmp/bandit_mcp_hub_pr1.json`. `git diff --check` passed.
