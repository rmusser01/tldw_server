# MCP Hub Setup Recovery UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MCP Hub setup and recovery task-led by adding a shared readiness model, clear no-auth/credential states, actionable Tool Catalog recovery, guided first-success setup, diagnostics, and truthful status surfaces.

**Architecture:** Start with a Stage 0 code-discovery checkpoint, then build a frontend-owned readiness mapper that can consume the existing server/catalog responses and later accept a backend readiness response without changing UI consumers. Wire the mapper into the smallest reviewable UX slice first: managed server rows and Tool Catalog recovery. Add backend endpoints only if Stage 0 proves current APIs cannot support the required state.

**Tech Stack:** React 18, TypeScript, Ant Design, Vitest/jsdom, Testing Library, FastAPI/Pydantic/Pytest if backend support is needed, Bandit for touched backend Python code.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-26-mcp-hub-setup-recovery-ux-design.md`
- Existing remediation spec: `Docs/superpowers/specs/2026-05-10-mcp-hub-walkthrough-remediation-design.md`
- Backlog: `TASK-223.2`

## Scope And Rollout

Implement this as phased reviewable slices. Do not attempt the full MCP Hub redesign in one PR.

- Stage 0 answers implementation questions and confirms whether backend changes are needed.
- Tasks 1-4 are the first PR-sized slice: readiness mapping, server-row state, Tool Catalog recovery, no-auth copy, and tests.
- Tasks 5-7 add the guided first-success setup flow and diagnostics.
- Task 8 handles status-card truthfulness, design-token cleanup, and final verification.

If a backend readiness/refresh endpoint is required, implement Task 2 before frontend fan-out. If Stage 0 proves existing APIs are sufficient for the first slice, skip Task 2 and record that decision in the Backlog task.

## Planned File Structure

- Create `apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts`
  - Pure readiness types and mapper functions.
  - No React imports.
  - Owns display states, reason codes, credential state, allowed actions, and message intent.
- Create `apps/packages/ui/src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts`
  - Unit coverage for mapper state, reason, credential, and action decisions.
- Modify `apps/packages/ui/src/services/tldw/mcp-hub.ts`
  - Add optional readiness/refresh client types only if backend support is implemented.
  - Otherwise keep this untouched for Phase 1 except type imports if needed.
- Modify `apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts`
  - Add service-client tests only if new endpoints are added.
- Modify `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`
  - Use readiness mapper for server row tags/actions.
  - Replace misleading no-auth/secret copy.
  - Fix literal icon color tokens.
  - Later: host guided setup and diagnostics affordances.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx`
  - Add no-auth, legacy fallback, and row-action tests.
- Modify `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx`
  - Load external server state alongside registry summary.
  - Use readiness mapper for empty/stale/discovery recovery state.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx`
  - Add empty, discovery-not-run, auth/runtime, and recovery-action tests.
  - Add no-tools tests only after Stage 0 confirms the data can distinguish
    successful zero-tool discovery from discovery not run.
- Modify `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
  - Later: make status cards data-backed or demote them to navigation.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
  - Later: status-card truthfulness tests.
- Optional create `apps/packages/ui/src/components/Option/MCPHub/McpHubSetupGuide.tsx`
  - Guided first-success setup UI, if splitting from `ExternalServersTab.tsx` keeps the file manageable.
- Optional create `apps/packages/ui/src/components/Option/MCPHub/McpHubDiagnosticsDrawer.tsx`
  - Sanitized diagnostics drawer, if splitting from `ExternalServersTab.tsx` keeps the file manageable.
- Optional backend files if Stage 0 requires backend support:
  - Modify `tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py`
  - Modify `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`
  - Modify or create service support under `tldw_Server_API/app/services/`
  - Modify `tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py`

## Commands

Frontend focused tests:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
```

Frontend service tests, if service client changes:

```bash
cd apps/packages/ui
bunx vitest run src/services/tldw/__tests__/mcp-hub.test.ts
```

Backend focused tests, if backend code changes:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -v
```

Backend security scan, if backend Python changes:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/app/services -f json -o /tmp/bandit_mcp_hub_setup_recovery.json
```

Design-system check after UI polish:

```bash
cd apps/packages/ui
bun run verify:design-system-state
```

## Task 0: Stage 0 Implementation Discovery

**Files:**
- Read: `Docs/superpowers/specs/2026-06-26-mcp-hub-setup-recovery-ux-design.md`
- Read: `apps/packages/ui/src/services/tldw/mcp-hub.ts`
- Read: `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`
- Read: `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx`
- Read: `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`
- Read: `tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py`
- Modify: `backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md`

- [ ] **Step 1: Confirm current API fields**

  Inspect `McpHubExternalServer`, `McpHubToolRegistrySummary`, and their backend schemas. Record whether the current fields can derive:
  `credentialState`, `discovery_not_run`, `config_changed`, `runtime_unavailable`, `no_tools_returned`, and `partial_capability`.
  If the current data cannot distinguish successful zero-tool discovery from
  discovery not run, Phase 1 must not emit `no_tools_returned`.

- [ ] **Step 2: Decide Phase 1 backend scope**

  If existing API responses are enough for the first slice, write in the Backlog task:
  `Stage 0 decision: Phase 1 uses frontend-normalized readiness from existing external server and tool registry APIs.`

  If not enough, write:
  `Stage 0 decision: Phase 1 requires backend readiness/refresh support before UI fan-out.`

- [ ] **Step 3: Decide permission names**

  Inspect existing MCP Hub route dependencies. Record which existing permission gates preflight and discovery refresh should use. Use the current admin/mutation permission unless a narrower existing permission is already present.

- [ ] **Step 4: Decide catalog staleness**

  Record that Phase 1 must not emit `catalog_expired` unless an explicit age threshold already exists. Otherwise use only `config_changed`, manual invalidation, failed refresh after prior success, and `discovery_not_run`. Record whether `no_tools_returned` is available; if not, zero tools maps to `discovery_not_run`.

- [ ] **Step 5: Decide toy MCP smoke path**

  Find the existing toy MCP walkthrough/server path or mark the browser smoke as a later Phase 2/4 verification if no stable toy server exists in the repo.

- [ ] **Step 6: Commit Stage 0 task note**

  ```bash
  git add "backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md"
  git commit -m "docs: record MCP Hub setup recovery Stage 0 decisions"
  ```

## Task 1: Add Shared Readiness Mapper

**Files:**
- Create: `apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts`
- Create: `apps/packages/ui/src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts`

- [ ] **Step 1: Write failing mapper tests for no-server and no-auth stdio**

  In `mcpHubReadiness.test.ts`, add tests like:

  ```ts
  import { describe, expect, it } from "vitest"
  import {
    getMcpHubReadiness,
    getMcpServerReadiness
  } from "../mcpHubReadiness"

  describe("mcpHubReadiness", () => {
    it("maps an empty hub to needs setup", () => {
      const readiness = getMcpHubReadiness({ servers: [], registryEntries: [] })

      expect(readiness.displayState).toBe("needs_setup")
      expect(readiness.primaryReasonCode).toBe("not_configured")
      expect(readiness.allowedActions).toContain("add_server")
    })

    it("maps no-auth stdio as credentials not required", () => {
      const readiness = getMcpServerReadiness({
        server: {
          id: "local-docs",
          name: "Local Docs",
          enabled: true,
          owner_scope_type: "global",
          transport: "stdio",
          config: {},
          secret_configured: false,
          server_source: "managed",
          runtime_executable: true,
          auth_template_present: false,
          auth_template_valid: false,
          auth_template_blocked_reason: "no_auth_template",
          credential_slots: []
        },
        registryEntries: []
      })

      expect(readiness.credentialState).toBe("not_required")
      expect(readiness.message).toContain("No credentials required")
    })
  })
  ```

- [ ] **Step 2: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts
  ```

  Expected: fails because `mcpHubReadiness.ts` does not exist.

- [ ] **Step 3: Implement mapper types and base functions**

  Create `mcpHubReadiness.ts` with:

  ```ts
  import type {
    McpHubExternalServer,
    McpHubToolRegistryEntry
  } from "@/services/tldw/mcp-hub"

  export type McpReadinessAction =
    | "add_server"
    | "edit_config"
    | "open_credentials"
    | "refresh_discovery"
    | "validate"
    | "view_details"
    | "open_tool_catalog"
    | "open_audit"

  export type McpDisplayState =
    | "needs_setup"
    | "checking"
    | "ready"
    | "needs_attention"
    | "no_tools"
    | "stale"

  export type McpReasonCode =
    | "not_configured"
    | "preflight_failed"
    | "discovery_not_run"
    | "auth_missing"
    | "runtime_unavailable"
    | "unreachable"
    | "discovery_failed"
    | "no_tools_returned"
    | "config_changed"
    | "catalog_expired"
    | "partial_capability"

  export type McpCredentialState =
    | "not_required"
    | "required_missing"
    | "configured"
    | "legacy_fallback"
    | "unknown"

  export type McpServerReadiness = {
    serverId: string
    displayName?: string
    transport?: string
    displayState: McpDisplayState
    primaryReasonCode?: McpReasonCode
    reasonCodes: McpReasonCode[]
    credentialState: McpCredentialState
    toolCount: number
    message: string
    allowedActions: McpReadinessAction[]
  }

  export type McpHubReadiness = {
    displayState: McpDisplayState
    primaryReasonCode?: McpReasonCode
    reasonCodes: McpReasonCode[]
    serverCount: number
    readyServerCount: number
    toolCount: number
    allowedActions: McpReadinessAction[]
  }

  export const getMcpHubReadiness = ({
    servers,
    registryEntries
  }: {
    servers: McpHubExternalServer[]
    registryEntries: McpHubToolRegistryEntry[]
  }): McpHubReadiness => {
    const managedServers = servers.filter((server) => server.server_source !== "legacy")
    if (managedServers.length === 0) {
      return {
        displayState: "needs_setup",
        primaryReasonCode: "not_configured",
        reasonCodes: ["not_configured"],
        serverCount: 0,
        readyServerCount: 0,
        toolCount: registryEntries.length,
        allowedActions: ["add_server"]
      }
    }

    const serverReadiness = managedServers.map((server) =>
      getMcpServerReadiness({ server, registryEntries })
    )
    return {
      displayState: serverReadiness.some((row) => row.displayState === "ready")
        ? "ready"
        : "needs_attention",
      primaryReasonCode: serverReadiness[0]?.primaryReasonCode,
      reasonCodes: [...new Set(serverReadiness.flatMap((row) => row.reasonCodes))],
      serverCount: managedServers.length,
      readyServerCount: serverReadiness.filter((row) => row.displayState === "ready").length,
      toolCount: registryEntries.length,
      allowedActions: ["add_server", "view_details"]
    }
  }
  ```

- [ ] **Step 4: Implement credential and reason helpers**

  Add helpers in the same file:

  ```ts
  const isNoAuthTemplate = (reason?: string | null) =>
    !reason || reason === "no_auth_template"

  export const getMcpCredentialState = (
    server: McpHubExternalServer
  ): McpCredentialState => {
    const slots = Array.isArray(server.credential_slots) ? server.credential_slots : []
    if (slots.some((slot) => slot.is_required && !slot.secret_configured)) {
      return "required_missing"
    }
    if (slots.some((slot) => slot.secret_configured) || server.auth_template_valid) {
      return "configured"
    }
    if (server.secret_configured && !server.auth_template_present && slots.length === 0) {
      return "legacy_fallback"
    }
    if (server.transport === "stdio" && isNoAuthTemplate(server.auth_template_blocked_reason)) {
      return "not_required"
    }
    return "unknown"
  }
  ```

- [ ] **Step 5: Implement server readiness mapping**

  Add `getMcpServerReadiness()` using the spec mapping:

  ```ts
  const getToolCountForServer = (
    server: McpHubExternalServer,
    registryEntries: McpHubToolRegistryEntry[]
  ) => {
    const serverId = server.id
    return registryEntries.filter(
      (entry) =>
        entry.module === serverId ||
        entry.module === `external.${serverId}` ||
        entry.tool_name.startsWith(`ext.${serverId}.`)
    ).length
  }

  export const getMcpServerReadiness = ({
    server,
    registryEntries
  }: {
    server: McpHubExternalServer
    registryEntries: McpHubToolRegistryEntry[]
  }): McpServerReadiness => {
    const credentialState = getMcpCredentialState(server)
    const toolCount = getToolCountForServer(server, registryEntries)

    if (credentialState === "required_missing") {
      return {
        serverId: server.id,
        displayName: server.name,
        transport: server.transport,
        displayState: "needs_attention",
        primaryReasonCode: "auth_missing",
        reasonCodes: ["auth_missing"],
        credentialState,
        toolCount,
        message: "Credentials are required before this server can be used.",
        allowedActions: ["open_credentials", "view_details"]
      }
    }

    if (server.runtime_executable === false) {
      return {
        serverId: server.id,
        displayName: server.name,
        transport: server.transport,
        displayState: "needs_attention",
        primaryReasonCode: "runtime_unavailable",
        reasonCodes: ["runtime_unavailable"],
        credentialState,
        toolCount,
        message: "Runtime is not available for this server.",
        allowedActions: ["edit_config", "view_details"]
      }
    }

    if (toolCount === 0) {
      return {
        serverId: server.id,
        displayName: server.name,
        transport: server.transport,
        displayState: "needs_attention",
        primaryReasonCode: "discovery_not_run",
        reasonCodes: ["discovery_not_run"],
        credentialState,
        toolCount,
        message:
          credentialState === "not_required"
            ? "No credentials required. Discover tools to make this server available."
            : "Server is saved, but tool discovery has not run.",
        allowedActions: ["refresh_discovery", "edit_config"]
      }
    }

    return {
      serverId: server.id,
      displayName: server.name,
      transport: server.transport,
      displayState: "ready",
      reasonCodes: [],
      credentialState,
      toolCount,
      message:
        credentialState === "not_required"
          ? "Ready. No credentials required."
          : "Ready.",
      allowedActions: ["open_tool_catalog", "view_details"]
    }
  }
  ```

  This provisional zero-tool mapping is only valid when Stage 0 found no
  explicit "discovery succeeded with zero tools" signal. If Stage 0 found such a
  signal, implement `no_tools_returned` as `displayState: "no_tools"` instead.

- [ ] **Step 6: Add mapper tests for remaining states**

  Add tests for:
  - `required_missing` -> `auth_missing`
  - `runtime_executable: false` -> `runtime_unavailable`
  - zero tools -> `discovery_not_run`
  - matching registry tool -> `ready`
  - `secret_configured` with no template/slots -> `legacy_fallback`
  - `no_tools_returned` only if Stage 0 found an explicit successful-zero-tools
    signal

- [ ] **Step 7: Run mapper tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts
  ```

  Expected: pass.

- [ ] **Step 8: Commit mapper**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts apps/packages/ui/src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts
  git commit -m "feat: add MCP Hub readiness mapper"
  ```

## Task 2: Optional Backend Readiness Or Refresh Contract

Only do this task if Stage 0 proves existing APIs cannot support the first readiness slice.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py`
- Modify: `apps/packages/ui/src/services/tldw/mcp-hub.ts`
- Modify: `apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts`

- [ ] **Step 1: Write backend API tests first**

  Add focused tests in `test_mcp_hub_management_api.py` for:
  - readiness response redacts secrets;
  - no-auth stdio returns `credential_state="not_required"`;
  - configured but undiscovered returns `primary_reason_code="discovery_not_run"`;
  - refresh/preflight route requires the same MCP Hub admin permission used by external server mutation routes.

- [ ] **Step 2: Run backend tests to verify failure**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -k "readiness or refresh" -v
  ```

  Expected: fail because schemas/routes do not exist.

- [ ] **Step 3: Add Pydantic response schemas**

  Add models matching the approved spec:
  `McpHubReadinessResponse`, `McpServerReadinessResponse`, and enum/string fields for display state, reason codes, credential state, operation state, and allowed actions.

- [ ] **Step 4: Add route with existing permission gate**

  Add a route such as:

  ```python
  @router.get("/readiness", response_model=McpHubReadinessResponse)
  async def get_mcp_hub_readiness(...):
      ...
  ```

  Keep it read-only unless implementing explicit refresh. Do not execute stdio processes in readiness.

- [ ] **Step 5: Add optional explicit refresh route only if needed**

  If refresh is needed in this slice, add:

  ```python
  @router.post("/external-servers/{server_id}/refresh-discovery", response_model=McpServerReadinessResponse)
  async def refresh_external_server_discovery(...):
      ...
  ```

  Requirements:
  - same MCP Hub mutation/admin permission as create/update/delete;
  - bounded timeout;
  - sanitized audit/log output;
  - no secret values in response.

- [ ] **Step 6: Add frontend service client tests**

  In `apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts`, verify exact paths and methods for any new client methods.

- [ ] **Step 7: Add frontend service methods**

  Add typed client methods in `apps/packages/ui/src/services/tldw/mcp-hub.ts` only for the backend routes actually added.

- [ ] **Step 8: Run focused tests**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -k "readiness or refresh" -v
  cd apps/packages/ui
  bunx vitest run src/services/tldw/__tests__/mcp-hub.test.ts
  ```

- [ ] **Step 9: Run Bandit for backend changes**

  ```bash
  source .venv/bin/activate
  python -m bandit -r tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/app/services -f json -o /tmp/bandit_mcp_hub_setup_recovery.json
  ```

- [ ] **Step 10: Commit backend/client contract**

  ```bash
  git add tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py apps/packages/ui/src/services/tldw/mcp-hub.ts apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts
  git commit -m "feat: add MCP Hub readiness contract"
  ```

## Task 3: Apply Readiness To External Server Rows

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx`
- Use: `apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts`

- [ ] **Step 1: Write failing no-auth row test**

  Add a test where a managed stdio server has no auth template, no secret, no slots, and `runtime_executable: true`.

  Assert:
  - `No credentials required` is visible.
  - `no secret` is not shown for that server.
  - `No auth template` is not used as the primary status.

- [ ] **Step 2: Write failing legacy fallback row test**

  Add a server with `secret_configured: true`, no template, and no slots.

  Assert:
  - `Legacy Secret Fallback` or `legacy secret fallback` appears only for that server.
  - It does not appear for the no-auth stdio server.

- [ ] **Step 3: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 4: Import and compute readiness**

  In `ExternalServersTab.tsx`, import:

  ```ts
  import { getMcpServerReadiness } from "./mcpHubReadiness"
  ```

  If registry entries are not loaded in this component yet, pass an empty array for Task 3 and let Task 4 wire registry-aware catalog recovery. Keep the row copy focused on credential/runtime states.

- [ ] **Step 5: Replace misleading credential tags**

  Replace unconditional `no secret` and `No auth template` tags for managed servers with tags derived from `credentialState`:
  - `not_required`: green or neutral `No credentials required`
  - `required_missing`: orange `Credentials required`
  - `configured`: green `credentials configured`
  - `legacy_fallback`: orange `Legacy Secret Fallback`
  - `unknown`: neutral `credential status unknown`

- [ ] **Step 6: Preserve expert detail without making it primary**

  Keep template validity and slot counts visible where useful, but do not present `No auth template` as an error for no-auth stdio. If retaining the raw template tag, make it secondary text or tooltip detail.

- [ ] **Step 7: Fix literal color tokens**

  Replace both `QuestionCircleOutlined style={{ color: "rgba(0,0,0,0.45)" }}` usages with an existing token/class. Prefer Ant Design token access if already used nearby; otherwise use `Typography.Text type="secondary"` wrapping or a local CSS class that maps to the design system.

- [ ] **Step 8: Run row tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

  Expected: pass.

- [ ] **Step 9: Commit server-row readiness**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  git commit -m "fix: clarify MCP Hub external server readiness"
  ```

## Task 4: Add Tool Catalog Recovery States

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx`
- Use: `apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts`

- [ ] **Step 1: Write failing empty-no-server test**

  Mock `getToolRegistrySummary()` to return no entries/modules and `listExternalServers()` to return `[]`.

  Assert:
  - `No MCP servers connected` or equivalent appears.
  - An `Add server` action appears.

- [ ] **Step 2: Write failing configured-undiscovered test**

  Mock no tools, one managed enabled server, no credential blockers, and `runtime_executable: true`.

  Assert:
  - The empty state says the server is saved but discovery has not run.
  - `Refresh discovery` appears.

- [ ] **Step 3: Write failing auth/runtime recovery tests**

  Add one test for required missing credentials and one for runtime unavailable.

  Assert:
  - Missing credentials offers `Fix credentials`.
  - Runtime unavailable offers `Open server config`.

- [ ] **Step 4: Add no-tools test only if data supports it**

  If Stage 0 found a backend field that distinguishes successful zero-tool
  discovery from discovery not run, add a test asserting that `no_tools_returned`
  renders a `no_tools` display state with explanatory copy. Otherwise do not add
  this test in Phase 1.

- [ ] **Step 5: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  ```

- [ ] **Step 6: Load external server state**

  In `ToolCatalogsTab.tsx`, import `listExternalServers` and store servers alongside registry entries/modules.

  Keep loading and error states independent enough that a registry failure still reports registry failure, while server-state failure reports recovery limitations.

- [ ] **Step 7: Compute hub and server readiness**

  Use `getMcpHubReadiness()` and `getMcpServerReadiness()` to derive empty-state copy and actions.

- [ ] **Step 8: Render actionable empty state**

  Replace the passive `Empty` copy with state-driven content:
  - no servers: `Add server`
  - discovery not run: `Refresh discovery` and `Open server config`
  - auth missing: `Fix credentials`
  - runtime unavailable: `Open server config`
  - no tools returned: only when Stage 0 found an explicit successful-zero-tools
    signal, explain that the server responded with no tools and offer `Refresh
    discovery`

  If refresh action is not implemented yet, render it disabled with copy such as `Refresh discovery coming from the runtime refresh slice`, or wire it only when Task 2 added a client method. Do not render a clickable no-op.

- [ ] **Step 9: Run catalog tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  ```

- [ ] **Step 10: Run first-slice frontend suite**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  ```

- [ ] **Step 11: Commit Tool Catalog recovery**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  git commit -m "feat: add MCP Hub tool catalog recovery states"
  ```

## Task 5: Add First-Success Setup Guide

**Files:**
- Optional Create: `apps/packages/ui/src/components/Option/MCPHub/McpHubSetupGuide.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx`

- [ ] **Step 1: Write failing setup-choice tests**

  Add tests that open the create flow and assert these choices are available:
  - `Local stdio`
  - `HTTP/SSE`
  - `Import config`
  - `Advanced/manual`

- [ ] **Step 2: Write failing expert bypass test**

  Assert that `Advanced/manual` exposes the existing raw Config JSON flow without requiring the guided fields.

- [ ] **Step 3: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 4: Add setup mode state**

  Add local state:

  ```ts
  type SetupMode = "choice" | "stdio" | "http" | "import" | "advanced"
  const [setupMode, setSetupMode] = useState<SetupMode>("choice")
  ```

- [ ] **Step 5: Add starter choice UI**

  When creating a new managed server, show the four setup choices first. Keep cards compact and work-focused. Avoid nested cards; use full-width rows or a simple segmented/button group inside the existing form area.

- [ ] **Step 6: Add guided stdio fields**

  Add fields for:
  - command
  - args
  - env vars
  - working directory
  - owner scope

  Convert them into the existing `config` object shape during save. Use the project’s observed external server config shape from Stage 0.

- [ ] **Step 7: Add guided HTTP/SSE fields**

  Add fields for:
  - URL
  - headers/auth choice
  - owner scope

  Convert them into the existing config object shape during save.

- [ ] **Step 8: Add safe preflight checks**

  Before save, validate:
  - required fields;
  - JSON object shape;
  - URL syntax for HTTP/SSE;
  - command presence for stdio.

  Do not execute commands or contact URLs.

- [ ] **Step 9: Rename primary/secondary actions**

  Use:
  - primary `Save and discover tools`
  - secondary `Save without discovery`

  If discovery refresh is not implemented, the primary action should save and show an honest post-save message that discovery refresh is not available yet. Do not claim discovery succeeded.

- [ ] **Step 10: Run setup tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 11: Commit setup guide**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  git commit -m "feat: add guided MCP Hub server setup"
  ```

## Task 6: Add Diagnostics And Environment Context

**Files:**
- Optional Create: `apps/packages/ui/src/components/Option/MCPHub/McpHubDiagnosticsDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx`
- Optional Modify: `apps/packages/ui/src/services/tldw/mcp-hub.ts`

- [ ] **Step 1: Write failing diagnostics tests**

  Add tests for:
  - details action opens diagnostics;
  - diagnostics show reason code, credential state, transport, tool count;
  - diagnostics do not show secret values from env, headers, URL query, or raw config;
  - diagnostics show API origin/health endpoint/deployment mode if the data source exists, or an explicit unavailable message if not.

- [ ] **Step 2: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 3: Add redaction helper**

  Put the helper in `mcpHubReadiness.ts` or a new pure helper file if it grows:

  ```ts
  export const redactMcpDiagnosticValue = (key: string, value: unknown): string => {
    const lowered = key.toLowerCase()
    if (/(token|secret|password|authorization|api[_-]?key)/.test(lowered)) {
      return "[redacted]"
    }
    const text = typeof value === "string" ? value : JSON.stringify(value)
    return text.replace(/([?&][^=]*(token|key|secret|password)[^=]*=)[^&]+/gi, "$1[redacted]")
  }
  ```

- [ ] **Step 4: Add diagnostics UI**

  Add a details drawer or modal that shows:
  - display state;
  - primary reason and reason codes;
  - credential state;
  - transport;
  - sanitized command/endpoint facts;
  - tool count;
  - last operation fields only if available;
  - Audit tab link or copy explaining where audit details live.

- [ ] **Step 5: Add environment diagnostics**

  Show effective deployment mode, frontend API origin, and health endpoint if current frontend settings expose them. If not, show `Not available in this client` and keep a Stage 0 note for a later shared diagnostics provider.

- [ ] **Step 6: Run diagnostics tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 7: Commit diagnostics**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  git commit -m "feat: add MCP Hub setup diagnostics"
  ```

## Task 7: Make Status Summary Truthful

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- Use: `apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts`

- [ ] **Step 1: Write failing status-card test**

  Add a test proving status cards do not imply readiness when no readiness data is available.

- [ ] **Step 2: Decide data-backed versus navigation**

  If `McpHubPage` can cheaply access external servers and registry summary without duplicating fetches or causing load churn, make the cards data-backed. Otherwise demote static cards to plain navigation.

- [ ] **Step 3: Implement the chosen path**

  For data-backed cards:
  - consume shared readiness object;
  - show counts/states that match server rows and Tool Catalog.

  For navigation:
  - remove readiness-like color/copy;
  - use plain section descriptions and links only.

- [ ] **Step 4: Run page tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
  ```

- [ ] **Step 5: Commit status truthfulness**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
  git commit -m "fix: make MCP Hub status summary truthful"
  ```

## Task 8: Responsive, Design-System, And Final Verification

**Files:**
- Modify as needed:
  - `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx`
  - `apps/packages/ui/src/components/Option/MCPHub/ToolCatalogsTab.tsx`
  - `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
  - relevant tests
- Modify: `backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md`

- [ ] **Step 1: Run focused MCP Hub test suite**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
  ```

- [ ] **Step 2: Run service tests if client changed**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/services/tldw/__tests__/mcp-hub.test.ts
  ```

- [ ] **Step 3: Run backend tests if backend changed**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -v
  ```

- [ ] **Step 4: Run Bandit if backend changed**

  ```bash
  source .venv/bin/activate
  python -m bandit -r tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/app/services -f json -o /tmp/bandit_mcp_hub_setup_recovery.json
  ```

- [ ] **Step 5: Run design-system check**

  ```bash
  cd apps/packages/ui
  bun run verify:design-system-state
  ```

- [ ] **Step 6: Check literal color cleanup**

  ```bash
  rg "rgba\\(0,0,0,0\\.45\\)" apps/packages/ui/src/components/Option/MCPHub
  ```

  Expected: no matches.

- [ ] **Step 7: Perform responsive/manual visual check**

  Start the local frontend if needed and inspect MCP Hub at desktop and mobile widths. Verify:
  - setup choices do not overflow;
  - primary actions remain visible;
  - server row tags wrap cleanly;
  - diagnostics drawer/modal does not hide controls;
  - Tool Catalog recovery actions are reachable.

- [ ] **Step 8: Update Backlog task**

  Add verification results, known skips, and final summary to `TASK-223.2`.

- [ ] **Step 9: Final commit**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub apps/packages/ui/src/services/tldw/mcp-hub.ts apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py "backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md"
  git commit -m "test: verify MCP Hub setup recovery UX"
  ```

  Only stage files that were actually changed.

## Implementation Notes

- Do not emit clickable refresh actions until there is a real refresh implementation behind them.
- Do not execute stdio commands from frontend preflight.
- Do not show raw env values, headers, tokens, URL query secrets, or secret-bearing config in diagnostics.
- Keep `catalog_expired` inactive unless an explicit freshness threshold is chosen.
- Keep `partial_capability` visually ready with warning detail, not fully healthy.
- Keep the advanced/manual path visible from the first setup choice screen.
- Preserve user changes in the dirty worktree; stage only files touched for this task.
