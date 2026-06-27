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

Task 2 is required for valid discovery-refresh actions. Skip backend work inside Task 2 only when Stage 0 finds an existing refresh/readiness endpoint that satisfies the spec; in that case, Task 2 still adds or verifies the frontend service client and tests. Do not leave refresh actions as disabled or no-op controls in the final implementation.

## Planned File Structure

- Create `apps/packages/ui/src/components/Option/MCPHub/mcpHubReadiness.ts`
  - Pure readiness types and mapper functions.
  - No React imports.
  - Owns display states, reason codes, credential state, allowed actions, and message intent.
- Create `apps/packages/ui/src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts`
  - Unit coverage for mapper state, reason, credential, and action decisions.
- Modify `apps/packages/ui/src/services/tldw/mcp-hub.ts`
  - Add or verify readiness/refresh client types for the endpoint selected in Stage 0.
- Modify `apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts`
  - Add service-client tests for refresh/readiness client behavior.
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
  - Add empty, discovery-not-run, auth/runtime, operation/failure, stale, and recovery-action tests.
  - Add no-tools tests only after Stage 0 confirms the data can distinguish
    successful zero-tool discovery from discovery not run.
- Modify `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
  - Later: make status cards data-backed or demote them to navigation.
- Modify `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
  - Later: status-card truthfulness tests.
- Modify `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`
  - Add first-run setup smoke coverage and responsive screenshot assertions.
- Modify `apps/tldw-frontend/e2e/utils/page-objects/MCPHubPage.ts`
  - Add page-object helpers for guided setup, Tool Catalog recovery, diagnostics, and screenshot targets as needed.
- Optional create `apps/packages/ui/src/components/Option/MCPHub/McpHubSetupGuide.tsx`
  - Guided first-success setup UI, if splitting from `ExternalServersTab.tsx` keeps the file manageable.
- Optional create `apps/packages/ui/src/components/Option/MCPHub/McpHubDiagnosticsDrawer.tsx`
  - Sanitized diagnostics drawer, if splitting from `ExternalServersTab.tsx` keeps the file manageable.
- Backend files if Stage 0 finds no existing endpoint or fields for valid refresh/stale/readiness behavior:
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

Frontend browser smoke and visual QA:

```bash
cd apps/tldw-frontend
bun run e2e:pw -- e2e/workflows/tier-2-features/mcp-hub.spec.ts --project=tier-2 --reporter=line
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
  `credentialState`, `checking`, `preflight_failed`, `discovery_not_run`, `config_changed`, `runtime_unavailable`, `unreachable`, `discovery_failed`, `no_tools_returned`, and `partial_capability`.
  If the current data cannot distinguish successful zero-tool discovery from
  discovery not run, Phase 1 must not emit `no_tools_returned`.

- [ ] **Step 2: Decide Phase 1 backend scope**

  If existing API responses are enough for the first slice, including stale/config-changed state and valid discovery refresh, write in the Backlog task:
  `Stage 0 decision: Phase 1 uses frontend-normalized readiness from existing external server and tool registry APIs.`

  If not enough, write:
  `Stage 0 decision: Phase 1 requires backend readiness/refresh support before UI fan-out.`

- [ ] **Step 3: Decide refresh action support**

  Find an existing discovery refresh endpoint/client or confirm Task 2 must add one. Record the chosen endpoint path and response shape. Final UI actions must call a real refresh operation or route to an existing implemented refresh flow; disabled "coming soon" refresh buttons are not acceptable for the final slice.

- [ ] **Step 4: Decide permission names**

  Inspect existing MCP Hub route dependencies. Record which existing permission gates preflight and discovery refresh should use. Use the current admin/mutation permission unless a narrower existing permission is already present.

- [ ] **Step 5: Decide catalog staleness**

  Record that Phase 1 must not emit `catalog_expired` unless an explicit age threshold already exists. Otherwise use only `config_changed`, manual invalidation, failed refresh after prior success, and `discovery_not_run`. Record whether `no_tools_returned` is available; if not, zero tools maps to `discovery_not_run`.

- [ ] **Step 6: Decide diagnostics data availability**

  Identify where the frontend can read deployment mode, API origin, health endpoint, operation timestamps, current operation metadata, and last sanitized error category/message. If any field is unavailable, record whether Task 2 adds it or Task 6 must display a precise unavailable state plus setup-isolation guidance.

- [ ] **Step 7: Decide permission-gated catalog state**

  Confirm whether the Tool Catalog API or frontend client can distinguish a permission-gated catalog from an ordinary empty/error state. If yes, record the response shape and expected recovery action in `TASK-223.2`; Task 4 must add a focused permission-boundary recovery test. If not, record that Phase 1 cannot emit a separate permission-gated catalog state.

- [ ] **Step 8: Decide toy MCP smoke path**

  Find the existing toy MCP walkthrough/server path. If no stable toy server exists in the repo, record an explicit skip reason in `TASK-223.2` and replace the smoke with the closest automated browser path that exercises the guided setup UI without mutating production data.

- [ ] **Step 9: Commit Stage 0 task note**

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

  export type McpCurrentOperationHint = {
    operationId?: string
    operationType: "preflight" | "discovery"
    startedAt?: string
    initiatedBy?: string
    timeoutMs?: number
  }

  export type McpServerReadinessHint = {
    currentOperation?: McpCurrentOperationHint
    preflightFailed?: boolean
    configChanged?: boolean
    manuallyInvalidated?: boolean
    failedRefreshAfterSuccess?: boolean
    unreachable?: boolean
    discoveryFailed?: boolean
    discoverySucceededWithNoTools?: boolean
    catalogExpired?: boolean
    partialCapability?: boolean
    lastValidationAt?: string
    lastDiscoveryAt?: string
    lastSuccessfulDiscoveryAt?: string
    lastErrorCategory?: string
    lastErrorMessage?: string
  }

  export type McpServerReadiness = {
    serverId: string
    displayName?: string
    transport?: string
    displayState: McpDisplayState
    primaryReasonCode?: McpReasonCode
    reasonCodes: McpReasonCode[]
    credentialState: McpCredentialState
    toolCount: number
    lastValidationAt?: string
    lastDiscoveryAt?: string
    lastSuccessfulDiscoveryAt?: string
    currentOperation?: McpCurrentOperationHint
    lastErrorCategory?: string
    lastErrorMessage?: string
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
    registryEntries,
    readinessHintsByServerId = {}
  }: {
    servers: McpHubExternalServer[]
    registryEntries: McpHubToolRegistryEntry[]
    readinessHintsByServerId?: Record<string, McpServerReadinessHint>
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
      getMcpServerReadiness({
        server,
        registryEntries,
        readinessHint: readinessHintsByServerId[server.id]
      })
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

  Add `getMcpServerReadiness()` using the spec mapping. The mapper must aggregate
  all applicable reason codes before selecting the primary reason. Do not
  implement this as a chain of early returns that drops secondary blockers.

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

  const REASON_PRIORITY: McpReasonCode[] = [
    "auth_missing",
    "runtime_unavailable",
    "preflight_failed",
    "unreachable",
    "discovery_failed",
    "config_changed",
    "discovery_not_run",
    "no_tools_returned",
    "catalog_expired",
    "partial_capability"
  ]

  const DISPLAY_STATE_BY_REASON: Record<McpReasonCode, McpDisplayState> = {
    not_configured: "needs_setup",
    preflight_failed: "needs_attention",
    discovery_not_run: "needs_attention",
    auth_missing: "needs_attention",
    runtime_unavailable: "needs_attention",
    unreachable: "needs_attention",
    discovery_failed: "needs_attention",
    no_tools_returned: "no_tools",
    config_changed: "stale",
    catalog_expired: "stale",
    partial_capability: "ready"
  }

  const ACTIONS_BY_REASON: Record<McpReasonCode, McpReadinessAction[]> = {
    not_configured: ["add_server"],
    preflight_failed: ["edit_config", "validate", "view_details"],
    discovery_not_run: ["refresh_discovery", "edit_config"],
    auth_missing: ["open_credentials", "view_details"],
    runtime_unavailable: ["edit_config", "view_details"],
    unreachable: ["edit_config", "refresh_discovery", "view_details"],
    discovery_failed: ["refresh_discovery", "view_details"],
    no_tools_returned: ["refresh_discovery", "view_details"],
    config_changed: ["refresh_discovery", "edit_config"],
    catalog_expired: ["refresh_discovery", "view_details"],
    partial_capability: ["open_tool_catalog", "view_details"]
  }

  const uniqueReasons = (reasons: McpReasonCode[]) =>
    [...new Set(reasons)].sort(
      (left, right) => REASON_PRIORITY.indexOf(left) - REASON_PRIORITY.indexOf(right)
    )

  const getPrimaryReasonCode = (reasonCodes: McpReasonCode[]) =>
    uniqueReasons(reasonCodes).find((reason) => REASON_PRIORITY.includes(reason))

  const getReasonActions = (reasonCodes: McpReasonCode[]) =>
    [...new Set(reasonCodes.flatMap((reason) => ACTIONS_BY_REASON[reason] ?? []))]

  const getReasonMessage = (
    primaryReasonCode: McpReasonCode | undefined,
    credentialState: McpCredentialState
  ) => {
    switch (primaryReasonCode) {
      case "auth_missing":
        return "Credentials are required before this server can be used."
      case "runtime_unavailable":
        return "Runtime is not available for this server."
      case "preflight_failed":
        return "Preflight validation failed. Check the server configuration."
      case "unreachable":
        return "Server cannot be reached."
      case "discovery_failed":
        return "Discovery ran but failed."
      case "config_changed":
        return "Server config or discovery state changed. Refresh discovery."
      case "discovery_not_run":
        return credentialState === "not_required"
          ? "No credentials required. Discover tools to make this server available."
          : "Server is saved, but tool discovery has not run."
      case "no_tools_returned":
        return "Server responded, but exposed no tools."
      case "catalog_expired":
        return "Tool catalog is stale. Refresh discovery."
      case "partial_capability":
        return "Ready with limited capability."
      default:
        return credentialState === "not_required"
          ? "Ready. No credentials required."
          : "Ready."
    }
  }

  const deriveReasonCodes = ({
    server,
    credentialState,
    toolCount,
    readinessHint
  }: {
    server: McpHubExternalServer
    credentialState: McpCredentialState
    toolCount: number
    readinessHint: McpServerReadinessHint
  }) => {
    const reasons: McpReasonCode[] = []
    if (credentialState === "required_missing") reasons.push("auth_missing")
    if (server.runtime_executable === false) reasons.push("runtime_unavailable")
    if (readinessHint.preflightFailed) reasons.push("preflight_failed")
    if (readinessHint.unreachable) reasons.push("unreachable")
    if (readinessHint.discoveryFailed) reasons.push("discovery_failed")
    if (
      readinessHint.configChanged ||
      readinessHint.manuallyInvalidated ||
      readinessHint.failedRefreshAfterSuccess
    ) {
      reasons.push("config_changed")
    }
    if (readinessHint.catalogExpired) reasons.push("catalog_expired")
    if (toolCount === 0 && readinessHint.discoverySucceededWithNoTools) {
      reasons.push("no_tools_returned")
    } else if (toolCount === 0) {
      reasons.push("discovery_not_run")
    }
    if (toolCount > 0 && readinessHint.partialCapability) {
      reasons.push("partial_capability")
    }
    return uniqueReasons(reasons)
  }

  export const getMcpServerReadiness = ({
    server,
    registryEntries,
    readinessHint = {}
  }: {
    server: McpHubExternalServer
    registryEntries: McpHubToolRegistryEntry[]
    readinessHint?: McpServerReadinessHint
  }): McpServerReadiness => {
    const credentialState = getMcpCredentialState(server)
    const toolCount = getToolCountForServer(server, registryEntries)

    if (readinessHint.currentOperation) {
      return {
        serverId: server.id,
        displayName: server.name,
        transport: server.transport,
        displayState: "checking",
        reasonCodes: [],
        credentialState,
        toolCount,
        currentOperation: readinessHint.currentOperation,
        lastValidationAt: readinessHint.lastValidationAt,
        lastDiscoveryAt: readinessHint.lastDiscoveryAt,
        lastSuccessfulDiscoveryAt: readinessHint.lastSuccessfulDiscoveryAt,
        message:
          readinessHint.currentOperation.operationType === "discovery"
            ? "Tool discovery is running."
            : "Preflight validation is running.",
        allowedActions: ["view_details"]
      }
    }

    const reasonCodes = deriveReasonCodes({
      server,
      credentialState,
      toolCount,
      readinessHint
    })
    const primaryReasonCode = getPrimaryReasonCode(reasonCodes)
    const displayState = primaryReasonCode
      ? DISPLAY_STATE_BY_REASON[primaryReasonCode]
      : "ready"

    return {
      serverId: server.id,
      displayName: server.name,
      transport: server.transport,
      displayState,
      primaryReasonCode,
      reasonCodes,
      credentialState,
      toolCount,
      lastValidationAt: readinessHint.lastValidationAt,
      lastDiscoveryAt: readinessHint.lastDiscoveryAt,
      lastSuccessfulDiscoveryAt: readinessHint.lastSuccessfulDiscoveryAt,
      lastErrorCategory: readinessHint.lastErrorCategory,
      lastErrorMessage: readinessHint.lastErrorMessage,
      message: getReasonMessage(primaryReasonCode, credentialState),
      allowedActions:
        reasonCodes.length > 0
          ? getReasonActions(reasonCodes)
          : ["open_tool_catalog", "view_details"]
    }
  }
  ```

  This provisional zero-tool mapping is only valid when Stage 0 found no
  explicit "discovery succeeded with zero tools" signal. If Stage 0 found such a
  signal, implement `no_tools_returned` as `displayState: "no_tools"` instead.

- [ ] **Step 6: Add mapper tests for remaining states**

  Add tests for:
  - `readinessHint.currentOperation` -> `checking`
  - `readinessHint.preflightFailed` -> `preflight_failed`
  - `required_missing` -> `auth_missing`
  - `runtime_executable: false` -> `runtime_unavailable`
  - `readinessHint.unreachable` -> `unreachable`
  - `readinessHint.discoveryFailed` -> `discovery_failed`
  - zero tools -> `discovery_not_run`
  - `readinessHint.configChanged` -> `stale` with `refresh_discovery`
  - multiple simultaneous reasons, such as missing required credentials plus
    `configChanged`, preserve both reason codes, choose the priority primary
    reason, and union allowed actions without duplicates
  - matching registry tool -> `ready`
  - matching registry tool plus `partialCapability` -> `ready` with `partial_capability`
  - `secret_configured` with no template/slots -> `legacy_fallback`
  - `no_tools_returned` only if Stage 0 found an explicit successful-zero-tools
    signal
  - every reason code selected by Stage 0 either maps to the spec's display
    state/action set or is intentionally not emitted because the required
    backend signal is unavailable

  When implementing, place reason-priority helpers above both hub and server
  mappers. `getMcpHubReadiness()` must also use the same priority helper across
  aggregated server reason codes instead of taking the first row's primary
  reason.

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

## Task 2: Ensure Backend Readiness And Discovery Refresh Contract

Do this task after Stage 0. If Stage 0 finds existing endpoints and fields that satisfy the requirements, this task verifies and adds the frontend service client/tests for those existing routes. If Stage 0 finds gaps, this task adds the smallest backend contract needed before UI fan-out. Do not continue to Tasks 3-5 with placeholder, disabled, or no-op refresh actions.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py`
- Modify: `apps/packages/ui/src/services/tldw/mcp-hub.ts`
- Modify: `apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts`

- [ ] **Step 1: Write or verify backend API tests first**

  Add or verify focused tests in `test_mcp_hub_management_api.py` for:
  - readiness response redacts secrets;
  - no-auth stdio returns `credential_state="not_required"`;
  - active validation/discovery can return `display_state="checking"` with current operation metadata;
  - preflight failure returns `primary_reason_code="preflight_failed"`;
  - configured but undiscovered returns `primary_reason_code="discovery_not_run"`;
  - config-changed or invalidated catalog returns `display_state="stale"` and `primary_reason_code="config_changed"`;
  - unreachable/discovery failure states return `primary_reason_code="unreachable"` or `primary_reason_code="discovery_failed"` when the backend exposes those causes;
  - partial capability returns `display_state="ready"` with `primary_reason_code="partial_capability"` when the backend exposes that warning state;
  - refresh/preflight route requires the same MCP Hub admin permission used by external server mutation routes;
  - refresh/preflight route rejects access to servers outside the caller's owner scope using the same visibility/mutation rules as external server update/delete;
  - concurrent refresh requests for the same server are serialized or deduplicated and return a stable current-operation response or bounded conflict, not two simultaneous runtime launches;
  - refresh honors the existing runtime execution and outbound/network safety policy before starting stdio or contacting HTTP/SSE endpoints;
  - refresh/preflight audit payloads redact env, headers, URL query secrets, args, and raw config values.

- [ ] **Step 2: Run backend tests to verify failure**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -k "readiness or refresh" -v
  ```

  Expected: fail if schemas/routes are missing. If existing tests already cover this behavior, record the passing command in `TASK-223.2` and continue to service-client verification.

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

- [ ] **Step 5: Add explicit refresh route if no compliant route exists**

  If no compliant refresh endpoint exists, add:

  ```python
  @router.post("/external-servers/{server_id}/refresh-discovery", response_model=McpServerReadinessResponse)
  async def refresh_external_server_discovery(...):
      ...
  ```

  Requirements:
  - same MCP Hub mutation/admin permission as create/update/delete;
  - same owner-scope visibility/mutation checks as external server update/delete;
  - per-server serialization or operation dedupe so one server cannot run multiple overlapping refresh jobs;
  - use the existing runtime execution and outbound/network safety policy; do not add a new ad hoc subprocess or HTTP client path that bypasses current MCP Hub runtime guards;
  - bounded timeout;
  - sanitized audit/log output;
  - no secret values in response.
  - returns enough state for the frontend to update the result panel and Tool Catalog recovery state.

- [ ] **Step 6: Add operation metadata and sanitized error fields when backend-owned**

  If the backend owns readiness, include optional fields for `last_validation_at`, `last_discovery_at`, `last_successful_discovery_at`, `current_operation`, `last_error_category`, and sanitized `last_error_message`. Do not include secret-bearing raw config.

- [ ] **Step 7: Add frontend service client tests**

  In `apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts`, verify exact paths and methods for any new client methods.

- [ ] **Step 8: Add frontend service methods**

  Add typed client methods in `apps/packages/ui/src/services/tldw/mcp-hub.ts` only for the backend routes actually added.

- [ ] **Step 9: Run focused tests**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -k "readiness or refresh" -v
  cd apps/packages/ui
  bunx vitest run src/services/tldw/__tests__/mcp-hub.test.ts
  ```

- [ ] **Step 10: Run Bandit for backend changes**

  ```bash
  source .venv/bin/activate
  python -m bandit -r tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/app/services -f json -o /tmp/bandit_mcp_hub_setup_recovery.json
  ```

- [ ] **Step 11: Commit backend/client contract**

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

- [ ] **Step 4: Load registry/readiness inputs for rows**

  In `ExternalServersTab.tsx`, import the registry client and readiness mapper:

  ```ts
  import {
    getToolRegistrySummary,
    ...
  } from "@/services/tldw/mcp-hub"
  import { getMcpServerReadiness } from "./mcpHubReadiness"
  ```

  Add state for registry entries and any backend readiness hints selected in Stage 0. Load them in the same refresh path as `loadServers()` so server rows and Tool Catalog can derive the same readiness state. Do not pass `[]` registry entries as a permanent shortcut.

- [ ] **Step 5: Add row/catalog consistency test**

  Add a test where a managed server has a matching registry tool and assert the server row shows `Ready` or equivalent ready copy, not `discovery_not_run`. Add a second test where Stage 0 readiness hints mark `configChanged` and assert the row shows stale/refresh copy.

- [ ] **Step 6: Write failing row recovery-action tests**

  Add tests that row-level actions are present and wired from `allowedActions`:
  - `validate` renders `Validate` and calls the real validation/preflight client selected in Task 2;
  - `refresh_discovery` renders `Refresh tools` and calls the real refresh client selected in Task 2;
  - `edit_config` renders `Edit config` and opens the existing edit flow for that server;
  - `open_credentials` renders `Credentials` and opens the credential/secret flow for that server;
  - `view_details` renders `Details` and opens diagnostics for that server;
  - `open_tool_catalog` renders `Tool Catalog` and switches to the Tool Catalog view.

  Tests must assert that these controls are not clickable no-ops. Mock the service
  method or parent view callback and verify the exact method/callback is invoked.

- [ ] **Step 7: Replace misleading credential tags**

  Replace unconditional `no secret` and `No auth template` tags for managed servers with tags derived from `credentialState`:
  - `not_required`: green or neutral `No credentials required`
  - `required_missing`: orange `Credentials required`
  - `configured`: green `credentials configured`
  - `legacy_fallback`: orange `Legacy Secret Fallback`
  - `unknown`: neutral `credential status unknown`

- [ ] **Step 8: Preserve expert detail without making it primary**

  Keep template validity and slot counts visible where useful, but do not present `No auth template` as an error for no-auth stdio. If retaining the raw template tag, make it secondary text or tooltip detail.

- [ ] **Step 9: Render row recovery actions**

  Add a compact row action group derived from `readiness.allowedActions`. Required mappings:
  - `validate` -> `Validate`, calls validation/preflight;
  - `refresh_discovery` -> `Refresh tools`, calls discovery refresh and reloads servers/registry/readiness;
  - `edit_config` -> `Edit config`, opens the existing edit flow;
  - `open_credentials` -> `Credentials`, opens existing credential/secret editing;
  - `view_details` -> `Details`, opens diagnostics;
  - `open_tool_catalog` -> `Tool Catalog`, switches to Tool Catalog.

  Preserve existing edit/delete controls, but do not make them the only recovery
  path. If a Task 2 client method is unavailable because Stage 0 selected an
  existing route with a different name, wire to that selected route instead of
  rendering a disabled placeholder.

- [ ] **Step 10: Fix literal color tokens**

  Replace both `QuestionCircleOutlined style={{ color: "rgba(0,0,0,0.45)" }}` usages with an existing token/class. Prefer Ant Design token access if already used nearby; otherwise use `Typography.Text type="secondary"` wrapping or a local CSS class that maps to the design system.

- [ ] **Step 11: Run row tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

  Expected: pass.

- [ ] **Step 12: Commit server-row readiness**

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

- [ ] **Step 4: Write failing operation/failure recovery tests**

  Add tests for any Stage 0 readiness signals exposed by the frontend/backend contract:
  - `checking`: shows in-progress discovery/preflight copy and no duplicate refresh action;
  - `preflight_failed`: offers `Open server config` and `View details`;
  - `unreachable`: offers `Open server config`, `Refresh discovery`, and `View details`;
  - `discovery_failed`: offers `Refresh discovery` and `View details`;
  - `partial_capability`: still allows Tool Catalog access and shows warning detail.

- [ ] **Step 5: Write failing stale recovery test**

  Mock one managed server with a Stage 0 readiness hint equivalent to `configChanged: true`, or the backend field selected in Stage 0. Assert:
  - the catalog recovery state says discovery is stale or config changed;
  - `Refresh discovery` appears;
  - `Open server config` appears.

- [ ] **Step 6: Add no-tools test only if data supports it**

  If Stage 0 found a backend field that distinguishes successful zero-tool
  discovery from discovery not run, add a test asserting that `no_tools_returned`
  renders a `no_tools` display state with explanatory copy. Otherwise do not add
  this test in Phase 1.

- [ ] **Step 7: Add permission-gated catalog test only if data supports it**

  If Stage 0 found a response shape that distinguishes a permission-gated Tool
  Catalog from an ordinary empty/error state, add a test asserting that the
  recovery state explains the permission boundary and offers the selected access
  action, such as switching to Access/Policy Assignments or opening Details. If
  Stage 0 found no such signal, do not infer this state from generic failures.

- [ ] **Step 8: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  ```

- [ ] **Step 9: Load external server state**

  In `ToolCatalogsTab.tsx`, import `listExternalServers` and store servers alongside registry entries/modules.

  Keep loading and error states independent enough that a registry failure still reports registry failure, while server-state failure reports recovery limitations.

- [ ] **Step 10: Compute hub and server readiness**

  Use `getMcpHubReadiness()` and `getMcpServerReadiness()` to derive empty-state copy and actions. Pass the same registry entries and Stage 0 readiness hints used by `ExternalServersTab` so rows and Tool Catalog agree.

- [ ] **Step 11: Render actionable empty state**

  Replace the passive `Empty` copy with state-driven content:
  - no servers: `Add server`
  - checking: show the current validation/discovery operation and `View details`
  - discovery not run: `Refresh discovery` and `Open server config`
  - preflight failed: `Open server config` and `View details`
  - stale/config changed: `Refresh discovery` and `Open server config`
  - auth missing: `Fix credentials`
  - runtime unavailable: `Open server config`
  - unreachable: `Open server config`, `Refresh discovery`, and `View details`
  - discovery failed: `Refresh discovery` and `View details`
  - no tools returned: only when Stage 0 found an explicit successful-zero-tools
    signal, explain that the server responded with no tools and offer `Refresh
    discovery`
  - partial capability: allow Tool Catalog access and show warning detail
  - permission-gated catalog: only when Stage 0 found an explicit signal, explain
    the permission boundary and offer the selected Access/Details action

  `Refresh discovery` must call the real client method selected or added in Task 2. Do not render a disabled placeholder or clickable no-op in the final implementation.

- [ ] **Step 12: Run catalog tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  ```

- [ ] **Step 13: Run first-slice frontend suite**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/mcpHubReadiness.test.ts src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx
  ```

- [ ] **Step 14: Commit Tool Catalog recovery**

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

- [ ] **Step 3: Write failing import config test**

  Assert that `Import config` exposes a pasted JSON/config input, previews the decoded server ID/name/transport when valid, reports validation errors for invalid JSON, and does not save invalid input.

- [ ] **Step 4: Write failing result-panel test**

  Mock create/import plus refresh. Assert that after `Save and discover tools` or a valid import:
  - the setup form closes or moves to a result panel;
  - the result panel reports what was saved/imported;
  - the result panel reports discovery result/readiness;
  - the next actions include Tool Catalog and the valid recovery action for the returned readiness state.

- [ ] **Step 5: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 6: Add setup mode state**

  Add local state:

  ```ts
  type SetupMode = "choice" | "stdio" | "http" | "import" | "advanced"
  const [setupMode, setSetupMode] = useState<SetupMode>("choice")
  ```

- [ ] **Step 7: Add starter choice UI**

  When creating a new managed server, show the four setup choices first. Keep cards compact and work-focused. Avoid nested cards; use full-width rows or a simple segmented/button group inside the existing form area.

- [ ] **Step 8: Add guided stdio fields**

  Add fields for:
  - display name, or reuse the existing name field in the same create form and
    keep it visible before save
  - command
  - args
  - env vars
  - working directory
  - owner scope

  Convert them into the existing `config` object shape during save. Use the project’s observed external server config shape from Stage 0.

- [ ] **Step 9: Add guided HTTP/SSE fields**

  Add fields for:
  - display name, or reuse the existing name field in the same create form and
    keep it visible before save
  - URL
  - headers/auth choice
  - owner scope

  Convert them into the existing config object shape during save.

- [ ] **Step 10: Add import config path**

  Implement pasted JSON import with preview and validation. Support the existing legacy import path separately from pasted managed config import:
  - pasted/imported managed config should produce the same payload shape as `createExternalServer`;
  - legacy row import still uses `importExternalServer(serverId)`;
  - invalid JSON or unsupported shape shows a validation error before save/import.

- [ ] **Step 11: Add safe preflight checks**

  Before save, validate:
  - required fields;
  - JSON object shape;
  - URL syntax for HTTP/SSE;
  - command presence for stdio.

  Do not execute commands or contact URLs.

- [ ] **Step 12: Implement save, refresh, and result panel**

  Use:
  - primary `Save and discover tools`
  - secondary `Save without discovery`

  `Save and discover tools` must save/import, call the real refresh client selected or added in Task 2, reload servers/registry, and show a result panel. `Save without discovery` saves/imports, skips refresh, reloads server state, and shows a `discovery_not_run` result panel with `Refresh discovery` as the next action.

- [ ] **Step 13: Run setup tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 14: Commit setup guide**

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
  - diagnostics show display state, primary reason, all reason codes, credential state, transport, tool count;
  - diagnostics show `lastValidationAt`, `lastDiscoveryAt`, `lastSuccessfulDiscoveryAt`, and current operation metadata when supplied by readiness hints/backend response;
  - diagnostics show last sanitized error category/message when supplied;
  - diagnostics do not show secret values from nested env, headers, URL query, args, or raw config objects/arrays;
  - diagnostics show API origin/health endpoint/deployment mode if the data source exists, or an explicit unavailable message if not;
  - diagnostics show setup isolation guidance or a link/copy explaining how local walkthrough/E2E setup should avoid unintended runtime database writes.

- [ ] **Step 2: Run tests to verify failure**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 3: Add redaction helper**

  Put the helper in `mcpHubReadiness.ts` or a new pure helper file if it grows:

  ```ts
  const SECRET_KEY_PATTERN = /(token|secret|password|authorization|api[_-]?key)/i
  const redactUrlSecrets = (value: string) =>
    value.replace(/([?&][^=]*(token|key|secret|password)[^=]*=)[^&]+/gi, "$1[redacted]")

  export const redactMcpDiagnosticValue = (key: string, value: unknown): unknown => {
    const lowered = key.toLowerCase()
    if (SECRET_KEY_PATTERN.test(lowered)) {
      return "[redacted]"
    }
    if (typeof value === "string") {
      return redactUrlSecrets(value)
    }
    if (Array.isArray(value)) {
      return value.map((entry, index) => redactMcpDiagnosticValue(`${key}.${index}`, entry))
    }
    if (value && typeof value === "object") {
      return Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([nestedKey, nestedValue]) => [
          nestedKey,
          redactMcpDiagnosticValue(nestedKey, nestedValue)
        ])
      )
    }
    return value
  }

  export const formatMcpDiagnosticValue = (key: string, value: unknown) =>
    JSON.stringify(redactMcpDiagnosticValue(key, value), null, 2)
  ```

- [ ] **Step 4: Add diagnostics UI**

  Add a details drawer or modal that shows:
  - display state;
  - primary reason and reason codes;
  - credential state;
  - transport;
  - sanitized command/endpoint facts;
  - tool count;
  - last validation/discovery/success timestamps;
  - current operation ID, type, start time, initiator, and timeout when available;
  - last sanitized error category/message;
  - Audit tab link or copy explaining where audit details live.

- [ ] **Step 5: Add environment diagnostics**

  Show effective deployment mode, frontend API origin, health endpoint, latest health result, and setup isolation guidance. If a field is unavailable in this client, show `Not available in this client` for that exact field and ensure the Stage 0 Backlog note records whether a later shared diagnostics provider is needed.

- [ ] **Step 6: Add backend RBAC/audit verification if Task 2 touched backend**

  If Task 2 added or modified backend validation/discovery routes, add or update tests proving:
  - validation/discovery actions require the selected MCP Hub permission;
  - audit events are emitted for refresh/validation attempts;
  - audit payloads and error responses redact nested env, headers, URL query,
    args, and raw config secrets.

- [ ] **Step 7: Run diagnostics tests**

  ```bash
  cd apps/packages/ui
  bunx vitest run src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx
  ```

- [ ] **Step 8: Run backend diagnostics tests if backend changed**

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py -k "readiness or refresh or audit" -v
  ```

- [ ] **Step 9: Commit diagnostics**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py
  git commit -m "feat: add MCP Hub setup diagnostics"
  ```

  Only stage backend test files if they were changed.

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
  - `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`
  - `apps/tldw-frontend/e2e/utils/page-objects/MCPHubPage.ts`
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

- [ ] **Step 8: Add or update first-run browser smoke**

  In `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`, add a focused smoke test using route mocks unless Stage 0 found a safe toy MCP server path. The test must avoid writing to the developer's normal runtime database.

  Cover:
  - fresh MCP Hub setup state with no external servers and empty tool registry;
  - open Servers & Credentials;
  - open the create flow;
  - verify `Local stdio`, `HTTP/SSE`, `Import config`, and `Advanced/manual`;
  - choose the Stage 0 toy/local stdio path or a mocked equivalent;
  - fill the minimum safe fields;
  - trigger `Save and discover tools`;
  - receive a mocked or toy-server refresh result;
  - verify the result panel and Tool Catalog recovery/ready state match the returned readiness.

  Use the existing `MCPHubPage` page object. Add helpers there instead of duplicating selectors in the spec when the helper will be reused by later MCP Hub E2E tests.

- [ ] **Step 9: Add responsive screenshot assertions**

  In the same Playwright spec, add a screenshot-oriented test that runs MCP Hub at:
  - desktop: `1440x900`
  - mobile: `390x844`

  For each viewport:
  - navigate to `/mcp-hub?workflow=setup&view=credentials`;
  - open the guided setup choices;
  - open diagnostics for a mocked server state;
  - switch to `/mcp-hub?workflow=setup&view=tool-catalogs`;
  - assert `document.documentElement.scrollWidth <= document.documentElement.clientWidth`;
  - attach screenshots with names such as `mcp-hub-setup-desktop.png` and `mcp-hub-setup-mobile.png` using Playwright `testInfo.attach`.

- [ ] **Step 10: Run first-run browser smoke and capture evidence**

  ```bash
  cd apps/tldw-frontend
  bun run e2e:pw -- e2e/workflows/tier-2-features/mcp-hub.spec.ts --project=tier-2 --reporter=line
  ```

  Expected: pass. Record the command, outcome, and screenshot artifact locations in `TASK-223.2`.

- [ ] **Step 11: Record explicit smoke skips only when unavoidable**

  If the full `Save and discover tools` smoke cannot run because no stable toy MCP server or safe mocked backend path exists, do not silently omit it. Record in `TASK-223.2`:
  - exact missing dependency;
  - why mocked route coverage cannot exercise the path;
  - closest browser test that was run instead;
  - follow-up task or issue needed to unblock full first-success smoke.

  Even when the full save/discovery smoke is skipped, the plan still requires a browser test for setup choices, recovery copy, no horizontal overflow, and desktop/mobile screenshots.

- [ ] **Step 12: Update Backlog task**

  Add verification results, known skips, and final summary to `TASK-223.2`.

- [ ] **Step 13: Final commit**

  ```bash
  git add apps/packages/ui/src/components/Option/MCPHub apps/packages/ui/src/services/tldw/mcp-hub.ts apps/packages/ui/src/services/tldw/__tests__/mcp-hub.test.ts apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts apps/tldw-frontend/e2e/utils/page-objects/MCPHubPage.ts tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py tldw_Server_API/tests/MCP_unified/test_mcp_hub_management_api.py "backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md"
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
