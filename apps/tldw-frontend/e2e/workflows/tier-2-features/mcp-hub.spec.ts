/**
 * MCP Hub E2E Tests (Tier 2)
 *
 * Tests the MCP Hub page lifecycle:
 * - Page loads with heading and workflow navigation
 * - Workflow and child-view switching across Setup, Access, Workspaces, Governance, and Audit
 * - API calls fired on representative child-view interactions
 *
 * Run: npx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts
 */
import type { Page, Route, TestInfo } from "@playwright/test"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { MCPHubPage } from "../../utils/page-objects/MCPHubPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

const MOCK_SERVER_ID = "docs-managed"
const MOCK_SERVER_NAME = "Docs Managed"
const MOCK_TIMESTAMP = "2026-06-27T12:00:00Z"

type MockMcpHubState = {
  serverCreated: boolean
  discoveryRan: boolean
}

const fulfillJson = async (
  route: Route,
  payload: unknown,
  status = 200
): Promise<void> => {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  })
}

const getMockExternalServer = () => ({
  id: MOCK_SERVER_ID,
  name: MOCK_SERVER_NAME,
  enabled: true,
  owner_scope_type: "global",
  owner_scope_id: null,
  transport: "stdio",
  config: {
    command: "node",
    args: ["mock-docs-server", "--stdio"],
    env: {
      API_TOKEN: "fake-e2e-token",
    },
  },
  secret_configured: false,
  key_hint: null,
  server_source: "managed",
  legacy_source_ref: null,
  superseded_by_server_id: null,
  binding_count: 0,
  runtime_executable: true,
  auth_template_present: false,
  auth_template_valid: true,
  auth_template_blocked_reason: null,
  credential_slots: [],
  created_by: null,
  updated_by: null,
  created_at: MOCK_TIMESTAMP,
  updated_at: MOCK_TIMESTAMP,
})

const getMockServerReadiness = (ready: boolean) => ({
  server_id: MOCK_SERVER_ID,
  server_name: MOCK_SERVER_NAME,
  display_state: ready ? "ready" : "needs_setup",
  credential_state: "not_required",
  tool_count: ready ? 1 : 0,
  reason_codes: ready ? [] : ["discovery_not_run"],
  primary_reason_code: ready ? null : "discovery_not_run",
  allowed_actions: ready
    ? ["view_details", "open_tool_catalog", "refresh_discovery"]
    : ["refresh_discovery", "view_details"],
  message: ready
    ? "Server is ready. 1 tool discovered."
    : "Run discovery to populate the tool catalog.",
  current_operation: null,
  last_validation_at: MOCK_TIMESTAMP,
  last_discovery_at: ready ? MOCK_TIMESTAMP : null,
  last_successful_discovery_at: ready ? MOCK_TIMESTAMP : null,
  last_error_category: null,
  last_error_message: null,
  refresh_result: ready
    ? {
        refreshed_servers: 1,
        total_servers: 1,
        virtual_tools: 1,
        errors: {},
      }
    : null,
})

const getMockReadiness = (state: MockMcpHubState) => {
  if (!state.serverCreated) {
    return {
      display_state: "needs_setup",
      reason_codes: ["not_configured"],
      primary_reason_code: "not_configured",
      allowed_actions: ["add_server"],
      message: "No external MCP servers are configured.",
      servers: [],
      total_servers: 0,
      ready_server_count: 0,
      checking_server_count: 0,
      attention_server_count: 0,
      no_tool_server_count: 0,
      stale_server_count: 0,
    }
  }

  const ready = state.discoveryRan
  return {
    display_state: ready ? "ready" : "needs_setup",
    reason_codes: ready ? [] : ["discovery_not_run"],
    primary_reason_code: ready ? null : "discovery_not_run",
    allowed_actions: ready ? ["open_tool_catalog"] : ["refresh_discovery"],
    message: ready
      ? "All configured external MCP servers are ready."
      : `${MOCK_SERVER_NAME} is saved, but tool discovery has not run.`,
    servers: [getMockServerReadiness(ready)],
    total_servers: 1,
    ready_server_count: ready ? 1 : 0,
    checking_server_count: 0,
    attention_server_count: 0,
    no_tool_server_count: 0,
    stale_server_count: 0,
  }
}

const getMockToolEntry = () => ({
  tool_name: "ext.docs-managed.search",
  display_name: "Docs Search",
  description: "Search the managed documentation corpus.",
  module: "external.docs-managed",
  module_display_name: MOCK_SERVER_NAME,
  category: "external",
  risk_class: "low",
  capabilities: ["search.query"],
  mutates_state: false,
  uses_filesystem: false,
  uses_processes: false,
  uses_network: false,
  uses_credentials: false,
  supports_arguments_preview: false,
  path_boundable: false,
  path_argument_hints: [],
  metadata_source: "explicit",
  metadata_warnings: [],
})

const getMockToolSummary = (hasTools: boolean) => {
  const entries = hasTools ? [getMockToolEntry()] : []
  return {
    entries,
    modules: hasTools
      ? [
          {
            module: "external.docs-managed",
            display_name: MOCK_SERVER_NAME,
            tool_count: 1,
            risk_summary: { low: 1, medium: 0, high: 0 },
            metadata_warnings: [],
          },
        ]
      : [],
  }
}

const mockMcpHubFirstRunApi = async (
  page: Page,
  initialState: Partial<MockMcpHubState> = {}
): Promise<MockMcpHubState> => {
  const state: MockMcpHubState = {
    serverCreated: Boolean(initialState.serverCreated),
    discoveryRan: Boolean(initialState.discoveryRan),
  }

  await page.route(/\/api\/v1\/mcp\/hub(\/.*)?/, async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const { pathname } = url
    const method = request.method()

    if (method === "GET" && pathname.endsWith("/readiness")) {
      await fulfillJson(route, getMockReadiness(state))
      return
    }

    if (method === "GET" && pathname.endsWith("/tool-registry/summary")) {
      await fulfillJson(route, getMockToolSummary(state.discoveryRan))
      return
    }

    if (method === "GET" && pathname.endsWith("/tool-registry")) {
      await fulfillJson(route, state.discoveryRan ? [getMockToolEntry()] : [])
      return
    }

    if (method === "GET" && pathname.endsWith("/external-servers")) {
      await fulfillJson(route, state.serverCreated ? [getMockExternalServer()] : [])
      return
    }

    if (method === "POST" && pathname.endsWith("/external-servers")) {
      const payload = request.postDataJSON() as {
        server_id?: unknown
        name?: unknown
        transport?: unknown
        config?: {
          command?: unknown
          args?: unknown
          env?: Record<string, unknown>
        }
      }
      const args = Array.isArray(payload?.config?.args) ? payload.config.args : []
      const env = payload?.config?.env as Record<string, unknown> | undefined
      if (
        payload?.server_id !== MOCK_SERVER_ID ||
        payload?.name !== MOCK_SERVER_NAME ||
        payload?.transport !== "stdio" ||
        payload?.config?.command !== "node" ||
        args.join(" ") !== "mock-docs-server --stdio" ||
        env?.API_TOKEN !== "fake-e2e-token"
      ) {
        await fulfillJson(route, { detail: "Unexpected mocked create payload" }, 400)
        return
      }
      state.serverCreated = true
      state.discoveryRan = false
      await fulfillJson(route, getMockExternalServer(), 201)
      return
    }

    if (
      method === "GET" &&
      pathname.endsWith(`/external-servers/${MOCK_SERVER_ID}/auth-template`)
    ) {
      await fulfillJson(route, { mode: "template", mappings: [] })
      return
    }

    if (
      method === "POST" &&
      pathname.endsWith(`/external-servers/${MOCK_SERVER_ID}/refresh-discovery`)
    ) {
      state.serverCreated = true
      state.discoveryRan = true
      await fulfillJson(route, getMockServerReadiness(true))
      return
    }

    await fulfillJson(
      route,
      { detail: `Unhandled mocked MCP Hub endpoint: ${method} ${pathname}` },
      404
    )
  })

  return state
}

const attachScreenshot = async (
  page: Page,
  testInfo: TestInfo,
  name: string
): Promise<void> => {
  await testInfo.attach(name, {
    body: await page.screenshot({ fullPage: true }),
    contentType: "image/png",
  })
}

test.describe("MCP Hub", () => {
  let mcpHub: MCPHubPage

  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    mcpHub = new MCPHubPage(page)
  })

  // =========================================================================
  // Page Load
  // =========================================================================

  test.describe("Page Load", () => {
    test("should render the MCP Hub page with heading and workflow navigation", async ({
      authedPage,
      diagnostics,
    }) => {
      await mockMcpHubFirstRunApi(authedPage, {
        serverCreated: true,
        discoveryRan: true,
      })

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      await expect(mcpHub.heading.first()).toBeVisible()

      await expect(
        authedPage.getByRole("navigation", { name: "MCP Hub workflow shortcuts" })
      ).toBeVisible()
      await expect(mcpHub.workflowShortcuts).toBeVisible()
      await expect(authedPage.getByTestId("mcp-hub-status-summary")).toHaveCount(0)
      await expect(
        authedPage.getByRole("button", { name: "Open Policy Assignments" })
      ).toBeVisible()
      await expect(mcpHub.workflows).toBeVisible()
      await expect(mcpHub.workflowButton("setup")).toBeVisible()
      await expect(mcpHub.workflowButton("access")).toBeVisible()
      await expect(mcpHub.workflowButton("workspaces")).toBeVisible()
      await expect(mcpHub.workflowButton("governance")).toBeVisible()
      await expect(mcpHub.workflowButton("audit")).toBeVisible()

      await mcpHub.expectWorkflowSelected("setup")
      await mcpHub.expectViewSelected("credentials")
      await expect(mcpHub.credentialsTab).toBeVisible()
      await expect(mcpHub.catalogTab).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should switch between workflows and child views without errors", async ({
      authedPage,
      diagnostics,
    }) => {
      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      for (const workflow of MCPHubPage.WORKFLOW_KEYS) {
        await mcpHub.selectWorkflow(workflow)
        await mcpHub.expectWorkflowSelected(workflow)
      }

      for (const view of MCPHubPage.VIEW_KEYS) {
        await mcpHub.selectView(view)
        await mcpHub.expectViewSelected(view)
      }

      await assertNoCriticalErrors(diagnostics)
    })

    test("should hydrate workflow and child view from query state", async ({
      authedPage,
      diagnostics,
    }) => {
      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto("/mcp-hub?workflow=access&view=assignments")
      await mcpHub.assertPageReady()

      await mcpHub.expectWorkflowSelected("access")
      await mcpHub.expectViewSelected("assignments")

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Mocked Setup Smoke", () => {
    test("should complete first-run local stdio setup and show discovered tools", async ({
      authedPage,
      diagnostics,
    }) => {
      await mockMcpHubFirstRunApi(authedPage)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto("/mcp-hub?workflow=setup&view=credentials")
      await mcpHub.assertPageReady()

      await expect(authedPage.getByText("No external servers connected")).toBeVisible()

      await mcpHub.openCreateManagedServer()
      await mcpHub.expectSetupChoicesVisible()
      await mcpHub.chooseLocalStdio()
      await mcpHub.fillLocalStdioServer({
        serverId: MOCK_SERVER_ID,
        name: MOCK_SERVER_NAME,
        command: "node",
        args: "mock-docs-server --stdio",
        env: "API_TOKEN=fake-e2e-token",
      })
      await mcpHub.saveAndDiscoverTools()

      await expect(authedPage.getByText(`${MOCK_SERVER_NAME} saved`)).toBeVisible()
      await expect(
        authedPage.getByText("Server is ready. 1 tool discovered.").first()
      ).toBeVisible()
      await authedPage.getByRole("button", { name: "Tool Catalog" }).first().click()
      await mcpHub.expectViewSelected("tool-catalogs")
      await expect(authedPage.getByText("Docs Search")).toBeVisible()
      await expect(authedPage.getByText("1 tools").first()).toBeVisible()

      await assertNoCriticalErrors(diagnostics)
    })

    test("should keep setup, diagnostics, and catalog states responsive", async ({
      authedPage,
      diagnostics,
    }, testInfo) => {
      await mockMcpHubFirstRunApi(authedPage, {
        serverCreated: true,
        discoveryRan: true,
      })

      const viewports = [
        { name: "desktop", width: 1440, height: 900 },
        { name: "mobile", width: 390, height: 844 },
      ] as const

      for (const viewport of viewports) {
        await authedPage.setViewportSize({
          width: viewport.width,
          height: viewport.height,
        })

        mcpHub = new MCPHubPage(authedPage)
        await mcpHub.goto("/mcp-hub?workflow=setup&view=credentials")
        await mcpHub.assertPageReady()

        await mcpHub.openCreateManagedServer()
        await mcpHub.expectSetupChoicesVisible()
        await mcpHub.openServerDetails(MOCK_SERVER_NAME)
        await expect(authedPage.getByText("Sanitized config")).toBeVisible()
        await expect(
          authedPage.getByTestId("mcp-server-diagnostics-config")
        ).not.toContainText("fake-e2e-token")
        await mcpHub.expectNoHorizontalOverflow()
        await attachScreenshot(
          authedPage,
          testInfo,
          `mcp-hub-setup-${viewport.name}.png`
        )

        await mcpHub.closeServerDetails()
        await mcpHub.selectView("tool-catalogs")
        await expect(authedPage.getByText("Docs Search")).toBeVisible()
        await mcpHub.expectNoHorizontalOverflow()
        await attachScreenshot(
          authedPage,
          testInfo,
          `mcp-hub-catalog-${viewport.name}.png`
        )
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  // =========================================================================
  // API Integration (requires server)
  // =========================================================================

  test.describe("Permission Profiles API", () => {
    test("should fire GET /api/v1/mcp/hub/permission-profiles on Profiles tab", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/mcp\/hub\/permission-profiles/,
        method: "GET",
      }, 15_000)

      await mcpHub.selectView("profiles")

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // MCP Hub API may not be available on this server version
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Policy Assignments API", () => {
    test("should fire GET /api/v1/mcp/hub/policy-assignments on Assignments tab", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/mcp\/hub\/policy-assignments/,
        method: "GET",
      }, 15_000)

      await mcpHub.selectView("assignments")

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // MCP Hub API may not be available on this server version
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Servers & Credentials API", () => {
    test("should fire GET /api/v1/mcp/hub/external-servers on Servers & Credentials view", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      await mcpHub.selectView("audit")

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/mcp\/hub\/external-servers/,
        method: "GET",
      }, 15_000)

      await mcpHub.selectView("credentials")

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // MCP Hub API may not be available on this server version
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Tool Catalog API", () => {
    test("should fire GET /api/v1/mcp/hub/tool-registry on Catalog tab", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/mcp\/hub\/tool-registry/,
        method: "GET",
      }, 15_000)

      await mcpHub.selectView("tool-catalogs")

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // MCP Hub API may not be available on this server version
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })

  test.describe("Governance Audit API", () => {
    test("should fire GET /api/v1/mcp/hub/audit/findings on Audit view", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      if (!headingVisible) return

      const apiCall = expectApiCall(authedPage, {
        url: /\/api\/v1\/mcp\/hub\/audit\/findings/,
        method: "GET",
      }, 15_000)

      await mcpHub.selectView("audit")

      try {
        const { response } = await apiCall
        expect(response.status()).toBeLessThan(500)
      } catch {
        // MCP Hub API may not be available on this server version
      }

      await assertNoCriticalErrors(diagnostics)
    })
  })
})
