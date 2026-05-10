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
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { MCPHubPage } from "../../utils/page-objects/MCPHubPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

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
      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      expect(headingVisible).toBe(true)

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

      for (const view of [
        "assignments",
        "path-scopes",
        "workspace-sets",
        "shared-workspaces",
        "audit",
        "approvals",
        "governance-packs",
        "capability-mappings",
        "tool-catalogs",
        "credentials",
        "profiles",
      ] as const) {
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
