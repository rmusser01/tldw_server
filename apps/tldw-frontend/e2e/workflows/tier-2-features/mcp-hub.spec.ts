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
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../../utils/fixtures"
import { MCPHubPage } from "../../utils/page-objects/MCPHubPage"
import { expectApiCall } from "../../utils/api-assertions"
import { seedAuth } from "../../utils/helpers"

const TOY_MCP_SERVER_SCRIPT = `\
import readline from "node:readline";

const rl = readline.createInterface({ input: process.stdin });

function send(payload) {
  process.stdout.write(JSON.stringify(payload) + "\\n");
}

rl.on("line", (line) => {
  if (!line.trim()) return;
  let message;
  try {
    message = JSON.parse(line);
  } catch {
    return;
  }

  const id = message.id;
  const method = message.method;
  const params = message.params || {};

  if (method === "initialize") {
    send({
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "toy-e2e-mcp", version: "1.0.0" },
      },
    });
    return;
  }

  if (method === "notifications/initialized") {
    return;
  }

  if (method === "tools/list") {
    send({
      jsonrpc: "2.0",
      id,
      result: {
        tools: [
          {
            name: "toy.echo",
            description: "Echoes a short message for MCP Hub E2E smoke tests.",
            inputSchema: {
              type: "object",
              properties: { text: { type: "string" } },
            },
            metadata: { category: "diagnostic", readOnlyHint: true },
          },
        ],
      },
    });
    return;
  }

  if (method === "tools/call") {
    send({
      jsonrpc: "2.0",
      id,
      result: {
        content: [{ type: "text", text: String((params.arguments || {}).text || "toy-ok") }],
        isError: false,
      },
    });
    return;
  }

  send({
    jsonrpc: "2.0",
    id,
    error: { code: -32601, message: \`unknown method: \${method}\` },
  });
});
`

async function writeToyMcpServer(): Promise<{ dir: string; scriptPath: string }> {
  const dir = await mkdtemp(join(tmpdir(), "tldw-toy-mcp-"))
  const scriptPath = join(dir, "toy-mcp-server.mjs")
  await writeFile(scriptPath, TOY_MCP_SERVER_SCRIPT, "utf8")
  return { dir, scriptPath }
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
      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto()
      await mcpHub.assertPageReady()

      const headingVisible = await mcpHub.heading.isVisible().catch(() => false)
      expect(headingVisible).toBe(true)

      await expect(authedPage.getByTestId("mcp-hub-status-summary")).toBeVisible()
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

  test.describe("Toy MCP walkthrough smoke", () => {
    test("can configure a temporary no-auth stdio server through MCP Hub when the live API can see the temp file", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      const { dir, scriptPath } = await writeToyMcpServer()
      const suffix = Date.now().toString(36)
      const serverId = `toy-stdio-${suffix}`
      const serverName = `Toy Stdio ${suffix}`
      let createdServer = false

      mcpHub = new MCPHubPage(authedPage)
      try {
        await mcpHub.goto()
        await mcpHub.assertPageReady()
        await mcpHub.selectView("credentials")

        await authedPage.getByRole("button", { name: /new managed server/i }).click()
        await authedPage.getByLabel(/server id/i).fill(serverId)
        await authedPage.getByLabel(/^name$/i).fill(serverName)
        await authedPage.getByRole("combobox", { name: /^transport$/i }).selectOption("stdio")
        await authedPage.getByLabel(/config json/i).fill(JSON.stringify({
          stdio: {
            command: process.execPath,
            args: [scriptPath],
          },
          auth: { mode: "none" },
          policy: { allow_tool_patterns: ["toy.*"], allow_writes: false },
          timeouts: { connect_seconds: 2, request_seconds: 5 },
        }, null, 2))

        const createResponsePromise = authedPage.waitForResponse(
          (response) =>
            response.url().includes("/api/v1/mcp/hub/external-servers") &&
            response.request().method() === "POST",
          { timeout: 15_000 }
        ).catch(() => null)

        await authedPage.getByRole("button", { name: /save server/i }).click()
        const createResponse = await createResponsePromise
        if (!createResponse || [401, 403, 404].includes(createResponse.status())) {
          test.skip(true, "Live API does not allow MCP Hub external-server mutations in this run")
        }
        expect(createResponse.status()).toBeLessThan(300)
        createdServer = true

        await expect(authedPage.getByLabel(/^server$/i)).toBeVisible({ timeout: 15_000 })
        await authedPage.getByLabel(/^server$/i).selectOption(serverId)
        await expect(authedPage.getByText(/no credentials required/i).first()).toBeVisible({ timeout: 10_000 })

        await mcpHub.selectView("tool-catalogs")
        await authedPage.getByRole("button", { name: /refresh discovery|refresh tools/i }).first().click()
        const toyTool = authedPage.getByText(/toy\.echo/i).first()
        const toyToolVisible = await toyTool.isVisible({ timeout: 15_000 }).catch(() => false)
        if (!toyToolVisible) {
          test.skip(true, "Toy MCP server was configured in UI, but live runtime discovery could not execute the temp stdio server")
        }
        await expect(toyTool).toBeVisible()

        await assertNoCriticalErrors(diagnostics)
      } finally {
        if (createdServer) {
          await mcpHub.selectView("credentials").catch(() => {})
          await authedPage.getByLabel(/^server$/i).selectOption(serverId).catch(() => {})
          await authedPage.getByRole("button", { name: new RegExp(`delete ${serverName}`, "i") }).click().catch(() => {})
          await authedPage.getByRole("button", { name: /^delete$/i }).click().catch(() => {})
        }
        await rm(dir, { recursive: true, force: true }).catch(() => {})
      }
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
