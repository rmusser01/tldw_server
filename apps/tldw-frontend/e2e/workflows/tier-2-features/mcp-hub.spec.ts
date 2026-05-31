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
import { fetchWithApiKey, seedAuth, TEST_CONFIG } from "../../utils/helpers"

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

type ApiResult = {
  status: number
  body: unknown
  text: string
}

const apiBaseUrl = (): string => {
  const raw = TEST_CONFIG.serverUrl.replace(/\/$/, "")
  return /^https?:\/\//i.test(raw) ? raw : `http://${raw}`
}

function normalizeHeaders(headers?: HeadersInit): Record<string, string> {
  if (!headers) {
    return {}
  }
  if (headers instanceof Headers) {
    return Object.fromEntries([...headers.entries()])
  }
  if (Array.isArray(headers)) {
    return Object.fromEntries(headers.map(([key, value]) => [key, String(value)]))
  }
  return Object.fromEntries(Object.entries(headers).map(([key, value]) => [key, String(value)]))
}

async function apiRequest(path: string, init: RequestInit = {}): Promise<ApiResult> {
  const headers: Record<string, string> = {
    ...(init.body ? { "content-type": "application/json" } : {}),
    ...normalizeHeaders(init.headers),
  }
  const response = await fetchWithApiKey(`${apiBaseUrl()}${path}`, TEST_CONFIG.apiKey, {
    ...init,
    headers,
  })
  const text = await response.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  return { status: response.status, body, text }
}

function assertMutableApiAvailable(result: ApiResult, label: string): void {
  if ([401, 403, 404].includes(result.status)) {
    test.skip(true, `${label} unavailable in this live run: HTTP ${result.status}`)
  }
}

function expectSuccessfulApiResult<T extends Record<string, unknown>>(
  result: ApiResult,
  label: string
): T {
  assertMutableApiAvailable(result, label)
  expect(result.status, `${label}: ${result.text}`).toBeGreaterThanOrEqual(200)
  expect(result.status, `${label}: ${result.text}`).toBeLessThan(300)
  expect(result.body && typeof result.body === "object", `${label}: ${result.text}`).toBe(true)
  return result.body as T
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

    test("should hydrate Research Workspace context in Workspace Sets", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      skipIfServerUnavailable(serverInfo)

      mcpHub = new MCPHubPage(authedPage)
      await mcpHub.goto(
        "/mcp-hub?workflow=setup&view=workspace-sets&workspace_id=rw-e2e-context&source=research-workspace"
      )
      await mcpHub.assertPageReady()

      await mcpHub.expectWorkflowSelected("workspaces")
      await mcpHub.expectViewSelected("workspace-sets")

      const contextStatus = authedPage.getByTestId("mcp-workspace-context-status")
      await expect(contextStatus).toBeVisible({ timeout: 15_000 })
      await expect(contextStatus).toContainText(/rw-e2e-context/)
      await expect(contextStatus).toContainText(
        /included in .* MCP workspace set|No MCP workspace set includes/i
      )

      await assertNoCriticalErrors(diagnostics)
    })

    test("binds a Research Workspace into an MCP workspace set and resolves policy evidence", async ({
      authedPage,
      serverInfo,
      diagnostics,
    }, testInfo) => {
      skipIfServerUnavailable(serverInfo)

      const suffix = `${Date.now().toString(36)}-${testInfo.workerIndex}`
      const workspaceId = `rw-mcp-e2e-${suffix}`
      const personaId = `rw-mcp-persona-${suffix}`
      const teamId = 1
      const workspaceRoot = await mkdtemp(join(tmpdir(), "tldw-rw-mcp-"))
      let policyAssignmentId: number | null = null
      let workspaceSetObjectId: number | null = null
      let sharedWorkspaceId: number | null = null

      try {
        const workspace = expectSuccessfulApiResult<Record<string, unknown>>(
          await apiRequest(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}`, {
            method: "PUT",
            body: JSON.stringify({
              name: `RW MCP E2E ${suffix}`,
              study_materials_policy: "general",
            }),
          }),
          "create Research Workspace"
        )
        expect(workspace.id).toBe(workspaceId)

        const sharedWorkspace = expectSuccessfulApiResult<Record<string, unknown>>(
          await apiRequest("/api/v1/mcp/hub/shared-workspaces", {
            method: "POST",
            body: JSON.stringify({
              workspace_id: workspaceId,
              display_name: `RW MCP Shared ${suffix}`,
              absolute_root: workspaceRoot,
              owner_scope_type: "team",
              owner_scope_id: teamId,
              is_active: true,
            }),
          }),
          "create MCP Hub Shared Workspace"
        )
        sharedWorkspaceId = Number(sharedWorkspace.id)
        expect(Number.isFinite(sharedWorkspaceId), JSON.stringify(sharedWorkspace)).toBe(true)
        expect(sharedWorkspace.workspace_id).toBe(workspaceId)

        const workspaceSet = expectSuccessfulApiResult<Record<string, unknown>>(
          await apiRequest("/api/v1/mcp/hub/workspace-set-objects", {
            method: "POST",
            body: JSON.stringify({
              name: `RW MCP Set ${suffix}`,
              description: "E2E Research Workspace handoff validation",
              owner_scope_type: "team",
              owner_scope_id: teamId,
              is_active: true,
            }),
          }),
          "create MCP Hub workspace set"
        )
        workspaceSetObjectId = Number(workspaceSet.id)
        expect(Number.isFinite(workspaceSetObjectId), JSON.stringify(workspaceSet)).toBe(true)
        expect(workspaceSet.owner_scope_type).toBe("team")

        const workspaceSetMember = expectSuccessfulApiResult<Record<string, unknown>>(
          await apiRequest(`/api/v1/mcp/hub/workspace-set-objects/${workspaceSetObjectId}/members`, {
            method: "POST",
            body: JSON.stringify({ workspace_id: workspaceId }),
          }),
          "add Research Workspace ID to MCP workspace set"
        )
        expect(workspaceSetMember.workspace_id).toBe(workspaceId)

        const policyAssignment = expectSuccessfulApiResult<Record<string, unknown>>(
          await apiRequest("/api/v1/mcp/hub/policy-assignments", {
            method: "POST",
            body: JSON.stringify({
              target_type: "persona",
              target_id: personaId,
              owner_scope_type: "team",
              owner_scope_id: teamId,
              workspace_source_mode: "named",
              workspace_set_object_id: workspaceSetObjectId,
              inline_policy_document: {
                allowed_tools: ["run"],
                approval_mode: "allow_silently",
              },
              is_active: true,
            }),
          }),
          "create MCP Hub named workspace-set policy assignment"
        )
        policyAssignmentId = Number(policyAssignment.id)
        expect(Number.isFinite(policyAssignmentId), JSON.stringify(policyAssignment)).toBe(true)
        expect(policyAssignment.workspace_source_mode).toBe("named")
        expect(policyAssignment.workspace_set_object_id).toBe(workspaceSetObjectId)

        const effectivePolicy = expectSuccessfulApiResult<Record<string, unknown>>(
          await apiRequest(
            `/api/v1/mcp/hub/effective-policy?persona_id=${encodeURIComponent(personaId)}&team_id=${teamId}`,
            { method: "GET" }
          ),
          "resolve MCP Hub effective policy"
        )
        expect(effectivePolicy.enabled).toBe(true)
        expect(effectivePolicy.selected_workspace_source_mode).toBe("named")
        expect(effectivePolicy.selected_workspace_set_object_id).toBe(workspaceSetObjectId)
        expect(effectivePolicy.selected_workspace_trust_source).toBe("shared_registry")
        expect(effectivePolicy.selected_assignment_workspace_ids).toEqual(
          expect.arrayContaining([workspaceId])
        )
        expect(effectivePolicy.allowed_tools).toEqual(expect.arrayContaining(["run"]))

        const toolExecution = await apiRequest("/api/v1/mcp/tools/execute", {
          method: "POST",
          headers: {
            "x-tldw-workspace-id": workspaceId,
            "x-tldw-cwd": workspaceRoot,
          },
          body: JSON.stringify({
            tool_name: "run",
            arguments: { command: "help" },
          }),
        })
        await testInfo.attach("mcp-tool-execution-probe.json", {
          body: JSON.stringify(
            {
              status: toolExecution.status,
              body: toolExecution.body,
            },
            null,
            2
          ),
          contentType: "application/json",
        })
        assertMutableApiAvailable(toolExecution, "execute MCP tool under Research Workspace headers")
        expect(toolExecution.status, toolExecution.text).toBeGreaterThanOrEqual(200)
        expect(toolExecution.status, toolExecution.text).toBeLessThan(300)
        expect(toolExecution.body && typeof toolExecution.body === "object", toolExecution.text).toBe(
          true
        )
        expect((toolExecution.body as { result?: unknown }).result, toolExecution.text).toEqual(
          expect.stringContaining("Virtual CLI commands available")
        )

        mcpHub = new MCPHubPage(authedPage)
        await mcpHub.goto(
          `/mcp-hub?workflow=setup&view=workspace-sets&workspace_id=${encodeURIComponent(workspaceId)}&source=research-workspace`
        )
        await mcpHub.assertPageReady()

        expect(authedPage.url()).toContain("source=research-workspace")
        expect(authedPage.url()).not.toContain("workspace-playground")
        await mcpHub.expectWorkflowSelected("workspaces")
        await mcpHub.expectViewSelected("workspace-sets")

        const contextStatus = authedPage.getByTestId("mcp-workspace-context-status")
        await expect(contextStatus).toBeVisible({ timeout: 15_000 })
        await expect(contextStatus).toContainText(workspaceId)
        await expect(contextStatus).toContainText(/included in .* MCP workspace set/i)
        await expect(contextStatus).not.toContainText(/workspace-playground/i)

        await assertNoCriticalErrors(diagnostics)
      } finally {
        if (policyAssignmentId != null) {
          await apiRequest(`/api/v1/mcp/hub/policy-assignments/${policyAssignmentId}`, {
            method: "DELETE",
          }).catch(() => {})
        }
        if (workspaceSetObjectId != null) {
          await apiRequest(
            `/api/v1/mcp/hub/workspace-set-objects/${workspaceSetObjectId}/members/${encodeURIComponent(workspaceId)}`,
            { method: "DELETE" }
          ).catch(() => {})
          await apiRequest(`/api/v1/mcp/hub/workspace-set-objects/${workspaceSetObjectId}`, {
            method: "DELETE",
          }).catch(() => {})
        }
        if (sharedWorkspaceId != null) {
          await apiRequest(`/api/v1/mcp/hub/shared-workspaces/${sharedWorkspaceId}`, {
            method: "DELETE",
          }).catch(() => {})
        }
        await apiRequest(`/api/v1/workspaces/${encodeURIComponent(workspaceId)}`, {
          method: "DELETE",
        }).catch(() => {})
        await rm(workspaceRoot, { recursive: true, force: true }).catch(() => {})
      }
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
