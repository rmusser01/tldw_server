/**
 * Page Object for MCP Hub workflow
 */
import { type Page, type Locator, expect } from "@playwright/test"
import { BasePage, type InteractiveElement } from "./BasePage"
import { waitForAppShell, waitForConnection } from "../helpers"

export type MCPHubWorkflowKey =
  | "setup"
  | "access"
  | "workspaces"
  | "governance"
  | "audit"

export type MCPHubViewKey =
  | "profiles"
  | "assignments"
  | "path-scopes"
  | "workspace-sets"
  | "shared-workspaces"
  | "audit"
  | "approvals"
  | "governance-packs"
  | "capability-mappings"
  | "tool-catalogs"
  | "credentials"

export type MCPHubLocalStdioServerInput = {
  serverId: string
  name: string
  command: string
  args?: string
  cwd?: string
  env?: string
}

const VIEW_TO_WORKFLOW: Record<MCPHubViewKey, MCPHubWorkflowKey> = {
  profiles: "access",
  assignments: "access",
  "path-scopes": "workspaces",
  "workspace-sets": "workspaces",
  "shared-workspaces": "workspaces",
  audit: "audit",
  approvals: "governance",
  "governance-packs": "governance",
  "capability-mappings": "governance",
  "tool-catalogs": "setup",
  credentials: "setup",
}

export class MCPHubPage extends BasePage {
  constructor(page: Page) {
    super(page)
  }

  // -- Navigation ------------------------------------------------------------

  async goto(path = "/mcp-hub"): Promise<void> {
    await this.page.goto(path, { waitUntil: "domcontentloaded" })
    await waitForConnection(this.page)
  }

  async assertPageReady(): Promise<void> {
    await waitForAppShell(this.page, 30_000)
    const heading = this.page.getByRole("heading", { name: /mcp hub/i })
    await heading.first().waitFor({ state: "visible", timeout: 20_000 }).catch(() => {})
  }

  // -- Locators --------------------------------------------------------------

  /** MCP Hub heading */
  get heading(): Locator {
    return this.page.getByRole("heading", { name: /mcp hub/i })
  }

  get workflows(): Locator {
    return this.page.getByTestId("mcp-hub-workflows")
  }

  get workflowShortcuts(): Locator {
    return this.page.getByTestId("mcp-hub-workflow-shortcuts")
  }

  workflowButton(workflow: MCPHubWorkflowKey): Locator {
    return this.page.getByTestId(`mcp-hub-workflow-${workflow}`)
  }

  viewTab(view: MCPHubViewKey): Locator {
    return this.page.getByTestId(`mcp-hub-tab-${view}`)
  }

  viewTabControl(view: MCPHubViewKey): Locator {
    return this.viewTab(view).locator("xpath=ancestor::*[@role='tab']")
  }

  /** Profiles tab */
  get profilesTab(): Locator {
    return this.viewTab("profiles")
  }

  /** Assignments tab */
  get assignmentsTab(): Locator {
    return this.viewTab("assignments")
  }

  /** Path Scopes tab */
  get pathScopesTab(): Locator {
    return this.viewTab("path-scopes")
  }

  /** Workspace Sets tab */
  get workspaceSetsTab(): Locator {
    return this.viewTab("workspace-sets")
  }

  /** Shared Workspaces tab */
  get sharedWorkspacesTab(): Locator {
    return this.viewTab("shared-workspaces")
  }

  /** Audit tab */
  get auditTab(): Locator {
    return this.viewTab("audit")
  }

  /** Approvals tab */
  get approvalsTab(): Locator {
    return this.viewTab("approvals")
  }

  /** Catalog tab */
  get catalogTab(): Locator {
    return this.viewTab("tool-catalogs")
  }

  /** Credentials tab */
  get credentialsTab(): Locator {
    return this.viewTab("credentials")
  }

  get newManagedServerButton(): Locator {
    return this.page.getByRole("button", { name: "New Managed Server" })
  }

  get saveAndDiscoverToolsButton(): Locator {
    return this.page.getByRole("button", { name: "Save and discover tools" })
  }

  // -- Helpers ---------------------------------------------------------------

  static readonly WORKFLOW_KEYS = [
    "setup",
    "access",
    "workspaces",
    "governance",
    "audit",
  ] as const

  static readonly VIEW_KEYS = [
    "profiles",
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
  ] as const

  async selectWorkflow(workflow: MCPHubWorkflowKey): Promise<void> {
    await this.workflowButton(workflow).click()
    await this.expectWorkflowSelected(workflow)
  }

  async selectView(view: MCPHubViewKey): Promise<void> {
    await this.selectWorkflow(VIEW_TO_WORKFLOW[view])
    await expect(this.viewTabControl(view)).toBeVisible()
    await this.viewTab(view).click()
  }

  async expectWorkflowSelected(workflow: MCPHubWorkflowKey): Promise<void> {
    await expect(this.workflowButton(workflow)).toHaveAttribute("aria-pressed", "true")
  }

  async expectViewSelected(view: MCPHubViewKey): Promise<void> {
    await expect(this.viewTabControl(view)).toHaveAttribute("aria-selected", "true")
  }

  /** Backward-compatible helper for older tests. Prefer selectView(). */
  async switchToTab(tab: MCPHubViewKey): Promise<void> {
    await this.selectView(tab)
  }

  setupChoice(name: "Local stdio" | "HTTP/SSE" | "Import config" | "Advanced/manual"): Locator {
    return this.page.getByRole("button", { name })
  }

  async openCreateManagedServer(): Promise<void> {
    await this.newManagedServerButton.click()
    await expect(this.page.getByText("Create Managed Server")).toBeVisible()
  }

  async expectSetupChoicesVisible(): Promise<void> {
    await expect(this.setupChoice("Local stdio")).toBeVisible()
    await expect(this.setupChoice("HTTP/SSE")).toBeVisible()
    await expect(this.setupChoice("Import config")).toBeVisible()
    await expect(this.setupChoice("Advanced/manual")).toBeVisible()
  }

  async chooseLocalStdio(): Promise<void> {
    await this.setupChoice("Local stdio").click()
    await expect(this.page.getByLabel("Server ID")).toBeVisible()
    await expect(this.page.getByRole("textbox", { name: "Command" })).toBeVisible()
  }

  async fillLocalStdioServer(input: MCPHubLocalStdioServerInput): Promise<void> {
    await this.page.getByLabel("Server ID").fill(input.serverId)
    await this.page.getByLabel("Name").fill(input.name)
    await this.page.getByRole("textbox", { name: "Command" }).fill(input.command)
    if (input.args) {
      await this.page.getByLabel("Args").fill(input.args)
    }
    if (input.cwd) {
      await this.page.getByLabel("Working Directory").fill(input.cwd)
    }
    if (input.env) {
      await this.page.getByLabel("Env vars").fill(input.env)
    }
  }

  async saveAndDiscoverTools(): Promise<void> {
    await this.saveAndDiscoverToolsButton.click()
  }

  async openServerDetails(serverName: string): Promise<void> {
    const row = this.page
      .getByText(serverName, { exact: true })
      .locator(
        "xpath=ancestor::*[contains(concat(' ', normalize-space(@class), ' '), ' ant-list-item ')][1]"
      )
    await expect(row).toBeVisible()
    await row.getByRole("button", { name: "Details" }).click()
    await expect(this.page.getByText(`${serverName} readiness details`)).toBeVisible()
  }

  async closeServerDetails(): Promise<void> {
    await this.page.getByRole("button", { name: "Close" }).last().click()
  }

  async expectNoHorizontalOverflow(): Promise<void> {
    const overflowPixels = await this.page.evaluate(() => {
      const root = document.documentElement
      return Math.ceil(root.scrollWidth) - Math.ceil(root.clientWidth)
    })
    expect(overflowPixels).toBeLessThanOrEqual(1)
  }

  // -- Interactive elements for assertAllButtonsWired() ----------------------

  async getInteractiveElements(): Promise<InteractiveElement[]> {
    return [
      {
        name: "Access workflow",
        locator: this.workflowButton("access"),
        expectation: {
          type: "state_change",
          stateCheck: async (page) =>
            page.locator('[data-testid="mcp-hub-workflow-access"]').getAttribute("aria-pressed"),
        },
      },
      {
        name: "Workspaces workflow",
        locator: this.workflowButton("workspaces"),
        expectation: {
          type: "state_change",
          stateCheck: async (page) =>
            page
              .locator('[data-testid="mcp-hub-workflow-workspaces"]')
              .getAttribute("aria-pressed"),
        },
      },
      {
        name: "Governance workflow",
        locator: this.workflowButton("governance"),
        expectation: {
          type: "state_change",
          stateCheck: async (page) =>
            page
              .locator('[data-testid="mcp-hub-workflow-governance"]')
              .getAttribute("aria-pressed"),
        },
      },
      {
        name: "Profiles view",
        locator: this.profilesTab,
        setup: async () => {
          await this.selectWorkflow("access")
        },
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/mcp\/hub\/permission-profiles/,
          method: "GET",
        },
      },
      {
        name: "Assignments view",
        locator: this.assignmentsTab,
        setup: async () => {
          await this.selectWorkflow("access")
        },
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/mcp\/hub\/policy-assignments/,
          method: "GET",
        },
      },
      {
        name: "Tool Catalog view",
        locator: this.catalogTab,
        setup: async () => {
          await this.selectWorkflow("setup")
        },
        expectation: {
          type: "api_call",
          apiPattern: /\/api\/v1\/mcp\/hub\/tool-registry/,
          method: "GET",
        },
      },
    ]
  }
}
