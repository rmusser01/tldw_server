// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, useLocation } from "react-router-dom"

vi.mock("../PermissionProfilesTab", () => ({
  PermissionProfilesTab: () => <div>profiles tab</div>
}))
vi.mock("../PolicyAssignmentsTab", () => ({
  PolicyAssignmentsTab: () => <div>assignments tab</div>
}))
vi.mock("../PathScopesTab", () => ({
  PathScopesTab: () => <div>path scopes tab</div>
}))
vi.mock("../WorkspaceSetsTab", () => ({
  WorkspaceSetsTab: ({
    focusWorkspaceId
  }: {
    focusWorkspaceId?: string | null
  }) => (
    <div>
      workspace sets tab
      <span data-testid="workspace-sets-focus">
        {focusWorkspaceId ?? "no workspace context"}
      </span>
    </div>
  )
}))
vi.mock("../SharedWorkspacesTab", () => ({
  SharedWorkspacesTab: () => <div>shared workspaces tab</div>
}))
vi.mock("../ApprovalPoliciesTab", () => ({
  ApprovalPoliciesTab: () => <div>approvals tab</div>
}))
vi.mock("../ToolCatalogsTab", () => ({
  ToolCatalogsTab: () => <div>catalog tab</div>
}))
vi.mock("../ExternalServersTab", () => ({
  ExternalServersTab: ({ onOpenToolCatalog }: { onOpenToolCatalog?: () => void }) => (
    <div>
      credentials tab
      <button type="button" onClick={onOpenToolCatalog}>
        open catalog from credentials row
      </button>
    </div>
  )
}))
vi.mock("../DeploymentDiagnosticsPanel", () => ({
  DeploymentDiagnosticsPanel: () => <div>deployment diagnostics</div>
}))
vi.mock("../GovernanceAuditTab", () => ({
  GovernanceAuditTab: ({ onOpen }: { onOpen?: (target: { tab: string; object_kind: string; object_id: string }) => void }) => (
    <button
      type="button"
      onClick={() =>
        onOpen?.({
          tab: "assignments",
          object_kind: "policy_assignment",
          object_id: "11"
        })
      }
    >
      open assignment from audit
    </button>
  )
}))
vi.mock("../GovernancePacksTab", () => ({
  GovernancePacksTab: () => <div>governance packs tab</div>
}))
vi.mock("../CapabilityMappingsTab", () => ({
  CapabilityMappingsTab: () => <div>capability mappings tab</div>
}))

const serviceMocks = vi.hoisted(() => ({
  getToolRegistrySummary: vi.fn()
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  getToolRegistrySummary: serviceMocks.getToolRegistrySummary
}))

import { McpHubPage } from "../McpHubPage"

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

const renderMcpHubPage = (initialEntry = "/mcp-hub") => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <McpHubPage />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe("McpHubPage", () => {
  beforeEach(() => {
    serviceMocks.getToolRegistrySummary.mockResolvedValue({
      total_tools: 0,
      modules: []
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("renders a shared recovery state when MCP Hub is unavailable", async () => {
    const user = userEvent.setup()
    serviceMocks.getToolRegistrySummary.mockRejectedValueOnce(
      Object.assign(new Error("Request failed: 404"), {
        response: { status: 404 }
      })
    )

    renderMcpHubPage()

    const recovery = await screen.findByTestId("mcp-hub-capability-recovery")

    expect(recovery).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(
      within(recovery).getByRole("heading", {
        name: "MCP Hub is unavailable on this server"
      })
    ).toBeTruthy()
    expect(
      within(recovery).getByText(
        "The connected server does not advertise MCP Hub management."
      )
    ).toBeTruthy()
    const diagnostics = within(recovery).getByLabelText("Diagnostics")
    expect(diagnostics).toHaveTextContent("[server-endpoint]")
    expect(diagnostics).toHaveTextContent("404")
    expect(diagnostics).toHaveTextContent("Request failed: 404")
    expect(diagnostics).not.toHaveTextContent(
      "/api/v1/mcp/hub/tool-registry/summary"
    )

    await user.click(
      within(recovery).getByRole("button", { name: "Try again" })
    )

    await waitFor(() => {
      expect(serviceMocks.getToolRegistrySummary).toHaveBeenCalledTimes(2)
    })
  })

  it("renders a compact status summary before workflow detail", () => {
    renderMcpHubPage()

    const summary = screen.getByTestId("mcp-hub-status-summary")
    const workflows = screen.getByTestId("mcp-hub-workflows")

    expect(summary.compareDocumentPosition(workflows)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING
    )
    expect(within(summary).getByText("Servers & Credentials")).toBeTruthy()
    expect(within(summary).getByText("Policy Assignments")).toBeTruthy()
    expect(within(summary).getByText("Approvals")).toBeTruthy()
    expect(within(summary).getByText("Workspace Boundaries")).toBeTruthy()
    expect(within(summary).getByText("Audit Findings")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-setup")).toBeTruthy()
  })

  it("opens existing workflow views from status summary actions", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    await user.click(screen.getByRole("button", { name: "Open Policy Assignments" }))

    expect(screen.getByText("assignments tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-access")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?workflow=access&view=assignments"
    )
  })

  it("renders workflow navigation with the current MCP Hub child views grouped inside it", async () => {
    renderMcpHubPage()

    expect(screen.getByTestId("mcp-hub-workflows")).toBeTruthy()
    expect(screen.getByText("Setup")).toBeTruthy()
    expect(screen.getByText("Access")).toBeTruthy()
    expect(screen.getByText("Workspaces")).toBeTruthy()
    expect(screen.getByText("Governance")).toBeTruthy()
    expect(screen.getByText("Audit")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-tab-credentials")).toHaveTextContent(
      "Servers & Credentials"
    )
    expect(screen.getByTestId("mcp-hub-tab-tool-catalogs")).toHaveTextContent(
      "Tool Catalog"
    )
  })

  it("defaults to Setup / Servers & Credentials", () => {
    renderMcpHubPage()

    expect(screen.getByText("credentials tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-setup")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
  })

  it("shows deployment diagnostics in the Setup workflow", () => {
    renderMcpHubPage()

    expect(screen.getByText("deployment diagnostics")).toBeTruthy()
  })

  it("derives the active workflow and view from query state", () => {
    renderMcpHubPage("/mcp-hub?workflow=access&view=assignments")

    expect(screen.getByText("assignments tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-access")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
  })

  it("passes research workspace query context into Workspace Sets", () => {
    renderMcpHubPage(
      "/mcp-hub?workflow=setup&view=workspace-sets&workspace_id=rw-1&source=research-workspace"
    )

    expect(screen.getByText("workspace sets tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-workspaces")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByTestId("workspace-sets-focus")).toHaveTextContent("rw-1")
  })

  it("updates query state when selecting a workflow", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    await user.click(screen.getByTestId("mcp-hub-workflow-workspaces"))

    expect(screen.getByText("path scopes tab")).toBeTruthy()
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?workflow=workspaces&view=path-scopes"
    )
  })

  it("keeps expert child views available inside their workflows", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    await user.click(screen.getByTestId("mcp-hub-workflow-access"))

    expect(screen.getByText("Profiles")).toBeTruthy()
    expect(screen.getByText("Assignments")).toBeTruthy()

    await user.click(screen.getByText("Assignments"))

    expect(screen.getByText("assignments tab")).toBeTruthy()
  })

  it("keeps governance child views available inside the Governance workflow", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    await user.click(screen.getByTestId("mcp-hub-workflow-governance"))

    expect(screen.getByText("Capability Mappings")).toBeTruthy()
    expect(screen.getByText("Governance Packs")).toBeTruthy()
  })

  it("opens the requested workflow child view from the audit view", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    await user.click(screen.getByTestId("mcp-hub-workflow-audit"))
    await user.click(screen.getByRole("button", { name: /open assignment from audit/i }))

    expect(screen.getByText("assignments tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-access")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
  })

  it("routes from external server row actions to the Tool Catalog view", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    await user.click(screen.getByRole("button", { name: /open catalog from credentials row/i }))

    expect(screen.getByText("catalog tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-setup")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?workflow=setup&view=tool-catalogs"
    )
  })
})
