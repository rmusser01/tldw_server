// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"
import { render, screen, within } from "@testing-library/react"
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
  ExternalServersTab: () => <div>credentials tab</div>
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

import { McpHubPage } from "../McpHubPage"

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

const renderMcpHubPage = (initialEntry = "/mcp-hub") =>
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <McpHubPage />
      <LocationProbe />
    </MemoryRouter>
  )

describe("McpHubPage", () => {
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
})
