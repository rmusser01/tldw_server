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
  WorkspaceSetsTab: () => <div>workspace sets tab</div>
}))
vi.mock("../SharedWorkspacesTab", () => ({
  SharedWorkspacesTab: () => <div>shared workspaces tab</div>
}))
vi.mock("../ApprovalPoliciesTab", () => ({
  ApprovalPoliciesTab: () => <div>approvals tab</div>
}))
vi.mock("../ToolCatalogsTab", () => ({
  ToolCatalogsTab: ({
    onOpenServerSetup,
    onOpenServerCredentials,
    onOpenServerConfig
  }: {
    onOpenServerSetup?: () => void
    onOpenServerCredentials?: (serverId: string) => void
    onOpenServerConfig?: (serverId: string) => void
  }) => (
    <div>
      catalog tab
      <button type="button" onClick={onOpenServerSetup}>
        catalog add server
      </button>
      <button type="button" onClick={() => onOpenServerCredentials?.("docs-managed")}>
        catalog fix credentials
      </button>
      <button type="button" onClick={() => onOpenServerConfig?.("docs-managed")}>
        catalog open config
      </button>
    </div>
  )
}))
vi.mock("../ExternalServersTab", () => ({
  ExternalServersTab: ({
    drillTarget,
    onOpenToolCatalog
  }: {
    drillTarget?: { action?: string; object_id?: string } | null
    onOpenToolCatalog?: () => void
  }) => (
    <div>
      credentials tab
      {drillTarget?.object_id ? (
        <span>{`drill ${drillTarget.object_id} ${drillTarget.action}`}</span>
      ) : null}
      <button type="button" onClick={onOpenToolCatalog}>
        open catalog from credentials row
      </button>
    </div>
  )
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
  it("renders workflow shortcuts without implying live readiness", () => {
    renderMcpHubPage()

    expect(screen.queryByLabelText("MCP Hub status summary")).toBeNull()
    const shortcuts = screen.getByRole("navigation", {
      name: "MCP Hub workflow shortcuts"
    })
    const workflows = screen.getByTestId("mcp-hub-workflows")

    expect(shortcuts.compareDocumentPosition(workflows)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING
    )
    expect(within(shortcuts).getByText("Servers & Credentials")).toBeTruthy()
    expect(within(shortcuts).getByText("Policy Assignments")).toBeTruthy()
    expect(within(shortcuts).getByText("Approvals")).toBeTruthy()
    expect(within(shortcuts).getByText("Workspace Boundaries")).toBeTruthy()
    expect(within(shortcuts).getByText("Audit Findings")).toBeTruthy()
    expect(
      within(shortcuts).queryByText(/ready|healthy|configured|degraded/i)
    ).toBeNull()
    expect(
      within(shortcuts).queryByText(/setup workflow|access workflow|governance workflow|workspaces workflow|audit workflow/i)
    ).toBeNull()
    expect(screen.getByTestId("mcp-hub-workflow-setup")).toBeTruthy()
  })

  it("opens existing workflow views from shortcut actions", async () => {
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

  it("derives the active workflow and view from query state", () => {
    renderMcpHubPage("/mcp-hub?workflow=access&view=assignments")

    expect(screen.getByText("assignments tab")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflow-access")).toHaveAttribute(
      "aria-pressed",
      "true"
    )
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

  it("routes Tool Catalog recovery actions back to server setup and credentials", async () => {
    const user = userEvent.setup()
    renderMcpHubPage("/mcp-hub?workflow=setup&view=tool-catalogs")

    await user.click(screen.getByRole("button", { name: /catalog add server/i }))

    expect(screen.getByText("credentials tab")).toBeTruthy()
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?workflow=setup&view=credentials"
    )

    await user.click(screen.getByRole("button", { name: /open catalog from credentials row/i }))
    await user.click(screen.getByRole("button", { name: /catalog fix credentials/i }))

    expect(screen.getByText("drill docs-managed focus")).toBeTruthy()
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?workflow=setup&view=credentials"
    )

    await user.click(screen.getByRole("button", { name: /open catalog from credentials row/i }))
    await user.click(screen.getByRole("button", { name: /catalog open config/i }))

    expect(screen.getByText("drill docs-managed edit")).toBeTruthy()
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?workflow=setup&view=credentials"
    )
  })
})
