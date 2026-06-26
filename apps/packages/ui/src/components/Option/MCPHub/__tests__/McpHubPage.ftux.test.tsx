// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"

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
  ToolCatalogsTab: () => <div>catalog tab</div>
}))
vi.mock("../ExternalServersTab", () => ({
  ExternalServersTab: () => <div>credentials tab</div>
}))
vi.mock("../GovernanceAuditTab", () => ({
  GovernanceAuditTab: () => <div>audit tab</div>
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

const renderMcpHubPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/mcp-hub"]}>
        <McpHubPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe("McpHubPage FTUX", () => {
  beforeEach(() => {
    localStorage.clear()
    serviceMocks.getToolRegistrySummary.mockResolvedValue({
      total_tools: 0,
      modules: []
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("renders the subtitle with Model Context Protocol text", () => {
    renderMcpHubPage()
    expect(screen.getByText(/Model Context Protocol/)).toBeTruthy()
  })

  it("shows the explainer card on first visit", () => {
    renderMcpHubPage()
    expect(screen.getByTestId("mcp-hub-explainer")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-status-summary")).toBeTruthy()
  })

  it("hides the explainer card after dismissal and persists to localStorage", async () => {
    const user = userEvent.setup()
    renderMcpHubPage()

    const explainer = screen.getByTestId("mcp-hub-explainer")
    expect(explainer).toBeTruthy()

    await user.click(within(explainer).getByRole("button", { name: "Dismiss" }))

    expect(screen.queryByTestId("mcp-hub-explainer")).toBeNull()
    expect(localStorage.getItem("tldw:mcp-hub:explainer-dismissed")).toBe("true")
  })

  it("does not show the explainer card if previously dismissed", () => {
    localStorage.setItem("tldw:mcp-hub:explainer-dismissed", "true")
    renderMcpHubPage()
    expect(screen.queryByTestId("mcp-hub-explainer")).toBeNull()
  })

  it("migrates the legacy explainer dismissal key on read", () => {
    localStorage.setItem("tldw_mcp_hub_explainer_dismissed", "true")

    renderMcpHubPage()

    expect(screen.queryByTestId("mcp-hub-explainer")).toBeNull()
    expect(localStorage.getItem("tldw:mcp-hub:explainer-dismissed")).toBe("true")
    expect(localStorage.getItem("tldw_mcp_hub_explainer_dismissed")).toBeNull()
  })

  it("defaults to the Servers & Credentials view", () => {
    renderMcpHubPage()
    expect(screen.getByText("credentials tab")).toBeTruthy()
  })

  it("has data-testid attributes on shell and workflow navigation", () => {
    renderMcpHubPage()
    expect(screen.getByTestId("mcp-hub-shell")).toBeTruthy()
    expect(screen.getByTestId("mcp-hub-workflows")).toBeTruthy()
  })
})
