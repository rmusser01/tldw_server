// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, useLocation } from "react-router-dom"
import type { ReactNode } from "react"

import type { McpToolsValidateResponse } from "@/types/setup-onboarding"

const { getMcpToolsRecoveryStatus, validateMcpToolsRecovery } = vi.hoisted(() => ({
  getMcpToolsRecoveryStatus: vi.fn(),
  validateMcpToolsRecovery: vi.fn()
}))

vi.mock("@/services/tldw/domains/setup-onboarding", () => ({
  setupOnboardingMethods: {
    getMcpToolsRecoveryStatus,
    validateMcpToolsRecovery
  }
}))

vi.mock("antd", () => ({
  Button: ({
    children,
    onClick,
    "aria-pressed": ariaPressed,
    "data-testid": dataTestId
  }: {
    children: ReactNode
    onClick?: () => void
    "aria-pressed"?: boolean
    "data-testid"?: string
  }) => (
    <button type="button" aria-pressed={ariaPressed} data-testid={dataTestId} onClick={onClick}>
      {children}
    </button>
  ),
  Tabs: ({
    activeKey,
    items,
    onChange
  }: {
    activeKey: string
    items: Array<{ key: string; label: ReactNode; children: ReactNode }>
    onChange?: (key: string) => void
  }) => (
    <div>
      {items.map((item) => (
        <button type="button" key={item.key} onClick={() => onChange?.(item.key)}>
          {item.label}
        </button>
      ))}
      <div>{items.find((item) => item.key === activeKey)?.children}</div>
    </div>
  ),
  Typography: {
    Title: ({ children }: { children: ReactNode }) => <h1>{children}</h1>,
    Text: ({ children }: { children: ReactNode }) => <span>{children}</span>
  }
}))

vi.mock("../PermissionProfilesTab", () => ({
  PermissionProfilesTab: ({
    drillTarget
  }: {
    drillTarget?: { tab?: string; object_kind?: string; object_id?: string; action?: string } | null
  }) => (
    <div>
      profiles tab
      {drillTarget ? (
        <span>{`profile drill ${drillTarget.tab} ${drillTarget.object_kind} ${drillTarget.object_id} ${drillTarget.action}`}</span>
      ) : null}
    </div>
  )
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

import { McpHubPage } from "../McpHubPage"

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="location-probe">{`${location.pathname}${location.search}`}</div>
}

const statusResponse = (
  overrides: Partial<McpToolsValidateResponse> = {}
): McpToolsValidateResponse => ({
  status: "saved",
  validation_state: "not_run",
  profile_id: 7,
  assignment_id: 9,
  catalog_version: "2026-07-04.v1",
  selected_pack_ids: ["research"],
  selected_addon_ids: [],
  effective_tool_count: 3,
  ...overrides
})

const renderMcpHubPage = (initialEntry = "/mcp-hub?source=first-run") =>
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <McpHubPage />
      <LocationProbe />
    </MemoryRouter>
  )

describe("McpHubPage first-run MCP tools status", () => {
  beforeEach(() => {
    getMcpToolsRecoveryStatus.mockReset()
    validateMcpToolsRecovery.mockReset()
  })

  it.each(["built_in_passed", "external_tool_passed"] as const)(
    "shows validated setup status for %s",
    async (validation_state) => {
      getMcpToolsRecoveryStatus.mockResolvedValueOnce(statusResponse({ validation_state }))

      renderMcpHubPage()

      expect(await screen.findByText("Validated during setup")).toBeInTheDocument()
    }
  )

  it("shows not validated setup status for not_run", async () => {
    getMcpToolsRecoveryStatus.mockResolvedValueOnce(
      statusResponse({ validation_state: "not_run" })
    )

    renderMcpHubPage()

    expect(await screen.findByText("Not validated during setup")).toBeInTheDocument()
  })

  it("shows validation failed status for failed", async () => {
    getMcpToolsRecoveryStatus.mockResolvedValueOnce(
      statusResponse({ validation_state: "failed" })
    )

    renderMcpHubPage()

    expect(await screen.findByText("Validation failed")).toBeInTheDocument()
  })

  it("shows external discovery incomplete status", async () => {
    getMcpToolsRecoveryStatus.mockResolvedValueOnce(
      statusResponse({ validation_state: "external_discovery_incomplete" })
    )

    renderMcpHubPage()

    expect(await screen.findByText("External discovery incomplete")).toBeInTheDocument()
  })

  it("shows profile manually changed when the backend reports the generated profile changed", async () => {
    const user = userEvent.setup()
    getMcpToolsRecoveryStatus.mockResolvedValueOnce(
      statusResponse({ status: "profile_manually_changed", validation_state: "not_run" })
    )

    renderMcpHubPage()

    expect(await screen.findByText("Profile manually changed")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Review profile" }))

    expect(screen.getByText("profiles tab")).toBeInTheDocument()
    expect(
      screen.getByText("profile drill profiles permission_profile 7 edit")
    ).toBeInTheDocument()
    expect(screen.getByTestId("location-probe")).toHaveTextContent(
      "/mcp-hub?source=first-run&workflow=access&view=profiles&profile_id=7"
    )
  })

  it.each(["skipped", "failed", "not_run", "external_discovery_incomplete"] as const)(
    "runs admin validation for recoverable %s status",
    async (validation_state) => {
      const user = userEvent.setup()
      getMcpToolsRecoveryStatus.mockResolvedValueOnce(statusResponse({ validation_state }))
      validateMcpToolsRecovery.mockResolvedValueOnce(
        statusResponse({ status: "validated", validation_state: "built_in_passed" })
      )

      renderMcpHubPage()

      await user.click(await screen.findByRole("button", { name: "Run validation" }))

      expect(validateMcpToolsRecovery).toHaveBeenCalledWith()
      await waitFor(() =>
        expect(screen.getByText("Validated during setup")).toBeInTheDocument()
      )
    }
  )

  it("shows a visible error when recovery validation fails", async () => {
    const user = userEvent.setup()
    getMcpToolsRecoveryStatus.mockResolvedValueOnce(
      statusResponse({ validation_state: "not_run" })
    )
    validateMcpToolsRecovery.mockRejectedValueOnce(new Error("validation unavailable"))

    renderMcpHubPage()

    await user.click(await screen.findByRole("button", { name: "Run validation" }))

    expect(
      await screen.findByText("Could not run first-run MCP tools validation.")
    ).toBeInTheDocument()
  })

  it("does not block MCP Hub management when status loading fails", async () => {
    getMcpToolsRecoveryStatus.mockRejectedValueOnce(new Error("status unavailable"))

    renderMcpHubPage()

    expect(await screen.findByText("credentials tab")).toBeInTheDocument()
    expect(screen.queryByText("First-run MCP tools")).not.toBeInTheDocument()
  })
})
