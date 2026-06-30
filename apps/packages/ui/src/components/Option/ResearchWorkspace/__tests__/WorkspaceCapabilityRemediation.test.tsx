import { fireEvent, render, screen, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import { getDesignSystemState } from "@/design-system"
import { WorkspaceCapabilityRemediation } from "../WorkspaceCapabilityRemediation"
import type { WorkspaceCapabilitiesResponse } from "@/services/tldw/domains/workspace-api"

const registryLabels = vi.hoisted(() => ({
  blocked: "Registry Blocked",
  degraded: "Registry Degraded"
}))

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label:
            key === "blocked"
              ? registryLabels.blocked
              : key === "degraded"
                ? registryLabels.degraded
                : state.label
        }
      }
    )
  }
})

vi.mock("../WorkspaceSandboxDiagnosticsPanel", () => ({
  WorkspaceSandboxDiagnosticsPanel: ({ workspaceId }: { workspaceId: string }) => (
    <div data-testid="workspace-sandbox-diagnostics-panel">
      Sandbox diagnostics for {workspaceId}
    </div>
  )
}))

const makeCapabilities = (
  overrides: Partial<WorkspaceCapabilitiesResponse> = {}
): WorkspaceCapabilitiesResponse => ({
  workspace_id: "workspace-1",
  workspace_kind: "research_workspace",
  workspace_profile: "research",
  access_level: "owner",
  source_summary: {
    total: 3,
    selected: 3,
    queryable: 1,
    partially_queryable: 0,
    processing: 1,
    failed: 0,
    missing: 1
  },
  workspace_services: {
    mcp: {
      state: "needs_approval",
      reason_code: "mcp_approval_required",
      management_surface: "mcp_hub"
    },
    acp: {
      state: "not_configured",
      reason_code: "acp_no_agents_configured",
      management_surface: "acp_workspace"
    },
    sandbox: {
      state: "blocked",
      reason_code: "sandbox_runtime_unavailable",
      management_surface: "sandbox_settings"
    },
    provider: {
      state: "degraded",
      reason_code: "external_provider_only",
      management_surface: "model_settings"
    }
  },
  allowed_actions: {
    ask_grounded_questions: {
      allowed: true,
      reason_code: null
    },
    run_mcp_tools: {
      allowed: false,
      reason_code: "mcp_approval_required"
    },
    use_acp_agents: {
      allowed: false,
      reason_code: "acp_no_agents_configured"
    },
    use_sandbox: {
      allowed: false,
      reason_code: "sandbox_runtime_unavailable"
    }
  },
  ...overrides
})

const renderRemediation = (capabilities: WorkspaceCapabilitiesResponse) =>
  render(
    <MemoryRouter>
      <WorkspaceCapabilityRemediation capabilities={capabilities} />
    </MemoryRouter>
  )

describe("WorkspaceCapabilityRemediation", () => {
  it("renders service capability states with user-facing remediation and route links", () => {
    renderRemediation(makeCapabilities())

    const panel = screen.getByTestId("workspace-capability-remediation")
    expect(panel).toHaveTextContent("Workspace readiness")
    expect(panel).toHaveTextContent("4 setup items")

    expect(within(panel).getByText("MCP Hub")).toBeInTheDocument()
    expect(within(panel).getByText("Needs approval")).toBeInTheDocument()
    expect(
      within(panel).getByText("Approve workspace tool use before running MCP actions.")
    ).toBeInTheDocument()
    expect(
      within(panel).getByRole("link", { name: "Open MCP Hub" })
    ).toHaveAttribute(
      "href",
      "/mcp-hub?workflow=workspaces&view=workspace-sets&workspace_id=workspace-1&source=research-workspace"
    )

    expect(within(panel).getByText("ACP Agents")).toBeInTheDocument()
    expect(
      within(panel).getByText("Configure an ACP agent before workspace agent runs.")
    ).toBeInTheDocument()
    expect(
      within(panel).getByRole("link", { name: "Open ACP Playground" })
    ).toHaveAttribute("href", "/acp-playground")

    expect(within(panel).getByText("Sandbox")).toBeInTheDocument()
    expect(within(panel).getByText(registryLabels.blocked)).toBeInTheDocument()
    expect(
      within(panel).getByText("Enable a sandbox runtime before sandboxed actions can run.")
    ).toBeInTheDocument()
    expect(
      within(panel).getByRole("link", { name: "Open Runtime Config" })
    ).toHaveAttribute("href", "/admin/runtime-config")

    expect(within(panel).getByText("Model Provider")).toBeInTheDocument()
    expect(within(panel).getByText(registryLabels.degraded)).toBeInTheDocument()
    expect(
      within(panel).getByText(
        "Only external providers are configured. Use a local provider for fully local answers."
      )
    ).toBeInTheDocument()
    expect(
      within(panel).getByRole("link", { name: "Open Model Settings" })
    ).toHaveAttribute("href", "/settings/model")
    expect(vi.mocked(getDesignSystemState)).toHaveBeenCalledWith("blocked")
    expect(vi.mocked(getDesignSystemState)).toHaveBeenCalledWith("degraded")
  })

  it("does not expose raw reason codes or workspace-playground routes", () => {
    renderRemediation(makeCapabilities())

    const panel = screen.getByTestId("workspace-capability-remediation")
    expect(panel).not.toHaveTextContent("mcp_approval_required")
    expect(panel).not.toHaveTextContent("acp_no_agents_configured")
    expect(panel).not.toHaveTextContent("sandbox_runtime_unavailable")
    expect(panel).not.toHaveTextContent("external_provider_only")

    for (const link of within(panel).getAllByRole("link")) {
      expect(link).not.toHaveAttribute(
        "href",
        expect.stringContaining("workspace-playground")
      )
    }
  })

  it("opens sandbox diagnostics from the sandbox remediation item without adding a trust bar", () => {
    renderRemediation(makeCapabilities())

    const panel = screen.getByTestId("workspace-capability-remediation")
    const diagnosticsButton = within(panel).getByRole("button", {
      name: "View Sandbox Diagnostics"
    })

    fireEvent.click(diagnosticsButton)

    expect(screen.getByTestId("workspace-sandbox-diagnostics-panel")).toHaveTextContent(
      "Sandbox diagnostics for workspace-1"
    )
    expect(panel).not.toHaveTextContent(/workspace trust/i)
  })

  it("encodes the active workspace id in the MCP Hub handoff link", () => {
    renderRemediation(
      makeCapabilities({
        workspace_id: "workspace one/with spaces"
      })
    )

    const panel = screen.getByTestId("workspace-capability-remediation")

    expect(
      within(panel).getByRole("link", { name: "Open MCP Hub" })
    ).toHaveAttribute(
      "href",
      "/mcp-hub?workflow=workspaces&view=workspace-sets&workspace_id=workspace+one%2Fwith+spaces&source=research-workspace"
    )
  })

  it("renders grounded-answer remediation when selected sources are not queryable", () => {
    renderRemediation(
      makeCapabilities({
        source_summary: {
          total: 2,
          selected: 2,
          queryable: 0,
          partially_queryable: 1,
          processing: 1,
          failed: 0,
          missing: 0
        },
        allowed_actions: {
          ask_grounded_questions: {
            allowed: false,
            reason_code: "no_queryable_sources"
          }
        }
      })
    )

    const panel = screen.getByTestId("workspace-capability-remediation")
    expect(within(panel).getByText("Grounded answers")).toBeInTheDocument()
    expect(within(panel).getAllByText(registryLabels.blocked).length).toBeGreaterThan(0)
    expect(
      within(panel).getByText(
        "Wait for extraction and indexing to finish before asking grounded questions."
      )
    ).toBeInTheDocument()
    expect(panel).not.toHaveTextContent("no_queryable_sources")
  })

  it("stays out of the composer when all tracked services are ready", () => {
    renderRemediation(
      makeCapabilities({
        workspace_services: {
          mcp: {
            state: "available",
            reason_code: null,
            management_surface: "mcp_hub"
          },
          acp: {
            state: "available",
            reason_code: null,
            management_surface: "acp_workspace"
          },
          sandbox: {
            state: "available",
            reason_code: null,
            management_surface: "sandbox_settings"
          },
          provider: {
            state: "available",
            reason_code: null,
            management_surface: "model_settings"
          }
        },
        allowed_actions: {}
      })
    )

    expect(screen.queryByTestId("workspace-capability-remediation")).not.toBeInTheDocument()
  })
})
