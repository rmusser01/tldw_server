import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspaceSandboxDiagnosticsPanel } from "../WorkspaceSandboxDiagnosticsPanel"

const { mockGetSandboxWorkspaceDiagnostics } = vi.hoisted(() => ({
  mockGetSandboxWorkspaceDiagnostics: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getSandboxWorkspaceDiagnostics: (...args: unknown[]) =>
      mockGetSandboxWorkspaceDiagnostics(...args)
  }
}))

const makeDiagnostics = (overrides: Record<string, unknown> = {}) => ({
  workspace_id: "workspace-alpha",
  source_label: "research_workspace",
  runtime: {
    state: "available",
    reason_code: null,
    message: "A sandbox runtime is available for workspace actions.",
    management_surface: "sandbox_settings"
  },
  admission: {
    state: "available",
    reason_code: null,
    message: "Sandboxed workspace actions may run.",
    management_surface: "sandbox_settings"
  },
  runs: {
    total: 0,
    limit: 10,
    has_more: false,
    items: []
  },
  links: {
    runtime_config: "/admin/runtime-config",
    admin_runs: "/admin/monitoring?focus=sandbox&workspace_id=workspace-alpha"
  },
  ...overrides
})

describe("WorkspaceSandboxDiagnosticsPanel", () => {
  beforeEach(() => {
    mockGetSandboxWorkspaceDiagnostics.mockReset()
  })

  it("loads workspace-scoped diagnostics with the active Research Workspace source label", async () => {
    mockGetSandboxWorkspaceDiagnostics.mockResolvedValue(makeDiagnostics())

    render(<WorkspaceSandboxDiagnosticsPanel workspaceId="workspace-alpha" />)

    expect(
      screen.getByText("Loading sandbox diagnostics for this workspace.")
    ).toBeInTheDocument()

    expect(
      await screen.findByText("No sandbox runs are linked to this workspace yet.")
    ).toBeInTheDocument()
    expect(mockGetSandboxWorkspaceDiagnostics).toHaveBeenCalledWith(
      "workspace-alpha",
      { sourceLabel: "research_workspace", limit: 10 }
    )
    expect(screen.queryByText(/workspace_playground/i)).toBeNull()
  })

  it("shows unavailable runtime state without raw endpoint dumps", async () => {
    mockGetSandboxWorkspaceDiagnostics.mockResolvedValue(
      makeDiagnostics({
        runtime: {
          state: "unavailable",
          reason_code: "sandbox_runtime_unavailable",
          message:
            "Sandbox runtime discovery failed. Workspace actions that require isolation are blocked until a runtime is healthy.",
          management_surface: "sandbox_settings"
        },
        admission: {
          state: "blocked",
          reason_code: "sandbox_runtime_unavailable",
          message:
            "Sandboxed workspace actions are blocked until a runtime is healthy.",
          management_surface: "sandbox_settings"
        }
      })
    )

    render(<WorkspaceSandboxDiagnosticsPanel workspaceId="workspace-alpha" />)

    expect(
      await screen.findByText(
        "Sandbox runtime discovery failed. Workspace actions that require isolation are blocked until a runtime is healthy."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("Blocked")).toBeInTheDocument()
    expect(screen.queryByText(/\/api\/v1\/sandbox/)).toBeNull()
  })

  it("distinguishes forbidden diagnostics from temporary backend failure", async () => {
    mockGetSandboxWorkspaceDiagnostics.mockRejectedValueOnce(
      new Error(
        "Request failed: 403 (GET /api/v1/sandbox/workspaces/workspace-alpha/diagnostics)"
      )
    )

    render(<WorkspaceSandboxDiagnosticsPanel workspaceId="workspace-alpha" />)

    expect(
      await screen.findByText(
        "You do not have permission to view sandbox diagnostics for this workspace."
      )
    ).toBeInTheDocument()
    expect(screen.queryByText(/\/api\/v1\/sandbox/)).toBeNull()
  })
})
