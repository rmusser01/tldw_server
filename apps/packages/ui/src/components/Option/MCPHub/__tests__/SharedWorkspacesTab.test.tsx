// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  listSharedWorkspaces: vi.fn(),
  createSharedWorkspace: vi.fn(),
  updateSharedWorkspace: vi.fn(),
  deleteSharedWorkspace: vi.fn()
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  listSharedWorkspaces: (...args: unknown[]) => mocks.listSharedWorkspaces(...args),
  createSharedWorkspace: (...args: unknown[]) => mocks.createSharedWorkspace(...args),
  updateSharedWorkspace: (...args: unknown[]) => mocks.updateSharedWorkspace(...args),
  deleteSharedWorkspace: (...args: unknown[]) => mocks.deleteSharedWorkspace(...args)
}))

import { SharedWorkspacesTab } from "../SharedWorkspacesTab"

describe("SharedWorkspacesTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listSharedWorkspaces.mockResolvedValue([
      {
        id: 71,
        workspace_id: "shared-docs",
        display_name: "Shared Docs",
        absolute_root: "/srv/shared/docs",
        owner_scope_type: "team",
        owner_scope_id: 21,
        is_active: true,
        readiness_summary: {
          is_multi_root_ready: false,
          warning_codes: ["multi_root_overlap_warning"],
          warning_message: "May conflict with other visible shared roots in multi-root assignments.",
          conflicting_workspace_ids: ["shared-docs", "shared-docs-nested"],
          conflicting_workspace_roots: ["/srv/shared/docs", "/srv/shared/docs/archive"],
          unresolved_workspace_ids: []
        }
      }
    ])
    mocks.createSharedWorkspace.mockResolvedValue({
      id: 72,
      workspace_id: "shared-research",
      display_name: "Shared Research",
      absolute_root: "/srv/shared/research",
      owner_scope_type: "org",
      owner_scope_id: 9,
      is_active: true
    })
  })

  it("renders existing shared workspaces and creates a new one", async () => {
    const user = userEvent.setup()
    render(<SharedWorkspacesTab />)

    expect(
      screen.getByText(/Shared Workspaces are MCP Hub path-trust records/i)
    ).toBeTruthy()
    expect(
      screen.getByRole("link", { name: /manage canonical Workspaces/i })
    ).toHaveAttribute("href", "#/workspaces")
    expect(await screen.findByText("Shared Docs")).toBeTruthy()
    expect(screen.getByText("/srv/shared/docs")).toBeTruthy()
    expect(await screen.findByText(/may conflict with other visible shared roots/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /new shared workspace/i }))
    await user.type(screen.getByLabelText(/shared workspace id/i), "shared-research")
    await user.type(screen.getByLabelText(/shared workspace display name/i), "Shared Research")
    await user.type(screen.getByLabelText(/shared workspace absolute root/i), "/srv/shared/research")
    await user.selectOptions(screen.getByLabelText("Shared Workspace Owner Scope"), "org")
    await user.clear(screen.getByLabelText("Shared Workspace Owner Scope Id"))
    await user.type(screen.getByLabelText("Shared Workspace Owner Scope Id"), "9")
    await user.click(screen.getByRole("button", { name: /save shared workspace/i }))

    expect(mocks.createSharedWorkspace).toHaveBeenCalledWith({
      workspace_id: "shared-research",
      display_name: "Shared Research",
      absolute_root: "/srv/shared/research",
      owner_scope_type: "org",
      owner_scope_id: 9,
      is_active: true
    })
  })

  it("opens the existing shared-workspace editor from a drill target", async () => {
    const onDrillHandled = vi.fn()
    render(
      <SharedWorkspacesTab
        drillTarget={{
          tab: "shared-workspaces",
          object_kind: "shared_workspace",
          object_id: "71",
          action: "edit",
          request_id: 9
        }}
        onDrillHandled={onDrillHandled}
      />
    )

    expect(await screen.findByDisplayValue("shared-docs")).toBeTruthy()
    expect(screen.getByDisplayValue("Shared Docs")).toBeTruthy()
    expect(onDrillHandled).toHaveBeenCalledWith(9)
  })
})
