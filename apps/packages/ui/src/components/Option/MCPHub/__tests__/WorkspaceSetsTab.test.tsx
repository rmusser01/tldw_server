// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

const mocks = vi.hoisted(() => ({
  listWorkspaceSetObjects: vi.fn(),
  createWorkspaceSetObject: vi.fn(),
  updateWorkspaceSetObject: vi.fn(),
  deleteWorkspaceSetObject: vi.fn(),
  listSharedWorkspaces: vi.fn(),
  listWorkspaceSetMembers: vi.fn(),
  addWorkspaceSetMember: vi.fn(),
  deleteWorkspaceSetMember: vi.fn()
}))

vi.mock("@/services/tldw/mcp-hub", () => ({
  listWorkspaceSetObjects: (...args: unknown[]) => mocks.listWorkspaceSetObjects(...args),
  createWorkspaceSetObject: (...args: unknown[]) => mocks.createWorkspaceSetObject(...args),
  updateWorkspaceSetObject: (...args: unknown[]) => mocks.updateWorkspaceSetObject(...args),
  deleteWorkspaceSetObject: (...args: unknown[]) => mocks.deleteWorkspaceSetObject(...args),
  listSharedWorkspaces: (...args: unknown[]) => mocks.listSharedWorkspaces(...args),
  listWorkspaceSetMembers: (...args: unknown[]) => mocks.listWorkspaceSetMembers(...args),
  addWorkspaceSetMember: (...args: unknown[]) => mocks.addWorkspaceSetMember(...args),
  deleteWorkspaceSetMember: (...args: unknown[]) => mocks.deleteWorkspaceSetMember(...args)
}))

import { WorkspaceSetsTab } from "../WorkspaceSetsTab"

describe("WorkspaceSetsTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listWorkspaceSetObjects.mockResolvedValue([
      {
        id: 51,
        name: "Primary Workspace Set",
        description: "User-scoped reusable workspaces",
        owner_scope_type: "user",
        owner_scope_id: 7,
        is_active: true,
        readiness_summary: {
          is_multi_root_ready: false,
          warning_codes: ["multi_root_overlap_warning"],
          warning_message: "May overlap with another trusted root in multi-root assignments.",
          conflicting_workspace_ids: ["workspace-alpha", "workspace-beta"],
          conflicting_workspace_roots: ["/repo", "/repo/docs"],
          unresolved_workspace_ids: []
        }
      }
    ])
    mocks.listSharedWorkspaces.mockResolvedValue([])
    mocks.listWorkspaceSetMembers.mockResolvedValue([
      {
        workspace_set_object_id: 51,
        workspace_id: "workspace-alpha"
      }
    ])
    mocks.createWorkspaceSetObject.mockResolvedValue({
      id: 52,
      name: "Docs Workspaces",
      owner_scope_type: "user",
      owner_scope_id: 7,
      is_active: true
    })
    mocks.addWorkspaceSetMember.mockResolvedValue({
      workspace_set_object_id: 52,
      workspace_id: "workspace-docs"
    })
    vi.stubGlobal("confirm", vi.fn(() => true))
  })

  it("renders existing workspace sets and creates a new one with members", async () => {
    const user = userEvent.setup()
    render(<WorkspaceSetsTab />)

    expect(await screen.findByText("Primary Workspace Set")).toBeTruthy()
    expect(screen.getAllByText(/workspace-alpha/i).length).toBeGreaterThan(0)
    expect(await screen.findByText(/may overlap with another trusted root/i)).toBeTruthy()

    await user.click(screen.getByRole("button", { name: /new workspace set/i }))
    await user.type(screen.getByLabelText(/workspace set name/i), "Docs Workspaces")
    await user.type(screen.getByLabelText(/workspace ids/i), "workspace-docs")
    await user.click(screen.getByRole("button", { name: /save workspace set/i }))

    expect(mocks.createWorkspaceSetObject).toHaveBeenCalledWith({
      name: "Docs Workspaces",
      description: null,
      owner_scope_type: "user",
      owner_scope_id: null,
      is_active: true
    })
    expect(mocks.addWorkspaceSetMember).toHaveBeenCalledWith(52, "workspace-docs")
  })

  it("opens the existing workspace-set editor from a drill target", async () => {
    const onDrillHandled = vi.fn()
    render(
      <WorkspaceSetsTab
        drillTarget={{
          tab: "workspace-sets",
          object_kind: "workspace_set_object",
          object_id: "51",
          action: "edit",
          request_id: 7
        }}
        onDrillHandled={onDrillHandled}
      />
    )

    expect(await screen.findByDisplayValue("Primary Workspace Set")).toBeTruthy()
    expect(screen.getByDisplayValue("User-scoped reusable workspaces")).toBeTruthy()
    expect(onDrillHandled).toHaveBeenCalledWith(7)
  })

  it("shows when the focused research workspace is already in a workspace set", async () => {
    render(<WorkspaceSetsTab {...({ focusWorkspaceId: "workspace-alpha" } as any)} />)

    expect(await screen.findByText("Primary Workspace Set")).toBeTruthy()
    expect(
      await screen.findByText(
        /workspace-alpha is included in 1 MCP workspace set/i
      )
    ).toBeTruthy()
    expect(screen.getByText(/Primary Workspace Set/)).toBeTruthy()
  })

  it("shows an MCP-owned no-binding state for the focused research workspace", async () => {
    mocks.listWorkspaceSetMembers.mockResolvedValue([])

    render(<WorkspaceSetsTab {...({ focusWorkspaceId: "workspace-gamma" } as any)} />)

    expect(await screen.findByText("Primary Workspace Set")).toBeTruthy()
    expect(
      screen.getByText(/No MCP workspace set includes workspace-gamma yet/i)
    ).toBeTruthy()
    expect(
      screen.getByText(
        /Create or edit a workspace set here to make this workspace available to MCP policy assignments/i
      )
    ).toBeTruthy()
  })
})
