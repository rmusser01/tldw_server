import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  WorkspaceApiResponse,
  WorkspaceContextResponse
} from "@/services/tldw/domains/workspace-api"
import { WORKSPACE_STORAGE_KEY } from "@/store/research-workspace-legacy-storage-inventory"

const apiMocks = vi.hoisted(() => ({
  listWorkspaces: vi.fn(),
  getWorkspaceContext: vi.fn(),
  upsertWorkspace: vi.fn(),
  patchWorkspace: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMocks
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => routerMocks.navigate
  }
})

import { WorkspacesManagerPage } from "../WorkspacesManagerPage"

const workspace = (
  overrides: Partial<WorkspaceApiResponse> = {}
): WorkspaceApiResponse => ({
  id: "ws-research",
  name: "Climate Review",
  archived: false,
  study_materials_policy: "workspace",
  workspace_profile: "research",
  deleted: false,
  banner_title: null,
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null,
  created_at: "2026-06-04T08:00:00Z",
  last_modified: "2026-06-04T09:00:00Z",
  version: 4,
  ...overrides
})

const contextFor = (
  item: WorkspaceApiResponse,
  overrides: Partial<WorkspaceContextResponse> = {}
): WorkspaceContextResponse => ({
  workspace_id: item.id,
  workspace_profile: item.workspace_profile,
  workspace_kind:
    item.workspace_profile === "project"
      ? "project_workspace"
      : "research_workspace",
  schema_version: 2,
  generated_at: "2026-06-04T09:01:00Z",
  workspace: item,
  attention_state: item.archived ? "archived" : "ready",
  resolution: { status: "complete", partial_errors: [] },
  project_root: {
    state: "not_configured",
    root_id: null,
    backend: null,
    display_name: null,
    path_hint: null,
    git_state: null,
    file_inventory_state: "not_started",
    file_inventory: {
      state: "not_started",
      indexed_file_count: null,
      total_file_count: null,
      updated_at: null,
      available: false
    },
    indexing_state: null,
    sandbox_mount_state: null,
    mcp_trust_state: null
  },
  sources: {
    items: [],
    summary: {
      total: 0,
      selected: 0,
      queryable: 0,
      partially_queryable: 0,
      processing: 0,
      failed: 0,
      missing: 0
    }
  },
  capabilities: {} as WorkspaceContextResponse["capabilities"],
  services: {},
  allowed_actions: {},
  active_jobs: [],
  active_operations: [],
  partial_errors: [],
  ...overrides
})

const renderManager = () =>
  render(
    <MemoryRouter initialEntries={["/workspaces"]}>
      <WorkspacesManagerPage />
    </MemoryRouter>
  )

describe("WorkspacesManagerPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.stubGlobal("crypto", {
      randomUUID: () => "test-workspace-id"
    })
    apiMocks.listWorkspaces.mockResolvedValue({ items: [], total: 0 })
    apiMocks.getWorkspaceContext.mockResolvedValue(
      contextFor(workspace({ id: "fallback" }))
    )
    apiMocks.upsertWorkspace.mockImplementation(
      async (id: string, payload: Partial<WorkspaceApiResponse>) =>
        workspace({
          id,
          name: String(payload.name ?? id),
          workspace_profile: payload.workspace_profile ?? "research"
        })
    )
    apiMocks.patchWorkspace.mockImplementation(
      async (_id: string, payload: Partial<WorkspaceApiResponse>) =>
        workspace(payload)
    )
  })

  it("shows loading, unavailable, and server-backed empty states", async () => {
    apiMocks.listWorkspaces.mockReturnValueOnce(new Promise(() => undefined))
    const { unmount } = renderManager()
    expect(screen.getByText("Loading Workspaces")).toBeInTheDocument()
    unmount()

    apiMocks.listWorkspaces.mockRejectedValueOnce(new Error("offline"))
    const { unmount: unmountUnavailable } = renderManager()
    expect(await screen.findByText("Workspaces are unavailable")).toBeVisible()
    expect(
      screen.getByText("Reconnect to your tldw server to manage Workspaces.")
    ).toBeVisible()
    unmountUnavailable()

    apiMocks.listWorkspaces.mockResolvedValueOnce({ items: [], total: 0 })
    renderManager()
    expect(await screen.findByText("No server-backed Workspaces yet")).toBeVisible()
    expect(
      screen.queryByText(/local-only Research Workspace/i)
    ).not.toBeInTheDocument()
  })

  it("renders server-backed rows with search, profile, archived, and attention filters", async () => {
    const research = workspace({
      id: "ws-research",
      name: "Climate Review",
      workspace_profile: "research"
    })
    const project = workspace({
      id: "ws-project",
      name: "Policy Website",
      workspace_profile: "project",
      last_modified: "2026-06-04T10:00:00Z"
    })
    const archived = workspace({
      id: "ws-archive",
      name: "Archived Notes",
      archived: true
    })
    apiMocks.listWorkspaces.mockResolvedValueOnce({
      items: [research, project, archived],
      total: 3
    })
    apiMocks.getWorkspaceContext.mockImplementation(async (id: string) => {
      if (id === "ws-project") {
        return contextFor(project, {
          attention_state: "needs_attention",
          project_root: {
            ...contextFor(project).project_root,
            state: "failed",
            backend: "sandbox_volume",
            display_name: "Project sandbox",
            file_inventory: {
              state: "failed",
              indexed_file_count: 8,
              total_file_count: 12,
              updated_at: "2026-06-04T09:55:00Z",
              available: false
            }
          },
          sources: {
            ...contextFor(project).sources,
            summary: {
              total: 3,
              selected: 2,
              queryable: 1,
              partially_queryable: 1,
              processing: 0,
              failed: 1,
              missing: 0
            }
          }
        })
      }
      if (id === "ws-archive") return contextFor(archived)
      return contextFor(research)
    })

    const user = userEvent.setup()
    renderManager()

    expect(await screen.findByText("Climate Review")).toBeVisible()
    expect(screen.getByText("Policy Website")).toBeVisible()
    expect(screen.queryByText("Archived Notes")).not.toBeInTheDocument()
    expect(screen.getByText("Sources 3")).toBeVisible()
    expect(screen.getByText("needs attention")).toBeVisible()
    expect(apiMocks.getWorkspaceContext).toHaveBeenCalledWith("ws-project")

    await user.type(screen.getByRole("searchbox", { name: "Search Workspaces" }), "policy")
    expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    expect(screen.getByText("Policy Website")).toBeVisible()

    await user.clear(screen.getByRole("searchbox", { name: "Search Workspaces" }))
    await user.click(screen.getByRole("button", { name: "Project" }))
    expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    expect(screen.getByText("Policy Website")).toBeVisible()

    await user.click(screen.getByRole("button", { name: "Needs attention" }))
    expect(screen.getByText("Policy Website")).toBeVisible()

    await user.click(screen.getByRole("button", { name: "All" }))
    await user.click(screen.getByRole("button", { name: "Needs attention" }))
    await user.click(screen.getByRole("checkbox", { name: "Show archived" }))
    expect(screen.getByText("Archived Notes")).toBeVisible()
  })

  it("creates Research Workspaces and Project Workspace shells through the server upsert contract", async () => {
    const user = userEvent.setup()
    renderManager()

    await screen.findByText("No server-backed Workspaces yet")
    await user.click(
      screen.getAllByRole("button", { name: "New Research Workspace" })[0]
    )
    await user.type(screen.getByLabelText("Workspace name"), "Migration Notes")
    await user.click(screen.getByRole("button", { name: "Create Workspace" }))

    await waitFor(() => {
      expect(apiMocks.upsertWorkspace).toHaveBeenCalledWith("test-workspace-id", {
        name: "Migration Notes",
        study_materials_policy: "workspace",
        workspace_profile: "research"
      })
    })

    await user.click(screen.getByRole("button", { name: "New Project Workspace" }))
    await user.type(screen.getByLabelText("Workspace name"), "Build Site")
    await user.click(screen.getByRole("button", { name: "Create Workspace" }))

    await waitFor(() => {
      expect(apiMocks.upsertWorkspace).toHaveBeenLastCalledWith(
        "test-workspace-id",
        {
          name: "Build Site",
          study_materials_policy: "workspace",
          workspace_profile: "project"
        }
      )
    })
  })

  it("uses getRandomValues fallback for workspace creation IDs", async () => {
    vi.stubGlobal("crypto", {
      getRandomValues: (values: Uint32Array) => {
        values[0] = 36
        values[1] = 1296
        return values
      }
    })
    const user = userEvent.setup()
    renderManager()

    await screen.findByText("No server-backed Workspaces yet")
    await user.click(
      screen.getAllByRole("button", { name: "New Research Workspace" })[0]
    )
    await user.type(screen.getByLabelText("Workspace name"), "Fallback ID")
    await user.click(screen.getByRole("button", { name: "Create Workspace" }))

    await waitFor(() => {
      expect(apiMocks.upsertWorkspace).toHaveBeenCalledWith(
        "workspace-10-100",
        expect.objectContaining({
          name: "Fallback ID",
          workspace_profile: "research"
        })
      )
    })
  })

  it("separates local-only Research Workspace entries from server-backed rows", async () => {
    window.localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      JSON.stringify({
        schema: "workspace_split_v1",
        version: 12,
        state: {
          workspaceId: "local-only",
          savedWorkspaces: [
            {
              id: "local-only",
              name: "Local Only Notes",
              sourceCount: 2
            }
          ],
          archivedWorkspaces: [],
          workspaceIds: ["local-only"],
          workspaceSnapshots: {},
          workspaceChatSessions: {}
        }
      })
    )
    const research = workspace({
      id: "ws-research",
      name: "Server Research"
    })
    apiMocks.listWorkspaces.mockResolvedValueOnce({
      items: [research],
      total: 1
    })
    apiMocks.getWorkspaceContext.mockResolvedValueOnce(contextFor(research))

    renderManager()

    expect(await screen.findByText("Local Research Workspaces")).toBeVisible()
    expect(screen.getByText("Local Only Notes")).toBeVisible()
    expect(screen.getByText("Server Research")).toBeVisible()
  })

  it("shows local-only entries even when no server-backed Workspaces exist yet", async () => {
    window.localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      JSON.stringify({
        schema: "workspace_split_v1",
        version: 12,
        state: {
          workspaceId: "local-first",
          savedWorkspaces: [
            {
              id: "local-first",
              name: "First Local Workspace",
              sourceCount: 1
            }
          ],
          archivedWorkspaces: [],
          workspaceIds: ["local-first"],
          workspaceSnapshots: {},
          workspaceChatSessions: {}
        }
      })
    )
    apiMocks.listWorkspaces.mockResolvedValueOnce({ items: [], total: 0 })

    renderManager()

    expect(await screen.findByText("Local Research Workspaces")).toBeVisible()
    expect(screen.getByText("First Local Workspace")).toBeVisible()
    expect(screen.getByText("No server-backed Workspaces yet")).toBeVisible()
  })

  it("edits metadata, archives, unarchives, and opens without hard-delete controls", async () => {
    const research = workspace()
    const archived = workspace({
      id: "ws-archive",
      name: "Archived Notes",
      archived: true,
      version: 2
    })
    apiMocks.listWorkspaces.mockResolvedValueOnce({
      items: [research, archived],
      total: 2
    })
    apiMocks.getWorkspaceContext.mockImplementation(async (id: string) =>
      id === "ws-archive" ? contextFor(archived) : contextFor(research)
    )

    const user = userEvent.setup()
    renderManager()

    const row = await screen.findByRole("row", { name: /Climate Review/i })
    expect(within(row).queryByRole("button", { name: /delete/i })).toBeNull()
    expect(screen.queryByText(/MCP policy/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/ACP launch/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/root setup/i)).not.toBeInTheDocument()

    await user.click(within(row).getByRole("button", { name: "Edit Climate Review" }))
    await user.clear(screen.getByLabelText("Workspace name"))
    await user.type(screen.getByLabelText("Workspace name"), "Climate Evidence")
    await user.click(screen.getByRole("button", { name: "Save metadata" }))

    await waitFor(() => {
      expect(apiMocks.patchWorkspace).toHaveBeenCalledWith("ws-research", {
        name: "Climate Evidence",
        version: 4
      })
    })

    await user.click(within(row).getByRole("button", { name: "Open Climate Evidence" }))
    expect(routerMocks.navigate).toHaveBeenCalledWith(
      "/research-workspace?source_workspace_id=ws-research"
    )

    await user.click(within(row).getByRole("button", { name: "Archive Climate Evidence" }))
    await waitFor(() => {
      expect(apiMocks.patchWorkspace).toHaveBeenCalledWith("ws-research", {
        archived: true,
        version: 4
      })
    })

    await user.click(screen.getByRole("checkbox", { name: "Show archived" }))
    const archivedRow = screen.getByRole("row", { name: /Archived Notes/i })
    await user.click(
      within(archivedRow).getByRole("button", { name: "Unarchive Archived Notes" })
    )

    await waitFor(() => {
      expect(apiMocks.patchWorkspace).toHaveBeenCalledWith("ws-archive", {
        archived: false,
        version: 2
      })
    })
  })
})
