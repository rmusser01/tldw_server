import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  WorkspaceApiResponse,
  WorkspaceContextResponse,
  WorkspaceRootsResponse
} from "@/services/tldw/domains/workspace-api"
import { WORKSPACE_STORAGE_KEY } from "@/store/research-workspace-legacy-storage-inventory"

const apiMocks = vi.hoisted(() => ({
  listWorkspaces: vi.fn(),
  getWorkspaceContext: vi.fn(),
  upsertWorkspace: vi.fn(),
  patchWorkspace: vi.fn(),
  attachWorkspacePrimaryRoot: vi.fn(),
  getWorkspaceOperation: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

const ownedMocks = vi.hoisted(() => ({
  directory: vi.fn(),
  lifecycle: vi.fn(),
  setArchived: vi.fn()
}))

vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceDirectoryContext: ownedMocks.directory,
  createOwnedWorkspaceLifecycleContext: ownedMocks.lifecycle
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMocks
}))

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
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
    <MemoryRouter>
      <WorkspacesManagerPage />
    </MemoryRouter>
  )

const scope = {
  serverBase: "https://server.example",
  principalId: "17",
  organizationId: "3"
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}

describe("WorkspacesManagerPage", () => {
  beforeEach(() => {
    vi.resetAllMocks()
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
    ownedMocks.directory.mockImplementation(async () => ({
      scope,
      list: apiMocks.listWorkspaces,
      getContext: apiMocks.getWorkspaceContext
    }))
    ownedMocks.lifecycle.mockImplementation(async (id: string) => ({
      scope,
      get: async () => workspace({ id }),
      setArchived: (archived: boolean, version: number) =>
        ownedMocks.setArchived(id, archived, version)
    }))
    ownedMocks.setArchived.mockImplementation(
      async (id: string, archived: boolean, version: number) =>
        workspace({ id, archived, version: version + 1 })
    )
  })

  it("shows loading, unavailable, and server-backed empty states", async () => {
    apiMocks.listWorkspaces.mockReturnValueOnce(new Promise(() => undefined))
    const { unmount } = renderManager()
    expect(screen.getByText("Loading Workspaces")).toBeInTheDocument()
    await waitFor(() =>
      expect(apiMocks.listWorkspaces).toHaveBeenCalledTimes(1)
    )
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
    expect(
      await screen.findByText("No server-backed Workspaces yet")
    ).toBeVisible()
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

    await user.type(
      screen.getByRole("searchbox", { name: "Search Workspaces" }),
      "policy"
    )
    expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    expect(screen.getByText("Policy Website")).toBeVisible()

    await user.clear(
      screen.getByRole("searchbox", { name: "Search Workspaces" })
    )
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
      expect(apiMocks.upsertWorkspace).toHaveBeenCalledWith(
        "test-workspace-id",
        {
          name: "Migration Notes",
          study_materials_policy: "workspace",
          workspace_profile: "research"
        }
      )
    })

    await user.click(
      screen.getByRole("button", { name: "New Project Workspace" })
    )
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

    await user.click(
      within(row).getByRole("button", { name: "Edit Climate Review" })
    )
    await user.clear(screen.getByLabelText("Workspace name"))
    await user.type(screen.getByLabelText("Workspace name"), "Climate Evidence")
    await user.click(screen.getByRole("button", { name: "Save metadata" }))

    await waitFor(() => {
      expect(apiMocks.patchWorkspace).toHaveBeenCalledWith("ws-research", {
        name: "Climate Evidence",
        version: 4
      })
    })

    await user.click(
      within(row).getByRole("button", { name: "Open Climate Evidence" })
    )
    expect(routerMocks.navigate).toHaveBeenCalledWith(
      "/research-workspace?source_workspace_id=ws-research"
    )

    await user.click(
      within(row).getByRole("button", { name: "Archive Climate Evidence" })
    )
    await waitFor(() => {
      expect(ownedMocks.setArchived).toHaveBeenCalledWith(
        "ws-research",
        true,
        4
      )
    })

    await user.click(screen.getByRole("checkbox", { name: "Show archived" }))
    const archivedRow = screen.getByRole("row", { name: /Archived Notes/i })
    await user.click(
      within(archivedRow).getByRole("button", {
        name: "Unarchive Archived Notes"
      })
    )

    await waitFor(() => {
      expect(ownedMocks.setArchived).toHaveBeenCalledWith(
        "ws-archive",
        false,
        2
      )
    })
  })

  it("reads the directory and details only through the captured account context", async () => {
    apiMocks.listWorkspaces.mockRejectedValue(
      new Error("mutable directory used")
    )
    apiMocks.getWorkspaceContext.mockRejectedValue(
      new Error("mutable context used")
    )
    ownedMocks.directory.mockResolvedValue({
      scope,
      list: async () => ({ items: [workspace()] }),
      getContext: async () => contextFor(workspace())
    })
    renderManager()
    expect(await screen.findByText("Climate Review")).toBeVisible()
    expect(apiMocks.listWorkspaces).not.toHaveBeenCalled()
    expect(apiMocks.getWorkspaceContext).not.toHaveBeenCalled()
    await userEvent.click(
      screen.getByRole("button", { name: "Archive Climate Review" })
    )
    expect(ownedMocks.lifecycle).toHaveBeenCalledWith(
      "ws-research",
      scope,
      expect.any(AbortSignal)
    )
    await waitFor(() =>
      expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    )
    expect(apiMocks.patchWorkspace).not.toHaveBeenCalled()
  })

  it.each([
    "tldw:config-updated",
    "tldw:auth-principal-changed",
    "focus",
    "pageshow"
  ])(
    "hides account rows and open edits on %s, clearing them for a different verified account",
    async (eventName) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      renderManager()
      await screen.findByText("Climate Review")
      await userEvent.click(
        screen.getByRole("button", { name: "Edit Climate Review" })
      )
      expect(ownedMocks.directory).toHaveBeenCalledTimes(1)
      const oldSignal = ownedMocks.directory.mock.calls[0][0] as AbortSignal
      const next = deferred<{ items: WorkspaceApiResponse[] }>()
      apiMocks.listWorkspaces.mockReturnValueOnce(next.promise)
      ownedMocks.directory.mockResolvedValueOnce({
        scope: { ...scope, principalId: "18" },
        list: apiMocks.listWorkspaces,
        getContext: apiMocks.getWorkspaceContext
      })
      act(() => window.dispatchEvent(new Event(eventName)))
      expect(oldSignal.aborted).toBe(true)
      expect(
        screen.queryByRole("row", { name: /Climate Review/ })
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole("textbox", { name: "Workspace name" })
      ).not.toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "New Research Workspace" })
      ).toBeDisabled()
      await act(async () =>
        next.resolve({ items: [workspace({ name: "Other account" })] })
      )
      expect(await screen.findByText("Other account")).toBeVisible()
      expect(screen.queryByLabelText("Workspace name")).not.toBeInTheDocument()
    }
  )

  it.each(["list", "context"])(
    "discards a late %s response after a new directory load",
    async (stage) => {
      const pending = deferred<
        { items: WorkspaceApiResponse[] } | WorkspaceContextResponse
      >()
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      if (stage === "list")
        apiMocks.listWorkspaces.mockReturnValueOnce(pending.promise)
      else apiMocks.getWorkspaceContext.mockReturnValueOnce(pending.promise)
      renderManager()
      await waitFor(() =>
        expect(
          stage === "list"
            ? apiMocks.listWorkspaces
            : apiMocks.getWorkspaceContext
        ).toHaveBeenCalled()
      )
      apiMocks.listWorkspaces.mockResolvedValue({
        items: [workspace({ name: "New account" })]
      })
      act(() => window.dispatchEvent(new Event("tldw:config-updated")))
      expect(await screen.findByText("New account")).toBeVisible()
      await act(async () =>
        pending.resolve(
          stage === "list" ? { items: [workspace()] } : contextFor(workspace())
        )
      )
      expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
      expect(screen.getByText("New account")).toBeVisible()
    }
  )

  it("denies logout without starting an unauthenticated directory fallback", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    renderManager()
    await screen.findByText("Climate Review")
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "logout" }
        })
      )
    )
    expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    expect(screen.getByRole("alert")).toBeVisible()
    expect(ownedMocks.directory).toHaveBeenCalledTimes(1)
  })

  it.each([401, 403, 412])(
    "does not expose fallback rows when context rejects the account with %s",
    async (status) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      apiMocks.getWorkspaceContext.mockRejectedValue({ status })
      renderManager()
      expect(
        await screen.findByText("Workspaces are unavailable")
      ).toBeVisible()
      expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    }
  )

  it("prevents synchronous duplicate archive requests and retains the canonical row while pending", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    const pending = deferred<WorkspaceApiResponse>()
    ownedMocks.setArchived.mockReturnValueOnce(pending.promise)
    renderManager()
    const archive = await screen.findByRole("button", {
      name: "Archive Climate Review"
    })
    act(() => {
      fireEvent.click(archive)
      fireEvent.click(archive)
    })
    await waitFor(() => expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1))
    expect(screen.getByText("Climate Review")).toBeVisible()
    await act(async () =>
      pending.resolve(workspace({ archived: true, version: 5 }))
    )
    expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
  })

  it.each([409, 500])(
    "requires explicit refresh and a fresh restore action after status %s",
    async (status) => {
      apiMocks.listWorkspaces.mockResolvedValue({
        items: [workspace({ archived: true })]
      })
      ownedMocks.setArchived.mockRejectedValueOnce(
        Object.assign(new Error("write failed"), { status })
      )
      renderManager()
      await screen.findByText(/Showing/)
      await userEvent.click(
        screen.getByRole("checkbox", { name: "Show archived" })
      )
      await userEvent.click(
        screen.getByRole("button", { name: "Unarchive Climate Review" })
      )
      expect(await screen.findByRole("alert")).toHaveTextContent(
        /refresh.*review/i
      )
      expect(
        screen.getByRole("button", { name: "Unarchive Climate Review" })
      ).toBeVisible()
      await userEvent.click(
        screen.getByRole("button", { name: "Unarchive Climate Review" })
      )
      expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
      apiMocks.listWorkspaces.mockResolvedValue({
        items: [workspace({ archived: true, version: 9 })]
      })
      await userEvent.click(
        screen.getByRole("button", { name: "Refresh and review" })
      )
      await screen.findByRole("button", { name: "Unarchive Climate Review" })
      expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
      await userEvent.click(
        screen.getByRole("button", { name: "Unarchive Climate Review" })
      )
      await waitFor(() =>
        expect(ownedMocks.setArchived).toHaveBeenLastCalledWith(
          "ws-research",
          false,
          9
        )
      )
      expect(
        await screen.findByRole("button", { name: "Archive Climate Review" })
      ).toBeVisible()
    }
  )

  it.each(["success", "failure"])(
    "discards a late lifecycle %s after switching accounts",
    async (result) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      const pending = deferred<WorkspaceApiResponse>()
      ownedMocks.setArchived.mockReturnValueOnce(pending.promise)
      renderManager()
      await userEvent.click(
        await screen.findByRole("button", { name: "Archive Climate Review" })
      )
      expect(ownedMocks.lifecycle).toHaveBeenCalledTimes(1)
      const signal = ownedMocks.lifecycle.mock.calls[0][2] as AbortSignal
      apiMocks.listWorkspaces.mockResolvedValue({
        items: [workspace({ name: "New account" })]
      })
      ownedMocks.directory.mockResolvedValueOnce({
        scope: { ...scope, principalId: "18" },
        list: apiMocks.listWorkspaces,
        getContext: apiMocks.getWorkspaceContext
      })
      act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
      expect(signal.aborted).toBe(true)
      await screen.findByText("New account")
      await act(async () =>
        result === "success"
          ? pending.resolve(workspace({ archived: true, version: 5 }))
          : pending.reject(new Error("old account failed"))
      )
      expect(screen.getByText("New account")).toBeVisible()
      expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    }
  )

  it("aborts directory and lifecycle requests on unmount", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    ownedMocks.setArchived.mockReturnValue(new Promise(() => undefined))
    const { unmount } = renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    expect(ownedMocks.lifecycle).toHaveBeenCalledTimes(1)
    const directorySignal = ownedMocks.directory.mock.calls[0][0] as AbortSignal
    const lifecycleSignal = ownedMocks.lifecycle.mock.calls[0][2] as AbortSignal
    unmount()
    expect(directorySignal.aborted).toBe(true)
    expect(lifecycleSignal.aborted).toBe(true)
  })

  it.each([
    "Archive Climate Review",
    "Edit Climate Review",
    "Open Climate Review",
    "New Research Workspace"
  ])(
    "makes the old %s action inert before the invalidated render commits",
    async (name) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      renderManager()
      await screen.findByText("Climate Review")
      const button = screen.getByRole("button", { name })
      apiMocks.listWorkspaces.mockResolvedValue({
        items: [workspace({ name: "New account" })]
      })
      act(() => {
        window.dispatchEvent(new Event("tldw:config-updated"))
        fireEvent.click(button)
      })
      await screen.findByText("New account")
      expect(ownedMocks.lifecycle).not.toHaveBeenCalled()
      expect(routerMocks.navigate).not.toHaveBeenCalled()
      expect(screen.queryByLabelText("Workspace name")).not.toBeInTheDocument()
    }
  )

  it.each(["create", "rename"])(
    "rejects stale %s dialog submission before starting a request",
    async (kind) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      renderManager()
      await screen.findByText("Climate Review")
      await userEvent.click(
        screen.getByRole("button", {
          name:
            kind === "create" ? "New Research Workspace" : "Edit Climate Review"
        })
      )
      await userEvent.clear(screen.getByLabelText("Workspace name"))
      await userEvent.type(screen.getByLabelText("Workspace name"), "Old draft")
      const submit = screen.getByRole("button", {
        name: kind === "create" ? "Create Workspace" : "Save metadata"
      })
      ownedMocks.directory.mockResolvedValueOnce({
        scope: { ...scope, principalId: "18" },
        list: apiMocks.listWorkspaces,
        getContext: apiMocks.getWorkspaceContext
      })
      act(() => {
        window.dispatchEvent(new Event("tldw:config-updated"))
        fireEvent.click(submit)
      })
      await screen.findByText("Climate Review")
      expect(apiMocks.upsertWorkspace).not.toHaveBeenCalled()
      expect(apiMocks.patchWorkspace).not.toHaveBeenCalled()
      expect(screen.queryByLabelText("Workspace name")).not.toBeInTheDocument()
    }
  )

  it.each(["create", "rename", "project"])(
    "ignores a stale %s completion in the next directory",
    async (kind) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      const pending = deferred<WorkspaceApiResponse>()
      if (kind === "create")
        apiMocks.upsertWorkspace.mockReturnValueOnce(pending.promise)
      else apiMocks.patchWorkspace.mockReturnValueOnce(pending.promise)
      renderManager()
      await screen.findByText("Climate Review")
      if (kind === "project") {
        await userEvent.click(
          screen.getByRole("button", { name: "Upgrade to Project Workspace" })
        )
      } else {
        await userEvent.click(
          screen.getByRole("button", {
            name:
              kind === "create"
                ? "New Research Workspace"
                : "Edit Climate Review"
          })
        )
        await userEvent.type(
          screen.getByLabelText("Workspace name"),
          "Old draft"
        )
        await userEvent.click(
          screen.getByRole("button", {
            name: kind === "create" ? "Create Workspace" : "Save metadata"
          })
        )
      }
      apiMocks.listWorkspaces.mockResolvedValue({
        items: [workspace({ name: "New account" })]
      })
      act(() => window.dispatchEvent(new Event("tldw:config-updated")))
      await screen.findByText("New account")
      await act(async () =>
        pending.resolve(workspace({ name: "Old response", version: 5 }))
      )
      expect(screen.getByText("New account")).toBeVisible()
      expect(screen.queryByText("Old response")).not.toBeInTheDocument()
    }
  )

  it("does not send an archive after lifecycle capture outlives its directory", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    const capture = deferred<{
      scope: typeof scope
      get: () => Promise<WorkspaceApiResponse>
      setArchived: typeof ownedMocks.setArchived
    }>()
    ownedMocks.lifecycle.mockReturnValueOnce(capture.promise)
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    act(() => window.dispatchEvent(new Event("tldw:config-updated")))
    await screen.findByText("Climate Review")
    await act(async () =>
      capture.resolve({
        scope,
        get: async () => workspace(),
        setArchived: ownedMocks.setArchived
      })
    )
    expect(ownedMocks.setArchived).not.toHaveBeenCalled()
  })

  it.each(["pagehide", "visibilitychange"])(
    "suspends account access on %s and reloads on restore",
    async (eventName) => {
      apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
      renderManager()
      await screen.findByText("Climate Review")
      const signal = ownedMocks.directory.mock.calls[0][0] as AbortSignal
      const visibility = vi.spyOn(document, "visibilityState", "get")
      visibility.mockReturnValue("hidden")
      act(() =>
        (eventName === "pagehide" ? window : document).dispatchEvent(
          new Event(eventName)
        )
      )
      expect(signal.aborted).toBe(true)
      expect(
        screen.queryByRole("row", { name: /Climate Review/ })
      ).not.toBeInTheDocument()
      visibility.mockReturnValue("visible")
      act(() => window.dispatchEvent(new Event("pageshow")))
      expect(await screen.findByText("Climate Review")).toBeVisible()
    }
  )

  it("invalidates account access on cross-tab configuration changes", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    renderManager()
    await screen.findByText("Climate Review")
    act(() =>
      window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig" }))
    )
    expect(
      screen.queryByRole("row", { name: /Climate Review/ })
    ).not.toBeInTheDocument()
    await screen.findByText("Climate Review")
    expect(ownedMocks.directory).toHaveBeenCalledTimes(2)
  })

  it("does not let automatic focus refresh bypass explicit conflict review", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    ownedMocks.setArchived.mockRejectedValueOnce({ status: 409 })
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    await screen.findByRole("alert")
    apiMocks.listWorkspaces.mockResolvedValue({
      items: [workspace({ version: 8 })]
    })
    act(() => window.dispatchEvent(new Event("focus")))
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
    expect(
      screen.getByRole("button", { name: "Refresh and review" })
    ).toBeVisible()
  })

  it("requires explicit review after an in-flight write is aborted by browser suspension", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    const pending = deferred<WorkspaceApiResponse>()
    ownedMocks.setArchived.mockReturnValueOnce(pending.promise)
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    act(() => window.dispatchEvent(new Event("pagehide")))
    act(() => window.dispatchEvent(new Event("pageshow")))
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
    expect(
      screen.getByRole("button", { name: "Refresh and review" })
    ).toBeVisible()
    await act(async () =>
      pending.resolve(workspace({ archived: true, version: 5 }))
    )
    expect(
      screen.getByRole("button", { name: "Archive Climate Review" })
    ).toBeVisible()
  })

  it("keeps archive blocked after a failed explicit refresh and retries reads only", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    ownedMocks.setArchived.mockRejectedValueOnce(
      new Error("network disconnected")
    )
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    await screen.findByRole("alert")
    apiMocks.listWorkspaces.mockRejectedValueOnce(new Error("still offline"))
    await userEvent.click(
      screen.getByRole("button", { name: "Refresh and review" })
    )
    expect(await screen.findByText("Workspaces are unavailable")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Archive Climate Review" })
    ).not.toBeInTheDocument()
    expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
    apiMocks.listWorkspaces.mockResolvedValue({
      items: [workspace({ version: 7 })]
    })
    await userEvent.click(screen.getByRole("button", { name: "Retry" }))
    await screen.findByRole("button", { name: "Archive Climate Review" })
    expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
    await userEvent.click(
      screen.getByRole("button", { name: "Archive Climate Review" })
    )
    await waitFor(() =>
      expect(ownedMocks.setArchived).toHaveBeenLastCalledWith(
        "ws-research",
        true,
        7
      )
    )
  })

  it("keeps ordinary detail failures partial without falling back to mutable reads", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    apiMocks.getWorkspaceContext.mockRejectedValueOnce(
      new Error("details unavailable")
    )
    renderManager()
    expect(await screen.findByText("Climate Review")).toBeVisible()
    expect(
      screen.getByText("Some Workspace details could not load.")
    ).toBeVisible()
  })

  it("rejects directory identity failure without listing or creating through mutable APIs", async () => {
    ownedMocks.directory.mockRejectedValueOnce({ reason: "denied" })
    renderManager()
    expect(await screen.findByText("Workspaces are unavailable")).toBeVisible()
    expect(
      screen.getByRole("button", { name: "New Research Workspace" })
    ).toBeDisabled()
    expect(apiMocks.listWorkspaces).not.toHaveBeenCalled()
    expect(apiMocks.upsertWorkspace).not.toHaveBeenCalled()
  })

  it("clears account rows when lifecycle capture reports an account mismatch", async () => {
    apiMocks.listWorkspaces.mockResolvedValue({ items: [workspace()] })
    ownedMocks.lifecycle.mockRejectedValueOnce({ status: 412 })
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    expect(await screen.findByText("Workspaces are unavailable")).toBeVisible()
    expect(screen.queryByText("Climate Review")).not.toBeInTheDocument()
    expect(ownedMocks.setArchived).not.toHaveBeenCalled()
  })

  it("does not treat automatic project-root completion as explicit lifecycle review", async () => {
    const project = workspace({ workspace_profile: "project" })
    apiMocks.listWorkspaces.mockResolvedValue({ items: [project] })
    apiMocks.getWorkspaceContext.mockResolvedValue(contextFor(project))
    const pending = deferred<WorkspaceRootsResponse>()
    apiMocks.attachWorkspacePrimaryRoot.mockReturnValueOnce(pending.promise)
    ownedMocks.setArchived.mockRejectedValueOnce({ status: 409 })
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Host-local root" })
    )
    await userEvent.type(screen.getByLabelText("Root path"), "/tmp/project")
    await userEvent.click(
      screen.getByRole("button", { name: "Attach host-local root" })
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Archive Climate Review" })
    )
    await screen.findByRole("button", { name: "Refresh and review" })
    await act(async () =>
      pending.resolve({
        workspace_id: project.id,
        workspace_profile: "project",
        primary_root: null,
        roots: []
      })
    )
    await userEvent.click(
      await screen.findByRole("button", { name: "Archive Climate Review" })
    )
    expect(ownedMocks.setArchived).toHaveBeenCalledTimes(1)
    expect(
      screen.getByRole("button", { name: "Refresh and review" })
    ).toBeVisible()
  })

  it.each([
    ["create", "focus"],
    ["rename", "focus"],
    ["root", "focus"],
    ["create", "visibility"],
    ["rename", "visibility"],
    ["root", "visibility"]
  ])(
    "preserves the %s draft through same-account %s revalidation",
    async (kind, trigger) => {
      const project = workspace({ workspace_profile: "project" })
      apiMocks.listWorkspaces.mockResolvedValue({ items: [project] })
      apiMocks.getWorkspaceContext.mockResolvedValue(contextFor(project))
      renderManager()
      await screen.findByText("Climate Review")
      await userEvent.click(
        screen.getByRole("button", {
          name:
            kind === "create"
              ? "New Research Workspace"
              : kind === "rename"
                ? "Edit Climate Review"
                : "Host-local root"
        })
      )
      const label = kind === "root" ? "Root path" : "Workspace name"
      const draft =
        kind === "root" ? "/tmp/unsaved-project" : "Unfinished research draft"
      await userEvent.clear(screen.getByLabelText(label))
      await userEvent.type(screen.getByLabelText(label), draft)
      const pending = deferred<{ items: WorkspaceApiResponse[] }>()
      apiMocks.listWorkspaces.mockReturnValueOnce(pending.promise)
      if (trigger === "visibility") {
        const visibility = vi.spyOn(document, "visibilityState", "get")
        visibility.mockReturnValue("hidden")
        act(() => document.dispatchEvent(new Event("visibilitychange")))
        expect(
          screen.queryByRole("textbox", { name: label })
        ).not.toBeInTheDocument()
        visibility.mockReturnValue("visible")
        act(() => document.dispatchEvent(new Event("visibilitychange")))
      } else {
        act(() => window.dispatchEvent(new Event("focus")))
      }
      expect(
        screen.queryByRole("textbox", { name: label })
      ).not.toBeInTheDocument()
      await act(async () => pending.resolve({ items: [project] }))
      expect(await screen.findByRole("textbox", { name: label })).toHaveValue(
        draft
      )
      expect(apiMocks.upsertWorkspace).not.toHaveBeenCalled()
      expect(apiMocks.patchWorkspace).not.toHaveBeenCalled()
      expect(apiMocks.attachWorkspacePrimaryRoot).not.toHaveBeenCalled()
    }
  )

  it("preserves the selected second project after attaching its root", async () => {
    const first = workspace({
      id: "first-project",
      name: "First project",
      workspace_profile: "project"
    })
    const second = workspace({
      id: "second-project",
      name: "Second project",
      workspace_profile: "project"
    })
    let attached = false
    apiMocks.listWorkspaces.mockResolvedValue({ items: [first, second] })
    apiMocks.getWorkspaceContext.mockImplementation(async (id: string) => {
      const context = contextFor(id === second.id ? second : first)
      if (id === second.id && attached) {
        context.project_root.display_name = "Second project root"
        context.project_root.state = "attached"
      }
      return context
    })
    apiMocks.attachWorkspacePrimaryRoot.mockImplementation(async () => {
      attached = true
      return {
        workspace_id: second.id,
        workspace_profile: "project",
        primary_root: null,
        roots: []
      }
    })
    renderManager()
    await userEvent.click(
      await screen.findByRole("button", { name: "Manage Second project" })
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Host-local root" })
    )
    await userEvent.type(
      screen.getByLabelText("Root path"),
      "/tmp/second-project"
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Attach host-local root" })
    )
    expect(apiMocks.attachWorkspacePrimaryRoot).toHaveBeenCalledWith(
      "second-project",
      expect.objectContaining({
        absolute_root: "/tmp/second-project"
      })
    )
    await waitFor(() =>
      expect(screen.getByTestId("workspace-root-summary")).toHaveTextContent(
        "Second project root"
      )
    )
  })

  it.each([
    ["create", { ...scope, principalId: "18" }],
    ["rename", { ...scope, serverBase: "https://other.example" }],
    ["root", { ...scope, organizationId: "9" }]
  ])(
    "discards the retained %s draft at a verified account boundary",
    async (kind, nextScope) => {
      const project = workspace({ workspace_profile: "project" })
      apiMocks.listWorkspaces.mockResolvedValue({ items: [project] })
      apiMocks.getWorkspaceContext.mockResolvedValue(contextFor(project))
      renderManager()
      await screen.findByText("Climate Review")
      await userEvent.click(
        screen.getByRole("button", {
          name:
            kind === "create"
              ? "New Research Workspace"
              : kind === "rename"
                ? "Edit Climate Review"
                : "Host-local root"
        })
      )
      const label = kind === "root" ? "Root path" : "Workspace name"
      await userEvent.type(screen.getByLabelText(label), "private draft")
      ownedMocks.directory.mockResolvedValueOnce({
        scope: nextScope,
        list: apiMocks.listWorkspaces,
        getContext: apiMocks.getWorkspaceContext
      })
      act(() => window.dispatchEvent(new Event("focus")))
      await screen.findByRole("button", { name: "Manage Climate Review" })
      expect(screen.queryByLabelText(label)).not.toBeInTheDocument()
      await userEvent.click(
        screen.getByRole("button", {
          name:
            kind === "create"
              ? "New Research Workspace"
              : kind === "rename"
                ? "Edit Climate Review"
                : "Host-local root"
        })
      )
      expect(screen.getByLabelText(label)).toHaveValue(
        kind === "rename" ? "Climate Review" : ""
      )
    }
  )

  it("does not poll a retained root panel while account revalidation is pending", async () => {
    const project = workspace({ workspace_profile: "project" })
    const operation = {
      operation_id: "op-1",
      workspace_id: project.id,
      command: "provision_sandbox_root",
      status: "running" as const,
      started_at: "2026-06-04T09:00:00Z",
      updated_at: "2026-06-04T09:00:01Z",
      retryable: false,
      diagnostics: {},
      poll_href: "/api/v1/workspaces/ws-research/operations/op-1"
    }
    apiMocks.listWorkspaces.mockResolvedValue({ items: [project] })
    apiMocks.getWorkspaceContext.mockResolvedValue(
      contextFor(project, { active_operations: [operation] })
    )
    apiMocks.getWorkspaceOperation.mockResolvedValue(operation)
    vi.useFakeTimers()
    try {
      await act(async () => {
        renderManager()
      })
      expect(
        screen.getByRole("button", { name: "Manage Climate Review" })
      ).toBeVisible()
      ownedMocks.directory.mockReturnValueOnce(new Promise(() => undefined))
      act(() => window.dispatchEvent(new Event("focus")))
      await act(async () => vi.advanceTimersByTimeAsync(800))
      expect(apiMocks.getWorkspaceOperation).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })
})
