import { act, renderHook, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type {
  WorkspaceApiResponse,
  WorkspaceCapabilitiesResponse,
  WorkspaceContextResponse,
  WorkspaceProjectRoot,
  WorkspaceSourceStatusSummary
} from "@/services/tldw/domains/workspace-api"
import { useActiveWorkspaceContext } from "../hooks"

const workspaceFixture = (
  overrides: Partial<WorkspaceApiResponse> = {}
): WorkspaceApiResponse => ({
  id: "ws-1",
  name: "Server Workspace",
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
  created_at: "2026-06-18T00:00:00Z",
  last_modified: "2026-06-18T00:10:00Z",
  version: 7,
  ...overrides
})

const sourceSummaryFixture = (
  overrides: Partial<WorkspaceSourceStatusSummary> = {}
): WorkspaceSourceStatusSummary => ({
  total: 0,
  selected: 0,
  queryable: 0,
  partially_queryable: 0,
  processing: 0,
  failed: 0,
  missing: 0,
  ...overrides
})

const projectRootFixture = (
  overrides: Partial<WorkspaceProjectRoot> = {}
): WorkspaceProjectRoot => ({
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
  mcp_trust_state: null,
  ...overrides
})

const capabilitiesFixture = (
  overrides: Partial<WorkspaceCapabilitiesResponse> = {}
): WorkspaceCapabilitiesResponse => ({
  workspace_id: "ws-1",
  workspace_profile: "research",
  workspace_kind: "research_workspace",
  access_level: "owner",
  source_summary: sourceSummaryFixture(),
  workspace_services: {},
  allowed_actions: {},
  ...overrides
})

const contextFixture = (
  overrides: Partial<WorkspaceContextResponse> = {}
): WorkspaceContextResponse => {
  const workspace = overrides.workspace ?? workspaceFixture()
  return {
    workspace_id: workspace.id,
    workspace_profile: workspace.workspace_profile,
    workspace_kind:
      workspace.workspace_profile === "project"
        ? "project_workspace"
        : "research_workspace",
    schema_version: 2,
    generated_at: "2026-06-18T00:11:00Z",
    workspace,
    attention_state: workspace.archived ? "archived" : "ready",
    resolution: { status: "complete", partial_errors: [] },
    project_root: projectRootFixture(),
    sources: {
      items: [],
      summary: sourceSummaryFixture()
    },
    capabilities: capabilitiesFixture({
      workspace_id: workspace.id,
      workspace_profile: workspace.workspace_profile
    }),
    services: {},
    allowed_actions: {},
    active_jobs: [],
    active_operations: [],
    partial_errors: [],
    ...overrides
  }
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })

  return { promise, resolve, reject }
}

describe("useActiveWorkspaceContext", () => {
  it("does not fetch when no active server workspace id exists", () => {
    const getWorkspaceContext = vi.fn()
    const { result } = renderHook(() =>
      useActiveWorkspaceContext({
        workspaceId: null,
        client: { getWorkspaceContext }
      })
    )

    expect(result.current.context.state).toBe("none")
    expect(result.current.loading).toBe(false)
    expect(getWorkspaceContext).not.toHaveBeenCalled()
  })

  it("normalizes fetched server context", async () => {
    const getWorkspaceContext = vi.fn(async () =>
      contextFixture({
        workspace: workspaceFixture({ id: "ws-1", name: "Fetched Workspace" })
      })
    )

    const { result } = renderHook(() =>
      useActiveWorkspaceContext({
        workspaceId: "ws-1",
        client: { getWorkspaceContext }
      })
    )

    expect(result.current.loading).toBe(true)

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
      expect(result.current.context.state).toBe("ready")
    })

    expect(result.current.context.workspace?.label).toBe("Fetched Workspace")
    expect(getWorkspaceContext).toHaveBeenCalledWith("ws-1")
  })

  it("surfaces failed server resolution as degraded context", async () => {
    const getWorkspaceContext = vi.fn(async () => {
      throw new Error("network down")
    })

    const { result } = renderHook(() =>
      useActiveWorkspaceContext({
        workspaceId: "ws-1",
        client: { getWorkspaceContext }
      })
    )

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
      expect(result.current.context.state).toBe("error")
    })

    expect(result.current.error).toBeInstanceOf(Error)
    expect(result.current.context.recovery.reasonCode).toBe("workspace_context_error")
  })

  it("refreshes context on demand", async () => {
    const getWorkspaceContext = vi.fn()
      .mockResolvedValueOnce(
        contextFixture({
          workspace: workspaceFixture({ id: "ws-1", name: "First" })
        })
      )
      .mockResolvedValueOnce(
        contextFixture({
          workspace: workspaceFixture({ id: "ws-1", name: "Second" })
        })
      )

    const { result } = renderHook(() =>
      useActiveWorkspaceContext({
        workspaceId: "ws-1",
        client: { getWorkspaceContext }
      })
    )

    await waitFor(() => {
      expect(result.current.context.workspace?.label).toBe("First")
    })

    await result.current.refresh()

    await waitFor(() => {
      expect(result.current.context.workspace?.label).toBe("Second")
    })
  })

  it("keeps the latest refresh result when workspace requests overlap", async () => {
    const firstRequest = deferred<WorkspaceContextResponse>()
    const secondRequest = deferred<WorkspaceContextResponse>()
    const getWorkspaceContext = vi.fn()
      .mockReturnValueOnce(firstRequest.promise)
      .mockReturnValueOnce(secondRequest.promise)

    const { result } = renderHook(() =>
      useActiveWorkspaceContext({
        workspaceId: "ws-1",
        client: { getWorkspaceContext }
      })
    )

    await waitFor(() => {
      expect(getWorkspaceContext).toHaveBeenCalledTimes(1)
    })

    let refreshPromise!: Promise<void>
    act(() => {
      refreshPromise = result.current.refresh()
    })

    await waitFor(() => {
      expect(getWorkspaceContext).toHaveBeenCalledTimes(2)
    })

    await act(async () => {
      secondRequest.resolve(
        contextFixture({
          workspace: workspaceFixture({ id: "ws-1", name: "Second" })
        })
      )
      await refreshPromise
    })

    await waitFor(() => {
      expect(result.current.context.workspace?.label).toBe("Second")
    })

    await act(async () => {
      firstRequest.resolve(
        contextFixture({
          workspace: workspaceFixture({ id: "ws-1", name: "First" })
        })
      )
      await firstRequest.promise
    })

    expect(result.current.context.workspace?.label).toBe("Second")
  })
})
