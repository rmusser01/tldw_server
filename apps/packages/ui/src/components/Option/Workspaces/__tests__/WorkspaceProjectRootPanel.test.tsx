import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  WorkspaceApiResponse,
  WorkspaceOperationResponse,
  WorkspaceRootsResponse
} from "@/services/tldw/domains/workspace-api"
import type { WorkspaceManagerItem } from "../workspace-manager-models"

const apiMocks = vi.hoisted(() => ({
  patchWorkspace: vi.fn(),
  attachWorkspacePrimaryRoot: vi.fn(),
  provisionWorkspaceSandboxRoot: vi.fn(),
  getWorkspaceOperation: vi.fn(),
  queueWorkspaceFileInventoryScan: vi.fn()
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMocks
}))

import { WorkspaceProjectRootPanel } from "../WorkspaceProjectRootPanel"

const workspaceResponse = (
  overrides: Partial<WorkspaceApiResponse> = {}
): WorkspaceApiResponse => ({
  id: "ws-project",
  name: "Policy Build",
  archived: false,
  study_materials_policy: "workspace",
  workspace_profile: "project",
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
  version: 5,
  ...overrides
})

const operation = (
  overrides: Partial<WorkspaceOperationResponse> = {}
): WorkspaceOperationResponse => ({
  operation_id: "op-1",
  workspace_id: "ws-project",
  command: "provision_sandbox_root",
  status: "running",
  started_at: "2026-06-04T09:00:00Z",
  updated_at: "2026-06-04T09:00:01Z",
  retryable: false,
  diagnostics: {},
  poll_href: "/api/v1/workspaces/ws-project/operations/op-1",
  ...overrides
})

const rootsResponse = (): WorkspaceRootsResponse => ({
  workspace_id: "ws-project",
  workspace_profile: "project",
  primary_root: {
    workspace_id: "ws-project",
    root_id: "root-1",
    backend: "host_local",
    state: "attached",
    display_name: "Repo",
    path_hint: "/redacted/repo",
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
    is_primary: true,
    version: 3,
    updated_at: "2026-06-04T09:00:00Z"
  },
  roots: []
})

const managerItem = (
  overrides: Partial<WorkspaceManagerItem> = {}
): WorkspaceManagerItem => ({
  id: "ws-project",
  name: "Policy Build",
  archived: false,
  profile: "project",
  attentionState: "setup_pending",
  projectRoot: {
    state: "not_configured",
    rootId: null,
    backend: null,
    displayName: null,
    pathHint: null,
    gitState: null,
    fileInventoryState: "not_started",
    fileInventory: {
      state: "not_started",
      indexedFileCount: null,
      totalFileCount: null,
      updatedAt: null,
      available: false
    },
    indexingState: null,
    sandboxMountState: null,
    mcpTrustState: null
  },
  fileInventoryAvailable: false,
  sourceCount: 0,
  selectedSourceCount: 0,
  activeOperations: [],
  updatedAt: "2026-06-04T09:00:00Z",
  version: 4,
  ...overrides
})

const renderPanel = (
  item: WorkspaceManagerItem,
  props: Partial<React.ComponentProps<typeof WorkspaceProjectRootPanel>> = {}
) =>
  render(
    <WorkspaceProjectRootPanel
      item={item}
      onWorkspaceUpdated={vi.fn()}
      onRootsUpdated={vi.fn()}
      onRefreshContext={vi.fn()}
      {...props}
    />
  )

describe("WorkspaceProjectRootPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("crypto", {
      randomUUID: () => "sandbox-root-op-1"
    })
    apiMocks.patchWorkspace.mockResolvedValue(workspaceResponse())
    apiMocks.attachWorkspacePrimaryRoot.mockResolvedValue(rootsResponse())
    apiMocks.provisionWorkspaceSandboxRoot.mockResolvedValue({
      workspace_id: "ws-project",
      workspace_profile: "project",
      operation: operation(),
      primary_root: null
    })
    apiMocks.getWorkspaceOperation.mockResolvedValue(operation())
    apiMocks.queueWorkspaceFileInventoryScan.mockResolvedValue({
      workspace_id: "ws-project",
      root_id: "root-1",
      state: "queued",
      durable_state: "queued",
      stale: false,
      last_scan_id: null,
      last_scan_started_at: null,
      last_scan_completed_at: null,
      root_version: 3,
      scan_root_version: null,
      ignore_policy_fingerprint: null,
      root_snapshot_token: null,
      counts: {
        files: 0,
        directories: 0,
        symlinks: 0,
        ignored: 0,
        indexing_candidates: 0,
        diagnostics: 0,
        total_entries: 0
      },
      diagnostics: [],
      job: null,
      updated_at: "2026-06-04T09:00:00Z"
    })
  })

  it("upgrades a Research Workspace to a Project Workspace", async () => {
    const onWorkspaceUpdated = vi.fn()
    const user = userEvent.setup()
    renderPanel(
      managerItem({
        id: "ws-research",
        name: "Climate Review",
        profile: "research",
        attentionState: "ready"
      }),
      { onWorkspaceUpdated }
    )

    await user.click(
      screen.getByRole("button", { name: "Upgrade to Project Workspace" })
    )

    await waitFor(() => {
      expect(apiMocks.patchWorkspace).toHaveBeenCalledWith("ws-research", {
        workspace_profile: "project",
        version: 4
      })
    })
    expect(onWorkspaceUpdated).toHaveBeenCalled()
  })

  it("attaches a host-local primary root with the expected workspace version", async () => {
    const user = userEvent.setup()
    renderPanel(managerItem())

    await user.click(screen.getByRole("button", { name: "Host-local root" }))
    await user.type(screen.getByLabelText("Root path"), "/Users/alice/repo")
    await user.type(screen.getByLabelText("Display name"), "Policy repo")
    await user.click(screen.getByRole("button", { name: "Attach host-local root" }))

    await waitFor(() => {
      expect(apiMocks.attachWorkspacePrimaryRoot).toHaveBeenCalledWith(
        "ws-project",
        {
          backend: "host_local",
          absolute_root: "/Users/alice/repo",
          display_name: "Policy repo",
          expected_workspace_version: 4
        }
      )
    })
  })

  it("provisions a sandbox-managed root with an idempotency key and polls status", async () => {
    const user = userEvent.setup()
    renderPanel(managerItem())

    await user.click(
      screen.getByRole("button", { name: "Sandbox-managed root" })
    )
    await user.type(screen.getByLabelText("Display name"), "Policy sandbox")
    await user.type(screen.getByLabelText("Requested runtime"), "python")
    await user.click(
      screen.getByRole("button", { name: "Provision sandbox root" })
    )

    await waitFor(() => {
      expect(apiMocks.provisionWorkspaceSandboxRoot).toHaveBeenCalledWith(
        "ws-project",
        {
          display_name: "Policy sandbox",
          requested_runtime: "python",
          expected_workspace_version: 4
        },
        "workspace-sandbox-root:ws-project:sandbox-root-op-1"
      )
    })
    expect(await screen.findByText("Provisioning sandbox root")).toBeVisible()
    await waitFor(() => {
      expect(apiMocks.getWorkspaceOperation).toHaveBeenCalledWith(
        "ws-project",
        "op-1"
      )
    })
  })

  it("recovers visible provisioning state from active operations", () => {
    renderPanel(
      managerItem({
        activeOperations: [operation()],
        projectRoot: {
          ...managerItem().projectRoot,
          state: "provisioning",
          backend: "sandbox_volume",
          displayName: "Policy sandbox"
        }
      })
    )

    expect(screen.getByText("Provisioning sandbox root")).toBeVisible()
    expect(screen.getByText("running")).toBeVisible()
  })

  it("disables inventory scans until file inventory is available", () => {
    renderPanel(
      managerItem({
        projectRoot: {
          ...managerItem().projectRoot,
          state: "provisioning",
          backend: "sandbox_volume",
          displayName: "Policy sandbox"
        }
      })
    )

    expect(screen.getByRole("button", { name: "Scan files" })).toBeDisabled()
    expect(
      screen.getByText(
        "File inventory is unavailable until the sandbox-managed root is mounted."
      )
    ).toBeVisible()
  })

  it("enables inventory scans only after the API marks inventory available", async () => {
    const user = userEvent.setup()
    renderPanel(
      managerItem({
        projectRoot: {
          ...managerItem().projectRoot,
          state: "attached",
          rootId: "root-1",
          backend: "sandbox_volume",
          displayName: "Policy sandbox",
          fileInventoryState: "current",
          fileInventory: {
            state: "current",
            indexedFileCount: 2,
            totalFileCount: 4,
            updatedAt: "2026-06-04T09:00:00Z",
            available: true
          }
        },
        fileInventoryAvailable: true
      })
    )

    await user.click(screen.getByRole("button", { name: "Scan files" }))

    await waitFor(() => {
      expect(apiMocks.queueWorkspaceFileInventoryScan).toHaveBeenCalledWith(
        "ws-project",
        { force: true }
      )
    })
  })

  it("does not expose raw host-local paths in passive root displays", () => {
    renderPanel(
      managerItem({
        projectRoot: {
          ...managerItem().projectRoot,
          state: "attached",
          rootId: "root-1",
          backend: "host_local",
          displayName: "Private repo",
          pathHint: "/Users/alice/private/customer-research"
        }
      })
    )

    const passiveSummary = screen.getByTestId("workspace-root-summary")
    expect(within(passiveSummary).getByText("Private repo")).toBeVisible()
    expect(passiveSummary).not.toHaveTextContent(
      "/Users/alice/private/customer-research"
    )
    expect(within(passiveSummary).getByText("Path hidden")).toBeVisible()
  })
})
