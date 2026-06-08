import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WORKSPACE_STORAGE_KEY } from "@/store/research-workspace-legacy-storage-inventory"
import { buildWorkspaceReconciliationMarkerStorageKey } from "../workspace-local-reconciliation"
import type { WorkspaceManagerItem } from "../workspace-manager-models"

const apiMocks = vi.hoisted(() => ({
  upsertWorkspace: vi.fn()
}))

vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => apiMocks
}))

import { WorkspaceReconciliationPanel } from "../WorkspaceReconciliationPanel"

const splitIndex = (savedWorkspaces: unknown[]) =>
  JSON.stringify({
    schema: "workspace_split_v1",
    version: 12,
    state: {
      workspaceId: "local-ready",
      savedWorkspaces,
      archivedWorkspaces: [],
      workspaceIds: savedWorkspaces
        .filter(
          (workspace): workspace is { id: string } =>
            typeof workspace === "object" &&
            workspace !== null &&
            typeof (workspace as { id?: unknown }).id === "string"
        )
        .map((workspace) => workspace.id),
      workspaceSnapshots: {},
      workspaceChatSessions: {}
    }
  })

const serverItem = (
  overrides: Partial<WorkspaceManagerItem> = {}
): WorkspaceManagerItem => ({
  id: "server-1",
  name: "Server Workspace",
  archived: false,
  profile: "research",
  attentionState: "ready",
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
  version: 1,
  ...overrides
})

describe("WorkspaceReconciliationPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    apiMocks.upsertWorkspace.mockResolvedValue({
      id: "local-ready",
      name: "Ready Local",
      workspace_profile: "research"
    })
  })

  it("creates server metadata for an eligible local Research Workspace and writes a marker", async () => {
    window.localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      splitIndex([
        {
          id: "local-ready",
          name: "Ready Local",
          sourceCount: 3
        }
      ])
    )
    const onServerWorkspaceCreated = vi.fn()
    const user = userEvent.setup()

    render(
      <WorkspaceReconciliationPanel
        serverWorkspaces={[]}
        onServerWorkspaceCreated={onServerWorkspaceCreated}
      />
    )

    const row = await screen.findByRole("row", { name: /Ready Local/i })
    expect(within(row).getByText("ready to create metadata")).toBeVisible()

    await user.click(
      within(row).getByRole("button", { name: "Create server metadata" })
    )

    await waitFor(() => {
      expect(apiMocks.upsertWorkspace).toHaveBeenCalledWith("local-ready", {
        name: "Ready Local",
        study_materials_policy: "workspace",
        workspace_profile: "research"
      })
    })
    const marker = JSON.parse(
      window.localStorage.getItem(
        buildWorkspaceReconciliationMarkerStorageKey("local-ready")
      ) ?? "{}"
    )
    expect(marker).toMatchObject({
      schemaVersion: 1,
      serverWorkspaceId: "local-ready",
      serverName: "Ready Local",
      serverProfile: "research",
      status: "metadata_promoted"
    })
    expect(onServerWorkspaceCreated).toHaveBeenCalled()
  })

  it("links a local entry to an existing server Workspace without creating metadata", async () => {
    window.localStorage.setItem(
      WORKSPACE_STORAGE_KEY,
      splitIndex([
        {
          id: "local-conflict",
          name: "Server Workspace",
          sourceCount: 2
        }
      ])
    )
    const user = userEvent.setup()

    render(<WorkspaceReconciliationPanel serverWorkspaces={[serverItem()]} />)

    const row = await screen.findByRole("row", { name: /Server Workspace/i })
    expect(within(row).getByText("name conflict")).toBeVisible()

    await user.click(
      within(row).getByRole("button", { name: "Link to existing Workspace" })
    )

    expect(apiMocks.upsertWorkspace).not.toHaveBeenCalled()
    const marker = JSON.parse(
      window.localStorage.getItem(
        buildWorkspaceReconciliationMarkerStorageKey("local-conflict")
      ) ?? "{}"
    )
    expect(marker).toMatchObject({
      schemaVersion: 1,
      serverWorkspaceId: "server-1",
      serverName: "Server Workspace",
      serverProfile: "research",
      status: "linked"
    })
  })
})
