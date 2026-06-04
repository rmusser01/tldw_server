import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { workspaceApiMethods } from "../tldw/domains/workspace-api"

const workspaceResponse = {
  id: "ws-1",
  name: "Workspace One",
  archived: false,
  study_materials_policy: "workspace",
  workspace_profile: "project",
  deleted: false,
  banner_title: "Workspace One",
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null,
  created_at: "2026-05-06T12:00:00Z",
  last_modified: "2026-05-06T12:00:00Z",
  version: 2
}

const rootsResponse = {
  workspace_id: "ws-1",
  workspace_profile: "project",
  primary_root: {
    workspace_id: "ws-1",
    root_id: "root-1",
    backend: "host_local",
    state: "attached",
    display_name: "Repo",
    path_hint: "/repo",
    git_state: null,
    file_inventory_state: "current",
    file_inventory: {
      state: "current",
      indexed_file_count: 4,
      total_file_count: 6,
      updated_at: "2026-06-04T00:00:00Z",
      available: true
    },
    indexing_state: null,
    sandbox_mount_state: null,
    mcp_trust_state: null,
    is_primary: true,
    version: 3,
    updated_at: "2026-06-04T00:00:00Z"
  },
  roots: []
}

describe("workspace API domain contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses the existing workspace endpoint for workspace upserts", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "ws-1",
      name: "Workspace One",
      archived: false,
      study_materials_policy: "workspace",
      workspace_profile: "research",
      deleted: false,
      banner_title: "Workspace One",
      banner_subtitle: null,
      banner_color: null,
      audio_provider: null,
      audio_model: null,
      audio_voice: null,
      audio_speed: null,
      created_at: "2026-05-06T12:00:00Z",
      last_modified: "2026-05-06T12:00:00Z",
      version: 1
    })

    const response = await workspaceApiMethods.upsertWorkspace("workspace with spaces", {
      name: "Workspace One",
      study_materials_policy: "workspace"
    })

    expect(response.version).toBe(1)
    expect(response.banner_title).toBe("Workspace One")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces",
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: {
          name: "Workspace One",
          study_materials_policy: "workspace"
        }
      })
    )
  })

  it("encodes workspace and nested resource IDs for legacy sub-resource methods", async () => {
    mocks.bgRequest.mockResolvedValue({})

    await workspaceApiMethods.getWorkspaceSources("workspace with spaces")
    await workspaceApiMethods.getWorkspaceSourcesStatus("workspace with spaces")
    await workspaceApiMethods.getWorkspaceCapabilities("workspace with spaces")
    await workspaceApiMethods.updateWorkspaceSource(
      "workspace with spaces",
      "source with spaces",
      { title: "Source", version: 1 }
    )
    await workspaceApiMethods.updateWorkspaceSourceSelection(
      "workspace with spaces",
      ["source with spaces"]
    )
    await workspaceApiMethods.deleteWorkspaceSource(
      "workspace with spaces",
      "source with spaces"
    )
    await workspaceApiMethods.getWorkspaceArtifacts("workspace with spaces")
    await workspaceApiMethods.updateWorkspaceArtifact(
      "workspace with spaces",
      "artifact with spaces",
      {
        title: "Artifact",
        status: "complete",
        content: "Updated body",
        content_type: "text/markdown",
        preview_text: "Updated body",
        summary: "Updated summary",
        review_state: "needs_revision",
        owner_scope: "workspace",
        owner_id: "ws-1",
        project_id: "project-1",
        task_id: "task-1",
        source_collection_id: "collection-1",
        producer_metadata: { producer: "workspace-manager" },
        source_lineage: { source_ids: ["source-1"] },
        review_metadata: { reviewer: "user" },
        version_metadata: { format: "markdown" },
        export_refs: [{ format: "md" }],
        redaction: {
          support_safe: true,
          redacted: false,
          retention_class: "local"
        },
        schema_version: 1,
        total_tokens: 12,
        total_cost_usd: 0.01,
        completed_at: "2026-06-04T00:00:00Z",
        version: 1
      }
    )
    await workspaceApiMethods.deleteWorkspaceArtifact(
      "workspace with spaces",
      "artifact with spaces"
    )
    await workspaceApiMethods.getWorkspaceNotes("workspace with spaces")
    await workspaceApiMethods.addWorkspaceNote("workspace with spaces", {
      title: "Note"
    })
    await workspaceApiMethods.updateWorkspaceNote("workspace with spaces", 42, {
      title: "Note",
      version: 1
    })
    await workspaceApiMethods.deleteWorkspaceNote("workspace with spaces", 42)

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/sources",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/sources/status",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/capabilities",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      4,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/sources/source%20with%20spaces",
        method: "PUT"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      5,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/sources/selection",
        method: "PUT"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      6,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/sources/source%20with%20spaces",
        method: "DELETE"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      7,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/artifacts",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      8,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/artifacts/artifact%20with%20spaces",
        method: "PUT",
        body: expect.objectContaining({
          content_type: "text/markdown",
          review_state: "needs_revision",
          producer_metadata: { producer: "workspace-manager" },
          version: 1
        })
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      9,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/artifacts/artifact%20with%20spaces",
        method: "DELETE"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      10,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/notes",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      11,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/notes",
        method: "POST"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      12,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/notes/42",
        method: "PUT"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      13,
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/notes/42",
        method: "DELETE"
      })
    )
  })

  it("rejects slash-delimited IDs before making segment-routed requests", async () => {
    await expect(
      workspaceApiMethods.deleteWorkspace("workspace/with/slash")
    ).rejects.toThrow("workspaceId")
    await expect(
      workspaceApiMethods.updateWorkspaceSource("ws-1", "source/with/slash", {
        title: "Source",
        version: 1
      })
    ).rejects.toThrow("sourceId")

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("keeps source status urls available from the backend contract", async () => {
    mocks.bgRequest.mockResolvedValue({
      workspace_id: "ws-1",
      sources: [
        {
          id: "source-1",
          workspace_id: "ws-1",
          media_id: 1,
          title: "Source One",
          source_type: "web",
          url: "https://example.test/source",
          selected: true,
          state: "queryable",
          status_reason: "ready",
          readiness: {
            metadata_ready: true,
            text_extracted: true,
            fts_ready: true,
            vector_ready: true,
            citation_ready: true,
            summary_ready: false,
            tool_accessible: true
          },
          progress_percent: null,
          progress_message: null,
          job: null,
          updated_at: "2026-06-04T00:00:00Z"
        }
      ],
      summary: {
        total: 1,
        selected: 1,
        queryable: 1,
        partially_queryable: 0,
        processing: 0,
        failed: 0,
        missing: 0
      }
    })

    const response = await workspaceApiMethods.getWorkspaceSourcesStatus("ws-1")

    expect(response.sources[0]?.url).toBe("https://example.test/source")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/sources/status",
        method: "GET"
      })
    )
  })

  it("uses workspace artifact sub-resource endpoints", async () => {
    mocks.bgRequest.mockResolvedValue({
      id: "artifact-1",
      workspace_id: "ws-1",
      artifact_type: "report",
      title: "Executive Brief",
      status: "draft",
      content: "Brief body",
      content_type: "text/markdown",
      preview_text: "Brief body",
      summary: "A concise brief",
      review_state: "accepted",
      owner_scope: "workspace",
      owner_id: "ws-1",
      project_id: "project-1",
      task_id: "task-1",
      source_collection_id: "collection-1",
      producer_metadata: { producer: "workspace-manager" },
      source_lineage: { source_ids: ["source-1"] },
      review_metadata: { reviewer: "user" },
      version_metadata: { format: "markdown" },
      export_refs: [{ format: "md" }],
      redaction: {
        support_safe: true,
        redacted: false,
        retention_class: "local"
      },
      schema_version: 1,
      total_tokens: 0,
      total_cost_usd: 0,
      created_at: "2026-05-06T12:00:00Z",
      completed_at: null,
      version: 1
    })

    const response = await workspaceApiMethods.addWorkspaceArtifact("ws-1", {
      id: "artifact-1",
      artifact_type: "report",
      title: "Executive Brief",
      status: "draft",
      content: "Brief body",
      content_type: "text/markdown",
      preview_text: "Brief body",
      summary: "A concise brief",
      review_state: "accepted",
      owner_scope: "workspace",
      owner_id: "ws-1",
      project_id: "project-1",
      task_id: "task-1",
      source_collection_id: "collection-1",
      producer_metadata: { producer: "workspace-manager" },
      source_lineage: { source_ids: ["source-1"] },
      review_metadata: { reviewer: "user" },
      version_metadata: { format: "markdown" },
      export_refs: [{ format: "md" }],
      redaction: {
        support_safe: true,
        redacted: false,
        retention_class: "local"
      },
      schema_version: 1
    })

    expect(response.total_tokens).toBe(0)
    expect(response.review_state).toBe("accepted")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/artifacts",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          id: "artifact-1",
          artifact_type: "report",
          title: "Executive Brief",
          status: "draft",
          content: "Brief body",
          content_type: "text/markdown",
          preview_text: "Brief body",
          summary: "A concise brief",
          review_state: "accepted",
          owner_scope: "workspace",
          owner_id: "ws-1",
          project_id: "project-1",
          task_id: "task-1",
          source_collection_id: "collection-1",
          producer_metadata: { producer: "workspace-manager" },
          source_lineage: { source_ids: ["source-1"] },
          review_metadata: { reviewer: "user" },
          version_metadata: { format: "markdown" },
          export_refs: [{ format: "md" }],
          redaction: {
            support_safe: true,
            redacted: false,
            retention_class: "local"
          },
          schema_version: 1
        }
      })
    )
  })

  it("patches workspace metadata through the canonical workspace endpoint", async () => {
    mocks.bgRequest.mockResolvedValue(workspaceResponse)

    const response = await workspaceApiMethods.patchWorkspace("ws-1", {
      name: "Renamed Workspace",
      archived: true,
      workspace_profile: "project",
      version: 1
    })

    expect(response.workspace_profile).toBe("project")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1",
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: {
          name: "Renamed Workspace",
          archived: true,
          workspace_profile: "project",
          version: 1
        }
      })
    )
  })

  it("exposes raw workspace delete API support without manager semantics", async () => {
    mocks.bgRequest.mockResolvedValue(undefined)

    await workspaceApiMethods.deleteWorkspace("workspace with spaces")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces",
        method: "DELETE"
      })
    )
  })

  it("fetches workspace project roots from the canonical roots endpoint", async () => {
    mocks.bgRequest.mockResolvedValue(rootsResponse)

    const response = await workspaceApiMethods.getWorkspaceRoots("ws-1")

    expect(response.primary_root?.file_inventory.available).toBe(true)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/roots",
        method: "GET"
      })
    )
  })

  it("attaches a workspace primary root through the canonical endpoint", async () => {
    mocks.bgRequest.mockResolvedValue(rootsResponse)

    await workspaceApiMethods.attachWorkspacePrimaryRoot("ws-1", {
      backend: "host_local",
      absolute_root: "/repo",
      display_name: "Repo",
      expected_workspace_version: 2
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/roots/primary",
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: {
          backend: "host_local",
          absolute_root: "/repo",
          display_name: "Repo",
          expected_workspace_version: 2
        }
      })
    )
  })

  it("queues workspace file inventory scans through the canonical endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({
      workspace_id: "ws-1",
      root_id: "root-1",
      state: "queued",
      stale: false,
      counts: { total_entries: 0 },
      diagnostics: [],
      job: null,
      updated_at: "2026-06-04T00:00:00Z"
    })

    await workspaceApiMethods.queueWorkspaceFileInventoryScan("ws-1", {
      force: true,
      expected_root_version: 3
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/file-inventory/scan",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          force: true,
          expected_root_version: 3
        }
      })
    )
  })

  it("fetches workspace file inventory status from the canonical endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({
      workspace_id: "ws-1",
      root_id: "root-1",
      state: "current",
      stale: false,
      counts: { files: 1, total_entries: 1 },
      diagnostics: [],
      job: null,
      updated_at: "2026-06-04T00:00:00Z"
    })

    await workspaceApiMethods.getWorkspaceFileInventoryStatus("ws-1")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/file-inventory/status",
        method: "GET"
      })
    )
  })

  it("fetches workspace file inventory items with filters", async () => {
    mocks.bgRequest.mockResolvedValue({
      workspace_id: "ws-1",
      root_id: "root-1",
      items: [
        {
          relative_path: "src/index.ts",
          entry_kind: "file",
          ignored: false,
          indexing_candidate: true
        }
      ],
      next_cursor: null,
      limit: 50
    })

    await workspaceApiMethods.getWorkspaceFileInventoryItems("ws-1", {
      prefix: "src/",
      limit: 50,
      cursor: "cursor-1",
      include_ignored: true,
      entry_kind: "file"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/file-inventory/items?prefix=src%2F&limit=50&cursor=cursor-1&include_ignored=true&entry_kind=file",
        method: "GET"
      })
    )
  })

})
