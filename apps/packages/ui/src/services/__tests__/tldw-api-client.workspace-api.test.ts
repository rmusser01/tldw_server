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

const workspaceAssistantDefaultsResponse = {
  ...workspaceResponse,
  assistant_defaults: {
    assistant_kind: "persona",
    assistant_id: "persona-1",
    persona_memory_mode: "read_only",
    voice: null,
    style: null,
    tool_policy_profile_id: null
  },
  effective_assistant_default: {
    status: "available",
    source: "workspace",
    assistant_kind: "persona",
    assistant_id: "persona-1",
    label: "Literature Review Assistant",
    persona_memory_mode: "read_only",
    degraded_reason: null
  }
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

  it("lists workspaces for destination pickers", async () => {
    mocks.bgRequest.mockResolvedValue({
      items: [
        {
          id: "workspace-alpha",
          name: "Research Workspace",
          archived: false,
          study_materials_policy: "workspace",
          deleted: false,
          banner_title: "Research Workspace",
          banner_subtitle: null,
          banner_color: null,
          audio_provider: null,
          audio_model: null,
          audio_voice: null,
          audio_speed: null,
          created_at: "2026-05-06T12:00:00Z",
          last_modified: "2026-05-06T12:00:00Z",
          version: 1
        }
      ],
      total: 1
    })

    const response = await workspaceApiMethods.listWorkspaces()

    expect(response.items).toHaveLength(1)
    expect(response.items[0]?.id).toBe("workspace-alpha")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces",
        method: "GET"
      })
    )
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

  it("updates workspace source review state through the batch endpoint", async () => {
    mocks.bgRequest.mockResolvedValue([])

    await workspaceApiMethods.updateWorkspaceSourceReviewState(
      "workspace with spaces",
      [" source with spaces "],
      "reviewed"
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/workspace%20with%20spaces/sources/review-state",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        source_ids: [" source with spaces "],
        review_state: "reviewed"
      }
    })
  })

  it("rejects slash-delimited IDs before making segment-routed requests", async () => {
    await expect(
      workspaceApiMethods.deleteWorkspace("workspace/with/slash")
    ).rejects.toThrow("workspaceId")
    await expect(
      workspaceApiMethods.getWorkspaceOperation("ws-1", "operation/with/slash")
    ).rejects.toThrow("operationId")
    await expect(
      workspaceApiMethods.updateWorkspaceSource("ws-1", "source/with/slash", {
        title: "Source",
        version: 1
      })
    ).rejects.toThrow("sourceId")

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("rejects blank IDs before making segment-routed requests", async () => {
    await expect(workspaceApiMethods.deleteWorkspace("   ")).rejects.toThrow(
      "workspaceId"
    )
    await expect(
      workspaceApiMethods.getWorkspaceOperation("ws-1", " ")
    ).rejects.toThrow("operationId")

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

  it("maps workspace assistant defaults response fields and serializes patch payloads", async () => {
    mocks.bgRequest.mockResolvedValue(workspaceAssistantDefaultsResponse)

    const response = await workspaceApiMethods.patchWorkspace("ws-1", {
      version: 2,
      assistantDefaults: {
        assistantKind: "persona",
        assistantId: "persona-2",
        personaMemoryMode: "read_write",
        voice: null,
        style: null,
        toolPolicyProfileId: null
      },
      confirmReadWriteAssistantDefault: true
    })

    expect(response.assistantDefaults).toEqual({
      assistantKind: "persona",
      assistantId: "persona-1",
      personaMemoryMode: "read_only",
      voice: null,
      style: null,
      toolPolicyProfileId: null
    })
    expect(response.effectiveAssistantDefault).toEqual({
      status: "available",
      source: "workspace",
      assistantKind: "persona",
      assistantId: "persona-1",
      label: "Literature Review Assistant",
      personaMemoryMode: "read_only",
      degradedReason: null
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1",
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: {
          version: 2,
          assistant_defaults: {
            assistant_kind: "persona",
            assistant_id: "persona-2",
            persona_memory_mode: "read_write",
            voice: null,
            style: null,
            tool_policy_profile_id: null
          },
          confirm_read_write_assistant_default: true
        }
      })
    )
  })

  it("rejects malformed assistant defaults patch payloads before sending", async () => {
    await expect(
      workspaceApiMethods.patchWorkspace("ws-1", {
        version: 2,
        assistantDefaults: {
          assistantKind: "persona",
          assistantId: "   ",
          personaMemoryMode: "read_only",
          voice: null,
          style: null,
          toolPolicyProfileId: null
        } as any
      })
    ).rejects.toThrow(/assistant_defaults/)

    expect(mocks.bgRequest).not.toHaveBeenCalled()
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

  it("provisions sandbox-managed workspace roots with an idempotency key", async () => {
    mocks.bgRequest.mockResolvedValue({
      workspace_id: "ws-1",
      workspace_profile: "project",
      operation: {
        operation_id: "op-1",
        workspace_id: "ws-1",
        command: "provision_sandbox_root",
        status: "running",
        started_at: "2026-06-04T00:00:00Z",
        updated_at: "2026-06-04T00:00:00Z",
        retryable: false,
        diagnostics: {},
        poll_href: "/api/v1/workspaces/ws-1/operations/op-1"
      },
      primary_root: rootsResponse.primary_root
    })

    await workspaceApiMethods.provisionWorkspaceSandboxRoot(
      "workspace with spaces",
      {
        display_name: "Project sandbox",
        requested_runtime: "python",
        expected_workspace_version: 2
      },
      "sandbox-root-idem-1"
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/workspace%20with%20spaces/roots/primary/sandbox-volume",
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": "sandbox-root-idem-1"
        },
        body: {
          display_name: "Project sandbox",
          requested_runtime: "python",
          expected_workspace_version: 2
        }
      })
    )
  })

  it("fetches workspace operation status from the canonical operation endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({
      operation_id: "operation with spaces",
      workspace_id: "ws-1",
      command: "provision_sandbox_root",
      status: "running",
      started_at: "2026-06-04T00:00:00Z",
      updated_at: "2026-06-04T00:00:00Z",
      retryable: false,
      diagnostics: {},
      poll_href: "/api/v1/workspaces/ws-1/operations/operation%20with%20spaces"
    })

    await workspaceApiMethods.getWorkspaceOperation(
      "ws-1",
      "operation with spaces"
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1/operations/operation%20with%20spaces",
        method: "GET"
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

  it("lists source saved views through the encoded workspace envelope", async () => {
    const invalidView = {
      id: "view-1",
      workspace_id: "workspace with spaces",
      name: "Needs review",
      schema_version: 2,
      version: 3,
      created_at: "2026-07-10T00:00:00Z",
      updated_at: "2026-07-10T01:00:00Z",
      state: null,
      valid: false,
      invalid_reason: "unsupported_schema_version"
    } as const
    mocks.bgRequest.mockResolvedValue({ items: [invalidView] })

    const response = await workspaceApiMethods.listWorkspaceSourceViews(
      "workspace with spaces"
    )

    expect(response).toEqual({ items: [invalidView] })
    expect(response.items[0]?.valid).toBe(false)
    expect(response.items[0]?.invalid_reason).toBe("unsupported_schema_version")
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/workspace%20with%20spaces/source-views",
      method: "GET",
      abortSignal: expect.any(AbortSignal)
    })
  })

  it("creates and updates source saved views with exact canonical bodies", async () => {
    const state = {
      type_filters: ["pdf"],
      status_filters: [],
      review_state_filters: ["needs_review"],
      lifecycle_state_filters: [],
      date_field: "added_at",
      date_from: null,
      date_to: null,
      require_url: false,
      require_file_size: false,
      require_duration: false,
      require_page_count: false,
      file_size_min: null,
      file_size_max: null,
      duration_min: null,
      duration_max: null,
      page_count_min: null,
      page_count_max: null,
      sort: "name_asc"
    } as const
    const response = {
      id: "view with spaces",
      workspace_id: "workspace with spaces",
      name: "Review PDFs",
      schema_version: 1,
      version: 1,
      created_at: "2026-07-10T00:00:00Z",
      updated_at: "2026-07-10T00:00:00Z",
      state,
      valid: true,
      invalid_reason: null
    } as const
    mocks.bgRequest.mockResolvedValue(response)

    expect(
      await workspaceApiMethods.createWorkspaceSourceView(
        "workspace with spaces",
        { name: "Review PDFs", schema_version: 1, state }
      )
    ).toEqual(response)
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/workspaces/workspace%20with%20spaces/source-views",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { name: "Review PDFs", schema_version: 1, state },
      expectedStatuses: [409]
    })

    await workspaceApiMethods.updateWorkspaceSourceView(
      "workspace with spaces",
      "view with spaces",
      {
        version: 2,
        name: "Review PDFs",
        schema_version: 1,
        state
      }
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/workspaces/workspace%20with%20spaces/source-views/view%20with%20spaces",
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: {
        version: 2,
        name: "Review PDFs",
        schema_version: 1,
        state
      },
      expectedStatuses: [404, 409]
    })
  })

  it("deletes source saved views without a request body", async () => {
    mocks.bgRequest.mockResolvedValue(undefined)

    await workspaceApiMethods.deleteWorkspaceSourceView(
      "workspace with spaces",
      "view with spaces"
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/workspace%20with%20spaces/source-views/view%20with%20spaces",
      method: "DELETE",
      expectedStatuses: [404]
    })
  })

  it.each(["", "   ", "view/child", " / "])(
    "rejects invalid source saved view id %j before requesting",
    async (viewId) => {
      await expect(
        workspaceApiMethods.updateWorkspaceSourceView("ws-1", viewId, {
          version: 1,
          name: "Name"
        })
      ).rejects.toThrow(/viewId/)
      await expect(
        workspaceApiMethods.deleteWorkspaceSourceView("ws-1", viewId)
      ).rejects.toThrow(/viewId/)
      expect(mocks.bgRequest).not.toHaveBeenCalled()
    }
  )
})
