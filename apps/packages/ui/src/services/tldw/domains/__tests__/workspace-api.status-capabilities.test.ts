import { beforeEach, describe, expect, it, vi } from "vitest"
import { bgRequest } from "@/services/background-proxy"
import { workspaceApiMethods } from "../workspace-api"

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn()
}))

describe("workspace API status and capabilities methods", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("fetches workspace source status from the authoritative endpoint", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      workspace_id: "ws-1",
      sources: [],
      summary: {
        total: 0,
        selected: 0,
        queryable: 0,
        partially_queryable: 0,
        processing: 0,
        failed: 0,
        missing: 0
      }
    })

    await workspaceApiMethods.getWorkspaceSourcesStatus("ws-1")

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/ws-1/sources/status",
      method: "GET"
    })
  })

  it("fetches workspace capability gates from the authoritative endpoint", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      workspace_id: "ws-1",
      workspace_kind: "research_workspace",
      access_level: "owner",
      source_summary: {
        total: 0,
        selected: 0,
        queryable: 0,
        partially_queryable: 0,
        processing: 0,
        failed: 0,
        missing: 0
      },
      workspace_services: {},
      allowed_actions: {}
    })

    await workspaceApiMethods.getWorkspaceCapabilities("ws-1")

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/ws-1/capabilities",
      method: "GET"
    })
  })

  it("fetches workspace context from the canonical page envelope endpoint", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      workspace_id: "ws-1",
      workspace_kind: "research_workspace",
      schema_version: 1,
      generated_at: "2026-05-25T00:00:00Z",
      workspace: {},
      sources: { items: [], summary: {} },
      capabilities: {},
      services: {},
      allowed_actions: {},
      active_jobs: [],
      partial_errors: []
    })

    await workspaceApiMethods.getWorkspaceContext("ws-1")

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/ws-1/context",
      method: "GET"
    })
  })

  it("fetches bounded workspace source preview detail", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      workspace_id: "ws-1",
      source_id: "src-1",
      media_id: 1,
      title: "Source",
      source_type: "pdf",
      state: "queryable",
      status_reason: "source_queryable",
      readiness: {},
      content_available: true,
      preview_mode: "available",
      text_preview: "Captured text",
      text_total_chars: 13,
      text_truncated: false,
      snippets: [],
      generated_at: "2026-05-25T00:00:00Z"
    })

    await workspaceApiMethods.getWorkspaceSourcePreview("ws-1", "src-1", {
      max_chars: 1200,
      chunk_limit: 2
    })

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/ws-1/sources/src-1/preview?max_chars=1200&chunk_limit=2",
      method: "GET"
    })
  })

  it("encodes workspace and source path parameters for source preview", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      workspace_id: "workspace with space",
      source_id: "source/with/slash",
      media_id: 1,
      title: "Source",
      source_type: "pdf",
      state: "queryable",
      status_reason: "source_queryable",
      readiness: {},
      content_available: true,
      preview_mode: "available",
      text_preview: "Captured text",
      text_total_chars: 13,
      text_truncated: false,
      snippets: [],
      generated_at: "2026-05-25T00:00:00Z"
    })

    await workspaceApiMethods.getWorkspaceSourcePreview(
      "workspace with space",
      "source/with/slash"
    )

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/workspace%20with%20space/sources/source%2Fwith%2Fslash/preview",
      method: "GET"
    })
  })

  it("creates a Research Workspace migration session through the canonical protocol endpoint", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      id: "mig-1",
      idempotency_key: "mig-1:aaaaaaaa",
      target_workspace_id: "ws-1",
      target_workspace_name: "Workspace",
      source_product: "research-workspace-webui",
      manifest_hash: "a".repeat(64),
      status: "created",
      declared_chunk_count: 0,
      accepted_chunk_count: 0,
      missing_chunk_ids: [],
      client_delete_eligible: false,
      created_at: "2026-05-26T00:00:00Z",
      updated_at: "2026-05-26T00:00:00Z",
      finalized_at: null,
      recovery_manifest: {},
      chunks: []
    })

    await workspaceApiMethods.createWorkspaceMigration({
      id: "mig-1",
      idempotency_key: "mig-1:aaaaaaaa",
      target_workspace_id: "ws-1",
      target_workspace_name: "Workspace",
      source_product: "research-workspace-webui",
      manifest_hash: "a".repeat(64),
      declared_chunks: [],
      manifest: {},
      diagnostics: {}
    })

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/migrations",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        id: "mig-1",
        idempotency_key: "mig-1:aaaaaaaa",
        target_workspace_id: "ws-1",
        target_workspace_name: "Workspace",
        source_product: "research-workspace-webui",
        manifest_hash: "a".repeat(64),
        declared_chunks: [],
        manifest: {},
        diagnostics: {}
      }
    })
  })

  it("records Research Workspace migration chunk receipts with encoded path ids", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({
      id: "chunk/1",
      migration_id: "mig/1",
      sha256: "b".repeat(64),
      byte_count: 12,
      chunk_kind: "workspace_bundle",
      metadata: {},
      status: "accepted",
      accepted_at: "2026-05-26T00:00:00Z"
    })

    await workspaceApiMethods.putWorkspaceMigrationChunk("mig/1", "chunk/1", {
      sha256: "b".repeat(64),
      byte_count: 12,
      chunk_kind: "workspace_bundle",
      metadata: {}
    })

    expect(bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/workspaces/migrations/mig%2F1/chunks/chunk%2F1",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: {
        sha256: "b".repeat(64),
        byte_count: 12,
        chunk_kind: "workspace_bundle",
        metadata: {}
      }
    })
  })

  it("finalizes, fetches, and acknowledges Research Workspace migration sessions", async () => {
    vi.mocked(bgRequest)
      .mockResolvedValueOnce({
        id: "mig-1",
        status: "finalized",
        client_delete_eligible: false,
        chunks: []
      })
      .mockResolvedValueOnce({
        id: "mig-1",
        status: "finalized",
        client_delete_eligible: false,
        chunks: []
      })
      .mockResolvedValueOnce({ ok: true })

    await workspaceApiMethods.finalizeWorkspaceMigration("mig-1", {
      manifest_hash: "a".repeat(64)
    })
    await workspaceApiMethods.getWorkspaceMigration("mig-1")
    await workspaceApiMethods.ackWorkspaceMigrationClientDelete("mig-1", {
      acknowledged_manifest_hash: "a".repeat(64)
    })

    expect(bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/workspaces/migrations/mig-1/finalize",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { manifest_hash: "a".repeat(64) }
    })
    expect(bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/workspaces/migrations/mig-1",
      method: "GET"
    })
    expect(bgRequest).toHaveBeenNthCalledWith(3, {
      path: "/api/v1/workspaces/migrations/mig-1/client-delete-ack",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { acknowledged_manifest_hash: "a".repeat(64) }
    })
  })
})
