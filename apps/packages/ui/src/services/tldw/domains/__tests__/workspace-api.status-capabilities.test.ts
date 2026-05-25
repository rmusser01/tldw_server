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
})
