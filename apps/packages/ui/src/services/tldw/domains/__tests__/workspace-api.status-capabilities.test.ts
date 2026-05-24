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
})
