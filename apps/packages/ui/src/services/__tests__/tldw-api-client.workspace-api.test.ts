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

    const response = await workspaceApiMethods.upsertWorkspace("ws-1", {
      name: "Workspace One",
      study_materials_policy: "workspace"
    })

    expect(response.version).toBe(1)
    expect(response.banner_title).toBe("Workspace One")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/workspaces/ws-1",
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: {
          name: "Workspace One",
          study_materials_policy: "workspace"
        }
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
      content: "Brief body"
    })

    expect(response.total_tokens).toBe(0)
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
          content: "Brief body"
        }
      })
    )
  })
})
