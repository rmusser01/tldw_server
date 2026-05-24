import { describe, expect, it, vi } from "vitest"
import {
  buildResearchWorkspaceServerSourceSignature,
  reconcileResearchWorkspaceServerState
} from "../workspace-server-reconcile"
import type { WorkspaceSource } from "@/types/workspace"

const makeSource = (
  overrides: Partial<WorkspaceSource> = {}
): WorkspaceSource => ({
  id: "source-1",
  mediaId: 101,
  title: "Source 1",
  type: "pdf",
  addedAt: new Date("2026-05-23T12:00:00.000Z"),
  ...overrides
})

const makeClient = (existingSources: Array<Record<string, unknown>> = []) => ({
  upsertWorkspace: vi.fn().mockResolvedValue({
    id: "workspace-1",
    name: "Notebook import",
    study_materials_policy: "workspace"
  }),
  getWorkspaceSources: vi.fn().mockResolvedValue(existingSources),
  addWorkspaceSource: vi.fn().mockImplementation(
    async (_workspaceId: string, source: Record<string, unknown>) => ({
      ...source,
      workspace_id: "workspace-1",
      added_at: "2026-05-23T12:00:00Z",
      version: 1
    })
  )
})

describe("research workspace server reconciliation", () => {
  it("upserts the workspace and adds missing local sources with valid media ids", async () => {
    const client = makeClient()
    const sources = [
      makeSource({
        id: "source-ready",
        mediaId: 101,
        title: "Ready Source",
        type: "pdf",
        url: "https://example.test/ready.pdf"
      }),
      makeSource({
        id: "source-web",
        mediaId: 102,
        title: "Web Source",
        type: "website",
        url: "https://example.test/web"
      })
    ]

    const result = await reconcileResearchWorkspaceServerState({
      client,
      workspaceId: "workspace-1",
      workspaceName: "Notebook import",
      sources
    })

    expect(client.upsertWorkspace).toHaveBeenCalledWith("workspace-1", {
      name: "Notebook import",
      study_materials_policy: "workspace"
    })
    expect(client.getWorkspaceSources).toHaveBeenCalledWith("workspace-1")
    expect(client.addWorkspaceSource).toHaveBeenNthCalledWith(1, "workspace-1", {
      id: "source-ready",
      media_id: 101,
      title: "Ready Source",
      source_type: "pdf",
      url: "https://example.test/ready.pdf",
      position: 0,
      selected: true
    })
    expect(client.addWorkspaceSource).toHaveBeenNthCalledWith(2, "workspace-1", {
      id: "source-web",
      media_id: 102,
      title: "Web Source",
      source_type: "website",
      url: "https://example.test/web",
      position: 1,
      selected: true
    })
    expect(result).toMatchObject({
      workspaceReady: true,
      sourceRowsChecked: true,
      addedSourceIds: ["source-ready", "source-web"],
      skippedSourceIds: [],
      errors: []
    })
  })

  it("skips sources already present by source id or media id and ignores invalid media ids", async () => {
    const client = makeClient([
      {
        id: "source-existing-id",
        media_id: 201
      },
      {
        id: "source-existing-media",
        media_id: 202
      }
    ])
    const sources = [
      makeSource({ id: "source-existing-id", mediaId: 999 }),
      makeSource({ id: "source-new-duplicate-media", mediaId: 202 }),
      makeSource({ id: "source-negative-media", mediaId: -1 }),
      makeSource({ id: "source-nan-media", mediaId: Number.NaN }),
      makeSource({ id: "source-new", mediaId: 203 })
    ]

    const result = await reconcileResearchWorkspaceServerState({
      client,
      workspaceId: "workspace-1",
      workspaceName: "Notebook import",
      sources
    })

    expect(client.addWorkspaceSource).toHaveBeenCalledTimes(1)
    expect(client.addWorkspaceSource).toHaveBeenCalledWith(
      "workspace-1",
      expect.objectContaining({
        id: "source-new",
        media_id: 203,
        position: 4
      })
    )
    expect(result.addedSourceIds).toEqual(["source-new"])
    expect(result.skippedSourceIds).toEqual([
      "source-existing-id",
      "source-new-duplicate-media",
      "source-negative-media",
      "source-nan-media"
    ])
    expect(result.errors).toEqual([])
  })

  it("reports source add failures and continues adding later sources", async () => {
    const client = makeClient()
    client.addWorkspaceSource
      .mockRejectedValueOnce(new Error("duplicate source row"))
      .mockResolvedValueOnce({
        id: "source-later",
        media_id: 302,
        workspace_id: "workspace-1",
        added_at: "2026-05-23T12:00:00Z",
        version: 1
      })

    const result = await reconcileResearchWorkspaceServerState({
      client,
      workspaceId: "workspace-1",
      workspaceName: "Notebook import",
      sources: [
        makeSource({ id: "source-fails", mediaId: 301 }),
        makeSource({ id: "source-later", mediaId: 302 })
      ]
    })

    expect(client.addWorkspaceSource).toHaveBeenCalledTimes(2)
    expect(result.workspaceReady).toBe(true)
    expect(result.sourceRowsChecked).toBe(true)
    expect(result.addedSourceIds).toEqual(["source-later"])
    expect(result.errors).toEqual([
      "Failed to add source source-fails: duplicate source row"
    ])
  })

  it("bounds repeated source add errors while continuing later source attempts", async () => {
    const client = makeClient()
    client.addWorkspaceSource.mockRejectedValue(new Error("source add unavailable"))
    const sources = Array.from({ length: 8 }, (_, index) =>
      makeSource({ id: `source-${index}`, mediaId: 400 + index })
    )

    const result = await reconcileResearchWorkspaceServerState({
      client,
      workspaceId: "workspace-1",
      workspaceName: "Notebook import",
      sources
    })

    expect(client.addWorkspaceSource).toHaveBeenCalledTimes(8)
    expect(result.errors).toHaveLength(5)
    expect(result.errors[0]).toBe(
      "Failed to add source source-0: source add unavailable"
    )
    expect(result.errors.at(-1)).toBe("Additional workspace sync errors omitted.")
  })

  it("builds a stable signature from source identity fields only", () => {
    const first = buildResearchWorkspaceServerSourceSignature([
      makeSource({ id: "source-a", mediaId: 1, title: "A", type: "pdf" }),
      makeSource({ id: "source-b", mediaId: 2, title: "B", type: "website" })
    ])
    const second = buildResearchWorkspaceServerSourceSignature([
      makeSource({
        id: "source-a",
        mediaId: 1,
        title: "A",
        type: "pdf",
        status: "error",
        statusMessage: "Changed status"
      }),
      makeSource({
        id: "source-b",
        mediaId: 2,
        title: "B",
        type: "website",
        status: "ready"
      })
    ])

    expect(second).toBe(first)
  })
})
