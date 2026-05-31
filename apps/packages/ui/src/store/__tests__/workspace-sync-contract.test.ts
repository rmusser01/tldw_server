import { describe, expect, it } from "vitest"
import {
  WORKSPACE_SYNC_PAYLOAD_VERSION,
  buildWorkspaceSyncPayload,
  isWorkspaceSyncPayload
} from "../workspace-sync-contract"

describe("workspace sync contract", () => {
  it("builds a versioned sync payload with normalized timestamps", () => {
    const payload = buildWorkspaceSyncPayload({
      workspaceId: "workspace-alpha",
      workspaceTag: "workspace:alpha",
      workspaceName: "Alpha Workspace",
      selectedSourceIds: ["source-1"],
      sources: [
        {
          id: "source-1",
          mediaId: 101,
          title: "Source One",
          type: "pdf",
          addedAt: new Date("2026-02-18T10:00:00.000Z")
        }
      ],
      generatedArtifacts: [
        {
          id: "artifact-1",
          type: "summary",
          title: "Summary",
          status: "completed",
          reviewStatus: "draft",
          content: "Summary content",
          totalTokens: 210,
          totalCostUsd: 0.042,
          createdAt: new Date("2026-02-18T10:05:00.000Z"),
          completedAt: new Date("2026-02-18T10:06:00.000Z")
        }
      ],
      currentNote: {
        title: "Alpha note",
        content: "Key findings",
        keywords: ["alpha"],
        isDirty: false
      },
      updatedAt: new Date("2026-02-18T10:10:00.000Z")
    })

    expect(payload.version).toBe(WORKSPACE_SYNC_PAYLOAD_VERSION)
    expect(payload.updatedAt).toBe("2026-02-18T10:10:00.000Z")
    expect(payload.snapshot.sources[0]?.addedAt).toBe("2026-02-18T10:00:00.000Z")
    expect(payload.snapshot.generatedArtifacts[0]?.completedAt).toBe(
      "2026-02-18T10:06:00.000Z"
    )
    expect(payload.snapshot.generatedArtifacts[0]?.reviewStatus).toBe("draft")
  })

  it("accepts valid payloads and rejects incompatible payloads", () => {
    const validPayload = buildWorkspaceSyncPayload({
      workspaceId: "workspace-beta",
      workspaceTag: "workspace:beta",
      workspaceName: "Beta Workspace",
      selectedSourceIds: [],
      sources: [],
      generatedArtifacts: [],
      currentNote: {
        title: "",
        content: "",
        keywords: [],
        isDirty: false
      }
    })

    expect(isWorkspaceSyncPayload(validPayload)).toBe(true)

    expect(
      isWorkspaceSyncPayload({
        ...validPayload,
        version: 99
      })
    ).toBe(false)

    expect(
      isWorkspaceSyncPayload({
        ...validPayload,
        snapshot: {
          ...validPayload.snapshot,
          currentNote: {
            ...validPayload.snapshot.currentNote,
            isDirty: "yes"
          }
        }
      })
    ).toBe(false)
  })
})
