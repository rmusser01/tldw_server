import { describe, it, expect, beforeEach } from "vitest"
import {
  buildResearchWorkspaceMigrationPlan,
  buildResearchWorkspaceMigrationTombstone,
  buildResearchWorkspaceMigrationTombstoneKey
} from "@/store/workspace-migration"

describe("Research Workspace migration manifest planning", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it("ignores the obsolete workspace_migrated flag as migration proof", async () => {
    localStorage.setItem("workspace_migrated", "true")

    const plan = await buildResearchWorkspaceMigrationPlan({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () =>
        JSON.stringify({ workspaces: [{ id: "ws-1", name: "Workspace One" }] })
    })

    expect(plan.declaredChunks).toHaveLength(1)
    expect(plan.manifestHash).toHaveLength(64)
    expect(plan.localDeletionEligibility.eligible).toBe(true)
    expect(localStorage.getItem("workspace_migrated")).toBe("true")
  })

  it("does not write the obsolete workspace_migrated flag while planning", async () => {
    await buildResearchWorkspaceMigrationPlan({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: [],
      readLocalStorageValue: async () => null
    })

    expect(localStorage.getItem("workspace_migrated")).toBeNull()
  })

  it("builds deterministic chunk declarations and manifest coverage for known local content surfaces", async () => {
    const values: Record<string, string> = {
      "tldw-workspace": JSON.stringify({ activeWorkspaceId: "ws-1" }),
      "tldw-workspace:workspace:ws-1:snapshot": JSON.stringify({
        sources: [{ id: "source-1", title: "Captured PDF" }]
      }),
      "tldw-workspace:workspace:ws-1:chat": JSON.stringify({
        messages: [{ role: "user", content: "Question" }]
      })
    }

    const first = await buildResearchWorkspaceMigrationPlan({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: Object.keys(values),
      readLocalStorageValue: async (key) => values[key] ?? null
    })
    const second = await buildResearchWorkspaceMigrationPlan({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: Object.keys(values),
      readLocalStorageValue: async (key) => values[key] ?? null
    })

    expect(first.migrationId).toBe(second.migrationId)
    expect(first.idempotencyKey).toBe(second.idempotencyKey)
    expect(first.manifestHash).toBe(second.manifestHash)
    expect(first.declaredChunks).toHaveLength(3)
    expect(first.declaredChunks.map((chunk) => chunk.byte_count)).toEqual(
      Object.values(values).map((value) => new TextEncoder().encode(value).byteLength)
    )
    expect(first.manifest.covered_surface_ids).toEqual([
      "localStorage:tldw-workspace",
      "localStorage:tldw-workspace:workspace:ws-1:snapshot",
      "localStorage:tldw-workspace:workspace:ws-1:chat"
    ])
    expect(first.localDeletionEligibility.eligible).toBe(true)
  })

  it("blocks local deletion when unknown workspace-prefixed storage is discovered", async () => {
    const plan = await buildResearchWorkspaceMigrationPlan({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: [
        "tldw-workspace:workspace:ws-1:unmapped-content"
      ],
      readLocalStorageValue: async () => "{}"
    })

    expect(plan.declaredChunks).toEqual([])
    expect(plan.localDeletionEligibility.eligible).toBe(false)
    expect(plan.localDeletionEligibility.unknownSurfaces).toEqual([
      expect.objectContaining({
        id: "unknown:localStorage:tldw-workspace:workspace:ws-1:unmapped-content"
      })
    ])
  })

  it("creates non-content tombstone keys and payloads", () => {
    expect(buildResearchWorkspaceMigrationTombstoneKey("legacy ws")).toBe(
      "tldw:research-workspace:migration:tombstone:legacy%20ws"
    )

    expect(
      buildResearchWorkspaceMigrationTombstone({
        legacyWorkspaceId: "legacy ws",
        serverWorkspaceId: "ws-1",
        migrationId: "mig-1",
        deletedAt: "2026-05-26T00:00:00Z"
      })
    ).toEqual({
      legacyWorkspaceId: "legacy ws",
      serverWorkspaceId: "ws-1",
      migrationId: "mig-1",
      deletedAt: "2026-05-26T00:00:00Z",
      contentRetained: false
    })
  })
})
