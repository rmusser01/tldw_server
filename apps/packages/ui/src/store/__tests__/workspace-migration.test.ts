import { describe, it, expect, beforeEach, vi } from "vitest"
import {
  buildResearchWorkspaceMigrationPlan,
  buildResearchWorkspaceMigrationTombstone,
  buildResearchWorkspaceMigrationTombstoneKey,
  runResearchWorkspaceMigration
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
        workspaceId: "ws-1",
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

  it("finalizes server migration but retains local content when server deletion eligibility is false", async () => {
    const createWorkspaceMigration = vi.fn(async (body) => ({
      ...body,
      status: "created",
      declared_chunk_count: body.declared_chunks.length,
      accepted_chunk_count: 0,
      missing_chunk_ids: body.declared_chunks.map((chunk: { id: string }) => chunk.id),
      client_delete_eligible: false,
      created_at: "2026-05-26T00:00:00Z",
      updated_at: "2026-05-26T00:00:00Z",
      finalized_at: null,
      recovery_manifest: {},
      chunks: []
    }))
    const putWorkspaceMigrationChunk = vi.fn(async () => ({
      id: "chunk-1",
      migration_id: "mig-1",
      sha256: "b".repeat(64),
      byte_count: 2,
      chunk_kind: "workspace_bundle",
      metadata: {},
      status: "accepted",
      accepted_at: "2026-05-26T00:00:00Z"
    }))
    const finalizeWorkspaceMigration = vi.fn(async () => ({
      id: "mig-1",
      status: "finalized",
      client_delete_eligible: false,
      chunks: []
    }))
    const getWorkspaceMigration = vi.fn(async () => ({
      id: "mig-1",
      status: "finalized",
      client_delete_eligible: false,
      chunks: []
    }))
    const compareAndDeleteLocalStorageValue = vi.fn(() => true)

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => JSON.stringify({ workspaceId: "ws-1" }),
      readLocalStorageValueSync: () => null,
      api: {
        createWorkspaceMigration,
        putWorkspaceMigrationChunk,
        finalizeWorkspaceMigration,
        getWorkspaceMigration,
        ackWorkspaceMigrationClientDelete: vi.fn()
      },
      compareAndDeleteLocalStorageValue,
      writeLocalStorageValue: vi.fn()
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(createWorkspaceMigration).toHaveBeenCalledOnce()
    expect(putWorkspaceMigrationChunk).toHaveBeenCalledOnce()
    expect(finalizeWorkspaceMigration).toHaveBeenCalledOnce()
    expect(getWorkspaceMigration).toHaveBeenCalledOnce()
    expect(compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("retains writable copies without markers or acknowledgement even when both receipt gates allow deletion", async () => {
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))
    const compareAndDeleteLocalStorageValue = vi.fn(() => true)
    const writeLocalStorageValue = vi.fn()

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      legacyWorkspaceId: "legacy-ws",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => JSON.stringify({ workspaceId: "ws-1" }),
      readLocalStorageValueSync: () => null,
      api: {
        createWorkspaceMigration: vi.fn(async (body) => ({
          ...body,
          status: "created",
          declared_chunk_count: body.declared_chunks.length,
          accepted_chunk_count: 0,
          missing_chunk_ids: [],
          client_delete_eligible: false,
          created_at: "2026-05-26T00:00:00Z",
          updated_at: "2026-05-26T00:00:00Z",
          finalized_at: null,
          recovery_manifest: {},
          chunks: []
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z"
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        ackWorkspaceMigrationClientDelete
      },
      compareAndDeleteLocalStorageValue,
      writeLocalStorageValue,
      now: () => "2026-05-26T00:00:00Z"
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(result.deletedSurfaceIds).toEqual([])
    expect(result.serverMigration?.client_delete_eligible).toBe(true)
    expect(result.message).toMatch(/automatic.*cleanup.*disabled/i)
    expect(compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("does not require a tombstone writer to retain remotely covered writable copies", async () => {
    const compareAndDeleteLocalStorageValue = vi.fn(() => true)
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => JSON.stringify({ workspaceId: "ws-1" }),
      readLocalStorageValueSync: () => null,
      api: {
        createWorkspaceMigration: vi.fn(async (body) => ({
          ...body,
          status: "created",
          declared_chunk_count: body.declared_chunks.length,
          accepted_chunk_count: 0,
          missing_chunk_ids: [],
          client_delete_eligible: false,
          created_at: "2026-05-26T00:00:00Z",
          updated_at: "2026-05-26T00:00:00Z",
          finalized_at: null,
          recovery_manifest: {},
          chunks: []
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z"
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        ackWorkspaceMigrationClientDelete
      },
      compareAndDeleteLocalStorageValue
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains shared IndexedDB stores without invoking any cleanup dependencies", async () => {
    const compareAndDeleteLocalStorageValue = vi.fn(() => true)
    const writeLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      discoveredIndexedDbStores: [
        {
          databaseName: "tldw-workspace-storage",
          storeName: "workspace-artifact-payloads"
        }
      ],
      readLocalStorageValue: async () => JSON.stringify({ workspaceId: "ws-1" }),
      readLocalStorageValueSync: () => null,
      readIndexedDbStorePayload: async () => "{\"artifact\":\"payload\"}",
      api: {
        createWorkspaceMigration: vi.fn(async (body) => ({
          ...body,
          status: "created",
          declared_chunk_count: body.declared_chunks.length,
          accepted_chunk_count: 0,
          missing_chunk_ids: [],
          client_delete_eligible: false,
          created_at: "2026-05-26T00:00:00Z",
          updated_at: "2026-05-26T00:00:00Z",
          finalized_at: null,
          recovery_manifest: {},
          chunks: []
        })),
        putWorkspaceMigrationChunk: vi.fn(async (migrationId, chunkId) => ({
          id: chunkId,
          migration_id: migrationId,
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z"
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        ackWorkspaceMigrationClientDelete
      },
      compareAndDeleteLocalStorageValue,
      writeLocalStorageValue
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("returns a failed state and retains local content when the migration API fails", async () => {
    const compareAndDeleteLocalStorageValue = vi.fn(() => true)

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => JSON.stringify({ workspaceId: "ws-1" }),
      readLocalStorageValueSync: () => null,
      api: {
        createWorkspaceMigration: vi.fn(async () => {
          throw new Error("conflict")
        }),
        putWorkspaceMigrationChunk: vi.fn(),
        finalizeWorkspaceMigration: vi.fn(),
        getWorkspaceMigration: vi.fn(),
        ackWorkspaceMigrationClientDelete: vi.fn()
      },
      compareAndDeleteLocalStorageValue,
      writeLocalStorageValue: vi.fn()
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("failed")
    expect(result.migrationId).toMatch(/^research-workspace-ws-1-/)
    expect(result.manifestHash).toHaveLength(64)
    expect(result.localDeletionEligibility?.eligible).toBe(true)
    expect(result.deletedSurfaceIds).toEqual([])
    expect(compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("never invokes the former destructive writer even when it would throw a quota error", async () => {
    const compareAndDeleteLocalStorageValue = vi.fn(() => true)
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => JSON.stringify({ workspaceId: "ws-1" }),
      readLocalStorageValueSync: () => null,
      api: {
        createWorkspaceMigration: vi.fn(async (body) => ({
          ...body,
          status: "created",
          declared_chunk_count: body.declared_chunks.length,
          accepted_chunk_count: 0,
          missing_chunk_ids: [],
          client_delete_eligible: false,
          created_at: "2026-05-26T00:00:00Z",
          updated_at: "2026-05-26T00:00:00Z",
          finalized_at: null,
          recovery_manifest: {},
          chunks: []
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z"
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        ackWorkspaceMigrationClientDelete
      },
      compareAndDeleteLocalStorageValue,
      writeLocalStorageValue: vi.fn(async () => {
        throw new Error("local-storage-quota")
      })
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(compareAndDeleteLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("rejects shared IndexedDB payloads without tombstones or whole-store deletion", async () => {
    const writeLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const input = {
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: [],
      discoveredIndexedDbStores: [
        {
          databaseName: "tldw-workspace-storage",
          storeName: "workspace-artifact-payloads"
        }
      ],
      readLocalStorageValue: async () => null,
      readIndexedDbStorePayload: async () => "{\"artifact\":\"payload\"}",
      api: {
        createWorkspaceMigration: vi.fn(async (body) => ({
          ...body,
          status: "created",
          declared_chunk_count: body.declared_chunks.length,
          accepted_chunk_count: 0,
          missing_chunk_ids: [],
          client_delete_eligible: false,
          created_at: "2026-05-26T00:00:00Z",
          updated_at: "2026-05-26T00:00:00Z",
          finalized_at: null,
          recovery_manifest: {},
          chunks: []
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "indexeddb_store",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z"
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: []
        })),
        ackWorkspaceMigrationClientDelete
      },
      writeLocalStorageValue,
      deleteIndexedDbStorePayload: vi.fn(async () => {
        throw new Error("indexeddb-delete-failed")
      })
    }
    const result = await runResearchWorkspaceMigration(input)

    expect(result.status).toBe("blocked")
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })
})
