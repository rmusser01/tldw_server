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
    const deleteLocalStorageValue = vi.fn()

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => "{}",
      api: {
        createWorkspaceMigration,
        putWorkspaceMigrationChunk,
        finalizeWorkspaceMigration,
        getWorkspaceMigration,
        ackWorkspaceMigrationClientDelete: vi.fn()
      },
      deleteLocalStorageValue,
      writeLocalStorageValue: vi.fn()
    })

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(createWorkspaceMigration).toHaveBeenCalledOnce()
    expect(putWorkspaceMigrationChunk).toHaveBeenCalledOnce()
    expect(finalizeWorkspaceMigration).toHaveBeenCalledOnce()
    expect(getWorkspaceMigration).toHaveBeenCalledOnce()
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("deletes covered local content, writes a tombstone, and acknowledges only when server and local gates allow it", async () => {
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))
    const deleteLocalStorageValue = vi.fn()
    const writeLocalStorageValue = vi.fn()

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      legacyWorkspaceId: "legacy-ws",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => "{}",
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
      deleteLocalStorageValue,
      writeLocalStorageValue,
      now: () => "2026-05-26T00:00:00Z"
    })

    expect(result.status).toBe("deleted")
    expect(deleteLocalStorageValue).toHaveBeenCalledWith("tldw-workspace")
    expect(writeLocalStorageValue).toHaveBeenCalledWith(
      "tldw:research-workspace:migration:tombstone:legacy-ws",
      JSON.stringify({
        legacyWorkspaceId: "legacy-ws",
        serverWorkspaceId: "ws-1",
        migrationId: result.migrationId,
        deletedAt: "2026-05-26T00:00:00Z",
        contentRetained: false
      })
    )
    expect(ackWorkspaceMigrationClientDelete).toHaveBeenCalledWith(
      result.migrationId,
      { acknowledged_manifest_hash: result.manifestHash }
    )
  })

  it("preflights tombstone writing before deleting remotely covered content", async () => {
    const deleteLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => "{}",
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
      deleteLocalStorageValue
    })

    expect(result.status).toBe("blocked")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("preflights every local deletion dependency before deleting any chunk", async () => {
    const deleteLocalStorageValue = vi.fn()
    const writeLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      discoveredIndexedDbStores: [
        {
          databaseName: "tldw-workspace-storage",
          storeName: "workspace-artifact-payloads"
        }
      ],
      readLocalStorageValue: async () => "{}",
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
      deleteLocalStorageValue,
      writeLocalStorageValue
    })

    expect(result.status).toBe("blocked")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("returns a failed state and retains local content when the migration API fails", async () => {
    const deleteLocalStorageValue = vi.fn()

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => "{}",
      api: {
        createWorkspaceMigration: vi.fn(async () => {
          throw new Error("conflict")
        }),
        putWorkspaceMigrationChunk: vi.fn(),
        finalizeWorkspaceMigration: vi.fn(),
        getWorkspaceMigration: vi.fn(),
        ackWorkspaceMigrationClientDelete: vi.fn()
      },
      deleteLocalStorageValue,
      writeLocalStorageValue: vi.fn()
    })

    expect(result.status).toBe("failed")
    expect(result.migrationId).toMatch(/^research-workspace-ws-1-/)
    expect(result.manifestHash).toHaveLength(64)
    expect(result.localDeletionEligibility?.eligible).toBe(true)
    expect(result.message).toBe("Research Workspace migration failed before local deletion.")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("fails safely when tombstone preflight write throws before local deletion", async () => {
    const deleteLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async () => "{}",
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
      deleteLocalStorageValue,
      writeLocalStorageValue: vi.fn(async () => {
        throw new Error("local-storage-quota")
      })
    })

    expect(result.status).toBe("failed")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("does not persist the final tombstone when indexeddb deletion cannot be preflight-cleaned", async () => {
    const writeLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
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
    })

    expect(result.status).toBe("blocked")
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })
})
