import { describe, it, expect, beforeEach, vi } from "vitest"
import {
  buildResearchWorkspaceMigrationPlan,
  buildResearchWorkspaceMigrationTombstone,
  buildResearchWorkspaceMigrationTombstoneKey,
  runResearchWorkspaceMigration,
  saveResearchWorkspaceChatReference,
  readSavedResearchWorkspaceChatReference
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
    const values = new Map([
      ["tldw-workspace", JSON.stringify({ workspaceId: "ws-1" })],
    ]);
    const deleteLocalStorageValue = vi.fn((key: string) => {
      values.delete(key);
    });

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async (key) => values.get(key) ?? null,
      api: {
        createWorkspaceMigration,
        putWorkspaceMigrationChunk,
        finalizeWorkspaceMigration,
        getWorkspaceMigration,
        ackWorkspaceMigrationClientDelete: vi.fn(),
      },
      deleteLocalStorageValue,
      writeLocalStorageValue: vi.fn(),
    });

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(createWorkspaceMigration).toHaveBeenCalledOnce()
    expect(putWorkspaceMigrationChunk).toHaveBeenCalledOnce()
    expect(finalizeWorkspaceMigration).toHaveBeenCalledOnce()
    expect(getWorkspaceMigration).toHaveBeenCalledOnce()
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("retains covered local content even when an older server advertises deletion", async () => {
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))
    const values = new Map([
      ["tldw-workspace", JSON.stringify({ workspaceId: "ws-1" })],
    ]);
    const deleteLocalStorageValue = vi.fn((key: string) => {
      values.delete(key);
    });
    const writeLocalStorageValue = vi.fn()

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      legacyWorkspaceId: "legacy-ws",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async (key) => values.get(key) ?? null,
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
          chunks: [],
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z",
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        ackWorkspaceMigrationClientDelete,
      },
      deleteLocalStorageValue,
      writeLocalStorageValue,
      now: () => "2026-05-26T00:00:00Z",
    });

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(values.get("tldw-workspace")).toBe(JSON.stringify({ workspaceId: "ws-1" }))
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains local content without a tombstone writer", async () => {
    const values = new Map([
      ["tldw-workspace", JSON.stringify({ workspaceId: "ws-1" })],
    ]);
    const deleteLocalStorageValue = vi.fn((key: string) => {
      values.delete(key);
    });
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async (key) => values.get(key) ?? null,
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
          chunks: [],
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z",
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        ackWorkspaceMigrationClientDelete,
      },
      deleteLocalStorageValue,
    });

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains both storage classes without invoking deletion dependencies", async () => {
    const values = new Map([
      ["tldw-workspace", JSON.stringify({ workspaceId: "ws-1" })],
    ]);
    const deleteLocalStorageValue = vi.fn((key: string) => {
      values.delete(key);
    });
    const writeLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      discoveredIndexedDbStores: [
        {
          databaseName: "tldw-workspace-storage",
          storeName: "workspace-artifact-payloads",
        },
      ],
      readLocalStorageValue: async (key) => values.get(key) ?? null,
      readIndexedDbStorePayload: async () =>
        JSON.stringify({ workspaceId: "ws-1", artifact: "payload" }),
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
          chunks: [],
        })),
        putWorkspaceMigrationChunk: vi.fn(async (migrationId, chunkId) => ({
          id: chunkId,
          migration_id: migrationId,
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z",
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        ackWorkspaceMigrationClientDelete,
      },
      deleteLocalStorageValue,
      writeLocalStorageValue,
    });

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("returns a failed state and retains local content when the migration API fails", async () => {
    const values = new Map([
      ["tldw-workspace", JSON.stringify({ workspaceId: "ws-1" })],
    ]);
    const deleteLocalStorageValue = vi.fn((key: string) => {
      values.delete(key);
    });

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async (key) => values.get(key) ?? null,
      api: {
        createWorkspaceMigration: vi.fn(async () => {
          throw new Error("conflict");
        }),
        putWorkspaceMigrationChunk: vi.fn(),
        finalizeWorkspaceMigration: vi.fn(),
        getWorkspaceMigration: vi.fn(),
        ackWorkspaceMigrationClientDelete: vi.fn(),
      },
      deleteLocalStorageValue,
      writeLocalStorageValue: vi.fn(),
    });

    expect(result.status).toBe("failed")
    expect(result.migrationId).toMatch(/^research-workspace-ws-1-/)
    expect(result.manifestHash).toHaveLength(64)
    expect(result.localDeletionEligibility?.eligible).toBe(true)
    expect(result.message).toBe("Research Workspace migration failed; local content is retained.")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
  })

  it("does not invoke a failing tombstone writer for receipt-only migration", async () => {
    const values = new Map([
      ["tldw-workspace", JSON.stringify({ workspaceId: "ws-1" })],
    ]);
    const deleteLocalStorageValue = vi.fn((key: string) => {
      values.delete(key);
    });
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async (key) => values.get(key) ?? null,
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
          chunks: [],
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "workspace_bundle",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z",
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        ackWorkspaceMigrationClientDelete,
      },
      deleteLocalStorageValue,
      writeLocalStorageValue: vi.fn(async () => {
        throw new Error("local-storage-quota");
      }),
    });

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(deleteLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })

  it("retains IndexedDB payloads without invoking the failing delete callback", async () => {
    const writeLocalStorageValue = vi.fn()
    const ackWorkspaceMigrationClientDelete = vi.fn(async () => ({ ok: true }))

    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-1",
      targetWorkspaceName: "Workspace One",
      discoveredLocalStorageKeys: [],
      discoveredIndexedDbStores: [
        {
          databaseName: "tldw-workspace-storage",
          storeName: "workspace-artifact-payloads",
        },
      ],
      readLocalStorageValue: async () => null,
      readIndexedDbStorePayload: async () =>
        JSON.stringify({ workspaceId: "ws-1", artifact: "payload" }),
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
          chunks: [],
        })),
        putWorkspaceMigrationChunk: vi.fn(async () => ({
          id: "chunk-1",
          migration_id: "mig-1",
          sha256: "b".repeat(64),
          byte_count: 2,
          chunk_kind: "indexeddb_store",
          metadata: {},
          status: "accepted",
          accepted_at: "2026-05-26T00:00:00Z",
        })),
        finalizeWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        getWorkspaceMigration: vi.fn(async () => ({
          id: "mig-1",
          status: "finalized",
          client_delete_eligible: true,
          chunks: [],
        })),
        ackWorkspaceMigrationClientDelete,
      },
      writeLocalStorageValue,
      deleteIndexedDbStorePayload: vi.fn(async () => {
        throw new Error("indexeddb-delete-failed");
      }),
    });

    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(writeLocalStorageValue).not.toHaveBeenCalled()
    expect(ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled()
  })
})


it.each([false, true])(
  "retains the complete local conversation without a destructive marker (late ACK=%s)",
  async (late) => {
    const writes = new Map<string, string>([
      [
        "tldw-workspace",
        JSON.stringify({
          workspaceId: "ws-hint",
          privateTranscript: "Do not retain this answer",
        }),
      ],
    ]);
    const receipt = {
      id: "migration-chat",
      status: "finalized",
      client_delete_eligible: true
    }
    const chatReference = {
      referenceId: "ws-hint",
      serverChatId: "11111111-2222-4333-8444-555555555555"
    }
    let currentReference = late
      ? { ...chatReference, serverChatId: null }
      : chatReference
    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-hint",
      targetWorkspaceName: "Retained chat",
      discoveredLocalStorageKeys: ["tldw-workspace"],
      readLocalStorageValue: async (key) => writes.get(key) ?? null,
      getChatReference: () => currentReference,
      api: {
        createWorkspaceMigration: vi.fn(async () => receipt),
        putWorkspaceMigrationChunk: vi.fn(async () => ({})),
        finalizeWorkspaceMigration: vi.fn(async () => receipt),
        getWorkspaceMigration: vi.fn(async () => receipt),
        ackWorkspaceMigrationClientDelete: vi.fn(async () => ({})),
      },
      deleteLocalStorageValue: (key) => {
        writes.delete(key);
        if (key === "tldw-workspace") currentReference = chatReference;
      },
      writeLocalStorageValue: (key, value) => {
        writes.set(key, value);
      },
    });
    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(writes.get("tldw-workspace")).toContain("Do not retain this answer")
    expect(writes.has(buildResearchWorkspaceMigrationTombstoneKey("ws-hint"))).toBe(false)
    expect(writes.size).toBe(1)
  }
)


describe("Account-scoped Chat reference inventory", () => {
  beforeEach(() => localStorage.clear())

  it("retains actual saved Chat reference metadata while planning another Workspace", async () => {
    const owner = { scopeKey: "owner:alice", isCurrent: () => true }
    saveResearchWorkspaceChatReference(
      "ws-alice",
      "ws-alice",
      "saved-chat",
      owner,
    )
    const hint = readSavedResearchWorkspaceChatReference(
      localStorage,
      "ws-alice",
      owner.scopeKey,
    )
    const contentKey = "tldw-workspace:workspace:ws-bob:snapshot"
    localStorage.setItem(contentKey, JSON.stringify({ workspaceId: "ws-bob" }))
    const plan = await buildResearchWorkspaceMigrationPlan({
      targetWorkspaceId: "ws-bob",
      targetWorkspaceName: "Bob",
      discoveredLocalStorageKeys: [hint.key, contentKey],
      readLocalStorageValue: async (key) => localStorage.getItem(key),
    })
    expect(plan.manifest.unknown_surface_ids).toEqual([])
    expect(plan.manifest.retained_local_surface_ids).toEqual([
      "localStorage:" + hint.key,
    ])
    expect(plan.manifest.covered_surface_ids).toEqual([
      "localStorage:" + contentKey,
    ])
    expect(plan.localDeletionEligibility.eligible).toBe(true)
    expect(localStorage.getItem(hint.key)).toBe(hint.raw)
  })

  it("never uploads the hint payload as a chunk or deletes its retained key", async () => {
    const owner = { scopeKey: "owner:alice", isCurrent: () => true }
    saveResearchWorkspaceChatReference(
      "ws-alice",
      "ws-alice",
      "saved-chat",
      owner,
    )
    const hint = readSavedResearchWorkspaceChatReference(
      localStorage,
      "ws-alice",
      owner.scopeKey,
    )
    const contentKey = "tldw-workspace:workspace:ws-bob:snapshot"
    localStorage.setItem(contentKey, JSON.stringify({ workspaceId: "ws-bob" }))
    const receipt = {
      id: "migration-bob",
      status: "finalized",
      client_delete_eligible: true,
    }
    const put = vi.fn(async () => ({}))
    const create = vi.fn(async (_body: unknown) => receipt)
    const result = await runResearchWorkspaceMigration({
      targetWorkspaceId: "ws-bob",
      targetWorkspaceName: "Bob",
      discoveredLocalStorageKeys: [hint.key, contentKey],
      readLocalStorageValue: async (key) => localStorage.getItem(key),
      api: {
        createWorkspaceMigration: create,
        putWorkspaceMigrationChunk: put,
        finalizeWorkspaceMigration: vi.fn(async () => receipt),
        getWorkspaceMigration: vi.fn(async () => receipt),
        ackWorkspaceMigrationClientDelete: vi.fn(async () => ({})),
      },
      deleteLocalStorageValue: (key) => localStorage.removeItem(key),
      writeLocalStorageValue: (key, value) => localStorage.setItem(key, value),
    })
    expect(result.status).toBe("finalized_not_delete_eligible")
    expect(localStorage.getItem(contentKey)).toBe(JSON.stringify({ workspaceId: "ws-bob" }))
    expect(localStorage.getItem(hint.key)).toBe(hint.raw)
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      manifest: { retained_local_surface_ids: ["localStorage:" + hint.key] }
    })
    expect(JSON.stringify(create.mock.calls)).not.toContain("saved-chat")
    expect(put).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(put.mock.calls)).not.toContain("saved-chat")
    expect(JSON.stringify(put.mock.calls)).not.toContain(hint.key)
  })
})
