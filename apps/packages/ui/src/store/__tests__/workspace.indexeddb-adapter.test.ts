import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import { DEFAULT_AUDIO_SETTINGS, DEFAULT_WORKSPACE_NOTE } from "@/types/workspace"
import { createWorkspaceStorage } from "../workspace"

const STORAGE_KEY = WORKSPACE_STORAGE_KEY
type WorkspaceStorageOptions = Parameters<typeof createWorkspaceStorage>[0]
type WorkspaceStorageIndexedDbAdapter =
  NonNullable<WorkspaceStorageOptions>["indexedDbAdapter"]
type WorkspaceChatRecord = Parameters<
  WorkspaceStorageIndexedDbAdapter["putChatRecord"]
>[0]
type WorkspaceArtifactRecord = Parameters<
  WorkspaceStorageIndexedDbAdapter["putArtifactPayloadRecord"]
>[0]

const snapshotKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:snapshot`
const chatKey = (workspaceId: string) =>
  `${STORAGE_KEY}:workspace:${encodeURIComponent(workspaceId)}:chat`
const chatRecordKey = (workspaceId: string) =>
  `workspace:${encodeURIComponent(workspaceId)}:chat`
const artifactRecordKey = (workspaceId: string, artifactId: string) =>
  `workspace:${encodeURIComponent(workspaceId)}:artifact:${encodeURIComponent(artifactId)}`

const buildEnvelope = (state: Record<string, unknown>, version = 1) =>
  JSON.stringify({
    state,
    version
  })

const createInMemoryIndexedDbAdapter = (initiallyAvailable = true) => {
  const chatRecords = new Map<string, WorkspaceChatRecord>()
  const artifactRecords = new Map<string, WorkspaceArtifactRecord>()
  let available = initiallyAvailable

  const adapter: WorkspaceStorageIndexedDbAdapter = {
    isAvailable: () => available,
    putChatRecord: async (record) => {
      if (!available || typeof record.key !== "string") return false
      chatRecords.set(record.key, record)
      return true
    },
    getChatRecord: async (key) => {
      if (!available) return null
      return chatRecords.get(key) || null
    },
    deleteChatRecord: async (key) => {
      if (!available) return false
      chatRecords.delete(key)
      return true
    },
    putArtifactPayloadRecord: async (record) => {
      if (!available || typeof record.key !== "string") return false
      artifactRecords.set(record.key, record)
      return true
    },
    getArtifactPayloadRecord: async (key) => {
      if (!available) return null
      return artifactRecords.get(key) || null
    },
    deleteArtifactPayloadRecord: async (key) => {
      if (!available) return false
      artifactRecords.delete(key)
      return true
    }
  }

  return {
    chatRecords,
    artifactRecords,
    setAvailable: (nextAvailable: boolean) => {
      available = nextAvailable
    },
    adapter
  }
}

const buildWorkspaceState = (workspaceId: string) => {
  const longChatMessage = "Chat payload ".repeat(1200)
  const longArtifactContent = "Artifact content ".repeat(1300)
  const artifactId = "artifact-1"

  return {
    workspaceId,
    savedWorkspaces: [
      {
        id: workspaceId,
        name: "Workspace A",
        tag: "workspace:a",
        collectionId: null,
        createdAt: "2026-02-22T00:00:00.000Z",
        lastAccessedAt: "2026-02-22T00:00:00.000Z",
        sourceCount: 0
      }
    ],
    archivedWorkspaces: [],
    workspaceCollections: [],
    workspaceSnapshots: {
      [workspaceId]: {
        workspaceId,
        workspaceName: "Workspace A",
        workspaceTag: "workspace:a",
        workspaceCreatedAt: "2026-02-22T00:00:00.000Z",
        workspaceChatReferenceId: workspaceId,
        sources: [],
        selectedSourceIds: [],
        generatedArtifacts: [
          {
            id: artifactId,
            type: "summary",
            title: "Large Artifact",
            status: "completed",
            content: longArtifactContent,
            data: {
              cards: Array.from({ length: 200 }, (_, index) => ({
                id: index,
                text: `Flashcard ${index}`
              }))
            },
            createdAt: "2026-02-22T00:00:00.000Z"
          }
        ],
        notes: "notes",
        currentNote: { ...DEFAULT_WORKSPACE_NOTE },
        leftPaneCollapsed: false,
        rightPaneCollapsed: false,
        audioSettings: { ...DEFAULT_AUDIO_SETTINGS }
      }
    },
    workspaceChatSessions: {
      [workspaceId]: {
        messages: [
          {
            isBot: false,
            name: "You",
            message: longChatMessage,
            sources: []
          }
        ],
        historyId: "history-a",
        serverChatId: "chat-a"
      }
    }
  }
}

describe("workspace IndexedDB offload adapter", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
  })

  it("offloads large chat/artifact payloads and rehydrates them transparently", async () => {
    const workspaceId = "workspace-a"
    const artifactId = "artifact-1"
    const idb = createInMemoryIndexedDbAdapter(true)
    const storage = createWorkspaceStorage({ indexedDbAdapter: idb.adapter })
    const state = buildWorkspaceState(workspaceId)

    await storage.setItem(STORAGE_KEY, buildEnvelope(state))

    const persistedChatRaw = localStorage.getItem(chatKey(workspaceId))
    const persistedChat = persistedChatRaw ? JSON.parse(persistedChatRaw) : null
    expect(persistedChat?.offloadType).toBe("workspace_chat_session_v1")
    expect(persistedChat?.key).toBe(chatRecordKey(workspaceId))

    const persistedSnapshotRaw = localStorage.getItem(snapshotKey(workspaceId))
    const persistedSnapshot = persistedSnapshotRaw
      ? JSON.parse(persistedSnapshotRaw)
      : null
    const persistedArtifact = persistedSnapshot?.generatedArtifacts?.[0]
    expect(persistedArtifact?.content).toBeUndefined()
    expect(persistedArtifact?.data).toBeUndefined()
    expect(
      persistedArtifact?.__tldwArtifactPayloadRef?.offloadType
    ).toBe("workspace_artifact_payload_v1")

    expect(idb.chatRecords.has(chatRecordKey(workspaceId))).toBe(true)
    expect(idb.artifactRecords.has(artifactRecordKey(workspaceId, artifactId))).toBe(
      true
    )

    const hydratedRaw = await Promise.resolve(storage.getItem(STORAGE_KEY))
    const hydrated = hydratedRaw ? JSON.parse(hydratedRaw) : null

    const hydratedChat = hydrated?.state?.workspaceChatSessions?.[workspaceId]
    expect(hydratedChat?.messages?.[0]?.message).toContain("Chat payload")
    expect(hydratedChat?.historyId).toBe("history-a")

    const hydratedArtifact =
      hydrated?.state?.workspaceSnapshots?.[workspaceId]?.generatedArtifacts?.[0]
    expect(hydratedArtifact?.content).toContain("Artifact content")
    expect(Array.isArray(hydratedArtifact?.data?.cards)).toBe(true)
    expect(hydratedArtifact?.__tldwArtifactPayloadRef).toBeUndefined()
  })

  it("falls back to inline localStorage payloads when IndexedDB is unavailable", async () => {
    const workspaceId = "workspace-a"
    const idb = createInMemoryIndexedDbAdapter(false)
    const storage = createWorkspaceStorage({ indexedDbAdapter: idb.adapter })
    const state = buildWorkspaceState(workspaceId)

    await storage.setItem(STORAGE_KEY, buildEnvelope(state))

    const persistedChatRaw = localStorage.getItem(chatKey(workspaceId))
    const persistedChat = persistedChatRaw ? JSON.parse(persistedChatRaw) : null
    expect(Array.isArray(persistedChat?.messages)).toBe(true)
    expect(persistedChat?.offloadType).toBeUndefined()

    const persistedSnapshotRaw = localStorage.getItem(snapshotKey(workspaceId))
    const persistedSnapshot = persistedSnapshotRaw
      ? JSON.parse(persistedSnapshotRaw)
      : null
    const persistedArtifact = persistedSnapshot?.generatedArtifacts?.[0]
    expect(typeof persistedArtifact?.content).toBe("string")
    expect(persistedArtifact?.__tldwArtifactPayloadRef).toBeUndefined()

    expect(idb.chatRecords.size).toBe(0)
    expect(idb.artifactRecords.size).toBe(0)
  })

  it("offloads multi-megabyte version metadata and restores every field exactly", async () => {
    const workspaceId = "large-section-analysis"
    const idb = createInMemoryIndexedDbAdapter(true)
    const storage = createWorkspaceStorage({ indexedDbAdapter: idb.adapter })
    const state = buildWorkspaceState(workspaceId)
    const artifact = Object.assign(state.workspaceSnapshots[workspaceId].generatedArtifacts[0], {
      serverId: "canonical-section-artifact",
      versionMetadata: { workspaceClient: { schemaVersion: 1, data: {
        sectionAnalysis: { evidence: "Exact source passage. ".repeat(300_000) }
      } } }
    })

    await storage.setItem(STORAGE_KEY, buildEnvelope(state))

    expect(localStorage.getItem(snapshotKey(workspaceId))!.length).toBeLessThan(10_000)
    expect(localStorage.getItem(STORAGE_KEY)!.length).toBeLessThan(10_000)
    const hydrated = JSON.parse((await storage.getItem(STORAGE_KEY))!)
    expect(hydrated.state.workspaceSnapshots[workspaceId].generatedArtifacts[0]).toEqual(artifact)

    // A metadata-only update must invalidate the offload dedup hash.
    artifact.versionMetadata.workspaceClient.data.sectionAnalysis.evidence += "Updated evidence."
    await storage.setItem(STORAGE_KEY, buildEnvelope(state))
    const updated = JSON.parse((await storage.getItem(STORAGE_KEY))!)
    expect(updated.state.workspaceSnapshots[workspaceId].generatedArtifacts[0]).toEqual(artifact)
  })

  it("cleans up offloaded records when workspace storage is removed", async () => {
    const workspaceId = "workspace-a"
    const artifactId = "artifact-1"
    const idb = createInMemoryIndexedDbAdapter(true)
    const storage = createWorkspaceStorage({ indexedDbAdapter: idb.adapter })

    await storage.setItem(STORAGE_KEY, buildEnvelope(buildWorkspaceState(workspaceId)))
    expect(idb.chatRecords.has(chatRecordKey(workspaceId))).toBe(true)
    expect(idb.artifactRecords.has(artifactRecordKey(workspaceId, artifactId))).toBe(
      true
    )

    await storage.removeItem(STORAGE_KEY)

    expect(localStorage.getItem(STORAGE_KEY)).toBeNull()
    expect(localStorage.getItem(snapshotKey(workspaceId))).toBeNull()
    expect(localStorage.getItem(chatKey(workspaceId))).toBeNull()
    expect(idb.chatRecords.size).toBe(0)
    expect(idb.artifactRecords.size).toBe(0)
  })
})
