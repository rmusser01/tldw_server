import { describe, expect, it } from "vitest"
import {
  SESSION_STATE_STORAGE_KEY,
  deserializeSessionState,
  emptySessionState,
  readPersistedSessionState,
  serializeIngestSessions,
  serializePendingReplay,
  serializeQuickIngestBatches,
  serializeQuickIngestSessions,
  writePersistedSessionState,
  type PersistedQuickIngestBatch,
  type SessionStorageArea
} from "@/entries/background-session-store"

const createMemoryArea = (): SessionStorageArea & {
  store: Record<string, unknown>
} => {
  const store: Record<string, unknown> = {}
  return {
    store,
    get: async (key: string) =>
      key in store ? { [key]: store[key] } : {},
    set: async (items: Record<string, unknown>) => {
      Object.assign(store, items)
    }
  }
}

describe("background-session-store serialization", () => {
  it("round-trips ingest sessions, pending replay, and quick-ingest sessions", () => {
    const ingest = new Map<string, Record<string, unknown>>([
      [
        "ingest-1",
        {
          funnelId: "ingest-1",
          url: "https://example.test/a",
          status: "queued",
          jobIds: [11, 22]
        }
      ]
    ])
    const replay = new Set<string>(["ingest-1", "ingest-1", "  "])
    const quick = new Map<
      string,
      { sessionId: string; cancelled: boolean; abortControllers: Set<unknown> }
    >([
      [
        "qi-1",
        { sessionId: "qi-1", cancelled: true, abortControllers: new Set() }
      ]
    ])

    const state = {
      ingestSessions: serializeIngestSessions(ingest),
      pendingAuthReplay: serializePendingReplay(replay),
      quickIngestSessions: serializeQuickIngestSessions(quick)
    }

    // AbortControllers are dropped (not serializable).
    expect(state.quickIngestSessions).toEqual([
      { sessionId: "qi-1", cancelled: true }
    ])
    // Blank + duplicate replay ids are collapsed.
    expect(state.pendingAuthReplay).toEqual(["ingest-1"])

    const restored = deserializeSessionState(state)
    expect(restored.ingestSessions["ingest-1"]).toMatchObject({
      funnelId: "ingest-1",
      status: "queued",
      jobIds: [11, 22]
    })
    expect(restored.pendingAuthReplay).toEqual(["ingest-1"])
    expect(restored.quickIngestSessions).toEqual([
      { sessionId: "qi-1", cancelled: true }
    ])
  })

  it("deserializes malformed input to an empty state", () => {
    expect(deserializeSessionState(null)).toEqual(emptySessionState())
    expect(deserializeSessionState("nope")).toEqual(emptySessionState())
    expect(
      deserializeSessionState({ ingestSessions: 5, pendingAuthReplay: {} })
    ).toEqual(emptySessionState())
  })

  it("persists to and rehydrates from an injected storage area", async () => {
    const area = createMemoryArea()
    const written = {
      ingestSessions: {
        "ingest-9": { funnelId: "ingest-9", status: "running", jobIds: [7] }
      },
      pendingAuthReplay: ["ingest-9"],
      quickIngestSessions: []
    }

    await writePersistedSessionState(written, area)
    expect(area.store[SESSION_STATE_STORAGE_KEY]).toEqual(written)

    const restored = await readPersistedSessionState(area)
    expect(restored.ingestSessions["ingest-9"]).toMatchObject({
      status: "running",
      jobIds: [7]
    })
    expect(restored.pendingAuthReplay).toEqual(["ingest-9"])
  })

  it("returns empty state when no storage area is available", async () => {
    expect(await readPersistedSessionState(null)).toEqual(emptySessionState())
    // Should not throw when writing without an area.
    await expect(
      writePersistedSessionState(emptySessionState(), null)
    ).resolves.toBeUndefined()
  })

  it("defaults quickIngestBatches to an empty array for legacy persisted state", () => {
    const restored = deserializeSessionState({
      ingestSessions: {},
      pendingAuthReplay: [],
      quickIngestSessions: []
    })
    expect(restored.quickIngestBatches).toEqual([])
  })
})

describe("quick-ingest batch resume persistence", () => {
  const buildBatch = (): PersistedQuickIngestBatch => ({
    sessionId: "qi-batch-1",
    totalCount: 3,
    processedCount: 1,
    ingestTimeoutMs: 300000,
    remoteJobs: [
      { jobId: 11, batchId: "batch-a", meta: { id: "e1", type: "video", url: "https://x.test/a" } },
      { jobId: 12, batchId: "batch-a", meta: { id: "e2", type: "audio", fileName: "clip.wav" } }
    ],
    collectedResults: [{ id: "e0", status: "ok", type: "html" }],
    plannedConferenceItems: [
      { key: "e1", collectionId: 7, itemId: 70, idempotencyKey: "idem-1" }
    ]
  })

  it("serializes a batch record map and drops malformed remote jobs", () => {
    const map = new Map<string, PersistedQuickIngestBatch>([
      ["qi-batch-1", buildBatch()],
      // Malformed: no sessionId -> dropped entirely.
      ["", { ...buildBatch(), sessionId: "" }]
    ])
    // Inject a malformed remote job that must be pruned.
    map.get("qi-batch-1")!.remoteJobs.push({
      jobId: 0,
      batchId: "",
      meta: {}
    } as never)

    const serialized = serializeQuickIngestBatches(map)
    expect(serialized).toHaveLength(1)
    expect(serialized[0].sessionId).toBe("qi-batch-1")
    // The jobId:0/batchId:"" entry is dropped; the two valid jobs survive.
    expect(serialized[0].remoteJobs.map((j) => j.jobId)).toEqual([11, 12])
  })

  it("round-trips a batch through persist -> restart -> rehydrate", async () => {
    const area = createMemoryArea()
    const batches = serializeQuickIngestBatches([buildBatch()])

    await writePersistedSessionState(
      { ...emptySessionState(), quickIngestBatches: batches },
      area
    )

    // Simulate a fresh worker: read the persisted state back from storage.
    const restored = await readPersistedSessionState(area)
    expect(restored.quickIngestBatches).toHaveLength(1)
    const batch = restored.quickIngestBatches[0]
    expect(batch.sessionId).toBe("qi-batch-1")
    // Progress cursor + queued/remote job ids survive so the poll can resume.
    expect(batch.processedCount).toBe(1)
    expect(batch.totalCount).toBe(3)
    expect(batch.remoteJobs.map((j) => j.jobId)).toEqual([11, 12])
    expect(batch.remoteJobs[0].meta).toMatchObject({ id: "e1", type: "video" })
    expect(batch.collectedResults).toEqual([
      { id: "e0", status: "ok", type: "html" }
    ])
    expect(batch.plannedConferenceItems[0]).toMatchObject({
      key: "e1",
      collectionId: 7,
      itemId: 70,
      idempotencyKey: "idem-1"
    })
  })
})
