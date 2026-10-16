import { describe, expect, it } from "vitest"
import {
  SESSION_STATE_INDEX_KEY,
  SESSION_STATE_STORAGE_KEY,
  createSerializedSessionStateWriter,
  emptySessionState,
  readPersistedSessionState,
  serializeQuickIngestBatches,
  serializeQuickIngestSessions,
  shardSessionState,
  type PersistedQuickIngestBatch,
  type PersistedSessionState,
  type SessionStorageArea
} from "@/entries/background-session-store"

const requestScope = { config: { serverUrl: "https://owned.test", authMode: "single-user" as const, expectedSingleUserApiKeyScope: "synthetic-scope" }, userId: null }

const createMemoryArea = (): SessionStorageArea & {
  store: Record<string, unknown>
} => {
  const store: Record<string, unknown> = {}
  return {
    store,
    get: async (keys: string | string[]) => {
      const wanted = Array.isArray(keys) ? keys : [keys]
      const out: Record<string, unknown> = {}
      for (const key of wanted) if (key in store) out[key] = store[key]
      return out
    },
    set: async (items: Record<string, unknown>) => {
      Object.assign(store, items)
    },
    remove: async (keys: string[] | string) => {
      for (const key of Array.isArray(keys) ? keys : [keys]) delete store[key]
    }
  }
}

// Track set/remove through wrappers so tests can assert exact write payloads.
const instrumentedArea = () => {
  const base = createMemoryArea()
  const setCalls: Array<Record<string, unknown>> = []
  const removed: string[][] = []
  const area: SessionStorageArea = {
    get: (keys) => base.get(keys),
    set: async (items) => {
      setCalls.push(items)
      await base.set(items)
    },
    remove: async (keys) => {
      removed.push(Array.isArray(keys) ? keys : [keys])
      await base.remove(keys)
    }
  }
  return { area, store: base.store, setCalls, removed }
}

const fatBody = (mediaId: number) => ({
  status: "completed",
  result: { media_id: mediaId, transcript: "x".repeat(8192) }
})

const buildBatch = (sessionId: string, resultCount = 0): PersistedQuickIngestBatch => ({
  sessionId,
  requestScope,
  totalCount: Math.max(resultCount, 1),
  processedCount: resultCount,
  ingestTimeoutMs: 300000,
  remoteJobs: [
    { jobId: 11, batchId: "batch-a", meta: { id: `${sessionId}-e1`, type: "video", url: "https://x.test/a" } }
  ],
  collectedResults: Array.from({ length: resultCount }, (_, index) => ({
    id: `${sessionId}-e${index}`,
    status: "ok",
    url: "https://x.test/a",
    type: "video",
    data: fatBody(index + 1)
  })),
  plannedConferenceItems: []
})

const stateWith = (parts: Partial<PersistedSessionState>): PersistedSessionState => ({
  ...emptySessionState(),
  ...parts
})

describe("sharded session-state persistence (WP-19)", () => {
  it("serializes at most once per drain window across 20 rapid persists", async () => {
    const { area, setCalls } = instrumentedArea()
    let buildCount = 0
    let revision = 0
    const write = createSerializedSessionStateWriter(area, () => {
      buildCount += 1
      return stateWith({ pendingAuthReplay: [`v${revision}`] })
    })
    for (let i = 1; i <= 20; i += 1) {
      revision = i
      write()
    }
    await new Promise((resolve) => setTimeout(resolve, 20))
    // One drain window absorbed all 20 persist requests.
    expect(buildCount).toBe(1)
    // One coalesced storage write landed the newest snapshot.
    expect(setCalls).toHaveLength(1)
    const restored = await readPersistedSessionState(area)
    expect(restored.pendingAuthReplay).toEqual(["v20"])
  })

  it("strips result.data from persisted collectedResults but not from the runtime records", async () => {
    const { area, setCalls } = instrumentedArea()
    const runtimeBatch = buildBatch("qi-fat", 100)
    let batches = [runtimeBatch]
    const write = createSerializedSessionStateWriter(area, () =>
      stateWith({ quickIngestBatches: batches })
    )
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    // Persisted shard contains no response bodies.
    const storedJson = JSON.stringify(setCalls)
    expect(storedJson).not.toContain("transcript")
    const storedBatch = (await readPersistedSessionState(area)).quickIngestBatches[0]
    expect(storedBatch.collectedResults).toHaveLength(100)
    for (const result of storedBatch.collectedResults) {
      expect(result.data).toBeUndefined()
      expect(result.id).toBeTruthy()
      expect(result.status).toBe("ok")
    }
    // The runtime object is untouched (strip happens at persist time only).
    expect(runtimeBatch.collectedResults[0].data).toEqual(fatBody(1))
    batches = []
  })

  it("writes only the changed funnel's key on incremental ticks", async () => {
    const { area, setCalls } = instrumentedArea()
    let state = stateWith({
      ingestSessions: {
        "ingest-a": { funnelId: "ingest-a", status: "queued", progressPercent: 0 },
        "ingest-b": { funnelId: "ingest-b", status: "queued", progressPercent: 0 }
      }
    })
    const write = createSerializedSessionStateWriter(area, () => state)
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))
    const initialWrites = setCalls.length

    // A status tick for funnel A only: B and the index are byte-identical.
    state = {
      ...state,
      ingestSessions: {
        ...state.ingestSessions,
        "ingest-a": { funnelId: "ingest-a", status: "running", progressPercent: 10 }
      }
    }
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    expect(setCalls.length).toBe(initialWrites + 1)
    expect(Object.keys(setCalls[setCalls.length - 1])).toEqual(["tldw:ingest:ingest-a"])
  })

  it("removes a completed funnel's key and rewrites the index", async () => {
    const { area, setCalls, removed } = instrumentedArea()
    let state = stateWith({
      ingestSessions: {
        "ingest-a": { funnelId: "ingest-a", status: "completed" },
        "ingest-b": { funnelId: "ingest-b", status: "running" }
      }
    })
    const write = createSerializedSessionStateWriter(area, () => state)
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    state = stateWith({
      ingestSessions: { "ingest-b": { funnelId: "ingest-b", status: "running" } }
    })
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    expect(removed).toEqual([["tldw:ingest:ingest-a"]])
    expect(setCalls[setCalls.length - 1]).toEqual({
      [SESSION_STATE_INDEX_KEY]: ["tldw:ingest:auth-replay", "tldw:ingest:ingest-b"]
    })
    const restored = await readPersistedSessionState(area)
    expect(Object.keys(restored.ingestSessions)).toEqual(["ingest-b"])
  })

  it("round-trips the sharded shape including quick-ingest sessions and batches", async () => {
    const { area } = instrumentedArea()
    const state = stateWith({
      ingestSessions: { "ingest-1": { funnelId: "ingest-1", status: "running", jobIds: [7] } },
      pendingAuthReplay: ["ingest-1"],
      quickIngestSessions: serializeQuickIngestSessions(
        new Map([[ "qi-1", { sessionId: "qi-1", requestScope, cancelled: false } ]])
      ),
      quickIngestBatches: serializeQuickIngestBatches([buildBatch("qi-1", 2)])
    })
    const shards = shardSessionState(state)
    expect(Object.keys(shards).sort()).toEqual([
      "tldw:ingest:auth-replay",
      "tldw:ingest:index",
      "tldw:ingest:ingest-1",
      "tldw:ingest:qi-batch:qi-1",
      "tldw:ingest:qi-session:qi-1"
    ].sort())

    const write = createSerializedSessionStateWriter(area, () => state)
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    const restored = await readPersistedSessionState(area)
    expect(restored.ingestSessions["ingest-1"]).toMatchObject({ status: "running", jobIds: [7] })
    expect(restored.pendingAuthReplay).toEqual(["ingest-1"])
    expect(restored.quickIngestSessions).toEqual([
      { sessionId: "qi-1", requestScope, cancelled: false }
    ])
    expect(restored.quickIngestBatches).toHaveLength(1)
    expect(restored.quickIngestBatches[0].remoteJobs[0].jobId).toBe(11)
    expect(restored.quickIngestBatches[0].collectedResults).toHaveLength(2)
  })

  it("migrates the old monolithic shape on read and removes the legacy key on the next drain", async () => {
    const { area, store, removed } = instrumentedArea()
    // Seed the OLD monolithic shape exactly as a pre-update worker wrote it,
    // including fat result bodies that only the old code persisted.
    const legacyState = stateWith({
      ingestSessions: { "ingest-legacy": { funnelId: "ingest-legacy", status: "running", jobIds: [42] } },
      pendingAuthReplay: ["ingest-legacy"],
      quickIngestSessions: [{ sessionId: "qi-legacy", requestScope, cancelled: false }],
      quickIngestBatches: [buildBatch("qi-legacy", 1)]
    })
    await area.set({ [SESSION_STATE_STORAGE_KEY]: legacyState })

    // Characterization: the old shape is consumed by the read path unchanged.
    const restored = await readPersistedSessionState(area)
    expect(restored.ingestSessions["ingest-legacy"]).toMatchObject({ status: "running", jobIds: [42] })
    expect(restored.pendingAuthReplay).toEqual(["ingest-legacy"])
    expect(restored.quickIngestSessions[0].sessionId).toBe("qi-legacy")
    expect(restored.quickIngestBatches[0].sessionId).toBe("qi-legacy")
    expect(restored.quickIngestBatches[0].remoteJobs[0].jobId).toBe(11)

    // Migration-on-read: the state was split into per-funnel keys + index…
    expect(store["tldw:ingest:ingest-legacy"]).toBeTruthy()
    expect(store["tldw:ingest:qi-batch:qi-legacy"]).toBeTruthy()
    expect(store[SESSION_STATE_INDEX_KEY]).toBeTruthy()
    // …with the fat response body stripped during the split.
    expect(JSON.stringify(store["tldw:ingest:qi-batch:qi-legacy"])).not.toContain("transcript")
    // …while the legacy blob is kept until the next write drain.
    expect(store[SESSION_STATE_STORAGE_KEY]).toBeTruthy()

    // The next drain (here: batch finalized, state empty) removes the legacy key.
    const write = createSerializedSessionStateWriter(area, () => emptySessionState())
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))
    expect(store[SESSION_STATE_STORAGE_KEY]).toBeUndefined()
    expect(removed.some((keys) => keys.includes(SESSION_STATE_STORAGE_KEY))).toBe(true)

    // And the re-read still resolves the migrated shard shape.
    const after = await readPersistedSessionState(area)
    expect(after).toEqual(emptySessionState())
  })

  it("falls back to nulling keys when the storage area has no remove", async () => {
    const area = createMemoryArea()
    // Simulate a storage area without `remove` (the SessionStorageArea minimum).
    const noRemoveArea: SessionStorageArea = {
      get: (keys) => area.get(keys),
      set: (items) => area.set(items)
    }
    let state = stateWith({
      ingestSessions: { "ingest-a": { funnelId: "ingest-a", status: "completed" } }
    })
    const write = createSerializedSessionStateWriter(noRemoveArea, () => state)
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    state = emptySessionState()
    write()
    await new Promise((resolve) => setTimeout(resolve, 20))

    // Removed keys are nulled (treated as absent on read) and the index drops them.
    expect(area.store["tldw:ingest:ingest-a"]).toBeNull()
    const restored = await readPersistedSessionState(noRemoveArea)
    expect(restored.ingestSessions).toEqual({})
  })
})
