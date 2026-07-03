import { describe, expect, it } from "vitest"
import {
  SESSION_STATE_STORAGE_KEY,
  createSerializedSessionStateWriter,
  deserializeSessionState,
  emptySessionState,
  isInterruptedPreSubmissionIngestSession,
  readPersistedSessionState,
  selectInterruptedIngestFunnelIds,
  serializeIngestSessions,
  serializePendingReplay,
  serializeQuickIngestBatches,
  serializeQuickIngestSessions,
  writePersistedSessionState,
  type PersistedQuickIngestBatch,
  type PersistedSessionState,
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

describe("interrupted pre-submission ingest sessions", () => {
  it("classifies a queued no-jobIds session as interrupted", () => {
    expect(
      isInterruptedPreSubmissionIngestSession({
        funnelId: "ingest-1",
        status: "queued",
        jobIds: []
      })
    ).toBe(true)
    expect(
      isInterruptedPreSubmissionIngestSession({
        funnelId: "ingest-2",
        status: "running"
        // jobIds absent entirely
      })
    ).toBe(true)
  })

  it("does not classify resumable, awaiting-auth, or terminal sessions", () => {
    // Has jobIds -> resumable by the poll, not interrupted.
    expect(
      isInterruptedPreSubmissionIngestSession({
        status: "queued",
        jobIds: [7]
      })
    ).toBe(false)
    // Awaiting auth -> driven by the auth-replay path.
    expect(
      isInterruptedPreSubmissionIngestSession({
        status: "running",
        jobIds: [],
        awaitingAuth: true
      })
    ).toBe(false)
    // Terminal states are already reported.
    expect(
      isInterruptedPreSubmissionIngestSession({ status: "failed", jobIds: [] })
    ).toBe(false)
    expect(
      isInterruptedPreSubmissionIngestSession({
        status: "completed",
        jobIds: []
      })
    ).toBe(false)
    expect(isInterruptedPreSubmissionIngestSession(null)).toBe(false)
    expect(isInterruptedPreSubmissionIngestSession(undefined)).toBe(false)
  })

  it("selects only stranded funnelIds and excludes auth-replay ids", () => {
    const ingestSessions = {
      "ingest-stranded": {
        funnelId: "ingest-stranded",
        status: "queued",
        jobIds: []
      },
      "ingest-polling": {
        funnelId: "ingest-polling",
        status: "running",
        jobIds: [42]
      },
      "ingest-auth": {
        funnelId: "ingest-auth",
        status: "queued",
        jobIds: []
      }
    }
    // ingest-auth is queued for auth replay; it must not be reported failed.
    const selected = selectInterruptedIngestFunnelIds(ingestSessions, [
      "ingest-auth"
    ])
    expect(selected).toEqual(["ingest-stranded"])
  })

  it("reports + clears a persisted no-jobIds queued session on resume", async () => {
    // Mirror the background rehydrate flow: read persisted state, select the
    // stranded funnelIds, then run the report-and-clear loop against a live Map.
    const area = createMemoryArea()
    await writePersistedSessionState(
      {
        ...emptySessionState(),
        ingestSessions: {
          "ingest-stuck": {
            funnelId: "ingest-stuck",
            url: "https://example.test/stuck",
            status: "queued",
            jobIds: []
          }
        }
      },
      area
    )

    const restored = await readPersistedSessionState(area)
    const funnelIds = selectInterruptedIngestFunnelIds(
      restored.ingestSessions,
      restored.pendingAuthReplay
    )
    expect(funnelIds).toEqual(["ingest-stuck"])

    // Simulate a fresh worker's in-memory Map + reporter.
    const live = new Map<string, Record<string, unknown>>(
      Object.entries(restored.ingestSessions)
    )
    const reported: Array<{ funnelId: string; status: string }> = []
    for (const funnelId of funnelIds) {
      reported.push({ funnelId, status: "failed" })
      live.delete(funnelId)
    }

    expect(reported).toEqual([{ funnelId: "ingest-stuck", status: "failed" }])
    // Session is cleared so the sidepanel is not left stuck.
    expect(live.has("ingest-stuck")).toBe(false)
  })
})

describe("serialized session state writer", () => {
  const versioned = (tag: string): PersistedSessionState => ({
    ...emptySessionState(),
    pendingAuthReplay: [tag]
  })

  it("coalesces rapid persists and ends on the latest snapshot", async () => {
    const store: Record<string, unknown> = {}
    const setOrder: string[] = []
    let setCount = 0
    // The first write is deliberately SLOW; a naive parallel writer would let
    // that stale early snapshot land last. The latch must serialize so only the
    // newest snapshot ("v5") is written last.
    const area: SessionStorageArea = {
      get: async (key) => (key in store ? { [key]: store[key] } : {}),
      set: async (items) => {
        setCount++
        const state = items[SESSION_STATE_STORAGE_KEY] as PersistedSessionState
        const tag = state.pendingAuthReplay[0]
        await new Promise((r) => setTimeout(r, tag === "v1" ? 25 : 1))
        Object.assign(store, items)
        setOrder.push(tag)
      }
    }

    const write = createSerializedSessionStateWriter(area)
    for (let i = 1; i <= 5; i++) {
      write(versioned(`v${i}`))
    }
    // Let the write chain drain.
    await new Promise((r) => setTimeout(r, 60))

    const stored = store[SESSION_STATE_STORAGE_KEY] as PersistedSessionState
    expect(stored.pendingAuthReplay).toEqual(["v5"])
    // Only the in-flight first write + one coalesced final write happen.
    expect(setOrder).toEqual(["v1", "v5"])
    expect(setCount).toBe(2)
  })

  it("no-ops without a storage area and reports write errors", async () => {
    // No area -> writePersistedSessionState no-ops; must not throw.
    const writeNull = createSerializedSessionStateWriter(null)
    expect(() => writeNull(emptySessionState())).not.toThrow()

    const errors: unknown[] = []
    const failingArea: SessionStorageArea = {
      get: async () => ({}),
      set: async () => {
        throw new Error("boom")
      }
    }
    const write = createSerializedSessionStateWriter(failingArea, (e) =>
      errors.push(e)
    )
    write(versioned("v1"))
    await new Promise((r) => setTimeout(r, 5))
    expect(errors).toHaveLength(1)
    expect((errors[0] as Error).message).toBe("boom")
  })
})
