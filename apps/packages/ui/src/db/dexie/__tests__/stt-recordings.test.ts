import { describe, it, expect, vi, beforeEach } from "vitest"

// ── Hoisted mock state (must be declared via vi.hoisted for vi.mock factory) ─

const { rows, mockTable, uuidState } = vi.hoisted(() => {
  const rows = new Map<string, any>()

  const mockTable = {
    put: vi.fn(async (row: any) => {
      rows.set(row.id, row)
      return row.id
    }),
    get: vi.fn(async (id: string) => rows.get(id)),
    delete: vi.fn(async (id: string) => {
      rows.delete(id)
    }),
    bulkDelete: vi.fn(async (ids: string[]) => {
      for (const id of ids) rows.delete(id)
    }),
    orderBy: vi.fn((field: string) => ({
      reverse: vi.fn(() => ({
        toArray: vi.fn(async () => {
          const all = Array.from(rows.values())
          all.sort((a, b) => b[field] - a[field])
          return all
        })
      })),
      toArray: vi.fn(async () => {
        const all = Array.from(rows.values())
        all.sort((a, b) => a[field] - b[field])
        return all
      })
    })),
    count: vi.fn(async () => rows.size)
  }

  const uuidState = { counter: 0 }

  return { rows, mockTable, uuidState }
})

// ── Mock the Dexie db singleton ──────────────────────────────────────────────

vi.mock("@/db/dexie/schema", () => ({
  db: {
    sttRecordings: mockTable,
    transaction: async (_mode: string, _tables: unknown[], work: () => unknown) => work()
  }
}))

// Stable UUID counter for deterministic tests
vi.stubGlobal("crypto", {
  ...globalThis.crypto,
  randomUUID: () => `uuid-${++uuidState.counter}`
})

// ── Import the module under test (after mocks) ──────────────────────────────

import {
  saveSttRecording,
  getSttRecording,
  deleteSttRecording,
  listSttRecordings,
  STT_RECORDING_CAP
} from "../stt-recordings"

// ── Helpers ──────────────────────────────────────────────────────────────────

function fakeBlob(size = 1024): Blob {
  return new Blob([new Uint8Array(size)], { type: "audio/webm" })
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("stt-recordings", () => {
  beforeEach(() => {
    rows.clear()
    uuidState.counter = 0
    vi.clearAllMocks()
  })

  // ── save + retrieve ──────────────────────────────────────────────────────

  it("saves a recording and retrieves it by id", async () => {
    const blob = fakeBlob()
    const id = await saveSttRecording({ ownerKey: "alice",
      blob,
      durationMs: 5000,
      mimeType: "audio/webm"
    })

    expect(id).toBe("uuid-1")
    expect(mockTable.put).toHaveBeenCalledTimes(1)

    const recording = await getSttRecording(id, "alice")
    expect(recording).toBeDefined()
    expect(recording!.blob).toBe(blob)
    expect(recording!.mimeType).toBe("audio/webm")
    expect(recording!.durationMs).toBe(5000)
    expect(typeof recording!.createdAt).toBe("number")
  })

  it("returns undefined for a non-existent id", async () => {
    const result = await getSttRecording("does-not-exist", "alice")
    expect(result).toBeUndefined()
  })

  // ── list ordering ────────────────────────────────────────────────────────

  it("lists recordings sorted by createdAt descending", async () => {
    const now = Date.now()

    rows.set("a", { ownerKey: "alice", id: "a", blob: fakeBlob(), mimeType: "audio/webm", durationMs: 100, createdAt: now - 2000 })
    rows.set("b", { ownerKey: "alice", id: "b", blob: fakeBlob(), mimeType: "audio/webm", durationMs: 200, createdAt: now - 1000 })
    rows.set("c", { ownerKey: "alice", id: "c", blob: fakeBlob(), mimeType: "audio/webm", durationMs: 300, createdAt: now })

    const list = await listSttRecordings("alice")
    expect(list).toHaveLength(3)
    // Most recent first
    expect(list[0].id).toBe("c")
    expect(list[1].id).toBe("b")
    expect(list[2].id).toBe("a")
  })

  // ── delete ───────────────────────────────────────────────────────────────

  it("deletes a recording by id", async () => {
    rows.set("x", { ownerKey: "alice", id: "x", blob: fakeBlob(), mimeType: "audio/webm", durationMs: 100, createdAt: Date.now() })

    await deleteSttRecording("x", "alice")
    expect(mockTable.delete).toHaveBeenCalledWith("x")
    expect(rows.has("x")).toBe(false)
  })

  // ── eviction at cap ──────────────────────────────────────────────────────

  it("evicts oldest recordings when cap is exceeded", async () => {
    const now = Date.now()
    for (let i = 1; i <= STT_RECORDING_CAP; i++) {
      rows.set(`old-${i}`, {
        ownerKey: "alice", id: `old-${i}`,
        blob: fakeBlob(),
        mimeType: "audio/webm",
        durationMs: 100,
        createdAt: now - (STT_RECORDING_CAP - i + 1) * 1000
      })
    }
    expect(rows.size).toBe(STT_RECORDING_CAP)

    // Save one more — should evict the oldest
    const id = await saveSttRecording({ ownerKey: "alice",
      blob: fakeBlob(),
      durationMs: 999,
      mimeType: "audio/webm"
    })

    expect(id).toBeDefined()
    // The oldest entry (old-1) should have been evicted
    expect(mockTable.bulkDelete).toHaveBeenCalled()
    const deletedIds: string[] = mockTable.bulkDelete.mock.calls[0][0]
    expect(deletedIds).toContain("old-1")
  })

  it("does not evict when under the cap", async () => {
    await saveSttRecording({ ownerKey: "alice", blob: fakeBlob(), durationMs: 100, mimeType: "audio/webm" })
    expect(mockTable.bulkDelete).not.toHaveBeenCalled()
  })

  // ── cap constant ─────────────────────────────────────────────────────────

  it("exports STT_RECORDING_CAP as 20", () => {
    expect(STT_RECORDING_CAP).toBe(20)
  })
  it("B05 refuses another owner's and unassigned recordings on read and delete", async () => {
    rows.set("private", { id: "private", ownerKey: "alice", createdAt: 1 });
    rows.set("legacy", { id: "legacy", createdAt: 0 });
    expect(await getSttRecording("private", "bob")).toBeUndefined();
    expect(await getSttRecording("legacy", "bob")).toBeUndefined();
    await deleteSttRecording("private", "bob");
    await deleteSttRecording("legacy", "bob");
    expect(rows.has("private") && rows.has("legacy")).toBe(true);
  })

  it("B05 recording cap preserves other owners and unassigned legacy audio", async () => {
    rows.set("private", { id: "private", ownerKey: "alice", createdAt: -2 });
    rows.set("legacy", { id: "legacy", createdAt: -1 });
    for (let i=0; i<STT_RECORDING_CAP; i++) rows.set(`bob-${i}`, { id: `bob-${i}`, ownerKey: "bob", createdAt: i });
    await saveSttRecording({ ownerKey: "bob", blob: fakeBlob(), durationMs: 1, mimeType: "audio/webm" });
    expect(rows.has("private") && rows.has("legacy")).toBe(true);
    expect(rows.has("bob-0")).toBe(false);
    expect((await listSttRecordings("bob")).every(row => row.ownerKey === "bob")).toBe(true);
  })

})
