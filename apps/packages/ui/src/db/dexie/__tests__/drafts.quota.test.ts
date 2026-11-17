import { beforeEach, describe, expect, it, vi } from "vitest"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * WP-29 / TASK-13523 — draft asset quota accounting must not materialize the
 * whole draftAssets table (blobs included) on every check. The sums stay
 * byte-identical; only the query shape changes to streaming accumulation.
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})
vi.mock("@/services/tldw/quick-ingest-authority", () => ({
  quickIngestAuthority: { capture: () => { throw new Error("Missing verified operation") } }
}))

import {
  DRAFT_STORAGE_CAP_BYTES,
  getDraftAssetsTotalBytes,
  storeDraftAsset
} from "../drafts"

const db = () => memory.db as MemoryDexie

const operation = (authorityKey: string) => {
  const controller = new AbortController()
  return {
    authorityKey,
    signal: controller.signal,
    isCurrent: () => !controller.signal.aborted,
    assertCurrent: () => { if (controller.signal.aborted) throw new Error("Account changed") }
  } as QuickIngestOperation
}
const alice = () => operation("server-one:alice")

const asset = (id: string, ownerScope: string | undefined, sizeBytes: number, blob?: Blob) => ({
  id,
  draftId: "d1",
  ownerScope,
  kind: "file" as const,
  fileName: id,
  mimeType: "application/octet-stream",
  sizeBytes,
  blob: blob ?? new Blob(["x"]),
  createdAt: 1
})

describe("draft asset quota (WP-29)", () => {
  beforeEach(() => {
    db().resetAll()
    db().contentDrafts.put({
      id: "d1",
      ownerScope: "server-one:alice",
      batchId: "b1",
      status: "draft",
      createdAt: 1,
      updatedAt: 1
    } as never)
  })

  it("getDraftAssetsTotalBytes sums only the verified account's assets", async () => {
    db().draftAssets.bulkPut([
      asset("a-alice", "server-one:alice", 100),
      asset("a-bob", "server-one:bob", 200),
      asset("a-legacy", undefined, 400)
    ] as never[])
    expect(await getDraftAssetsTotalBytes(alice())).toBe(100)
  })

  it("getDraftAssetsTotalBytes is zero for an account with no assets", async () => {
    expect(await getDraftAssetsTotalBytes(alice())).toBe(0)
  })

  it("storeDraftAsset rejects an upload that would exceed the whole-browser cap", async () => {
    // The cap is browser-wide: every account's and preserved legacy assets count.
    db().draftAssets.bulkPut([
      asset("a-alice", "server-one:alice", 60 * 1024 * 1024),
      asset("a-bob", "server-one:bob", 36 * 1024 * 1024)
    ] as never[])
    const oversized = new Blob(["payload"])
    Object.defineProperty(oversized, "size", { value: 5 * 1024 * 1024 })
    const result = await storeDraftAsset("d1", oversized as Blob & { name?: string }, alice())
    expect(result.stored).toBe(false)
    expect(result.asset).toBeNull()
    expect((await db().draftAssets.toArray()).map(row => row.id)).toEqual(["a-alice", "a-bob"])
  })

  it("storeDraftAsset accepts an upload that fits under the cap and accounts for it", async () => {
    db().draftAssets.put(asset("a-alice", "server-one:alice", 100) as never)
    const file = new Blob(["payload"], { type: "text/plain" })
    const result = await storeDraftAsset("d1", file as Blob & { name?: string }, alice())
    expect(result.stored).toBe(true)
    expect(result.asset?.sizeBytes).toBe(file.size)
    expect(await getDraftAssetsTotalBytes(alice())).toBe(100 + file.size)
  })

  it("quota accounting never materializes the whole table at once (mechanism, post-rewrite)", async () => {
    db().draftAssets.bulkPut([
      asset("a-alice", "server-one:alice", 100),
      asset("a-bob", "server-one:bob", 200)
    ] as never[])
    const toArray = vi.spyOn(db().draftAssets, "toArray")
    toArray.mockImplementation(async () => {
      throw new Error("full-table materialization: draftAssets.toArray() must not be used by quota accounting")
    })
    const tableFilter = vi.spyOn(db().draftAssets, "filter")
    tableFilter.mockImplementation(() => {
      throw new Error("full-table JS scan: draftAssets.filter() must not be used by quota accounting")
    })
    try {
      expect(await getDraftAssetsTotalBytes(alice())).toBe(100)
      const file = new Blob(["ok"])
      const result = await storeDraftAsset("d1", file as Blob & { name?: string }, alice())
      expect(result.stored).toBe(true)
    } finally {
      toArray.mockRestore()
      tableFilter.mockRestore()
    }
  })
})
