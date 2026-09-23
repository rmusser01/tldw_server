import { beforeEach, describe, expect, it, vi } from "vitest"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import type { ContentDraft, DraftBatch, DraftAsset } from "../types"

type StoredRow = { id: string; [key: string]: unknown }

// The adapter implements only Dexie's storage boundary; authorization stays real.
const memory = vi.hoisted(() => {
  const table = () => {
    const rows = new Map<string, StoredRow>()
    const query = (predicate = (_row: StoredRow) => true) => ({
      toArray: async () => [...rows.values()].filter(predicate),
      first: async () => [...rows.values()].find(predicate),
      filter: (next: (row: StoredRow) => boolean) => query(row => predicate(row) && next(row)),
      and: (next: (row: StoredRow) => boolean) => query(row => predicate(row) && next(row)),
      reverse: () => query(predicate),
      delete: async () => { for (const [id, row] of rows) if (predicate(row)) rows.delete(id) }
    })
    return { rows, get: async (id: string) => rows.get(id), put: async (row: StoredRow) => { rows.set(row.id, row) },
      update: async (id: string, update: Partial<StoredRow>) => { if (rows.has(id)) rows.set(id, { ...rows.get(id), ...update }) },
      delete: async (id: string) => { rows.delete(id) }, bulkDelete: async (ids: string[]) => { ids.forEach(id => rows.delete(id)) },
      clear: async () => rows.clear(), toArray: query().toArray, filter: query().filter,
      orderBy: () => query(), where: (key: string) => ({ equals: (value: unknown) => query(row => row[key] === value) }) }
  }
  return { batches: table(), drafts: table(), assets: table() }
})
vi.mock("../schema", () => ({ db: { draftBatches: memory.batches, contentDrafts: memory.drafts, draftAssets: memory.assets,
  transaction: async (_mode: unknown, _tables: unknown, run: (tx: unknown) => unknown) => run({ abort: () => {} }) } }))
vi.mock("@/services/tldw/quick-ingest-authority", () => ({ quickIngestAuthority: { capture: () => { throw new Error("Missing verified operation") } } }))
import * as drafts from "../drafts"

const operation = (authorityKey: string) => {
  const controller = new AbortController()
  return { authorityKey, signal: controller.signal, requestScope: {},
    isCurrent: () => !controller.signal.aborted,
    assertCurrent: () => { if (controller.signal.aborted) throw new Error("Account changed") },
    abort: () => controller.abort()
  } as QuickIngestOperation & { abort: () => void }
}
const alice = () => operation("server-one:alice")
const seed = () => {
  for (const [id, ownerScope] of [["alice", "server-one:alice"], ["bob", "server-one:bob"], ["elsewhere", "server-two:alice"], ["legacy", undefined]]) {
    memory.batches.rows.set(id!, { id, ownerScope, createdAt: 1 })
    memory.drafts.rows.set(id!, { id, ownerScope, batchId: id, content: id })
    memory.assets.rows.set(id!, { id, ownerScope, draftId: id, sizeBytes: 3 })
  }
}
describe("Content Review account storage (UAT400)", () => {
  beforeEach(() => { for (const table of Object.values(memory)) table.rows.clear(); seed() })
  it("lists only the verified server/account and hides foreign or unowned direct IDs", async () => {
    const owner = alice()
    expect((await drafts.getDraftBatches(owner)).map(row => row.id)).toEqual(["alice"])
    for (const id of ["bob", "elsewhere", "legacy"]) {
      expect(await drafts.getDraftBatchById(id, owner)).toBeUndefined()
      expect(await drafts.getDraftById(id, owner)).toBeUndefined()
      expect(await drafts.getDraftAsset(id, owner)).toBeUndefined()
      expect(await drafts.getDraftsByBatch(id, owner)).toEqual([])
      expect(await drafts.getDraftAssetsByDraftId(id, owner)).toEqual([])
    }
  })
  it("refuses foreign and legacy overwrite even when incoming ownership is forged", async () => {
    for (const id of ["bob", "legacy"]) {
      const owner = alice()
      await expect(drafts.upsertContentDraft({ id, ownerScope: owner.authorityKey, content: "stolen" } as ContentDraft, owner)).rejects.toThrow()
      await expect(drafts.upsertDraftBatch({ id, ownerScope: owner.authorityKey } as DraftBatch, owner)).rejects.toThrow()
      await expect(drafts.upsertDraftAsset({ id, ownerScope: owner.authorityKey } as DraftAsset, owner)).rejects.toThrow()
      expect(memory.drafts.rows.get(id).content).toBe(id)
    }
  })
  it("cannot update, delete or cascade a foreign ID", async () => {
    const owner = alice()
    await drafts.updateContentDraft("bob", { content: "changed" }, owner)
    await drafts.deleteContentDraft("bob", owner)
    await drafts.deleteDraftsByBatch("bob", owner)
    await drafts.deleteDraftAsset("bob", owner)
    await drafts.deleteDraftAssetsByDraftId("bob", owner)
    await drafts.deleteDraftBatch("bob", owner)
    await drafts.deleteDraftBatchWithContents("bob", owner)
    expect(memory.drafts.rows.get("bob").content).toBe("bob")
    expect(memory.batches.rows.has("bob") && memory.assets.rows.has("bob")).toBe(true)
  })
  it("clears only the current owner's batches, drafts and assets", async () => {
    await drafts.clearDrafts(alice())
    for (const table of Object.values(memory)) expect([...table.rows.keys()]).toEqual(["bob", "elsewhere", "legacy"])
  })
  it("stamps new records and preserves owner identity on updates", async () => {
    const owner = alice()
    await drafts.upsertDraftBatch({ id: "new" } as DraftBatch, owner)
    await drafts.upsertContentDraft({ id: "new", batchId: "new" } as ContentDraft, owner)
    await drafts.upsertDraftAsset({ id: "new", draftId: "new" } as DraftAsset, owner)
    await drafts.updateContentDraft("new", { ownerScope: "server-one:bob", content: "retained" }, owner)
    for (const table of Object.values(memory)) expect(table.rows.get("new").ownerScope).toBe(owner.authorityKey)
    expect((await drafts.getDraftById("new", owner))?.content).toBe("retained")
  })
  it("rejects a retired operation before reading or writing", async () => {
    const owner = alice(); owner.abort()
    await expect(drafts.getDraftBatches(owner)).rejects.toThrow("Account changed")
    await expect(drafts.upsertContentDraft({ id: "late" } as ContentDraft, owner)).rejects.toThrow("Account changed")
    expect(memory.drafts.rows.has("late")).toBe(false)
  })
})
