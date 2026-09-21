import type { Table, Transaction } from "dexie"
import { quickIngestAuthority, type QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { db } from "./schema"
import type { ContentDraft, DraftAsset, DraftBatch } from "./types"

export const DRAFT_STORAGE_CAP_BYTES = 100 * 1024 * 1024

type OwnedDraftRecord = ContentDraft | DraftAsset | DraftBatch
const owned = (row: { ownerScope?: string } | undefined, operation: QuickIngestOperation) =>
  Boolean(row && row.ownerScope === operation.authorityKey)

/** Keep the captured account valid through the IndexedDB commit, including
 * an account change after the final request but before transaction completion. */
async function transact<T>(operation: QuickIngestOperation, mode: "r" | "rw", run: () => Promise<T>): Promise<T> {
  operation.assertCurrent()
  let active: Transaction | undefined
  const abort = () => { active?.abort() }
  operation.signal.addEventListener("abort", abort, { once: true })
  try {
    const result = await db.transaction(mode, [db.draftBatches, db.contentDrafts, db.draftAssets], async transaction => {
      active = transaction
      operation.assertCurrent()
      const value = await run()
      operation.assertCurrent()
      return value
    })
    operation.assertCurrent()
    return result
  } finally { operation.signal.removeEventListener("abort", abort) }
}

async function putOwned<T extends OwnedDraftRecord>(table: Table<T>, row: T, operation: QuickIngestOperation): Promise<void> {
  await transact(operation, "rw", async () => {
    const previous = await table.get(row.id)
    if ((previous && !owned(previous, operation)) || (row.ownerScope && !owned(row, operation))) {
      throw new Error("This draft belongs to another account or has no verified owner.")
    }
    await table.put({ ...row, ownerScope: operation.authorityKey })
  })
}

async function readOwned<T extends OwnedDraftRecord>(table: Table<T>, id: string, operation: QuickIngestOperation): Promise<T | undefined> {
  return transact(operation, "r", async () => {
    const row = await table.get(id)
    return owned(row, operation) ? row : undefined
  })
}
async function deleteOwned<T extends OwnedDraftRecord>(table: Table<T>, id: string, operation: QuickIngestOperation): Promise<void> {
  await transact(operation, "rw", async () => {
    if (owned(await table.get(id), operation)) await table.delete(id)
  })
}

export async function getDraftBatches(operation = quickIngestAuthority.capture()): Promise<DraftBatch[]> {
  return transact(operation, "r", () => db.draftBatches.orderBy("createdAt").reverse().filter(row => owned(row, operation)).toArray())
}
export async function getDraftBatchById(id: string, operation = quickIngestAuthority.capture()) {
  return readOwned(db.draftBatches, id, operation)
}
export async function upsertDraftBatch(batch: DraftBatch, operation = quickIngestAuthority.capture()): Promise<void> {
  await putOwned(db.draftBatches, batch, operation)
}
export async function deleteDraftBatch(id: string, operation = quickIngestAuthority.capture()): Promise<void> {
  await deleteOwned(db.draftBatches, id, operation)
}
export async function deleteDraftBatchWithContents(batchId: string, operation = quickIngestAuthority.capture()): Promise<void> {
  await transact(operation, "rw", async () => {
    if (!owned(await db.draftBatches.get(batchId), operation)) return
    const rows = await db.contentDrafts.where("batchId").equals(batchId).filter(row => owned(row, operation)).toArray()
    for (const draft of rows) await db.draftAssets.where("draftId").equals(draft.id).filter(row => owned(row, operation)).delete()
    await db.contentDrafts.bulkDelete(rows.map(row => row.id))
    await db.draftBatches.delete(batchId)
  })
}
export async function getDraftsByBatch(batchId: string, operation = quickIngestAuthority.capture()): Promise<ContentDraft[]> {
  return transact(operation, "r", () => db.contentDrafts.where("batchId").equals(batchId).filter(row => owned(row, operation)).toArray())
}
export async function getDraftById(id: string, operation = quickIngestAuthority.capture()) {
  return readOwned(db.contentDrafts, id, operation)
}
export async function upsertContentDraft(draft: ContentDraft, operation = quickIngestAuthority.capture()): Promise<void> {
  await putOwned(db.contentDrafts, draft, operation)
}
export async function updateContentDraft(id: string, updates: Partial<ContentDraft>, operation = quickIngestAuthority.capture()): Promise<void> {
  await transact(operation, "rw", async () => {
    if (owned(await db.contentDrafts.get(id), operation)) {
      await db.contentDrafts.update(id, { ...updates, id, ownerScope: operation.authorityKey })
    }
  })
}
export async function deleteContentDraft(id: string, operation = quickIngestAuthority.capture()): Promise<void> {
  await deleteOwned(db.contentDrafts, id, operation)
}
export async function deleteDraftsByBatch(batchId: string, operation = quickIngestAuthority.capture()): Promise<void> {
  await transact(operation, "rw", async () => { await db.contentDrafts.where("batchId").equals(batchId).filter(row => owned(row, operation)).delete() })
}
export async function getDraftAsset(id: string, operation = quickIngestAuthority.capture()) {
  return readOwned(db.draftAssets, id, operation)
}
export async function getDraftAssetsByDraftId(draftId: string, operation = quickIngestAuthority.capture()): Promise<DraftAsset[]> {
  return transact(operation, "r", () => db.draftAssets.where("draftId").equals(draftId).filter(row => owned(row, operation)).toArray())
}
export async function upsertDraftAsset(asset: DraftAsset, operation = quickIngestAuthority.capture()): Promise<void> {
  await putOwned(db.draftAssets, asset, operation)
}
export async function deleteDraftAsset(id: string, operation = quickIngestAuthority.capture()): Promise<void> {
  await deleteOwned(db.draftAssets, id, operation)
}
export async function deleteDraftAssetsByDraftId(draftId: string, operation = quickIngestAuthority.capture()): Promise<void> {
  await transact(operation, "rw", async () => { await db.draftAssets.where("draftId").equals(draftId).filter(row => owned(row, operation)).delete() })
}
export async function getDraftAssetsTotalBytes(operation = quickIngestAuthority.capture()): Promise<number> {
  return transact(operation, "r", async () => (await db.draftAssets.filter(row => owned(row, operation)).toArray()).reduce((sum, asset) => sum + (asset.sizeBytes || 0), 0))
}
export async function clearDrafts(operation = quickIngestAuthority.capture()): Promise<void> {
  await transact(operation, "rw", async () => {
    await db.contentDrafts.filter(row => owned(row, operation)).delete()
    await db.draftAssets.filter(row => owned(row, operation)).delete()
    await db.draftBatches.filter(row => owned(row, operation)).delete()
  })
}

type DraftAssetInput = Blob & { name?: string; lastModified?: number }
export async function storeDraftAsset(draftId: string, file: DraftAssetInput, operation = quickIngestAuthority.capture()): Promise<{ asset: DraftAsset | null; stored: boolean }> {
  return transact(operation, "rw", async () => {
    const draft = await db.contentDrafts.get(draftId)
    if (draft && !owned(draft, operation)) throw new Error("Source draft belongs to another account.")
    const sizeBytes = Number(file.size || 0)
    // The cap covers this browser's physical storage, including preserved legacy assets.
    const totalBytes = (await db.draftAssets.toArray()).reduce((sum, asset) => sum + (asset.sizeBytes || 0), 0)
    if (totalBytes + sizeBytes > DRAFT_STORAGE_CAP_BYTES) return { asset: null, stored: false }
    const asset: DraftAsset = {
      id: crypto.randomUUID(), draftId, ownerScope: operation.authorityKey, kind: "file",
      fileName: file.name || "upload", mimeType: file.type || "application/octet-stream",
      sizeBytes, blob: file, createdAt: Date.now()
    }
    await db.draftAssets.put(asset)
    return { asset, stored: true }
  })
}

/** Publish a complete review batch atomically, including its original files. */
export async function withDraftTransaction<T>(operation: QuickIngestOperation, run: () => Promise<T>): Promise<T> {
  return transact(operation, "rw", run)
}
