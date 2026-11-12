import { db } from "./schema"
import type { SttRecordingRow } from "./types"

/** Maximum number of STT recordings kept for each verified account. */
export const STT_RECORDING_CAP = 20

/** Persist audio and evict only the same owner's oldest recordings atomically. */
export async function saveSttRecording(input: {
  ownerKey: string
  blob: Blob
  durationMs: number
  mimeType: string
}): Promise<string> {
  if (!input.ownerKey) throw new Error("A verified recording owner is required")
  const id = crypto.randomUUID()
  const row: SttRecordingRow = { ...input, id, createdAt: Date.now() }
  await db.transaction("rw", [db.sttRecordings], async () => {
    const owned = await listSttRecordings(input.ownerKey)
    const overflow = owned.length - STT_RECORDING_CAP + 1
    if (overflow > 0) {
      await db.sttRecordings.bulkDelete(owned.slice(-overflow).map(r => r.id))
    }
    await db.sttRecordings.put(row)
  })
  return id
}

/** Unassigned legacy rows are preserved without assigning them to a login. */
export async function getSttRecording(
  id: string, ownerKey: string
): Promise<SttRecordingRow | undefined> {
  if (!ownerKey) return undefined
  const row = await db.sttRecordings.get(id)
  return row?.ownerKey === ownerKey ? row : undefined
}

export async function deleteSttRecording(id: string, ownerKey: string): Promise<void> {
  await db.transaction("rw", [db.sttRecordings], async () => {
    if (await getSttRecording(id, ownerKey)) await db.sttRecordings.delete(id)
  })
}

/** List this owner's recordings, most recent first. */
export async function listSttRecordings(ownerKey: string): Promise<SttRecordingRow[]> {
  if (!ownerKey) return []
  const rows = await db.sttRecordings.orderBy("createdAt").reverse().toArray()
  return rows.filter(row => row.ownerKey === ownerKey)
}
