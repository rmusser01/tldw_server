import type { ClipDraft } from "./draft-builder"
import { normalizeClipDraft } from "./draft-builder"

export const CLIPPER_PENDING_DRAFT_STORAGE_KEY =
  "tldw:web-clipper:pendingDraft"
export const CLIPPER_PENDING_DRAFT_EVENT = "tldw:web-clipper-pending-draft"
export const CLIPPER_CAPTURE_MESSAGE_TYPE = "save-to-clipper"

export type PendingClipDraft = ClipDraft

const hasWindow = () => typeof window !== "undefined"
let inMemoryPendingClipDraft: PendingClipDraft | null = null

export const normalizePendingClipDraft = (
  raw: unknown
): PendingClipDraft | null => normalizeClipDraft(raw)

export const readPendingClipDraft = (): PendingClipDraft | null => {
  if (!hasWindow()) return null
  if (inMemoryPendingClipDraft) {
    return inMemoryPendingClipDraft
  }
  try {
    const raw = window.sessionStorage.getItem(CLIPPER_PENDING_DRAFT_STORAGE_KEY)
    if (!raw) return null
    return normalizePendingClipDraft(JSON.parse(raw))
  } catch {
    return inMemoryPendingClipDraft
  }
}

export const writePendingClipDraft = (draft: PendingClipDraft): void => {
  if (!hasWindow()) return
  inMemoryPendingClipDraft = draft
  try {
    window.sessionStorage.setItem(
      CLIPPER_PENDING_DRAFT_STORAGE_KEY,
      JSON.stringify(draft)
    )
  } catch {
    // ignore storage failures
  }
  try {
    window.dispatchEvent(
      new CustomEvent<PendingClipDraft>(CLIPPER_PENDING_DRAFT_EVENT, {
        detail: draft
      })
    )
  } catch {
    // ignore event dispatch failures
  }
}

export const clearPendingClipDraft = (clipId?: string): void => {
  if (!hasWindow()) return
  if (clipId) {
    const current = readPendingClipDraft()
    if (current && current.clipId !== clipId) {
      return
    }
  }
  inMemoryPendingClipDraft = null
  try {
    window.sessionStorage.removeItem(CLIPPER_PENDING_DRAFT_STORAGE_KEY)
  } catch {
    // ignore storage failures
  }
}
