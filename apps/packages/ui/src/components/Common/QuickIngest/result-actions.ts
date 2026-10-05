import { getEligibleQueueItems } from "./queue-items"
import { classifyError } from "./ErrorClassification"
import type { WizardQueueItem, WizardResultItem } from "./types"

export const LOCAL_QUEUE_DUPLICATE_SKIP_MESSAGE =
  "Already queued. Use Process again to include this duplicate, or remove it from the queue."

export const LIBRARY_DUPLICATE_SKIP_MESSAGE =
  "Already in library. Enable Overwrite existing in Configure to replace it."

export const GENERIC_SKIPPED_MESSAGE =
  "Skipped. Review the item settings and retry if needed."

export type SkippedResultReason =
  | "local-queue-duplicate"
  | "library-duplicate"
  | "other"

export const canOpenMedia = (item: WizardResultItem): boolean =>
  item.mediaId != null &&
  item.status !== "error" &&
  item.outcome !== "failed" &&
  item.outcome !== "cancelled"

export const resolveSkippedResultReason = (item: WizardResultItem): SkippedResultReason => {
  const message = typeof item.message === "string" ? item.message.trim() : ""
  const dbMessage =
    item.data != null &&
    typeof item.data === "object" &&
    typeof (item.data as Record<string, unknown>).db_message === "string"
      ? String((item.data as Record<string, unknown>).db_message)
      : ""
  const combined = `${message} ${dbMessage}`.toLowerCase()

  if (
    combined.includes("already queued") ||
    combined.includes("duplicate url") ||
    combined.includes("duplicate file")
  ) {
    return "local-queue-duplicate"
  }

  if (
    combined.includes("already in library") ||
    combined.includes("already exists") ||
    combined.includes("overwrite not enabled")
  ) {
    return "library-duplicate"
  }

  return "other"
}

/** Only confirmed successful stored identities may be handed to the saved viewer. */
export const getSavedMediaIds = (results: WizardResultItem[]): Array<string | number> => {
  const ids = new Map<string, string | number>()
  for (const item of results) {
    if (item.status !== "ok" || !canOpenMedia(item) || item.persisted === false || item.outcome === "submit_failed") continue
    const id = item.mediaId
    if (typeof id === "number" ? !Number.isFinite(id) || id <= 0 : typeof id !== "string" || !id.trim()) continue
    const key = String(id).trim()
    if (!ids.has(key)) ids.set(key, typeof id === "string" ? key : id)
  }
  return [...ids.values()]
}

export const canRetryWizardResult = (
  item: WizardResultItem,
  queueItems: WizardQueueItem[]
): boolean => {
  const queued = getEligibleQueueItems(queueItems).find((entry) => entry.id === item.id)
  return (
    (item.status === "error" || item.outcome === "failed" || item.outcome === "submit_failed") &&
    item.outcome !== "cancelled" &&
    classifyError(item.error, item.data).retryable &&
    Boolean(
      queued?.validation.valid &&
      queued.conferenceOverride?.selected !== false &&
      (queued.url || queued.file)
    )
  )
}
