import type { WizardResultItem } from "./types"

export const LOCAL_QUEUE_DUPLICATE_SKIP_MESSAGE =
  "Already queued. Remove the duplicate queue item or keep one copy before processing."

export const LIBRARY_DUPLICATE_SKIP_MESSAGE =
  "Already in library. Enable Overwrite existing or use the Deep preset to replace it."

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
