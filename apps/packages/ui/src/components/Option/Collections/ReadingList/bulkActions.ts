import type { ReadingItemsBulkResponse } from "@/types/collections"
import { dedupeTags } from "@/utils/tag-identity"

export const normalizeBulkTags = (raw: string): string[] => (raw ? dedupeTags(raw.split(",")) : [])

export const getBulkFailureLines = (
  response: ReadingItemsBulkResponse,
  maxLines = 10
): string[] => {
  if (!Array.isArray(response.results) || maxLines <= 0) return []
  return response.results
    .filter((entry) => !entry.success)
    .slice(0, maxLines)
    .map((entry) => `#${entry.item_id}: ${entry.error || "update_failed"}`)
}
