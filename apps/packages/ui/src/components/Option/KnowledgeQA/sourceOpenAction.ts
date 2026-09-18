import type { RagResult } from "./types"
import { getResultSourceId } from "./sourceListUtils"

/** Resolve a source action without treating uploaded filenames as web routes. */
export function getSourceOpenAction(result: RagResult): { href: string; label: string } | null {
  const rawUrl = result.metadata?.url
  if (typeof rawUrl === "string") {
    try {
      const url = new URL(rawUrl)
      if (url.protocol === "http:" || url.protocol === "https:") {
        return { href: url.href, label: "Open original" }
      }
    } catch {
      // Uploaded filenames and relative API source references need a media view.
    }
  }
  const sourceType = result.sourceType ?? result.metadata?.source_type ?? "media_db"
  const sourceId = getResultSourceId(result)
  if (sourceType === "media_db" && sourceId && /^[1-9]\d*$/.test(sourceId)) {
    return { href: `/media?id=${sourceId}`, label: "Open in Media" }
  }
  return null
}
