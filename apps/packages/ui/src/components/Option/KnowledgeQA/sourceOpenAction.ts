import type { RagResult } from "./types"
import { buildChatThreadPath, buildDocumentPagePath } from "@/routes/route-paths"
import { getResultSourceId } from "./sourceListUtils"

/** Resolve a source action without treating uploaded filenames as web routes. */
export function getSourceOpenAction(result: RagResult): { href: string; label: string } | null {
  const sourceType = result.sourceType ?? result.metadata?.source_type ?? "media_db"
  const sourceId = getResultSourceId(result)
  const page = result.metadata?.page_number
  const mediaType = result.metadata?.type ?? result.metadata?.media_type
  const pdfPath = sourceType === "media_db" && mediaType === "pdf" ? buildDocumentPagePath(sourceId, page) : null
  if (pdfPath) {
    return { href: pdfPath, label: `Open PDF page ${page}` }
  }
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
  if (sourceType === "media_db" && sourceId && /^[1-9]\d*$/.test(sourceId)) {
    return { href: `/media?id=${sourceId}`, label: "Open in Media" }
  }
  if (sourceType === "characters" && sourceId && /^[1-9]\d*$/.test(sourceId)) {
    return { href: `/characters?focusCharacterId=${sourceId}`, label: "Open in Characters" }
  }
  if (sourceType === "chats" && sourceId) {
    return { href: buildChatThreadPath({ serverChatId: sourceId }), label: "Open in Chat" }
  }
  return null
}
