import { inferUploadMediaTypeFromFile } from "@/services/tldw/media-routing"

export const QUICK_INGEST_ACCEPT_STRING =
  ".pdf,.txt,.rtf,.docx,.md,.markdown,.html,.htm,.xhtml,.xml,.json,.epub,.mp3,.wav,.m4a,.flac,.aac,.ogg,.mp4,.webm,.mkv,.ogm,.mov,.avi,application/pdf,text/plain,text/markdown,text/html,text/xml,application/xml,application/xhtml+xml,application/json,application/rtf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/epub+zip,audio/*,video/*"

const API_FILE_LIMIT_MIB = 1024
const configuredDocumentLimitMiB = (() => {
  try {
    return Number(process.env.NEXT_PUBLIC_TLDW_QUICK_INGEST_DOCUMENT_MAX_MIB)
  } catch {
    // Standalone extensions may have no process environment.
    return NaN
  }
})()
const documentLimitMiB = Number.isInteger(configuredDocumentLimitMiB) && configuredDocumentLimitMiB > 0
  ? Math.min(configuredDocumentLimitMiB, API_FILE_LIMIT_MIB)
  : 250

export const QUICK_INGEST_AV_LIMIT = { bytes: API_FILE_LIMIT_MIB * 1024 ** 2, label: "1 GB" }
export const QUICK_INGEST_DOCUMENT_LIMIT = {
  bytes: documentLimitMiB * 1024 ** 2,
  label: documentLimitMiB === API_FILE_LIMIT_MIB ? QUICK_INGEST_AV_LIMIT.label : `${documentLimitMiB} MB`
}

export const getQuickIngestFileLimit = (file: { name: string; type?: string }) => {
  const type = inferUploadMediaTypeFromFile(file.name, file.type)
  return type === "audio" || type === "video" ? QUICK_INGEST_AV_LIMIT : QUICK_INGEST_DOCUMENT_LIMIT
}

export const getQuickIngestFileSizeError = (
  file: { name: string; type?: string; size: number },
  subject = "File"
): string | null => {
  const limit = getQuickIngestFileLimit(file)
  return file.size > limit.bytes ? `${subject} exceeds ${limit.label} quick-ingest limit` : null
}

// ---------------------------------------------------------------------------
// Duplicate / skip detection
// ---------------------------------------------------------------------------

/** Default English message shown when an item is skipped as a duplicate. */
export const DUPLICATE_SKIP_MESSAGE =
  "This item already exists in your library. Use the \u2018Deep\u2019 preset to overwrite."

/** Classify the backend outcome without treating its quoted title as an outcome. */
export const isDbMessageDuplicate = (data: Record<string, unknown> | null | undefined): boolean => {
  if (typeof data?.db_message !== "string") return false
  return data.db_message.replace(/^Media '[\s\S]*' /i, "").toLowerCase().includes("already exists")
}
