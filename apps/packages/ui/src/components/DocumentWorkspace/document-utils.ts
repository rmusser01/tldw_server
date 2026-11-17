import type { DocumentType } from "./types"

const PDF_EXTENSIONS = [".pdf"]
const EPUB_EXTENSIONS = [".epub"]

const normalizeFilename = (filename: string): string => {
  const normalized = filename.trim().toLowerCase()
  const separatorIndex = normalized.search(/[?#]/)
  return separatorIndex === -1
    ? normalized
    : normalized.slice(0, separatorIndex)
}

const inferFromFilename = (filename?: string): DocumentType | null => {
  if (!filename) return null
  const name = normalizeFilename(filename)
  if (PDF_EXTENSIONS.some((ext) => name.endsWith(ext))) return "pdf"
  if (EPUB_EXTENSIONS.some((ext) => name.endsWith(ext))) return "epub"
  return null
}

export const inferDocumentTypeFromMedia = (
  mediaType?: string,
  filename?: string
): DocumentType | null => {
  const type = String(mediaType || "").toLowerCase()
  if (type.includes("pdf")) return "pdf"
  if (type.includes("epub") || type.includes("ebook")) return "epub"
  return inferFromFilename(filename)
}

export const getDocumentMimeType = (docType: DocumentType): string =>
  docType === "pdf" ? "application/pdf" : "application/epub+zip"

export function extractMediaId(data: unknown): string | number | null {
  const visited = new WeakSet<object>()
  return extractMediaIdInternal(data, visited)
}

function extractMediaIdInternal(
  data: unknown,
  visited: WeakSet<object>
): string | number | null {
  if (!data || typeof data !== "object") return null
  const record = data as Record<string, unknown>
  if (Array.isArray(data)) {
    return data.length > 0 ? extractMediaIdInternal(data[0], visited) : null
  }
  if (visited.has(record)) return null
  visited.add(record)

  if (
    "results" in record &&
    Array.isArray(record.results) &&
    record.results.length > 0
  ) {
    return extractMediaIdInternal(record.results[0], visited)
  }
  if ("result" in record && record.result) {
    return extractMediaIdInternal(record.result, visited)
  }
  if ("media" in record && record.media) {
    return extractMediaIdInternal(record.media, visited)
  }

  const direct =
    record.id ?? record.media_id ?? record.db_id ?? record.pk ?? record.uuid
  if (typeof direct === "string" || typeof direct === "number") return direct

  return null
}

export function extractUploadResult(
  data: unknown
): Record<string, unknown> | null {
  if (!data || typeof data !== "object") return null
  const record = data as Record<string, unknown>
  if (Array.isArray(record.results) && record.results.length > 0) {
    return record.results[0] as Record<string, unknown>
  }
  if (record.result && typeof record.result === "object") {
    return record.result as Record<string, unknown>
  }
  if (record.media && typeof record.media === "object") {
    return record.media as Record<string, unknown>
  }
  return record
}

export function extractWarnings(
  result: Record<string, unknown> | null
): string[] {
  if (!result) return []
  const warnings = result?.warnings
  if (!Array.isArray(warnings)) return []
  return warnings.map((warning) => String(warning || "").toLowerCase())
}

/** A persisted ID can accompany a 207 failure; it is never a success receipt. */
export function getDocumentUploadProblem(
  result: Record<string, unknown> | null
): string | null {
  switch (
    String(result?.status ?? "")
      .trim()
      .toLowerCase()
  ) {
    case "success":
      return result?.error ? "Import reported an error." : null
    case "error":
      return "Import failed."
    case "warning":
      return "Import completed with warnings."
    case "cancelled":
      return "Import was cancelled."
    default:
      return "Import completion was not confirmed."
  }
}
