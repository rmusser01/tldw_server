import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type DetectedMediaType =
  | "audio"
  | "video"
  | "pdf"
  | "ebook"
  | "document"
  | "html"
  | "auto"

export type IngestEntryType =
  | "auto"
  | "html"
  | "pdf"
  | "document"
  | "audio"
  | "video"

export type DraftMediaType = "html" | "pdf" | "document" | "audio" | "video"

export type UploadMediaType = "audio" | "video" | "pdf" | "ebook" | "document"

const AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]
const VIDEO_EXTENSIONS = [".mp4", ".webm", ".mkv", ".mov", ".avi"]
const PDF_EXTENSIONS = [".pdf"]
const EBOOK_EXTENSIONS = [".epub", ".mobi"]
const DOCUMENT_EXTENSIONS = [
  ".doc",
  ".docx",
  ".rtf",
  ".odt",
  ".txt",
  ".md",
  ".markdown"
]

const isSubdomainOf = (host: string, domain: string): boolean =>
  host === domain || host.endsWith(`.${domain}`)

const isYouTubeVideoUrl = (url: URL): boolean => {
  const host = url.hostname.toLowerCase()
  if (isSubdomainOf(host, "youtu.be")) {
    return url.pathname.length > 1
  }
  if (
    !isSubdomainOf(host, "youtube.com") &&
    !isSubdomainOf(host, "youtube-nocookie.com")
  ) {
    return false
  }
  const path = url.pathname.toLowerCase()
  if (path === "/watch" || path === "/watch/") {
    return url.searchParams.has("v")
  }
  const videoPrefixes = ["/shorts/", "/live/", "/embed/", "/v/", "/clip/"]
  return videoPrefixes.some(
    (prefix) => path.startsWith(prefix) && path.length > prefix.length
  )
}

const endsWithAny = (value: string, exts: string[]): boolean =>
  exts.some((ext) => value.endsWith(ext))

const inferMediaTypeFromPath = (
  raw: string,
  fallback: "html" | "auto"
): DetectedMediaType => {
  const path = String(raw || "").toLowerCase()
  if (endsWithAny(path, AUDIO_EXTENSIONS)) return "audio"
  if (endsWithAny(path, VIDEO_EXTENSIONS)) return "video"
  if (endsWithAny(path, PDF_EXTENSIONS)) return "pdf"
  if (endsWithAny(path, EBOOK_EXTENSIONS)) return "ebook"
  if (endsWithAny(path, DOCUMENT_EXTENSIONS)) return "document"
  return path ? fallback : "auto"
}

export const inferMediaTypeFromUrl = (raw: string): DetectedMediaType => {
  try {
    const parsed = new URL(raw)
    if (isYouTubeVideoUrl(parsed)) return "video"
    return inferMediaTypeFromPath(parsed.pathname, "html")
  } catch {
    return inferMediaTypeFromPath(String(raw || ""), "html")
  }
}

export const inferMediaTypeFromFilename = (raw: string): DetectedMediaType =>
  inferMediaTypeFromPath(raw, "auto")

export const inferMediaTypeFromMime = (raw: string): DetectedMediaType => {
  const t = String(raw || "").toLowerCase()
  if (!t) return "auto"
  if (t.startsWith("audio/")) return "audio"
  if (t.startsWith("video/")) return "video"
  if (t.includes("pdf")) return "pdf"
  if (t.includes("epub") || t.includes("mobi")) return "ebook"
  if (t.includes("markdown")) return "document"
  if (t.includes("html")) return "html"
  return "auto"
}

export const normalizeMediaType = (rawType: string): string => {
  if (!rawType) return rawType
  const t = rawType.toLowerCase()
  if (t === "html") return "document"
  return t
}

const INGEST_TYPE_MAP: Record<DetectedMediaType, IngestEntryType> = {
  audio: "audio",
  video: "video",
  pdf: "pdf",
  ebook: "document",
  document: "document",
  html: "html",
  auto: "auto"
}

const DRAFT_MEDIA_TYPES = new Set<DraftMediaType>([
  "html",
  "pdf",
  "document",
  "audio",
  "video"
])

const UPLOAD_MEDIA_TYPES = new Set<UploadMediaType>([
  "audio",
  "video",
  "pdf",
  "ebook",
  "document"
])

/** Raw (pre-normalization) types whose original file should be preserved for Document Workspace. */
const WORKSPACE_FILE_TYPES = new Set(["pdf", "ebook", "epub", "document"])

/** Returns true if this raw media type's original file should be stored on the server. */
export const shouldKeepOriginalFile = (type: string): boolean => {
  const t = String(type || "").trim().toLowerCase()
  return WORKSPACE_FILE_TYPES.has(t)
}

export const inferIngestTypeFromUrl = (url: string): IngestEntryType => {
  try {
    // eslint-disable-next-line no-new
    new URL(url)
  } catch {
    return "auto"
  }
  return INGEST_TYPE_MAP[inferMediaTypeFromUrl(url)]
}

export const inferIngestTypeFromFilename = (name: string): IngestEntryType =>
  INGEST_TYPE_MAP[inferMediaTypeFromFilename(name)]

export const coerceDraftMediaType = (rawType?: string): DraftMediaType => {
  const t = String(rawType || "").toLowerCase()
  if (DRAFT_MEDIA_TYPES.has(t as DraftMediaType)) {
    return t as DraftMediaType
  }
  if (t === "ebook") return "document"
  return "document"
}

export const normalizeMediaTypeForUpload = (
  rawType?: string
): UploadMediaType => {
  const normalized = normalizeMediaType(String(rawType || ""))
  if (!normalized || normalized === "auto") return "document"
  if (UPLOAD_MEDIA_TYPES.has(normalized as UploadMediaType)) {
    return normalized as UploadMediaType
  }
  return "document"
}

export const inferUploadMediaTypeFromFile = (
  name?: string,
  mimeType?: string
): UploadMediaType => {
  const fromName = inferMediaTypeFromFilename(String(name || ""))
  const inferred = fromName === "auto"
    ? inferMediaTypeFromMime(String(mimeType || ""))
    : fromName
  return normalizeMediaTypeForUpload(inferred)
}

export const inferUploadMediaTypeFromUrl = (
  rawUrl?: string
): UploadMediaType =>
  normalizeMediaTypeForUpload(inferMediaTypeFromUrl(String(rawUrl || "")))

export const getProcessPathForType = (rawType: string): AllowedPath => {
  const t = normalizeMediaType(rawType)
  switch (t) {
    case "audio":
      return "/api/v1/media/process-audios"
    case "video":
      return "/api/v1/media/process-videos"
    case "pdf":
      return "/api/v1/media/process-pdfs"
    case "ebook":
      return "/api/v1/media/process-ebooks"
    default:
      return "/api/v1/media/process-documents"
  }
}

export const getProcessPathForUrl = (url: string): AllowedPath => {
  const detected = inferMediaTypeFromUrl(url)
  if (detected === "html" || detected === "auto") {
    return "/api/v1/media/process-web-scraping"
  }
  return getProcessPathForType(detected)
}
