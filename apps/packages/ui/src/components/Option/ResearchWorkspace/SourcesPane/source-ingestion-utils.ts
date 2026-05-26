import type { UploadProps } from "antd"

export const DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB = 500

const ACCEPTED_EXTENSIONS = [
  "pdf",
  "doc",
  "docx",
  "txt",
  "md",
  "markdown",
  "epub",
  "html",
  "htm",
  "mp3",
  "wav",
  "m4a",
  "ogg",
  "flac",
  "mp4",
  "webm",
  "mkv",
  "avi",
  "mov"
] as const

const ACCEPTED_EXTENSION_SET = new Set<string>(ACCEPTED_EXTENSIONS)

const ACCEPTED_MIME_PREFIXES = ["audio/", "video/"] as const

const ACCEPTED_MIME_TYPES = new Set<string>([
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/epub+zip",
  "text/plain",
  "text/markdown",
  "text/html"
])

const clampPositiveInt = (value: number): number | null => {
  if (!Number.isFinite(value)) return null
  if (value <= 0) return null
  return Math.floor(value)
}

const parseTimestampNumber = (value: number): number | null => {
  if (!Number.isFinite(value) || value <= 0) return null
  // Most APIs return UNIX seconds, milliseconds, or microseconds.
  if (value > 1_000_000_000_000_000) return null
  if (value >= 1_000_000_000_000) return Math.trunc(value)
  if (value >= 1_000_000_000) return Math.trunc(value * 1000)
  return null
}

export const parseSourceCreatedAt = (value: unknown): Date | undefined => {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? undefined : value
  }

  if (typeof value === "number") {
    const timestamp = parseTimestampNumber(value)
    if (timestamp == null) return undefined
    const parsed = new Date(timestamp)
    return Number.isNaN(parsed.getTime()) ? undefined : parsed
  }

  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return undefined

    const numeric = Number(trimmed)
    if (Number.isFinite(numeric)) {
      const timestamp = parseTimestampNumber(numeric)
      if (timestamp != null) {
        const parsedFromNumeric = new Date(timestamp)
        if (!Number.isNaN(parsedFromNumeric.getTime())) {
          return parsedFromNumeric
        }
      }
    }

    const parsed = new Date(trimmed)
    return Number.isNaN(parsed.getTime()) ? undefined : parsed
  }

  return undefined
}

export const resolveSourceUploadMaxSizeMb = (value: unknown): number => {
  if (typeof value === "number") {
    return clampPositiveInt(value) ?? DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB
    const parsed = clampPositiveInt(Number.parseFloat(trimmed))
    if (parsed != null) return parsed
  }
  return DEFAULT_SOURCE_UPLOAD_MAX_SIZE_MB
}

export const resolveSourceUploadMaxSizeBytes = (value: unknown): number =>
  resolveSourceUploadMaxSizeMb(value) * 1024 * 1024

export const getConfiguredSourceUploadMaxSizeBytes = (): number => {
  const importMetaEnv =
    typeof import.meta !== "undefined"
      ? ((import.meta as unknown as { env?: Record<string, unknown> }).env ?? undefined)
      : undefined
  const envValue =
    importMetaEnv?.VITE_WORKSPACE_UPLOAD_MAX_SIZE_MB

  return resolveSourceUploadMaxSizeBytes(envValue)
}

const getExtension = (name: string): string => {
  const segment = name.trim().split(".").pop()?.toLowerCase()
  return segment || ""
}

export const isSupportedSourceUploadType = (file: {
  name?: string
  type?: string
}): boolean => {
  const fileName = String(file.name || "")
  const extension = getExtension(fileName)
  const mimeType = String(file.type || "").toLowerCase()

  if (extension && ACCEPTED_EXTENSION_SET.has(extension)) return true
  if (ACCEPTED_MIME_TYPES.has(mimeType)) return true
  return ACCEPTED_MIME_PREFIXES.some((prefix) => mimeType.startsWith(prefix))
}

export type SourceUploadValidationResult =
  | { valid: true }
  | {
      valid: false
      code: "unsupported_file_type" | "file_too_large"
      maxSizeBytes?: number
      fileName: string
    }

export const validateSourceUploadFile = (
  file: { name?: string; type?: string; size?: number },
  maxSizeBytes: number
): SourceUploadValidationResult => {
  const fileName = String(file.name || "file")
  const size = Number(file.size ?? 0)

  if (!isSupportedSourceUploadType(file)) {
    return {
      valid: false,
      code: "unsupported_file_type",
      fileName
    }
  }

  if (Number.isFinite(size) && size > maxSizeBytes) {
    return {
      valid: false,
      code: "file_too_large",
      maxSizeBytes,
      fileName
    }
  }

  return { valid: true }
}

type SourceIngestionErrorLike = {
  status?: number
  message?: string
}

const extractErrorStatus = (error: unknown): number | null => {
  const maybeError = error as SourceIngestionErrorLike | null
  const status = maybeError?.status
  if (!Number.isFinite(status as number)) return null
  return Number(status)
}

const extractErrorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  const maybeError = error as SourceIngestionErrorLike | null
  if (typeof maybeError?.message === "string") return maybeError.message
  return "Unknown error"
}

const matchesError = (value: string, pattern: RegExp): boolean =>
  pattern.test(value.toLowerCase())

export const mapSourceIngestionError = (error: unknown): string => {
  const status = extractErrorStatus(error)
  const message = extractErrorMessage(error)

  if (status === 400) {
    return "Request was invalid. Verify the URL or file and try again."
  }
  if (status === 401 || status === 403) {
    return "You do not have permission to add this source. Check your session and retry."
  }
  if (status === 404) {
    return "Source endpoint was not found. Check server configuration and try again."
  }
  if (status === 408 || status === 504 || matchesError(message, /timeout/)) {
    return "Request timed out. Retry, or try a smaller source."
  }
  if (status === 413 || matchesError(message, /too large|payload too large|size limit/)) {
    return "File is too large for upload. Use a smaller file or split it before retrying."
  }
  if (status === 415 || matchesError(message, /unsupported|file type/)) {
    return "File type is not supported. Upload PDF, DOCX, text, audio, or video formats."
  }
  if (status === 429) {
    return "Too many requests right now. Wait a moment and retry."
  }
  if (
    status === 0 ||
    status === 502 ||
    status === 503 ||
    matchesError(message, /network|failed to fetch|connection|offline|cors|abort/)
  ) {
    return "Unable to reach the server. Check your connection and retry."
  }
  if (status != null && status >= 500) {
    return "Server error while ingesting this source. Retry in a few moments."
  }

  const trimmed = message.trim()
  return trimmed.length > 0
    ? `Failed to add source: ${trimmed}`
    : "Failed to add source. Please retry."
}

export const formatSourceUploadSizeLimit = (bytes: number): string => {
  const mb = bytes / (1024 * 1024)
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`
  }
  return `${Math.round(mb)} MB`
}

export const buildSourceUploadAccept = (): UploadProps["accept"] =>
  ".pdf,.doc,.docx,.txt,.md,.epub,.html,.htm,.mp3,.wav,.m4a,.ogg,.flac,.mp4,.webm,.mkv,.avi,.mov"
