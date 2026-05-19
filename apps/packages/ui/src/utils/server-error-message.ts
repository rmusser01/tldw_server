import { formatErrorMessage } from "@/utils/format-error-message"

const MAX_ERROR_LENGTH = 220

const normalizeWhitespace = (value: string): string =>
  value.replace(/\s+/g, " ").trim()

const normalizeRawError = (error: unknown): string =>
  formatErrorMessage(error, "").trim()

/**
 * Sanitize backend error details before rendering them in user-facing UI.
 * Preserves a concise first-line message and redacts endpoints/paths.
 */
export const sanitizeServerErrorMessage = (
  error: unknown,
  fallbackMessage: string
): string => {
  const raw = formatErrorMessage(error, fallbackMessage)
  if (!raw.trim()) return fallbackMessage

  const firstLine = raw
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0)

  let cleaned = normalizeWhitespace((firstLine || raw).replace(/^Error:\s*/i, ""))

  cleaned = cleaned.replace(
    /\b(GET|POST|PUT|PATCH|DELETE)\s+\/api\/v1\/[^\s)]+/gi,
    "$1 [server-endpoint]"
  )
  cleaned = cleaned.replace(/\b\/api\/v1\/[^\s)]+/gi, "[server-endpoint]")
  cleaned = cleaned.replace(/\bhttps?:\/\/[^\s)]+/gi, "[server-url]")

  cleaned = cleaned.replace(
    /\/(?:Users|home|var|etc|opt|tmp|private|srv)\/[^\s)]+/g,
    "[redacted-path]"
  )
  cleaned = cleaned.replace(
    /[A-Za-z]:\\(?:[^\\\s]+\\)+[^\\\s)]+/g,
    "[redacted-path]"
  )

  if (cleaned.length > MAX_ERROR_LENGTH) {
    cleaned = `${cleaned.slice(0, MAX_ERROR_LENGTH - 3)}...`
  }

  return cleaned || fallbackMessage
}

export const extractServerCorrelationId = (error: unknown): string | null => {
  const raw = normalizeRawError(error)
  if (!raw) return null

  const patterns = [
    /(?:request[_\s-]?id|correlation[_\s-]?id)\s*[:=]\s*([a-zA-Z0-9-]{6,})/i,
    /\btrace[_\s-]?id\s*[:=]\s*([a-zA-Z0-9-]{6,})/i
  ]

  for (const pattern of patterns) {
    const match = raw.match(pattern)
    if (match?.[1]) {
      return match[1]
    }
  }
  return null
}

export const buildServerLogHint = (
  error: unknown,
  fallbackHint: string
): string => {
  const correlationId = extractServerCorrelationId(error)
  if (!correlationId) return fallbackHint
  return `Check server logs with correlation ID: ${correlationId}.`
}

/**
 * Map common backend error patterns to user-friendly messages.
 * Returns null if no known pattern is matched.
 */
// TODO(i18n): extract user-facing strings to i18n resources
export const humanizeBackendError = (error: unknown): string | null => {
  const raw = normalizeRawError(error)
  if (!raw) return null
  const lower = raw.toLowerCase()

  if (
    lower.includes("chunkererror") ||
    lower.includes("chunker error") ||
    lower.includes("chunking failed")
  ) {
    return "This file couldn't be processed. Try a different format."
  }
  if (
    lower.includes("timeouterror") ||
    lower.includes("timed out") ||
    lower.includes("request timeout") ||
    lower.includes("gateway timeout")
  ) {
    return "Processing took too long. Try a smaller file."
  }
  if (
    lower.includes("connectionerror") ||
    lower.includes("connection refused") ||
    lower.includes("econnrefused") ||
    lower.includes("failed to fetch") ||
    lower.includes("network error")
  ) {
    return "Lost connection to the server."
  }
  if (
    lower.includes("413") ||
    lower.includes("payload too large") ||
    lower.includes("request entity too large")
  ) {
    return "File is too large for the server to accept."
  }
  if (
    lower.includes("unsupported") &&
    (lower.includes("format") || lower.includes("type"))
  ) {
    return "This file format is not supported."
  }

  return null
}
