import { formatErrorMessage } from "@/utils/format-error-message"

export type FlashcardsUiErrorCode =
  | "FLASHCARDS_VERSION_CONFLICT"
  | "FLASHCARDS_NETWORK"
  | "FLASHCARDS_VALIDATION"
  | "FLASHCARDS_NOT_FOUND"
  | "FLASHCARDS_SERVER"
  | "FLASHCARDS_UNKNOWN"

export interface FlashcardsUiError {
  code: FlashcardsUiErrorCode
  message: string
  actionLabel: string
  status?: number
  rawMessage: string
}

export interface FlashcardsErrorMappingOptions {
  operation: string
  fallback: string
}

const NETWORK_ERROR_PATTERN =
  /(networkerror|failed to fetch|network error|load failed|err_connection|connection refused|receiving end does not exist|messaging timeout|timed out|timeout)/i
const VERSION_CONFLICT_PATTERN =
  /(version mismatch|version changed|optimistic lock|expected version|conflict)/i
const VALIDATION_PATTERN =
  /(invalid|missing required|required field|unprocessable|validation|bad request|must contain)/i

const asHttpStatus = (value: unknown): number | undefined => {
  const num =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value.trim())
        : Number.NaN
  if (!Number.isFinite(num) || !Number.isInteger(num)) return undefined
  if (num < 100 || num > 599) return undefined
  return num
}

const extractStatusFromRecord = (
  record: Record<string, unknown>
): number | undefined => {
  return (
    asHttpStatus(record.status) ??
    asHttpStatus(record.statusCode) ??
    asHttpStatus(record.code)
  )
}

export const extractFlashcardsErrorStatus = (error: unknown): number | undefined => {
  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>
    const direct = extractStatusFromRecord(record)
    if (direct !== undefined) return direct

    if (record.response && typeof record.response === "object") {
      const nested = extractStatusFromRecord(record.response as Record<string, unknown>)
      if (nested !== undefined) return nested
    }
  }

  const message = error instanceof Error ? error.message : String(error || "")
  const matches = [
    message.match(/\b(?:status|statusCode)\b\s*[:=]?\s*(\d{3})\b/i)?.[1],
    message.match(/\bHTTP(?:\/\d(?:\.\d)?)?\s*(\d{3})\b/i)?.[1],
    message.match(/\b(?:request|upload)\s+failed\b\s*[:(]\s*(\d{3})\b/i)?.[1],
    message.match(/\((\d{3})\)/)?.[1]
  ]

  for (const candidate of matches) {
    const status = asHttpStatus(candidate)
    if (status !== undefined) return status
  }

  return undefined
}

const stripMethodPathSuffix = (message: string): string =>
  message
    .replace(/\s+\((GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+\/api\/[^\)]*\)$/i, "")
    .trim()

const appendRetryHint = (message: string): string =>
  message.endsWith(".") ? `${message} Please retry.` : `${message}. Please retry.`

export const mapFlashcardsUiError = (
  error: unknown,
  options: FlashcardsErrorMappingOptions
): FlashcardsUiError => {
  const status = extractFlashcardsErrorStatus(error)
  const rawMessage = stripMethodPathSuffix(
    formatErrorMessage(error, options.fallback)
  )
  const normalized = rawMessage.toLowerCase()

  if (status === 409 || VERSION_CONFLICT_PATTERN.test(normalized)) {
    return {
      code: "FLASHCARDS_VERSION_CONFLICT",
      message: "This card was modified since you opened it. Refresh and try again.",
      actionLabel: "Refresh",
      status,
      rawMessage
    }
  }

  if (status === 0 || NETWORK_ERROR_PATTERN.test(normalized)) {
    return {
      code: "FLASHCARDS_NETWORK",
      message: `Couldn't reach the server while ${options.operation}. Check your connection and retry.`,
      actionLabel: "Retry",
      status,
      rawMessage
    }
  }

  if (status === 400 || status === 422 || VALIDATION_PATTERN.test(normalized)) {
    const baseMessage = rawMessage || options.fallback
    return {
      code: "FLASHCARDS_VALIDATION",
      message: `${baseMessage} Fix the input and retry.`,
      actionLabel: "Fix input",
      status,
      rawMessage
    }
  }

  if (status === 404) {
    return {
      code: "FLASHCARDS_NOT_FOUND",
      message: "This card is no longer available. Reload and try again.",
      actionLabel: "Reload card",
      status,
      rawMessage
    }
  }

  if (typeof status === "number" && status >= 500) {
    return {
      code: "FLASHCARDS_SERVER",
      message: `Server error while ${options.operation}. Please retry in a moment.`,
      actionLabel: "Retry",
      status,
      rawMessage
    }
  }

  return {
    code: "FLASHCARDS_UNKNOWN",
    message: appendRetryHint(options.fallback),
    actionLabel: "Retry",
    status,
    rawMessage
  }
}

export const formatFlashcardsUiErrorMessage = (error: FlashcardsUiError): string =>
  `${error.message} [${error.code}]`
