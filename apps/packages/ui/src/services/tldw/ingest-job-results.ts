import { isDbMessageDuplicate } from "@/components/Common/QuickIngest/constants"

type RecordLike = Record<string, unknown>

const FAILURE_STATUS_TOKENS = new Set([
  "error",
  "failed",
  "failure",
  "quarantined",
  "timeout",
])

const SKIPPED_STATUS_TOKENS = new Set(["skipped", "duplicate"])

const asRecord = (value: unknown): RecordLike | undefined =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as RecordLike)
    : undefined

const firstNonEmptyString = (...values: unknown[]): string | undefined => {
  for (const value of values) {
    if (typeof value === "string") {
      const normalized = value.trim()
      if (normalized) return normalized
    }
  }
  return undefined
}

const firstStringFromArray = (value: unknown): string | undefined => {
  if (!Array.isArray(value)) return undefined
  for (const item of value) {
    if (typeof item !== "string") continue
    const normalized = item.trim()
    if (normalized) return normalized
  }
  return undefined
}

export const extractCompletedIngestJobPayload = (
  value: unknown
): RecordLike | undefined => {
  const record = asRecord(value)
  const nested = asRecord(record?.result)
  return nested ?? record
}

export const extractCompletedIngestJobTerminalData = (value: unknown): unknown => {
  const payload = extractCompletedIngestJobPayload(value)
  return payload ?? value
}

export const extractCompletedIngestJobStatusToken = (
  value: unknown
): string => {
  const payload = extractCompletedIngestJobPayload(value)
  return String(payload?.status || "").trim().toLowerCase()
}

export const extractCompletedIngestJobError = (
  value: unknown
): string | undefined => {
  const payload = extractCompletedIngestJobPayload(value)
  const record = asRecord(value)
  return firstNonEmptyString(
    payload?.error,
    payload?.detail,
    firstStringFromArray(payload?.errors),
    record?.error_message,
    record?.cancellation_reason,
    record?.error,
    firstStringFromArray(record?.errors)
  )
}

export const extractCompletedIngestJobMediaId = (
  value: unknown
): string | number | null => {
  const payload = extractCompletedIngestJobPayload(value)
  const record = asRecord(value)
  const results = Array.isArray(value)
    ? value
    : Array.isArray(payload?.results)
      ? payload.results
      : Array.isArray(record?.results)
        ? record.results
        : []
  const firstResult = asRecord(results[0])
  const mediaId =
    payload?.media_id ??
    payload?.mediaId ??
    payload?.db_id ??
    firstResult?.media_id ??
    firstResult?.mediaId ??
    firstResult?.db_id ??
    null

  return typeof mediaId === "string" || typeof mediaId === "number"
    ? mediaId
    : null
}

export const completedIngestJobIndicatesSkipped = (
  value: unknown
): boolean => {
  const payload = extractCompletedIngestJobPayload(value)
  if (isDbMessageDuplicate(payload)) return true
  return SKIPPED_STATUS_TOKENS.has(extractCompletedIngestJobStatusToken(value))
}

export const completedIngestJobIndicatesFailure = (
  value: unknown
): boolean => {
  if (completedIngestJobIndicatesSkipped(value)) return false

  const status = extractCompletedIngestJobStatusToken(value)
  if (FAILURE_STATUS_TOKENS.has(status)) return true

  return Boolean(extractCompletedIngestJobError(value))
}
