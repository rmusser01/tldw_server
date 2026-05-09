export const UNKNOWN_LAST_MODIFIED_LABEL = "Unknown"

const SECONDS_TO_MS = 1000
const SECOND_TIMESTAMP_CUTOFF = 1_000_000_000_000
const MINUTES_TO_MS = 60 * SECONDS_TO_MS
const HOURS_TO_MS = 60 * MINUTES_TO_MS
const DAYS_TO_MS = 24 * HOURS_TO_MS
const MONTHS_TO_MS = 30 * DAYS_TO_MS
const YEARS_TO_MS = 365 * DAYS_TO_MS

export const parseWorldBookTimestamp = (value: unknown): number | null => {
  if (value == null) return null
  if (value instanceof Date) {
    const ts = value.getTime()
    return Number.isFinite(ts) ? ts : null
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value < SECOND_TIMESTAMP_CUTOFF ? value * SECONDS_TO_MS : value
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return null
    const parsed = Date.parse(trimmed)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

const formatUtcTimestamp = (timestamp: number): string =>
  new Date(timestamp)
    .toISOString()
    .replace(/\.\d{3}Z$/, " UTC")
    .replace("T", " ")

const formatRelativeTimestamp = (timestamp: number, nowMs: number): string => {
  const diffMs = timestamp - nowMs
  const absMs = Math.abs(diffMs)
  const suffixRelative = (label: string): string =>
    diffMs > 0 ? `in ${label}` : `${label} ago`
  const unitRelative = (count: number, singular: string, pluralUnit: string): string =>
    suffixRelative(count === 1 ? singular : `${count} ${pluralUnit}`)

  if (absMs < 45 * SECONDS_TO_MS) return suffixRelative("a few seconds")
  if (absMs < 90 * SECONDS_TO_MS) return suffixRelative("a minute")

  if (absMs < 45 * MINUTES_TO_MS) {
    return unitRelative(Math.round(absMs / MINUTES_TO_MS), "a minute", "minutes")
  }
  if (absMs < 90 * MINUTES_TO_MS) return suffixRelative("an hour")

  if (absMs < 22 * HOURS_TO_MS) {
    return unitRelative(Math.round(absMs / HOURS_TO_MS), "an hour", "hours")
  }
  if (absMs < 36 * HOURS_TO_MS) return suffixRelative("a day")

  if (absMs < 26 * DAYS_TO_MS) {
    return unitRelative(Math.round(absMs / DAYS_TO_MS), "a day", "days")
  }
  if (absMs < 46 * DAYS_TO_MS) return suffixRelative("a month")

  const months = Math.round(absMs / MONTHS_TO_MS)
  if (months < 11) return unitRelative(months, "a month", "months")
  if (months < 18) return suffixRelative("a year")

  const years = Math.round(absMs / YEARS_TO_MS)
  return unitRelative(years, "a year", "years")
}

export const formatWorldBookLastModified = (
  value: unknown,
  options?: { nowMs?: number }
): { relative: string; absolute: string | null; timestamp: number | null } => {
  const timestamp = parseWorldBookTimestamp(value)
  if (!timestamp) {
    return {
      relative: UNKNOWN_LAST_MODIFIED_LABEL,
      absolute: null,
      timestamp: null
    }
  }

  const nowMs =
    typeof options?.nowMs === "number" && Number.isFinite(options.nowMs)
      ? options.nowMs
      : Date.now()

  return {
    relative: formatRelativeTimestamp(timestamp, nowMs),
    absolute: formatUtcTimestamp(timestamp),
    timestamp
  }
}
