const SECONDS_TO_MS = 1000
// Epoch seconds remain below this until 2286; millisecond timestamps exceed it after 1970.
const SECOND_TIMESTAMP_CUTOFF = 10_000_000_000
const MINUTES_TO_MS = 60 * SECONDS_TO_MS
const HOURS_TO_MS = 60 * MINUTES_TO_MS
const DAYS_TO_MS = 24 * HOURS_TO_MS
const MONTHS_TO_MS = 30 * DAYS_TO_MS
const YEARS_TO_MS = 365 * DAYS_TO_MS

export interface FlashcardTimestampDisplay {
  absolute: string
  relative: string
  timestamp: number
}

export const parseFlashcardTimestamp = (value: unknown): number | null => {
  if (value == null) return null
  if (value instanceof Date) {
    const timestamp = value.getTime()
    return Number.isFinite(timestamp) ? timestamp : null
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.abs(value) < SECOND_TIMESTAMP_CUTOFF
      ? value * SECONDS_TO_MS
      : value
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return null
    const timestamp = Date.parse(trimmed)
    return Number.isFinite(timestamp) ? timestamp : null
  }
  return null
}

const padDatePart = (value: number): string => value.toString().padStart(2, "0")

const formatFlashcardAbsoluteDateTimeFromMs = (timestamp: number): string => {
  const date = new Date(timestamp)
  const year = date.getFullYear()
  const month = padDatePart(date.getMonth() + 1)
  const day = padDatePart(date.getDate())
  const hours = padDatePart(date.getHours())
  const minutes = padDatePart(date.getMinutes())

  return `${year}-${month}-${day} ${hours}:${minutes}`
}

export const formatFlashcardAbsoluteDateTime = (value: unknown): string | null => {
  const timestamp = parseFlashcardTimestamp(value)
  return timestamp == null ? null : formatFlashcardAbsoluteDateTimeFromMs(timestamp)
}

const formatFlashcardRelativeTimeFromMs = (
  timestamp: number,
  options?: { nowMs?: number }
): string => {
  const nowMs =
    typeof options?.nowMs === "number" && Number.isFinite(options.nowMs)
      ? options.nowMs
      : Date.now()
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

export const formatFlashcardRelativeTime = (
  value: unknown,
  options?: { nowMs?: number }
): string | null => {
  const timestamp = parseFlashcardTimestamp(value)
  return timestamp == null ? null : formatFlashcardRelativeTimeFromMs(timestamp, options)
}

export const formatFlashcardTimestampWithRelative = (
  value: unknown,
  options?: { nowMs?: number }
): FlashcardTimestampDisplay | null => {
  const timestamp = parseFlashcardTimestamp(value)
  if (timestamp == null) return null

  return {
    absolute: formatFlashcardAbsoluteDateTimeFromMs(timestamp),
    relative: formatFlashcardRelativeTimeFromMs(timestamp, options),
    timestamp
  }
}
