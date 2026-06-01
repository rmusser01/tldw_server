export type ReminderRecurrencePreset = "daily" | "weekly" | "custom"

export type ReminderWeekdayToken = "SUN" | "MON" | "TUE" | "WED" | "THU" | "FRI" | "SAT"

export type CronValidationResult =
  | { valid: true; error: null }
  | { valid: false; error: string }

const WEEKDAY_MAP: Record<string, ReminderWeekdayToken> = {
  "0": "SUN",
  "1": "MON",
  "2": "TUE",
  "3": "WED",
  "4": "THU",
  "5": "FRI",
  "6": "SAT",
  "7": "SUN",
  SUN: "SUN",
  MON: "MON",
  TUE: "TUE",
  WED: "WED",
  THU: "THU",
  FRI: "FRI",
  SAT: "SAT"
}

const CRON_TOKEN_PATTERN = /^[A-Za-z0-9*#/,\-]+$/
const WEEKDAY_LABELS: Record<ReminderWeekdayToken, string> = {
  SUN: "Sunday",
  MON: "Monday",
  TUE: "Tuesday",
  WED: "Wednesday",
  THU: "Thursday",
  FRI: "Friday",
  SAT: "Saturday"
}

const clampInteger = (value: unknown, min: number, max: number): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return min
  return Math.min(max, Math.max(min, Math.floor(parsed)))
}

export const getDefaultReminderTimezone = (): string => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
  } catch {
    return "UTC"
  }
}

export const normalizeReminderWeekday = (value: unknown): ReminderWeekdayToken => {
  if (typeof value !== "string") return "MON"
  return WEEKDAY_MAP[value.toUpperCase()] || "MON"
}

export const buildDailyCron = (hour: unknown, minute: unknown): string => {
  return `${clampInteger(minute, 0, 59)} ${clampInteger(hour, 0, 23)} * * *`
}

export const buildWeeklyCron = (
  weekday: unknown,
  hour: unknown,
  minute: unknown
): string => {
  return `${clampInteger(minute, 0, 59)} ${clampInteger(hour, 0, 23)} * * ${normalizeReminderWeekday(weekday)}`
}

export const datetimeLocalToIsoString = (value: string | null | undefined): string | null => {
  const trimmed = value?.trim() || ""
  if (!trimmed) return null

  const date = new Date(trimmed)
  if (Number.isNaN(date.getTime())) return null
  return date.toISOString()
}

export const isoStringToDatetimeLocal = (value: string | null | undefined): string => {
  if (!value) return ""
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ""

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  const hour = String(date.getHours()).padStart(2, "0")
  const minute = String(date.getMinutes()).padStart(2, "0")
  return `${year}-${month}-${day}T${hour}:${minute}`
}

export const validateCronExpression = (
  expression: string | null | undefined
): CronValidationResult => {
  const normalized = expression?.trim() || ""
  if (!normalized) {
    return { valid: false, error: "Cron is required for recurring reminders" }
  }

  const fields = normalized.split(/\s+/)
  if (fields.length !== 5) {
    return { valid: false, error: "Cron must have exactly 5 fields" }
  }
  if (fields.some((field) => field.includes("?"))) {
    return { valid: false, error: "Cron field '?' is not supported by the scheduler." }
  }
  if (fields.some((field) => !CRON_TOKEN_PATTERN.test(field))) {
    return {
      valid: false,
      error: "Cron tokens can only include letters, numbers, *, /, -, #, and comma."
    }
  }
  return { valid: true, error: null }
}

export const buildReminderCron = (
  preset: ReminderRecurrencePreset,
  weekday: unknown,
  hour: unknown,
  minute: unknown,
  customCron: string | null | undefined
): string => {
  if (preset === "weekly") return buildWeeklyCron(weekday, hour, minute)
  if (preset === "custom") return customCron?.trim() || ""
  return buildDailyCron(hour, minute)
}

export type ParsedReminderCron = {
  preset: ReminderRecurrencePreset
  hour: number
  minute: number
  weekday: ReminderWeekdayToken
}

export const parseReminderCron = (
  expression: string | null | undefined
): ParsedReminderCron | null => {
  const fields = expression?.trim().split(/\s+/) || []
  if (fields.length !== 5) return null

  const [minuteToken, hourToken, dayOfMonthToken, monthToken, dayOfWeekToken] = fields
  if (dayOfMonthToken !== "*" || monthToken !== "*") return null

  const minute = Number(minuteToken)
  const hour = Number(hourToken)
  if (!Number.isInteger(minute) || minute < 0 || minute > 59) return null
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) return null

  if (dayOfWeekToken === "*") {
    return { preset: "daily", hour, minute, weekday: "MON" }
  }

  const weekday = WEEKDAY_MAP[dayOfWeekToken.toUpperCase()]
  if (!weekday) return null
  return { preset: "weekly", hour, minute, weekday }
}

export const getOneTimePreviewCopy = (runAt: string | null | undefined): string => {
  const iso = datetimeLocalToIsoString(runAt)
  if (!iso) return "Choose a date and time to preview the first run."
  return `Runs once at ${new Date(iso).toLocaleString()}`
}

export const getRecurringPreviewCopy = (
  preset: ReminderRecurrencePreset,
  cron: string | null | undefined,
  timezone: string | null | undefined
): string => {
  const normalizedCron = cron?.trim() || ""
  const normalizedTimezone = timezone?.trim() || ""
  if (!normalizedCron) return "Choose a repeat schedule to preview the next eligible run."
  const validation = validateCronExpression(normalizedCron)
  if (!validation.valid) return validation.error

  const parsedCron = parseReminderCron(normalizedCron)
  const timezoneCopy = normalizedTimezone || "the selected timezone"

  if (preset === "daily" && parsedCron?.preset === "daily") {
    const time = `${String(parsedCron.hour).padStart(2, "0")}:${String(parsedCron.minute).padStart(2, "0")}`
    return `Next run: the next daily ${time} occurrence in ${timezoneCopy}.`
  }

  if (preset === "weekly" && parsedCron?.preset === "weekly") {
    const time = `${String(parsedCron.hour).padStart(2, "0")}:${String(parsedCron.minute).padStart(2, "0")}`
    return `Next run: the next ${WEEKDAY_LABELS[parsedCron.weekday]} ${time} occurrence in ${timezoneCopy}.`
  }

  return `Next eligible run: the scheduler will evaluate custom cron "${normalizedCron}" in ${timezoneCopy}.`
}
