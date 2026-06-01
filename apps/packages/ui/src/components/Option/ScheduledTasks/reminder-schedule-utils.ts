export type ReminderRecurrencePreset = "daily" | "weekly" | "custom"

export type ReminderWeekdayToken = "SUN" | "MON" | "TUE" | "WED" | "THU" | "FRI" | "SAT"

export type CronValidationResult =
  | { valid: true; error: null }
  | { valid: false; error: string }

const WEEKDAY_MAP: Record<string, ReminderWeekdayToken> = {
  "0": "MON",
  "1": "TUE",
  "2": "WED",
  "3": "THU",
  "4": "FRI",
  "5": "SAT",
  "6": "SUN",
  SUN: "SUN",
  MON: "MON",
  TUE: "TUE",
  WED: "WED",
  THU: "THU",
  FRI: "FRI",
  SAT: "SAT"
}

const CRON_TOKEN_PATTERN = /^[A-Za-z0-9*#/,\-]+$/
const MONTH_MAP: Record<string, number> = {
  JAN: 1,
  FEB: 2,
  MAR: 3,
  APR: 4,
  MAY: 5,
  JUN: 6,
  JUL: 7,
  AUG: 8,
  SEP: 9,
  OCT: 10,
  NOV: 11,
  DEC: 12
}
const WEEKDAY_ORDER_MAP: Record<string, number> = {
  "0": 0,
  "1": 1,
  "2": 2,
  "3": 3,
  "4": 4,
  "5": 5,
  "6": 6,
  MON: 0,
  TUE: 1,
  WED: 2,
  THU: 3,
  FRI: 4,
  SAT: 5,
  SUN: 6
}
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

type CronFieldOptions = {
  label: string
  min: number
  max: number
  names?: Record<string, unknown>
  orderValues?: Record<string, number>
  allowNthWeekday?: boolean
}

const parseCronFieldValue = (
  value: string,
  options: CronFieldOptions
): CronValidationResult => {
  const upperValue = value.toUpperCase()
  if (options.names?.[upperValue] !== undefined) {
    return { valid: true, error: null }
  }

  if (!/^\d+$/.test(value)) {
    return {
      valid: false,
      error: `Cron ${options.label} must be a number, range, step, list, or wildcard.`
    }
  }

  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < options.min || parsed > options.max) {
    return {
      valid: false,
      error: `Cron ${options.label} must be between ${options.min} and ${options.max}.`
    }
  }

  return { valid: true, error: null }
}

const getCronFieldOrderValue = (
  value: string,
  options: CronFieldOptions
): number | null => {
  const upperValue = value.toUpperCase()
  const namedOrder = options.orderValues?.[upperValue]
  if (namedOrder !== undefined) return namedOrder

  if (!/^\d+$/.test(value)) return null
  return Number(value)
}

const isNamedCronFieldValue = (
  value: string,
  options: CronFieldOptions
): boolean => {
  return !/^\d+$/.test(value) && options.names?.[value.toUpperCase()] !== undefined
}

const validateCronFieldBase = (
  base: string,
  options: CronFieldOptions
): CronValidationResult => {
  if (base === "*") {
    return { valid: true, error: null }
  }

  if (options.allowNthWeekday && base.includes("#")) {
    const [weekday, nth, extra] = base.split("#")
    if (extra !== undefined || !weekday || !nth) {
      return {
        valid: false,
        error: "Cron day of week nth-weekday syntax must look like mon#2."
      }
    }

    const weekdayResult = parseCronFieldValue(weekday, options)
    if (!weekdayResult.valid) return weekdayResult

    const nthNumber = Number(nth)
    if (!Number.isInteger(nthNumber) || nthNumber < 0 || nthNumber > 6) {
      return {
        valid: false,
        error: "Cron day of week nth-weekday value must be between 0 and 6."
      }
    }
    return { valid: true, error: null }
  }

  if (base.includes("-")) {
    const [start, end, extra] = base.split("-")
    if (extra !== undefined || !start) {
      return {
        valid: false,
        error: `Cron ${options.label} range must look like ${options.min}-${options.max}.`
      }
    }

    const startResult = parseCronFieldValue(start, options)
    if (!startResult.valid) return startResult

    if (!end) {
      if (isNamedCronFieldValue(start, options)) {
        return { valid: true, error: null }
      }
      return {
        valid: false,
        error: `Cron ${options.label} range must look like ${options.min}-${options.max}.`
      }
    }

    const endResult = parseCronFieldValue(end, options)
    if (!endResult.valid) return endResult

    if (/^\d+$/.test(start) && isNamedCronFieldValue(end, options)) {
      return {
        valid: false,
        error: `Cron ${options.label} range cannot start with a number and end with a name.`
      }
    }

    const startOrder = getCronFieldOrderValue(start, options)
    const endOrder = getCronFieldOrderValue(end, options)
    const isNameToNumericRange = isNamedCronFieldValue(start, options) && /^\d+$/.test(end)
    if (!isNameToNumericRange && startOrder !== null && endOrder !== null && startOrder > endOrder) {
      return {
        valid: false,
        error: `Cron ${options.label} range start must be less than or equal to the end.`
      }
    }
    return { valid: true, error: null }
  }

  return parseCronFieldValue(base, options)
}

const validateCronField = (
  field: string,
  options: CronFieldOptions
): CronValidationResult => {
  if (!field) {
    return {
      valid: false,
      error: `Cron ${options.label} is required.`
    }
  }

  for (const item of field.split(",")) {
    if (!item) {
      return {
        valid: false,
        error: `Cron ${options.label} list contains an empty value.`
      }
    }

    const [base, step, extra] = item.split("/")
    if (extra !== undefined || !base) {
      return {
        valid: false,
        error: `Cron ${options.label} step must look like */5 or 1-10/2.`
      }
    }

    if (step !== undefined) {
      if (!/^\d+$/.test(step) || Number(step) < 1) {
        return {
          valid: false,
          error: `Cron ${options.label} step must be a positive number.`
        }
      }
    }

    const baseResult = validateCronFieldBase(base, options)
    if (!baseResult.valid) return baseResult
  }

  return { valid: true, error: null }
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

  const [minute, hour, day, month, dayOfWeek] = fields
  const fieldValidations = [
    validateCronField(minute, { label: "minute", min: 0, max: 59 }),
    validateCronField(hour, { label: "hour", min: 0, max: 23 }),
    validateCronField(day, { label: "day", min: 1, max: 31 }),
    validateCronField(month, {
      label: "month",
      min: 1,
      max: 12,
      names: MONTH_MAP,
      orderValues: MONTH_MAP
    }),
    validateCronField(dayOfWeek, {
      label: "day of week",
      min: 0,
      max: 6,
      names: WEEKDAY_MAP,
      orderValues: WEEKDAY_ORDER_MAP,
      allowNthWeekday: true
    })
  ]
  const invalidField = fieldValidations.find((result) => !result.valid)
  if (invalidField) return invalidField

  return { valid: true, error: null }
}

export const validateReminderTimezone = (
  timezone: string | null | undefined
): CronValidationResult => {
  const normalized = timezone?.trim() || ""
  if (!normalized) {
    return { valid: false, error: "Timezone is required for recurring reminders" }
  }

  try {
    Intl.DateTimeFormat(undefined, { timeZone: normalized }).format(new Date())
    return { valid: true, error: null }
  } catch {
    return { valid: false, error: "Timezone must be a valid IANA timezone." }
  }
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
