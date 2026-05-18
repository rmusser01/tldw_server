import { MIN_SCHEDULE_INTERVAL_MINUTES } from "./schedule-frequency"

export type SchedulePresetKey = "interval" | "daily" | "weekdays" | "weekly"

export type ScheduleIntervalUnit = "minutes" | "hours"

export type WeekdayToken = "SUN" | "MON" | "TUE" | "WED" | "THU" | "FRI" | "SAT"

export interface PresetScheduleState {
  preset: SchedulePresetKey
  intervalValue: number
  intervalUnit: ScheduleIntervalUnit
  hour: number
  minute: number
  weekday: WeekdayToken
}

export const INTERVAL_MINUTES_MIN = MIN_SCHEDULE_INTERVAL_MINUTES
export const INTERVAL_MINUTES_MAX = 59
export const INTERVAL_HOURS_MIN = 1
export const INTERVAL_HOURS_MAX = 23

const WEEKDAY_MAP: Record<string, WeekdayToken> = {
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

const DEFAULT_PRESET_STATE: PresetScheduleState = {
  preset: "daily",
  intervalValue: 1,
  intervalUnit: "hours",
  hour: 9,
  minute: 0,
  weekday: "MON"
}

const clampInteger = (value: unknown, min: number, max: number): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return min
  return Math.min(max, Math.max(min, Math.floor(parsed)))
}

export const normalizeWeekdayToken = (value: unknown): WeekdayToken => {
  if (typeof value !== "string") return DEFAULT_PRESET_STATE.weekday
  return WEEKDAY_MAP[value.toUpperCase()] || DEFAULT_PRESET_STATE.weekday
}

export const normalizeIntervalUnit = (value: unknown): ScheduleIntervalUnit => {
  return value === "minutes" ? "minutes" : "hours"
}

const parseStepToken = (token: string, min: number, max: number): number | null => {
  if (!token.startsWith("*/")) return null
  const parsed = Number(token.slice(2))
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) return null
  return parsed
}

export const buildCronFromPreset = (state: PresetScheduleState): string => {
  const minute = clampInteger(state.minute, 0, 59)
  const hour = clampInteger(state.hour, 0, 23)
  const weekday = normalizeWeekdayToken(state.weekday)
  const intervalUnit = normalizeIntervalUnit(state.intervalUnit)

  switch (state.preset) {
    case "interval": {
      if (intervalUnit === "minutes") {
        const intervalMinutes = clampInteger(
          state.intervalValue,
          INTERVAL_MINUTES_MIN,
          INTERVAL_MINUTES_MAX
        )
        return `*/${intervalMinutes} * * * *`
      }
      const intervalHours = clampInteger(
        state.intervalValue,
        INTERVAL_HOURS_MIN,
        INTERVAL_HOURS_MAX
      )
      return `${minute} */${intervalHours} * * *`
    }
    case "weekdays":
      return `${minute} ${hour} * * MON-FRI`
    case "weekly":
      return `${minute} ${hour} * * ${weekday}`
    case "daily":
    default:
      return `${minute} ${hour} * * *`
  }
}

export const parsePresetFromCron = (
  expression: string | null | undefined
): PresetScheduleState | null => {
  if (!expression) return null
  const parts = expression.trim().split(/\s+/)
  if (parts.length !== 5) return null

  const [minuteToken, hourToken, dayOfMonthToken, monthToken, dayOfWeekToken] = parts
  if (dayOfMonthToken !== "*" || monthToken !== "*") return null

  const minuteStep = parseStepToken(
    minuteToken,
    INTERVAL_MINUTES_MIN,
    INTERVAL_MINUTES_MAX
  )
  if (minuteStep !== null && hourToken === "*" && dayOfWeekToken === "*") {
    return {
      ...DEFAULT_PRESET_STATE,
      preset: "interval",
      intervalValue: minuteStep,
      intervalUnit: "minutes"
    }
  }

  const minute = Number(minuteToken)
  if (!Number.isInteger(minute) || minute < 0 || minute > 59) return null

  if (hourToken === "*" && dayOfWeekToken === "*") {
    return {
      ...DEFAULT_PRESET_STATE,
      preset: "interval",
      intervalValue: 1,
      intervalUnit: "hours",
      minute
    }
  }

  const hourStep = parseStepToken(hourToken, 1, 23)
  if (hourStep !== null && dayOfWeekToken === "*") {
    return {
      ...DEFAULT_PRESET_STATE,
      preset: "interval",
      intervalValue: hourStep,
      intervalUnit: "hours",
      minute
    }
  }

  const hour = Number(hourToken)
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) return null

  if (dayOfWeekToken === "*") {
    return {
      ...DEFAULT_PRESET_STATE,
      preset: "daily",
      hour,
      minute
    }
  }

  if (dayOfWeekToken.toUpperCase() === "MON-FRI") {
    return {
      ...DEFAULT_PRESET_STATE,
      preset: "weekdays",
      hour,
      minute
    }
  }

  const weekday = WEEKDAY_MAP[dayOfWeekToken.toUpperCase()]
  if (!weekday) return null
  return {
    ...DEFAULT_PRESET_STATE,
    preset: "weekly",
    hour,
    minute,
    weekday
  }
}

export const createDefaultPresetState = (): PresetScheduleState => ({
  ...DEFAULT_PRESET_STATE
})
