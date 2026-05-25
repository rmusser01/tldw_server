import type {
  WatchlistJobCreate,
  WatchlistSourceCreate,
  SourceType
} from "@/types/watchlists"
import {
  buildCronFromPreset,
  formatScheduleTimeValue,
  INTERVAL_HOURS_MAX,
  INTERVAL_HOURS_MIN,
  INTERVAL_MINUTES_MAX,
  INTERVAL_MINUTES_MIN,
  normalizeWeekdayToken,
  parseScheduleTime,
  validateCronSchedule,
  type ScheduleIntervalUnit,
  type WeekdayToken
} from "../JobsTab/schedule-utils"
import { normalizeWatchlistTemplateName } from "../shared/templateNames"

export type QuickSetupSchedulePreset = "none" | "hourly" | "daily" | "weekdays"
export type QuickSetupScheduleMode = WatchlistCadenceDraft["kind"]
export type WatchlistCadenceIntervalUnit = "minute" | "minutes" | "hour" | "hours"
export type WatchlistCadenceDraft =
  | { kind: "manual" }
  | { kind: "interval"; every: number; unit: WatchlistCadenceIntervalUnit }
  | { kind: "daily"; time?: string }
  | { kind: "weekdays"; time?: string }
  | { kind: "weekly"; weekday: string; time?: string }
  | { kind: "advanced"; cron: string }
export type QuickSetupGoal = "briefing" | "triage"

export interface QuickSetupValues {
  sourceName: string
  sourceUrl: string
  extraSourceUrls: string
  sourceType: SourceType
  monitorName: string
  schedulePreset: QuickSetupSchedulePreset
  scheduleCadence?: WatchlistCadenceDraft
  runNow: boolean
  setupGoal: QuickSetupGoal
  includeAudioBriefing: boolean
}

export const QUICK_SETUP_DEFAULT_VALUES: QuickSetupValues = {
  sourceName: "",
  sourceUrl: "",
  extraSourceUrls: "",
  sourceType: "rss",
  monitorName: "",
  schedulePreset: "daily",
  scheduleCadence: { kind: "daily", time: "08:00" },
  runNow: true,
  setupGoal: "briefing",
  includeAudioBriefing: true
}

const presetToCron: Record<Exclude<QuickSetupSchedulePreset, "none">, string> = {
  hourly: "0 * * * *",
  daily: "0 8 * * *",
  weekdays: "0 8 * * MON-FRI"
}

const normalizeCadenceIntervalUnit = (
  unit: WatchlistCadenceIntervalUnit
): ScheduleIntervalUnit => {
  return unit === "minute" || unit === "minutes" ? "minutes" : "hours"
}

const isWatchlistCadenceDraft = (
  value: QuickSetupSchedulePreset | WatchlistCadenceDraft | null | undefined
): value is WatchlistCadenceDraft =>
  Boolean(value) && typeof value === "object" && "kind" in value

const clampInteger = (value: unknown, min: number, max: number): number => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return min
  return Math.min(max, Math.max(min, Math.floor(parsed)))
}

export const normalizeQuickSetupIntervalCadence = (
  every: unknown,
  unit: WatchlistCadenceIntervalUnit
): Extract<WatchlistCadenceDraft, { kind: "interval" }> => {
  const normalizedUnit = normalizeCadenceIntervalUnit(unit)
  const min = normalizedUnit === "minutes" ? INTERVAL_MINUTES_MIN : INTERVAL_HOURS_MIN
  const max = normalizedUnit === "minutes" ? INTERVAL_MINUTES_MAX : INTERVAL_HOURS_MAX
  return {
    kind: "interval",
    every: clampInteger(every, min, max),
    unit: normalizedUnit
  }
}

export const createDefaultQuickSetupCadenceDraft = (
  kind: QuickSetupScheduleMode,
  current?: WatchlistCadenceDraft
): WatchlistCadenceDraft => {
  const time =
    "time" in (current || {})
      ? formatScheduleTimeValue((current as Extract<WatchlistCadenceDraft, { time?: string }>).time, 8, 0)
      : "08:00"
  if (kind === "manual") return { kind: "manual" }
  if (kind === "interval") {
    return normalizeQuickSetupIntervalCadence(
      current?.kind === "interval" ? current.every : 5,
      current?.kind === "interval" ? current.unit : "hours"
    )
  }
  if (kind === "daily") return { kind: "daily", time }
  if (kind === "weekdays") return { kind: "weekdays", time }
  if (kind === "weekly") {
    return {
      kind: "weekly",
      weekday: current?.kind === "weekly" ? normalizeWeekdayToken(current.weekday) : "MON",
      time
    }
  }
  return {
    kind: "advanced",
    cron: current?.kind === "advanced" ? current.cron : ""
  }
}

export const legacyPresetToQuickSetupCadenceDraft = (
  preset: QuickSetupSchedulePreset | null | undefined
): WatchlistCadenceDraft => {
  if (preset === "none") return { kind: "manual" }
  if (preset === "hourly") return { kind: "interval", every: 1, unit: "hours" }
  if (preset === "weekdays") return { kind: "weekdays", time: "08:00" }
  return { kind: "daily", time: "08:00" }
}

export const cadenceDraftToLegacyPreset = (
  draft: WatchlistCadenceDraft | null | undefined
): QuickSetupSchedulePreset => {
  if (!draft || draft.kind === "daily") return "daily"
  if (draft.kind === "manual") return "none"
  if (draft.kind === "weekdays") return "weekdays"
  if (draft.kind === "interval" && Number(draft.every) === 1 && normalizeCadenceIntervalUnit(draft.unit) === "hours") {
    return "hourly"
  }
  return "daily"
}

const WEEKDAY_LABELS: Record<WeekdayToken, string> = {
  SUN: "Sunday",
  MON: "Monday",
  TUE: "Tuesday",
  WED: "Wednesday",
  THU: "Thursday",
  FRI: "Friday",
  SAT: "Saturday"
}

export const formatQuickSetupCadenceLabel = (
  draft: WatchlistCadenceDraft | QuickSetupSchedulePreset | null | undefined,
  copy?: Partial<{
    manual: string
    hourly: string
    daily: (time: string) => string
    weekdays: (time: string) => string
    weekly: (weekday: WeekdayToken, time: string) => string
    interval: (value: number, unit: ScheduleIntervalUnit) => string
    advanced: (cron: string) => string
  }>
): string => {
  const cadence = isWatchlistCadenceDraft(draft)
    ? draft
    : legacyPresetToQuickSetupCadenceDraft(draft)
  if (cadence.kind === "manual") return copy?.manual || "Manual only"
  if (cadence.kind === "interval") {
    const interval = normalizeQuickSetupIntervalCadence(cadence.every, cadence.unit)
    const every = interval.every
    const unit = normalizeCadenceIntervalUnit(interval.unit)
    if (every === 1 && unit === "hours" && copy?.hourly) return copy.hourly
    if (copy?.interval) return copy.interval(every, unit)
    return `Every ${every} ${unit === "minutes" ? "minute" : "hour"}${every === 1 ? "" : "s"}`
  }
  if (cadence.kind === "advanced") {
    const cron = String(cadence.cron || "").trim()
    if (copy?.advanced) return copy.advanced(cron)
    return cron ? `Custom cron: ${cron}` : "Advanced cron"
  }
  const time = formatScheduleTimeValue("time" in cadence ? cadence.time : undefined, 8, 0)
  if (cadence.kind === "weekdays") {
    return copy?.weekdays ? copy.weekdays(time) : `Weekdays at ${time}`
  }
  if (cadence.kind === "weekly") {
    const weekday = normalizeWeekdayToken(cadence.weekday)
    return copy?.weekly ? copy.weekly(weekday, time) : `${WEEKDAY_LABELS[weekday]} at ${time}`
  }
  return copy?.daily ? copy.daily(time) : `Daily at ${time}`
}

export const getLocalTimezone = (): string => {
  const resolved = Intl.DateTimeFormat().resolvedOptions().timeZone
  return resolved || "UTC"
}

export const resolveQuickSetupSchedule = (
  schedule: QuickSetupSchedulePreset | WatchlistCadenceDraft
): { schedule_expr?: string; timezone?: string } => {
  if (isWatchlistCadenceDraft(schedule)) {
    if (schedule.kind === "manual") return {}
    if (schedule.kind === "advanced") {
      const cron = String(schedule.cron || "").trim()
      return cron && !validateCronSchedule(cron)
        ? { schedule_expr: cron, timezone: getLocalTimezone() }
        : {}
    }
    const time = parseScheduleTime("time" in schedule ? schedule.time : undefined)
    if (schedule.kind === "interval") {
      const interval = normalizeQuickSetupIntervalCadence(schedule.every, schedule.unit)
      return {
        schedule_expr: buildCronFromPreset({
          preset: "interval",
          intervalValue: interval.every,
          intervalUnit: normalizeCadenceIntervalUnit(interval.unit),
          hour: time.hour,
          minute: time.minute,
          weekday: "MON"
        }),
        timezone: getLocalTimezone()
      }
    }
    if (schedule.kind === "weekdays") {
      return {
        schedule_expr: buildCronFromPreset({
          preset: "weekdays",
          intervalValue: 1,
          intervalUnit: "hours",
          hour: time.hour,
          minute: time.minute,
          weekday: "MON"
        }),
        timezone: getLocalTimezone()
      }
    }
    return {
      schedule_expr: buildCronFromPreset({
        preset: schedule.kind === "weekly" ? "weekly" : "daily",
        intervalValue: 1,
        intervalUnit: "hours",
        hour: time.hour,
        minute: time.minute,
        weekday: schedule.kind === "weekly" ? normalizeWeekdayToken(schedule.weekday) : "MON"
      }),
      timezone: getLocalTimezone()
    }
  }

  if (schedule === "none") return {}
  return {
    schedule_expr: presetToCron[schedule],
    timezone: getLocalTimezone()
  }
}

export const toQuickSetupSourcePayload = (
  values: Pick<QuickSetupValues, "sourceName" | "sourceUrl" | "sourceType">,
  watchlistId?: number | null
): WatchlistSourceCreate => {
  const payload: WatchlistSourceCreate = {
    name: String(values.sourceName || "").trim(),
    url: String(values.sourceUrl || "").trim(),
    source_type: values.sourceType || "rss",
    active: true
  }
  const normalizedWatchlistId = Number(watchlistId)
  if (Number.isFinite(normalizedWatchlistId) && normalizedWatchlistId > 0) {
    payload.watchlist_id = normalizedWatchlistId
  }
  return payload
}

export const toQuickSetupJobPayload = (
  values: Pick<QuickSetupValues, "monitorName" | "schedulePreset" | "scheduleCadence" | "setupGoal" | "includeAudioBriefing">,
  sourceIds: number[],
  watchlistId?: number | null
): WatchlistJobCreate => {
  const uniqueSourceIds = Array.from(
    new Set((Array.isArray(sourceIds) ? sourceIds : []).filter((id) => Number.isFinite(id) && id > 0))
  )
  const payload: WatchlistJobCreate = {
    name: String(values.monitorName || "").trim(),
    scope: { sources: uniqueSourceIds },
    active: true,
    ...resolveQuickSetupSchedule(
      values.scheduleCadence || legacyPresetToQuickSetupCadenceDraft(values.schedulePreset || "daily")
    )
  }
  const normalizedWatchlistId = Number(watchlistId)
  if (Number.isFinite(normalizedWatchlistId) && normalizedWatchlistId > 0) {
    payload.watchlist_id = normalizedWatchlistId
  }

  if ((values.setupGoal || "briefing") === "briefing") {
    const templateName = normalizeWatchlistTemplateName("briefing_md")
    payload.output_prefs = {
      template_name: templateName,
      template: {
        default_name: templateName
      },
      generate_audio: Boolean(values.includeAudioBriefing)
    }
  }

  return payload
}

export const parseQuickSetupExtraSourceUrls = (value: string): string[] => {
  const entries = String(value || "")
    .split(/\r?\n|,/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0)

  const normalized = entries.filter((entry) => {
    try {
      const parsed = new URL(entry)
      return parsed.protocol === "http:" || parsed.protocol === "https:"
    } catch {
      return false
    }
  })

  return Array.from(new Set(normalized))
}
