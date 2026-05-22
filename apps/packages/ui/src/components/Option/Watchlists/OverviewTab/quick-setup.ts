import type {
  WatchlistJobCreate,
  WatchlistSourceCreate,
  SourceType
} from "@/types/watchlists"
import {
  buildCronFromPreset,
  normalizeWeekdayToken,
  parseScheduleTime,
  validateCronSchedule,
  type ScheduleIntervalUnit
} from "../JobsTab/schedule-utils"
import { normalizeWatchlistTemplateName } from "../shared/templateNames"

export type QuickSetupSchedulePreset = "none" | "hourly" | "daily" | "weekdays"
type WatchlistCadenceIntervalUnit = "minute" | "minutes" | "hour" | "hours"
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

export const getLocalTimezone = (): string => {
  const resolved = Intl.DateTimeFormat().resolvedOptions().timeZone
  return resolved || "UTC"
}

export const resolveQuickSetupSchedule = (
  schedule: QuickSetupSchedulePreset | WatchlistCadenceDraft
): { schedule_expr?: string; timezone?: string } => {
  if (typeof schedule === "object" && schedule != null) {
    if (schedule.kind === "manual") return {}
    if (schedule.kind === "advanced") {
      const cron = String(schedule.cron || "").trim()
      return cron && !validateCronSchedule(cron)
        ? { schedule_expr: cron, timezone: getLocalTimezone() }
        : {}
    }
    const time = parseScheduleTime("time" in schedule ? schedule.time : undefined)
    if (schedule.kind === "interval") {
      return {
        schedule_expr: buildCronFromPreset({
          preset: "interval",
          intervalValue: schedule.every,
          intervalUnit: normalizeCadenceIntervalUnit(schedule.unit),
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
  values: Pick<QuickSetupValues, "monitorName" | "schedulePreset" | "setupGoal" | "includeAudioBriefing">,
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
    ...resolveQuickSetupSchedule(values.schedulePreset || "daily")
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
