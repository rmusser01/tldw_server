import type {
  WatchlistJobCreate,
  WatchlistSourceCreate,
  SourceType
} from "@/types/watchlists"
import { normalizeWatchlistTemplateName } from "../shared/templateNames"

export type QuickSetupSchedulePreset = "none" | "hourly" | "daily" | "weekdays"
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

export const getLocalTimezone = (): string => {
  const resolved = Intl.DateTimeFormat().resolvedOptions().timeZone
  return resolved || "UTC"
}

export const resolveQuickSetupSchedule = (
  preset: QuickSetupSchedulePreset
): { schedule_expr?: string; timezone?: string } => {
  if (preset === "none") return {}
  return {
    schedule_expr: presetToCron[preset],
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
