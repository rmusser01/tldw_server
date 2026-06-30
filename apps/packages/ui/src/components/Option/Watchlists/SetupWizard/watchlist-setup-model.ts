import type {
  SourceType,
  WatchlistCreate,
  WatchlistDomain,
  WatchlistJobCreate,
  WatchlistPriority,
  WatchlistSourceCreate
} from "@/types/watchlists"

export type WatchlistSetupPreset = "cti_osint" | "news" | "general" | "blank"
export type WatchlistSetupStartMode = "sources" | "topic" | "report_goal"
export type WatchlistSetupDestination = "sources" | "jobs" | "outputs"
export type WatchlistSetupSchedulePreset = "none" | "hourly" | "daily" | "weekdays"

interface WatchlistSetupPresetDefaults {
  domain: WatchlistDomain
  priority: WatchlistPriority
  tags: string[]
  objectivePlaceholder: string
  trackedScopePlaceholder: string
  reportGoalPlaceholder: string
}

export interface WatchlistSetupValues {
  preset: WatchlistSetupPreset
  startMode: WatchlistSetupStartMode
  name: string
  objective: string
  trackedScopeText: string
  sourceUrlsText: string
  sourceName?: string
  sourceType?: SourceType
  monitorName?: string
  reportGoal?: string
  includeAudioBriefing?: boolean
  schedulePreset?: WatchlistSetupSchedulePreset
}

export interface WatchlistSetupPlan {
  watchlist: WatchlistCreate
  sources: WatchlistSourceCreate[]
  canCreateMonitor: boolean
  destination: WatchlistSetupDestination
}

export const WATCHLIST_SETUP_PRESETS: Record<WatchlistSetupPreset, WatchlistSetupPresetDefaults> = {
  cti_osint: {
    domain: "cti_osint",
    priority: "high",
    tags: ["cti", "osint"],
    objectivePlaceholder: "Track vulnerabilities, malware, actors, advisories, and source changes.",
    trackedScopePlaceholder: "CVEs, ransomware families, sectors, regions, vendors",
    reportGoalPlaceholder: "Daily situational brief with source provenance"
  },
  news: {
    domain: "news",
    priority: "medium",
    tags: ["news"],
    objectivePlaceholder: "Track a developing event, person, organization, or topic.",
    trackedScopePlaceholder: "People, organizations, locations, storylines",
    reportGoalPlaceholder: "Concise briefing with recency and source diversity"
  },
  general: {
    domain: "general",
    priority: "medium",
    tags: ["research"],
    objectivePlaceholder: "Track a topic and collect updates for review.",
    trackedScopePlaceholder: "Topics, sources, keywords, organizations",
    reportGoalPlaceholder: "Summary of relevant changes"
  },
  blank: {
    domain: "general",
    priority: "medium",
    tags: [],
    objectivePlaceholder: "",
    trackedScopePlaceholder: "",
    reportGoalPlaceholder: ""
  }
}

const presetOrDefault = (preset: WatchlistSetupPreset): WatchlistSetupPresetDefaults =>
  WATCHLIST_SETUP_PRESETS[preset] || WATCHLIST_SETUP_PRESETS.blank

const trimText = (value: unknown): string => String(value || "").trim()

const unique = (values: string[]): string[] => Array.from(new Set(values.filter(Boolean)))

const parseTrackedScopeTags = (value: string): string[] =>
  unique(
    trimText(value)
      .split(/\r?\n|,/)
      .map((entry) => entry.trim().toLowerCase())
      .filter((entry) => entry.length > 0)
  )

const getLocalTimezone = (): string => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
  } catch {
    return "UTC"
  }
}

const resolveSchedule = (
  preset: WatchlistSetupSchedulePreset | undefined
): Pick<WatchlistJobCreate, "schedule_expr" | "timezone"> => {
  switch (preset || "daily") {
    case "none":
      return {}
    case "hourly":
      return { schedule_expr: "0 * * * *", timezone: getLocalTimezone() }
    case "weekdays":
      return { schedule_expr: "0 8 * * MON-FRI", timezone: getLocalTimezone() }
    case "daily":
    default:
      return { schedule_expr: "0 8 * * *", timezone: getLocalTimezone() }
  }
}

const sourceNameFromUrl = (url: string, fallback: string): string => {
  try {
    return new URL(url).hostname || fallback
  } catch {
    return fallback
  }
}

export const applyWatchlistSetupPreset = (
  preset: WatchlistSetupPreset
): WatchlistSetupPresetDefaults => ({
  ...presetOrDefault(preset),
  tags: [...presetOrDefault(preset).tags]
})

export const parseSetupSourceUrls = (value: string): string[] =>
  Array.from(
    new Set(
      trimText(value)
        .split(/\r?\n|,/)
        .map((entry) => entry.trim())
        .filter((entry) => {
          try {
            const parsed = new URL(entry)
            return parsed.protocol === "http:" || parsed.protocol === "https:"
          } catch {
            return false
          }
        })
    )
  )

export const buildWatchlistSetupPlan = (
  values: WatchlistSetupValues
): WatchlistSetupPlan => {
  const preset = presetOrDefault(values.preset)
  const scopeText = trimText(values.trackedScopeText)
  const reportGoal = trimText(values.reportGoal)
  const descriptionParts = [
    scopeText ? `Tracked scope: ${scopeText}` : "",
    reportGoal ? `Report goal: ${reportGoal}` : ""
  ].filter(Boolean)
  const scopeTags = parseTrackedScopeTags(scopeText)
  const sources = parseSetupSourceUrls(values.sourceUrlsText).map((url, index) => ({
    name:
      index === 0 && trimText(values.sourceName)
        ? trimText(values.sourceName)
        : sourceNameFromUrl(url, `Feed ${index + 1}`),
    url,
    source_type: values.sourceType || "rss",
    active: true
  }))

  const canCreateMonitor = sources.length > 0 && trimText(values.monitorName || values.name).length > 0
  const destination: WatchlistSetupDestination =
    values.startMode === "topic"
      ? "sources"
      : values.startMode === "report_goal" && sources.length === 0
        ? "outputs"
        : canCreateMonitor
          ? "jobs"
          : "sources"

  return {
    watchlist: {
      name: trimText(values.name),
      description: descriptionParts.length > 0 ? descriptionParts.join("\n") : undefined,
      objective: trimText(values.objective) || undefined,
      domain: preset.domain,
      status: "active",
      priority: preset.priority,
      tags: unique([...preset.tags, ...scopeTags])
    },
    sources,
    canCreateMonitor,
    destination
  }
}

export const buildWatchlistSetupJobPayload = (
  values: WatchlistSetupValues,
  sourceIds: number[]
): WatchlistJobCreate => {
  const normalizedSourceIds = Array.from(
    new Set(
      (Array.isArray(sourceIds) ? sourceIds : [])
        .map((id) => Number(id))
        .filter((id) => Number.isFinite(id) && id > 0)
    )
  )
  const reportGoal = trimText(values.reportGoal)
  const payload: WatchlistJobCreate = {
    name: trimText(values.monitorName) || `${trimText(values.name) || "Watchlist"} monitor`,
    description: reportGoal ? `Report goal: ${reportGoal}` : undefined,
    scope: { sources: normalizedSourceIds },
    active: true,
    ...resolveSchedule(values.schedulePreset)
  }

  if (values.startMode === "report_goal" || reportGoal.length > 0) {
    payload.output_prefs = {
      template_name: "briefing_md",
      generate_audio: Boolean(values.includeAudioBriefing)
    }
  }

  return payload
}
