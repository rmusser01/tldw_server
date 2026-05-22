import type { JobOutputPrefs, JobScope, WatchlistFilter } from "@/types/watchlists"

type Translator = (...args: any[]) => unknown

const toText = (value: unknown): string =>
  typeof value === "string" ? value : String(value)

export interface ScopeNameCatalog {
  sources: Record<number, string>
  groups: Record<number, string>
}

export interface OverflowSummary {
  visible: string[]
  hiddenCount: number
  text: string
}

const pluralize = (count: number, singular: string): string =>
  `${count} ${singular}${count === 1 ? "" : "s"}`

const asStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0)
}

const summarizeValueList = (values: string[], maxVisible = 2): OverflowSummary => {
  const visible = values.slice(0, maxVisible)
  const hiddenCount = Math.max(0, values.length - visible.length)
  const text = hiddenCount > 0 ? `${visible.join(", ")} +${hiddenCount}` : visible.join(", ")
  return { visible, hiddenCount, text }
}

const extractFilterValueSummary = (filter: WatchlistFilter): string | null => {
  const value = filter.value || {}

  if (filter.type === "keyword") {
    const keywords = asStringArray((value as Record<string, unknown>).keywords)
    return keywords.length ? summarizeValueList(keywords).text : null
  }

  if (filter.type === "author") {
    const authors = asStringArray((value as Record<string, unknown>).authors)
    return authors.length ? summarizeValueList(authors).text : null
  }

  if (filter.type === "regex") {
    const pattern = (value as Record<string, unknown>).pattern
    if (typeof pattern === "string" && pattern.trim()) return pattern.trim()
    return null
  }

  if (filter.type === "date_range") {
    const start = (value as Record<string, unknown>).start_date
    const end = (value as Record<string, unknown>).end_date
    const from = typeof start === "string" && start.trim() ? start.trim() : "..."
    const to = typeof end === "string" && end.trim() ? end.trim() : "..."
    if (from === "..." && to === "...") return null
    return `${from} - ${to}`
  }

  const valueStrings = Object.values(value as Record<string, unknown>)
    .flatMap((entry) => (Array.isArray(entry) ? entry : [entry]))
    .map((entry) => (typeof entry === "string" ? entry.trim() : ""))
    .filter((entry) => entry.length > 0)

  return valueStrings.length ? summarizeValueList(valueStrings).text : null
}

const filterTypeLabel = (type: WatchlistFilter["type"], t: Translator): string => {
  const map: Record<WatchlistFilter["type"], string> = {
    keyword: toText(t("watchlists:jobs.filters.type.keyword", "keyword")),
    author: toText(t("watchlists:jobs.filters.type.author", "author")),
    date_range: toText(t("watchlists:jobs.filters.type.dateRange", "date range")),
    regex: toText(t("watchlists:jobs.filters.type.regex", "regex")),
    all: toText(t("watchlists:jobs.filters.type.all", "all"))
  }
  return map[type]
}

const filterActionLabel = (action: WatchlistFilter["action"], t: Translator): string => {
  const map: Record<WatchlistFilter["action"], string> = {
    include: toText(t("watchlists:jobs.filters.action.include", "Include")),
    exclude: toText(t("watchlists:jobs.filters.action.exclude", "Exclude")),
    flag: toText(t("watchlists:jobs.filters.action.flag", "Flag"))
  }
  return map[action]
}

const resolveNames = (
  ids: number[] | undefined,
  catalog: Record<number, string>,
  fallbackPrefix = "#"
): string[] => {
  if (!Array.isArray(ids)) return []
  return ids.map((id) => catalog[id] || `${fallbackPrefix}${id}`)
}

export const summarizeScopeCounts = (scope: JobScope, t: Translator): string => {
  const parts: string[] = []
  if (scope.sources?.length) {
    parts.push(
      pluralize(
        scope.sources.length,
        toText(t("watchlists:jobs.scope.summary.feed", "feed"))
      )
    )
  }
  if (scope.groups?.length) {
    parts.push(
      pluralize(
        scope.groups.length,
        toText(t("watchlists:jobs.scope.summary.group", "group"))
      )
    )
  }
  if (scope.tags?.length) {
    parts.push(
      pluralize(
        scope.tags.length,
        toText(t("watchlists:jobs.scope.summary.tag", "tag"))
      )
    )
  }
  return parts.length > 0
    ? parts.join(", ")
    : toText(t("watchlists:jobs.noFeeds", "No feeds selected"))
}

export const summarizeOverflowList = (
  values: string[],
  maxVisible = 3
): OverflowSummary => summarizeValueList(values, maxVisible)

export const buildScopeTooltipLines = (
  scope: JobScope,
  catalog: ScopeNameCatalog,
  t: Translator,
  maxVisiblePerSection = 3
): string[] => {
  const lines: string[] = []

  const sourceNames = resolveNames(scope.sources, catalog.sources)
  const groupNames = resolveNames(scope.groups, catalog.groups)
  const tagNames = scope.tags || []

  if (sourceNames.length > 0) {
    const summary = summarizeOverflowList(sourceNames, maxVisiblePerSection)
    lines.push(`${toText(t("watchlists:jobs.scope.sources", "Feeds"))}: ${summary.text}`)
  }
  if (groupNames.length > 0) {
    const summary = summarizeOverflowList(groupNames, maxVisiblePerSection)
    lines.push(`${toText(t("watchlists:jobs.scope.groups", "Groups"))}: ${summary.text}`)
  }
  if (tagNames.length > 0) {
    const summary = summarizeOverflowList(tagNames, maxVisiblePerSection)
    lines.push(`${toText(t("watchlists:jobs.scope.tags", "Tags"))}: ${summary.text}`)
  }

  if (lines.length === 0) {
    lines.push(toText(t("watchlists:jobs.noFeeds", "No feeds selected")))
  }

  return lines
}

export interface FilterSummary {
  count: number
  preview: string
  tooltipLines: string[]
}

export const summarizeFilters = (
  filters: WatchlistFilter[] | undefined,
  t: Translator
): FilterSummary => {
  const list = Array.isArray(filters) ? filters : []
  if (list.length === 0) {
    return { count: 0, preview: "-", tooltipLines: [] }
  }

  const tooltipLines = list.map((filter) => {
    const actionLabel = filterActionLabel(filter.action, t)
    const typeLabel = filterTypeLabel(filter.type, t)
    const valueSummary = extractFilterValueSummary(filter)
    return valueSummary
      ? `${actionLabel} ${typeLabel}: ${valueSummary}`
      : `${actionLabel} ${typeLabel}`
  })

  const preview =
    list.length === 1
      ? tooltipLines[0]
      : `${tooltipLines[0]} (${list.length - 1} more)`

  return {
    count: list.length,
    preview,
    tooltipLines
  }
}

const resolveTemplateName = (outputPrefs: JobOutputPrefs | null | undefined): string | null => {
  const nestedTemplate = outputPrefs?.template?.default_name
  if (typeof nestedTemplate === "string" && nestedTemplate.trim().length > 0) {
    return nestedTemplate.trim()
  }
  const legacyTemplate = outputPrefs?.template_name
  if (typeof legacyTemplate === "string" && legacyTemplate.trim().length > 0) {
    return legacyTemplate.trim()
  }
  return null
}

const isEnabledRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" &&
  value !== null &&
  !Array.isArray(value) &&
  (value as Record<string, unknown>).enabled !== false

export const summarizeOutputLinkage = (
  outputPrefs: JobOutputPrefs | null | undefined,
  t: Translator
): string => {
  const autoOutputSummary = outputPrefs?.auto_output?.enabled
    ? toText(
        t(
          "watchlists:jobs.outputLinkage.scheduledReport",
          "Create a report after each scheduled run"
        )
      )
    : toText(
        t(
          "watchlists:jobs.outputLinkage.manualReports",
          "Manual/test reports only"
        )
      )
  const templateName = resolveTemplateName(outputPrefs)
  const templateSummary = templateName
    ? toText(
        t("watchlists:jobs.outputLinkage.templateNamed", "Template: {{name}}", {
          name: templateName
        })
      )
    : toText(
        t("watchlists:jobs.outputLinkage.templateDefault", "Template: default")
      )

  const deliveryParts: string[] = []
  if (isEnabledRecord(outputPrefs?.deliveries?.email)) {
    deliveryParts.push(
      toText(t("watchlists:jobs.outputLinkage.email", "Deliver by email"))
    )
  }
  if (isEnabledRecord(outputPrefs?.deliveries?.chatbook)) {
    deliveryParts.push(
      toText(t("watchlists:jobs.outputLinkage.chatbook", "Save to Chatbook"))
    )
  }
  if (deliveryParts.length === 0) {
    deliveryParts.push(
      toText(t("watchlists:jobs.outputLinkage.reportsTabOnly", "Reports tab only"))
    )
  }

  const audioSummary = outputPrefs?.generate_audio
    ? toText(
        t("watchlists:jobs.outputLinkage.audioRequested", "Audio briefing requested")
      )
    : toText(t("watchlists:jobs.outputLinkage.audioTextOnly", "Text report only"))

  return [autoOutputSummary, templateSummary, ...deliveryParts, audioSummary].join(" • ")
}
