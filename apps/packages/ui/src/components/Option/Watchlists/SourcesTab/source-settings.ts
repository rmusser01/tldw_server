export interface SourceSettingsFormValues {
  scrape_list_url?: unknown
  scrape_item_selector?: unknown
  scrape_link_selector?: unknown
  scrape_title_selector?: unknown
  scrape_summary_selector?: unknown
  scrape_content_selector?: unknown
  scrape_date_selector?: unknown
  scrape_guid_selector?: unknown
  scrape_limit?: unknown
  source_top_n?: unknown
  discover_method?: unknown
  skip_article_fetch?: unknown
}

export interface SourceSettingsFormState {
  scrape_list_url: string
  scrape_item_selector: string
  scrape_link_selector: string
  scrape_title_selector: string
  scrape_summary_selector: string
  scrape_content_selector: string
  scrape_date_selector: string
  scrape_guid_selector: string
  scrape_limit: number | null
  source_top_n: number | null
  discover_method: string
  skip_article_fetch: boolean
}

export const SOURCE_SETTINGS_FORM_FIELDS = [
  "scrape_list_url",
  "scrape_item_selector",
  "scrape_link_selector",
  "scrape_title_selector",
  "scrape_summary_selector",
  "scrape_content_selector",
  "scrape_date_selector",
  "scrape_guid_selector",
  "scrape_limit",
  "source_top_n",
  "discover_method",
  "skip_article_fetch"
] as const

const UI_OWNED_SCRAPE_RULE_KEYS = [
  "list_url",
  "item_selector",
  "link_xpath",
  "url_xpath",
  "title_selector",
  "title_xpath",
  "summary_selector",
  "summary_xpath",
  "content_selector",
  "content_xpath",
  "date_selector",
  "date_xpath",
  "guid_xpath",
  "id_xpath",
  "limit",
  "skip_article_fetch"
] as const

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const cleanString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const cleanPositiveInteger = (value: unknown): number | undefined => {
  if (value === null || value === undefined || value === "") return undefined
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return undefined
  const normalized = Math.floor(parsed)
  return normalized > 0 ? normalized : undefined
}

const addStringRule = (
  target: Record<string, unknown>,
  key: string,
  value: unknown
): void => {
  const cleaned = cleanString(value)
  if (cleaned) target[key] = cleaned
}

export const buildScrapeRulesFromForm = (
  values: SourceSettingsFormValues
): Record<string, unknown> | undefined => {
  const scrapeRules: Record<string, unknown> = {}

  addStringRule(scrapeRules, "list_url", values.scrape_list_url)
  addStringRule(scrapeRules, "item_selector", values.scrape_item_selector)
  addStringRule(scrapeRules, "link_xpath", values.scrape_link_selector)
  addStringRule(scrapeRules, "title_selector", values.scrape_title_selector)
  addStringRule(scrapeRules, "summary_selector", values.scrape_summary_selector)
  addStringRule(scrapeRules, "content_selector", values.scrape_content_selector)
  addStringRule(scrapeRules, "date_selector", values.scrape_date_selector)
  addStringRule(scrapeRules, "guid_xpath", values.scrape_guid_selector)

  const limit = cleanPositiveInteger(values.scrape_limit)
  if (limit) scrapeRules.limit = limit
  if (values.skip_article_fetch === true) scrapeRules.skip_article_fetch = true

  return Object.keys(scrapeRules).length > 0 ? scrapeRules : undefined
}

export const buildSourceSettingsPayload = (
  existing: Record<string, unknown> | null | undefined,
  values: SourceSettingsFormValues
): Record<string, unknown> | undefined => {
  const next: Record<string, unknown> = isRecord(existing) ? { ...existing } : {}
  const scrapeRules = buildScrapeRulesFromForm(values)
  const retainedScrapeRules = isRecord(next.scrape_rules) ? { ...next.scrape_rules } : {}

  for (const key of UI_OWNED_SCRAPE_RULE_KEYS) {
    delete retainedScrapeRules[key]
  }

  const mergedScrapeRules = scrapeRules
    ? { ...retainedScrapeRules, ...scrapeRules }
    : retainedScrapeRules

  if (Object.keys(mergedScrapeRules).length > 0) {
    next.scrape_rules = mergedScrapeRules
  } else {
    delete next.scrape_rules
  }

  const topN = cleanPositiveInteger(values.source_top_n)
  if (topN) {
    next.top_n = topN
  } else {
    delete next.top_n
  }

  const discoverMethod = cleanString(values.discover_method)
  if (discoverMethod && discoverMethod !== "auto") {
    next.discover_method = discoverMethod
  } else {
    delete next.discover_method
  }

  return Object.keys(next).length > 0 ? next : undefined
}

export const sourceSettingsToFormValues = (
  settings: Record<string, unknown> | null | undefined
): SourceSettingsFormState => {
  const safeSettings = isRecord(settings) ? settings : {}
  const scrapeRules = isRecord(safeSettings.scrape_rules) ? safeSettings.scrape_rules : {}

  return {
    scrape_list_url: cleanString(scrapeRules.list_url) || "",
    scrape_item_selector: cleanString(scrapeRules.item_selector) || "",
    scrape_link_selector:
      cleanString(scrapeRules.link_xpath) || cleanString(scrapeRules.url_xpath) || "",
    scrape_title_selector:
      cleanString(scrapeRules.title_selector) || cleanString(scrapeRules.title_xpath) || "",
    scrape_summary_selector:
      cleanString(scrapeRules.summary_selector) || cleanString(scrapeRules.summary_xpath) || "",
    scrape_content_selector:
      cleanString(scrapeRules.content_selector) || cleanString(scrapeRules.content_xpath) || "",
    scrape_date_selector:
      cleanString(scrapeRules.date_selector) || cleanString(scrapeRules.date_xpath) || "",
    scrape_guid_selector:
      cleanString(scrapeRules.guid_xpath) || cleanString(scrapeRules.id_xpath) || "",
    scrape_limit: cleanPositiveInteger(scrapeRules.limit) || null,
    source_top_n: cleanPositiveInteger(safeSettings.top_n) || null,
    discover_method: cleanString(safeSettings.discover_method) || "auto",
    skip_article_fetch: scrapeRules.skip_article_fetch === true
  }
}

export const sourceSettingsAreEqual = (
  left: Record<string, unknown> | null | undefined,
  right: Record<string, unknown> | null | undefined
): boolean => {
  const normalizedLeft = isRecord(left) && Object.keys(left).length > 0 ? left : null
  const normalizedRight = isRecord(right) && Object.keys(right).length > 0 ? right : null
  return (
    JSON.stringify(sortJsonKeys(normalizedLeft)) ===
    JSON.stringify(sortJsonKeys(normalizedRight))
  )
}

const sortJsonKeys = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortJsonKeys)
  if (!isRecord(value)) return value
  return Object.keys(value)
    .sort()
    .reduce<Record<string, unknown>>((acc, key) => {
      acc[key] = sortJsonKeys(value[key])
      return acc
    }, {})
}
