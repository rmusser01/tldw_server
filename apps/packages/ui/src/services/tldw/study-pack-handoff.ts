export type StudyPackHandoffSourceType = "media" | "note" | "message"

export type StudyPackHandoffSourceItem = {
  sourceType: StudyPackHandoffSourceType
  sourceId: string
  sourceTitle?: string
  excerptText?: string
  locator?: Record<string, unknown>
}

export type StudyPackIntent = {
  title: string
  sourceItems: StudyPackHandoffSourceItem[]
}

const toNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const normalizeSourceType = (value: unknown): StudyPackHandoffSourceType | undefined => {
  if (value === "media" || value === "note" || value === "message") {
    return value
  }
  return undefined
}

const extractSearchFromHash = (hash: string): string => {
  const questionMarkIndex = hash.indexOf("?")
  if (questionMarkIndex < 0) return ""
  return hash.slice(questionMarkIndex)
}

const toSearchParams = (search: string): URLSearchParams => {
  const normalized = search.startsWith("?") ? search.slice(1) : search
  return new URLSearchParams(normalized)
}

const normalizeSourceItem = (value: unknown): StudyPackHandoffSourceItem | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const payload = value as Record<string, unknown>
  const sourceType = normalizeSourceType(payload.sourceType ?? payload.source_type)
  const sourceId = toNonEmptyString(payload.sourceId ?? payload.source_id)
  if (!sourceType || !sourceId) return null

  const normalized: StudyPackHandoffSourceItem = {
    sourceType,
    sourceId
  }

  const sourceTitle = toNonEmptyString(payload.sourceTitle ?? payload.source_title)
  if (sourceTitle) {
    normalized.sourceTitle = sourceTitle
  }

  const excerptText = toNonEmptyString(payload.excerptText ?? payload.excerpt_text)
  if (excerptText) {
    normalized.excerptText = excerptText
  }

  const locator = payload.locator
  if (locator && typeof locator === "object" && !Array.isArray(locator)) {
    normalized.locator = locator as Record<string, unknown>
  }

  return normalized
}

const normalizeIntent = (value: unknown): StudyPackIntent | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const payload = value as Record<string, unknown>
  const title = toNonEmptyString(payload.title ?? payload.study_pack_title)
  const rawItems = payload.sourceItems ?? payload.source_items
  if (!title || !Array.isArray(rawItems)) return null

  const sourceItems = rawItems
    .map((item) => normalizeSourceItem(item))
    .filter((item): item is StudyPackHandoffSourceItem => item != null)

  if (sourceItems.length === 0) return null

  return {
    title,
    sourceItems
  }
}

export const parseStudyPackIntentFromSearch = (search: string): StudyPackIntent | null => {
  const params = toSearchParams(search)
  const hasSignal =
    params.get("study_pack") === "1" ||
    params.has("study_pack_payload") ||
    params.has("study_pack_title")
  if (!hasSignal) return null

  const rawPayload = toNonEmptyString(params.get("study_pack_payload"))
  if (!rawPayload) return null

  try {
    return normalizeIntent(JSON.parse(rawPayload))
  } catch {
    return null
  }
}

export const parseStudyPackIntentFromLocation = (locationLike: {
  search?: string
  hash?: string
}): StudyPackIntent | null => {
  const fromSearch = parseStudyPackIntentFromSearch(locationLike.search || "")
  if (fromSearch) return fromSearch

  return parseStudyPackIntentFromSearch(extractSearchFromHash(locationLike.hash || ""))
}

export const buildStudyPackRoute = (intent: StudyPackIntent): string => {
  const title = toNonEmptyString(intent.title)
  const sourceItems = Array.isArray(intent.sourceItems)
    ? intent.sourceItems
        .map((item) => normalizeSourceItem(item))
        .filter((item): item is StudyPackHandoffSourceItem => item != null)
    : []

  if (!title || sourceItems.length === 0) {
    return "/flashcards?tab=importExport"
  }

  const params = new URLSearchParams()
  params.set("tab", "importExport")
  params.set("study_pack", "1")
  params.set("study_pack_title", title)
  params.set(
    "study_pack_payload",
    JSON.stringify({
      title,
      sourceItems
    })
  )

  return `/flashcards?${params.toString()}`
}
