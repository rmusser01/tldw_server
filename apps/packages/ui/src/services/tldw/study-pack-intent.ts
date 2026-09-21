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

export const normalizeStudyPackIntent = (value: unknown): StudyPackIntent | null => {
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

