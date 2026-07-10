import type {
  SourceReviewActivity,
  StudyPackSourceSelection
} from "@/services/flashcards"

const SOURCE_REVIEW_HANDOFF_PREFIX = "tldw:source-review-handoff:"
const SOURCE_REVIEW_HANDOFF_TTL_MS = 30 * 60 * 1000
const SOURCE_REVIEW_GENERATION_TEXT_LIMIT = 20_000

export type SourceReviewHandoffPayload = {
  occurrence_id: number
  plan_id: number
  plan_title?: string | null
  activity_type: SourceReviewActivity
  source_bundle: {
    items: StudyPackSourceSelection[]
  }
}

export type SourceReviewFlashcardsIntent = {
  activity_type: "flashcards" | "cloze"
  text: string
  source_items: StudyPackSourceSelection[]
}

type StoredSourceReviewHandoff = {
  expires_at: number
  payload: SourceReviewHandoffPayload
}

const storage = (): Storage | null => {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage
  } catch {
    return null
  }
}

const token = (): string => {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

const isSourceReviewHandoff = (
  value: unknown
): value is SourceReviewHandoffPayload => {
  if (!value || typeof value !== "object") return false
  const candidate = value as Partial<SourceReviewHandoffPayload>
  return (
    typeof candidate.occurrence_id === "number" &&
    typeof candidate.plan_id === "number" &&
    typeof candidate.activity_type === "string" &&
    Array.isArray(candidate.source_bundle?.items)
  )
}

export const getSourceReviewItems = (
  payload: SourceReviewHandoffPayload
): StudyPackSourceSelection[] => payload.source_bundle.items

const sourceLabel = (item: StudyPackSourceSelection): string =>
  item.label?.trim() ||
  item.source_title?.trim() ||
  `${item.source_type} ${item.source_id}`

const buildBoundedGenerationText = (
  items: StudyPackSourceSelection[]
): string => {
  const sources = items.map((item) => ({
    label: sourceLabel(item),
    excerpt: item.excerpt_text?.trim() || ""
  }))
  if (sources.length === 0) return ""

  const separatorLength = Math.max(0, sources.length - 1) * 2
  let remaining = Math.max(
    0,
    SOURCE_REVIEW_GENERATION_TEXT_LIMIT - separatorLength
  )
  let sourcesRemaining = sources.length

  const blocks = sources.map((source) => {
    const allocation = Math.floor(remaining / sourcesRemaining)
    const label = source.label.slice(0, allocation)
    const excerptBudget = Math.max(0, allocation - label.length - 1)
    const excerpt = source.excerpt.slice(0, excerptBudget)
    const block = excerpt ? `${label}\n${excerpt}` : label
    remaining -= block.length
    sourcesRemaining -= 1
    return block
  })
  return blocks.join("\n\n").slice(0, SOURCE_REVIEW_GENERATION_TEXT_LIMIT)
}

export function buildSourceReviewRereadContent(
  payload: SourceReviewHandoffPayload
): string {
  return payload.source_bundle.items
    .map((item) => {
      const parts = [sourceLabel(item)]
      if (item.excerpt_text?.trim()) parts.push(item.excerpt_text.trim())
      if (item.locator && Object.keys(item.locator).length > 0) {
        parts.push(JSON.stringify(item.locator))
      }
      return parts.join("\n")
    })
    .join("\n\n")
}

export function buildSourceReviewFlashcardsIntent(
  payload: SourceReviewHandoffPayload
): SourceReviewFlashcardsIntent {
  const activityType = payload.activity_type === "cloze" ? "cloze" : "flashcards"
  const text = buildBoundedGenerationText(payload.source_bundle.items)

  return {
    activity_type: activityType,
    text,
    source_items: getSourceReviewItems(payload)
  }
}

export function saveSourceReviewHandoff(
  payload: SourceReviewHandoffPayload
): string {
  const session = storage()
  if (!session) return ""
  const handoffToken = token()
  const stored: StoredSourceReviewHandoff = {
    expires_at: Date.now() + SOURCE_REVIEW_HANDOFF_TTL_MS,
    payload
  }
  try {
    session.setItem(
      `${SOURCE_REVIEW_HANDOFF_PREFIX}${handoffToken}`,
      JSON.stringify(stored)
    )
    return handoffToken
  } catch {
    return ""
  }
}

export function loadSourceReviewHandoff(
  handoffToken: string
): SourceReviewHandoffPayload | null {
  if (!handoffToken.trim()) return null
  const session = storage()
  if (!session) return null
  const key = `${SOURCE_REVIEW_HANDOFF_PREFIX}${handoffToken}`
  const removeStoredHandoff = () => {
    try {
      session.removeItem(key)
    } catch {
      // Storage cleanup is best-effort when browser privacy settings block access.
    }
  }
  try {
    const raw = session.getItem(key)
    if (!raw) return null
    const stored = JSON.parse(raw) as Partial<StoredSourceReviewHandoff>
    if (
      typeof stored.expires_at !== "number" ||
      stored.expires_at <= Date.now() ||
      !isSourceReviewHandoff(stored.payload)
    ) {
      removeStoredHandoff()
      return null
    }
    return stored.payload
  } catch {
    removeStoredHandoff()
    return null
  }
}

export function buildSourceReviewQuizRoute(
  payload: SourceReviewHandoffPayload
): string {
  const handoffToken = saveSourceReviewHandoff(payload)
  if (!handoffToken) return "/quiz?tab=generate&source_review=1"
  return `/quiz?tab=generate&source_review=1&source_review_token=${encodeURIComponent(
    handoffToken
  )}`
}
