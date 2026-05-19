import type { RagResult } from "./types"

export type SourceSortMode = "relevance" | "title" | "date" | "cited"
export type SourceDateFilter = "all" | "last_30d" | "last_365d" | "older_365d"
export type SourceContentFacet =
  | "all"
  | "pdf"
  | "transcript"
  | "video"
  | "audio"
  | "note"
  | "web"
  | "other"

export type SourceListItem = {
  result: RagResult
  originalIndex: number
}

export type CitationUsageAnchor = {
  sentenceNumber: number
  occurrence: number
  sentencePreview: string
}

const SOURCE_TYPE_LABELS: Record<
  string,
  { singular: string; plural: string }
> = {
  media_db: { singular: "Document", plural: "Documents" },
  notes: { singular: "Note", plural: "Notes" },
  note: { singular: "Note", plural: "Notes" },
  characters: { singular: "Story Character", plural: "Story Characters" },
  chats: { singular: "Conversation", plural: "Conversations" },
  kanban: { singular: "Board item", plural: "Board items" },
  web: { singular: "Web", plural: "Web" },
  pdf: { singular: "PDF", plural: "PDFs" },
  transcript: { singular: "Transcript", plural: "Transcripts" },
  video: { singular: "Video", plural: "Videos" },
  audio: { singular: "Audio", plural: "Audio" },
  other: { singular: "Other", plural: "Other" },
  unknown: { singular: "Other", plural: "Other" },
}

const CONTENT_FACET_LABELS: Record<SourceContentFacet, string> = {
  all: "Any type",
  pdf: "PDF",
  transcript: "Transcript",
  video: "Video",
  audio: "Audio",
  note: "Note",
  web: "Web",
  other: "Other",
}

const DATE_METADATA_KEYS = [
  "published_at",
  "publishedAt",
  "created_at",
  "createdAt",
  "updated_at",
  "updatedAt",
  "date",
  "source_date",
] as const

export function normalizeSourceType(sourceType: unknown): string {
  const normalized = String(sourceType || "").trim().toLowerCase()
  if (!normalized) return "unknown"
  if (SOURCE_TYPE_LABELS[normalized]) return normalized
  return "unknown"
}

export function getSourceTypeLabel(
  sourceType: unknown,
  options: { plural?: boolean } = {}
): string {
  const normalized = normalizeSourceType(sourceType)
  const labels = SOURCE_TYPE_LABELS[normalized] || SOURCE_TYPE_LABELS.unknown
  return options.plural ? labels.plural : labels.singular
}

export function buildSourceTypeCounts(results: RagResult[]): Record<string, number> {
  return results.reduce<Record<string, number>>((acc, result) => {
    const sourceType = normalizeSourceType(result.metadata?.source_type)
    acc[sourceType] = (acc[sourceType] || 0) + 1
    return acc
  }, {})
}

function normalizeText(value: unknown): string {
  if (typeof value !== "string") return ""
  return value.trim().toLowerCase()
}

export function detectSourceContentFacet(result: RagResult): SourceContentFacet {
  const metadata = result.metadata || {}
  const sourceType = normalizeSourceType(metadata.source_type)

  if (sourceType === "notes") return "note"
  if (sourceType === "web") return "web"

  const title = normalizeText(metadata.title)
  const source = normalizeText(metadata.source)
  const url = normalizeText(metadata.url)
  const mimeType = normalizeText(
    (metadata as Record<string, unknown>).mime_type ??
      (metadata as Record<string, unknown>).content_type
  )
  const mediaType = normalizeText((metadata as Record<string, unknown>).media_type)
  const fileType = normalizeText(
    (metadata as Record<string, unknown>).file_type ??
      (metadata as Record<string, unknown>).format
  )

  const haystack = [title, source, url, mimeType, mediaType, fileType]
    .filter((value) => value.length > 0)
    .join(" ")

  if (/pdf|\b\.pdf\b/.test(haystack)) return "pdf"
  if (/transcript|subtitle|\bsrt\b|\bvtt\b/.test(haystack)) return "transcript"
  if (/video|youtube|vimeo|\bmp4\b|\bmkv\b|\bmov\b|\bavi\b|\bwebm\b/.test(haystack)) {
    return "video"
  }
  if (/audio|\bmp3\b|\bwav\b|\bm4a\b|\baac\b|\bflac\b|\bogg\b/.test(haystack)) {
    return "audio"
  }

  if (sourceType === "unknown" && /^https?:\/\//.test(url)) {
    return "web"
  }

  return "other"
}

export function getSourceContentFacetLabel(facet: SourceContentFacet): string {
  return CONTENT_FACET_LABELS[facet]
}

export function buildSourceContentFacetCounts(
  results: RagResult[]
): Record<SourceContentFacet, number> {
  return results.reduce<Record<SourceContentFacet, number>>(
    (acc, result) => {
      const facet = detectSourceContentFacet(result)
      acc[facet] = (acc[facet] || 0) + 1
      return acc
    },
    {
      all: 0,
      pdf: 0,
      transcript: 0,
      video: 0,
      audio: 0,
      note: 0,
      web: 0,
      other: 0,
    }
  )
}

function parseDateValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value !== "string" || value.trim().length === 0) {
    return null
  }
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) return null
  return parsed
}

export function getSourceTimestamp(result: RagResult): number | null {
  const metadata = result.metadata
  if (!metadata || typeof metadata !== "object") {
    return null
  }

  for (const key of DATE_METADATA_KEYS) {
    const value = metadata[key]
    const parsed = parseDateValue(value)
    if (parsed != null) return parsed
  }
  return null
}

export function formatSourceDate(result: RagResult): string | null {
  const timestamp = getSourceTimestamp(result)
  if (timestamp == null) return null
  return new Date(timestamp).toLocaleDateString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
  })
}

export type RelevanceDescriptor = {
  percent: number
  level: "high" | "moderate" | "low"
  label: string
  className: string
}

export type FreshnessDescriptor = {
  label: string
  className: string
}

export function getFreshnessDescriptor(
  result: RagResult,
  nowMs = Date.now()
): FreshnessDescriptor | null {
  const timestamp = getSourceTimestamp(result)
  if (timestamp == null) return null

  const ageMs = Math.max(0, nowMs - timestamp)
  const ageDays = Math.floor(ageMs / (1000 * 60 * 60 * 24))
  const ageYears = ageDays / 365
  const year = new Date(timestamp).getUTCFullYear()

  if (ageDays <= 7) {
    return {
      label: ageDays <= 1 ? "Updated today" : `Updated ${ageDays}d ago`,
      className: "border-success/30 bg-success/10 text-success",
    }
  }

  if (ageDays <= 30) {
    return {
      label: `Updated ${ageDays}d ago`,
      className: "border-success/30 bg-success/10 text-success",
    }
  }

  if (ageYears < 1) {
    const months = Math.max(1, Math.floor(ageDays / 30))
    return {
      label: `Updated ${months}mo ago`,
      className: "border-primary/30 bg-primary/10 text-primary",
    }
  }

  if (ageYears > 3) {
    return {
      label: `From ${year}`,
      className: "border-danger/30 bg-danger/10 text-danger",
    }
  }

  return {
    label: `From ${year}`,
    className: "border-warn/30 bg-warn/10 text-warn",
  }
}

export function getRelevanceDescriptor(
  score: number | undefined
): RelevanceDescriptor | null {
  if (typeof score !== "number" || !Number.isFinite(score)) return null
  const percent = Math.max(0, Math.min(100, Math.round(score * 100)))
  if (score >= 0.8) {
    return {
      percent,
      level: "high",
      label: "Strong relevance",
      className: "bg-success/15 text-success border border-success/30",
    }
  }
  if (score >= 0.5) {
    return {
      percent,
      level: "moderate",
      label: "Moderate relevance",
      className: "bg-warn/15 text-warn border border-warn/30",
    }
  }
  return {
    percent,
    level: "low",
    label: "Weak relevance",
    className: "bg-danger/15 text-danger border border-danger/30",
  }
}

export function formatChunkPosition(chunkId: unknown): string | null {
  if (typeof chunkId !== "string" || chunkId.trim().length === 0) return null
  const normalized = chunkId.trim()

  const slashPattern = normalized.match(/\b(\d+)\s*\/\s*(\d+)\b/)
  if (slashPattern) {
    return `Section ${slashPattern[1]} of ${slashPattern[2]}`
  }

  const chunkPattern = normalized.match(
    /chunk[_\s-]?(\d+)(?:[_\s-]?(?:of)?[_\s-]?(\d+))?/i
  )
  if (chunkPattern) {
    if (chunkPattern[2]) {
      return `Section ${chunkPattern[1]} of ${chunkPattern[2]}`
    }
    return `Section ${chunkPattern[1]}`
  }

  if (/^\d{1,4}$/.test(normalized)) {
    return `Section ${normalized}`
  }

  return null
}

export function filterItemsBySourceType(
  items: SourceListItem[],
  sourceType: string
): SourceListItem[] {
  if (sourceType === "all") return items
  return items.filter(
    (item) => normalizeSourceType(item.result.metadata?.source_type) === sourceType
  )
}

export function filterItemsByContentFacet(
  items: SourceListItem[],
  facet: SourceContentFacet
): SourceListItem[] {
  if (facet === "all") return items
  return items.filter((item) => detectSourceContentFacet(item.result) === facet)
}

function buildKeywordHaystack(result: RagResult): string {
  const metadata = result.metadata || {}
  const fields = [
    result.content,
    result.text,
    result.chunk,
    result.excerpt,
    typeof metadata.title === "string" ? metadata.title : "",
    typeof metadata.source === "string" ? metadata.source : "",
    typeof metadata.url === "string" ? metadata.url : "",
    typeof metadata.chunk_id === "string" ? metadata.chunk_id : "",
    typeof metadata.page_number === "number" ? String(metadata.page_number) : "",
  ]

  return fields
    .filter((value): value is string => typeof value === "string" && value.length > 0)
    .join(" ")
    .toLowerCase()
}

export function filterItemsByKeyword(
  items: SourceListItem[],
  keyword: string
): SourceListItem[] {
  const normalized = keyword.trim().toLowerCase()
  if (!normalized) return items

  const terms = normalized.split(/\s+/).filter(Boolean)
  if (terms.length === 0) return items

  return items.filter((item) => {
    const haystack = buildKeywordHaystack(item.result)
    return terms.every((term) => haystack.includes(term))
  })
}

export function filterItemsByDateRange(
  items: SourceListItem[],
  dateFilter: SourceDateFilter,
  nowMs = Date.now()
): SourceListItem[] {
  if (dateFilter === "all") return items

  return items.filter((item) => {
    const timestamp = getSourceTimestamp(item.result)
    if (timestamp == null) return false

    const ageDays = Math.floor(Math.max(0, nowMs - timestamp) / (1000 * 60 * 60 * 24))
    if (dateFilter === "last_30d") {
      return ageDays <= 30
    }
    if (dateFilter === "last_365d") {
      return ageDays <= 365
    }
    return ageDays > 365
  })
}

export function sortSourceItems(
  items: SourceListItem[],
  mode: SourceSortMode,
  citedIndices: Set<number>
): SourceListItem[] {
  const copy = [...items]

  if (mode === "relevance") {
    return copy
  }

  if (mode === "title") {
    return copy.sort((left, right) => {
      const titleLeft = left.result.metadata?.title || left.result.metadata?.source || ""
      const titleRight =
        right.result.metadata?.title || right.result.metadata?.source || ""
      return titleLeft.localeCompare(titleRight)
    })
  }

  if (mode === "date") {
    return copy.sort((left, right) => {
      const dateLeft = getSourceTimestamp(left.result)
      const dateRight = getSourceTimestamp(right.result)
      if (dateLeft == null && dateRight == null) {
        return left.originalIndex - right.originalIndex
      }
      if (dateLeft == null) return 1
      if (dateRight == null) return -1
      return dateRight - dateLeft
    })
  }

  return copy.sort((left, right) => {
    const leftCited = citedIndices.has(left.originalIndex)
    const rightCited = citedIndices.has(right.originalIndex)
    if (leftCited === rightCited) {
      return left.originalIndex - right.originalIndex
    }
    return leftCited ? -1 : 1
  })
}

const HIGHLIGHT_STOP_WORDS = new Set([
  "the",
  "a",
  "an",
  "and",
  "or",
  "to",
  "of",
  "in",
  "on",
  "for",
  "with",
  "is",
  "are",
  "was",
  "were",
  "be",
  "this",
  "that",
])

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export function buildHighlightTerms(
  query: string,
  expandedQueries: string[] = []
): string[] {
  const rawSegments = [query, ...expandedQueries]
    .map((value) => String(value || "").trim())
    .filter((value) => value.length > 0)

  const candidateTerms = new Set<string>()

  for (const segment of rawSegments) {
    if (segment.length >= 3 && segment.length <= 80) {
      candidateTerms.add(segment.toLowerCase())
    }

    const words = segment
      .split(/[\s,.;:!?()[\]{}"“”'`/\\|<>+=_-]+/)
      .map((word) => word.trim().toLowerCase())
      .filter((word) => word.length >= 3)
      .filter((word) => !HIGHLIGHT_STOP_WORDS.has(word))

    for (const word of words) {
      candidateTerms.add(word)
    }
  }

  return Array.from(candidateTerms)
    .sort((left, right) => right.length - left.length)
    .slice(0, 20)
}

export function splitTextByHighlights(
  text: string,
  terms: string[]
): Array<{ text: string; highlight: boolean }> {
  if (!text || terms.length === 0) {
    return [{ text, highlight: false }]
  }

  const escapedTerms = terms
    .map((term) => term.trim())
    .filter((term) => term.length >= 3)
    .map((term) => {
      const escaped = escapeRegex(term)
      return /^[A-Za-z0-9]+$/.test(term) ? `\\b${escaped}\\b` : escaped
    })

  if (escapedTerms.length === 0) {
    return [{ text, highlight: false }]
  }

  const pattern = new RegExp(`(${escapedTerms.join("|")})`, "gi")
  const normalizedTermSet = new Set(terms.map((term) => term.toLowerCase()))
  const parts = text.split(pattern)

  return parts
    .filter((part) => part.length > 0)
    .map((part) => ({
      text: part,
      highlight: normalizedTermSet.has(part.toLowerCase()),
    }))
}

const SENTENCE_SPLIT_PATTERN = /(?<=[.!?])\s+|\n+/
const CITATION_PATTERN = /\[(\d+)\]/g
const MAX_SENTENCE_PREVIEW_LENGTH = 140

function toSentencePreview(sentence: string): string {
  const normalized = sentence.replace(/\s+/g, " ").trim()
  if (normalized.length <= MAX_SENTENCE_PREVIEW_LENGTH) {
    return normalized
  }
  return `${normalized.slice(0, MAX_SENTENCE_PREVIEW_LENGTH - 1).trimEnd()}…`
}

export function buildCitationUsageAnchors(
  answer: string | null | undefined
): Record<number, CitationUsageAnchor[]> {
  if (typeof answer !== "string" || answer.trim().length === 0) {
    return {}
  }

  const sentences = answer
    .split(SENTENCE_SPLIT_PATTERN)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 0)

  if (sentences.length === 0) {
    return {}
  }

  const usageByCitation: Record<number, CitationUsageAnchor[]> = {}
  const occurrenceByCitation = new Map<number, number>()

  sentences.forEach((sentence, sentenceIndex) => {
    const seenCitationInSentence = new Set<number>()
    let match: RegExpExecArray | null

    while ((match = CITATION_PATTERN.exec(sentence)) !== null) {
      const citationIndex = Number.parseInt(match[1], 10)
      if (!Number.isFinite(citationIndex) || citationIndex < 1) {
        continue
      }

      const nextOccurrence = (occurrenceByCitation.get(citationIndex) || 0) + 1
      occurrenceByCitation.set(citationIndex, nextOccurrence)

      if (seenCitationInSentence.has(citationIndex)) {
        continue
      }
      seenCitationInSentence.add(citationIndex)

      if (!usageByCitation[citationIndex]) {
        usageByCitation[citationIndex] = []
      }
      usageByCitation[citationIndex].push({
        sentenceNumber: sentenceIndex + 1,
        occurrence: nextOccurrence,
        sentencePreview: toSentencePreview(sentence),
      })
    }

    CITATION_PATTERN.lastIndex = 0
  })

  return usageByCitation
}
