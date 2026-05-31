import { createSafeStorage } from "@/utils/safe-storage"
import type { WorkspaceSourceType } from "@/types/workspace"

const PREFILL_KEY = "__tldw_research_workspace_prefill"
const storage = createSafeStorage({ area: "local" })

type KnowledgeQaResultLike = {
  id?: string
  metadata?: {
    title?: unknown
    source?: unknown
    source_type?: unknown
    url?: unknown
    page_number?: unknown
    media_id?: unknown
    mediaId?: unknown
    document_id?: unknown
    doc_id?: unknown
    [key: string]: unknown
  }
}

export type WorkspaceKnowledgeQaPrefillSource = {
  mediaId: number | null
  title: string
  type: WorkspaceSourceType
  sourceType: string | null
  url?: string
  pageNumber?: number
  citationIndex?: number
}

export type ResearchWorkspacePrefill =
  | {
      kind: "knowledge_qa_thread"
      createdAt: string
      threadId: string | null
      query: string
      answer: string | null
      citations: number[]
      sources: WorkspaceKnowledgeQaPrefillSource[]
    }

export type BuildKnowledgeQaWorkspacePrefillInput = {
  threadId: string | null
  query: string
  answer: string | null
  citations: number[]
  results: KnowledgeQaResultLike[]
}

const normalizeString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.trim()
  return normalized.length > 0 ? normalized : null
}

const parseNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.floor(value)
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    return Number.parseInt(value, 10)
  }
  return null
}

const toWorkspaceSourceType = (
  sourceTypeRaw: string | null,
  url: string | null
): WorkspaceSourceType => {
  const sourceType = (sourceTypeRaw || "").toLowerCase()
  if (sourceType.includes("pdf")) return "pdf"
  if (sourceType.includes("video")) return "video"
  if (sourceType.includes("audio")) return "audio"
  if (
    sourceType.includes("website") ||
    sourceType.includes("web") ||
    sourceType.includes("url")
  ) {
    return "website"
  }
  if (sourceType.includes("text")) return "text"
  if (!sourceType && url) return "website"
  return "document"
}

const resolveMediaId = (result: KnowledgeQaResultLike): number | null => {
  const metadata = result.metadata || {}
  const candidates = [
    metadata.media_id,
    metadata.mediaId,
    metadata.document_id,
    metadata.doc_id,
    result.id,
  ]
  for (const candidate of candidates) {
    const parsed = parseNumber(candidate)
    if (parsed != null) return parsed
  }
  return null
}

const toPrefillSource = (
  result: KnowledgeQaResultLike,
  index: number,
  citedIndices: Set<number>
): WorkspaceKnowledgeQaPrefillSource => {
  const metadata = result.metadata || {}
  const sourceType = normalizeString(metadata.source_type)
  const url = normalizeString(metadata.url)
  const pageNumber = parseNumber(metadata.page_number)
  const fallbackTitle = normalizeString(metadata.source) || `Source ${index + 1}`
  const title = normalizeString(metadata.title) || fallbackTitle
  const citationIndex = citedIndices.has(index + 1) ? index + 1 : undefined

  return {
    mediaId: resolveMediaId(result),
    title,
    type: toWorkspaceSourceType(sourceType, url),
    sourceType,
    ...(url ? { url } : {}),
    ...(pageNumber != null ? { pageNumber } : {}),
    ...(citationIndex != null ? { citationIndex } : {}),
  }
}

export const buildKnowledgeQaWorkspacePrefill = (
  input: BuildKnowledgeQaWorkspacePrefillInput
): ResearchWorkspacePrefill => {
  const citedIndices = new Set(input.citations)
  const sources = input.results.map((result, index) =>
    toPrefillSource(result, index, citedIndices)
  )

  return {
    kind: "knowledge_qa_thread",
    createdAt: new Date().toISOString(),
    threadId: input.threadId,
    query: input.query.trim(),
    answer: input.answer,
    citations: [...new Set(input.citations)].filter(
      (index) => Number.isFinite(index) && index > 0
    ),
    sources,
  }
}

export const queueResearchWorkspacePrefill = async (
  payload: ResearchWorkspacePrefill
): Promise<void> => {
  try {
    await storage.set(PREFILL_KEY, payload)
  } catch {
    // Prefill is optional and should not block navigation.
  }
}

export const consumeResearchWorkspacePrefill =
  async (): Promise<ResearchWorkspacePrefill | null> => {
    try {
      const payload = await storage.get<ResearchWorkspacePrefill | null>(PREFILL_KEY)
      if (payload) {
        await storage.remove(PREFILL_KEY)
        return payload
      }
    } catch {
      // Ignore storage failures and continue without prefill.
    }
    return null
  }

const truncate = (value: string, max: number): string =>
  value.length > max ? `${value.slice(0, max - 1)}...` : value

export const buildKnowledgeQaSeedNote = (
  payload: Extract<ResearchWorkspacePrefill, { kind: "knowledge_qa_thread" }>
): string => {
  const question = payload.query.trim()
  const answer = (payload.answer || "").trim()

  const lines: string[] = []
  lines.push("Imported from Knowledge QA")

  if (question) {
    lines.push(`Question: ${question}`)
  }

  if (answer) {
    lines.push("")
    lines.push("Answer:")
    lines.push(answer)
  }

  if (payload.sources.length > 0) {
    lines.push("")
    lines.push("Sources:")
    for (const source of payload.sources.slice(0, 20)) {
      const citationPrefix =
        source.citationIndex != null ? `[${source.citationIndex}] ` : ""
      const pageSuffix =
        source.pageNumber != null ? ` (p. ${source.pageNumber})` : ""
      const urlSuffix = source.url ? ` - ${source.url}` : ""
      lines.push(
        `- ${citationPrefix}${truncate(source.title, 140)}${pageSuffix}${urlSuffix}`
      )
    }
  }

  return lines.join("\n")
}
