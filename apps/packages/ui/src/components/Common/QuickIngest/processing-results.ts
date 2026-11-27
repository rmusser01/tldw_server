import { extractCompletedIngestJobTerminalData } from "@/services/tldw/ingest-job-results"

export type ProcessingItem = {
  id?: string | number
  media_id?: string | number
  pk?: string | number
  uuid?: string | number
  media?: ProcessingItem
  status?: string
  url?: string
  input_ref?: string
  media_type?: string
  content?: string | Array<string | number>
  text?: string
  transcript?: string
  transcription?: string
  summary?: string
  analysis_content?: string | null
  analysis?: string | null
  analysis_status?: string
  analysis_error?: string
  warnings?: string[]
  prompt?: string
  custom_prompt?: string
  title?: string
  metadata?: Record<string, unknown> & { keywords?: string[] | string }
  keywords?: string[] | string
  segments?: Record<string, unknown>[]
}

export type ProcessingResultPayload =
  | ProcessingItem[]
  | ProcessingItem
  | {
      results?: ProcessingItem[]
      articles?: ProcessingItem[]
      result?: ProcessingItem | ProcessingItem[]
    }
  | null
  | undefined

const isResultsWrapper = (value: unknown): value is { results: ProcessingItem[] } => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Array.isArray((value as { results?: unknown }).results)
}

const isArticlesWrapper = (value: unknown): value is { articles: ProcessingItem[] } => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  return Array.isArray((value as { articles?: unknown }).articles)
}

const isResultWrapper = (value: unknown): value is { result: ProcessingItem | ProcessingItem[] } => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const result = (value as { result?: unknown }).result
  return Array.isArray(result) || (typeof result === "object" && result !== null)
}

export const isProcessingItem = (value: unknown): value is ProcessingItem => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  if (isResultsWrapper(value) || isArticlesWrapper(value) || isResultWrapper(value)) return false
  return (
    "id" in value || "media_id" in value || "pk" in value || "uuid" in value ||
    "content" in value || "text" in value || "transcript" in value || "transcription" in value ||
    "analysis" in value || "analysis_content" in value || "segments" in value
  )
}

export const extractProcessingItems = (data: unknown): ProcessingItem[] => {
  if (!data) return []
  if (Array.isArray(data)) return data
  if (isResultsWrapper(data)) return data.results
  if (isArticlesWrapper(data)) return data.articles
  if (isResultWrapper(data)) return Array.isArray(data.result) ? data.result : [data.result]
  if (isProcessingItem(data)) return [data]
  return []
}

export const resolveContent = (item: ProcessingItem, includeAnalysis = true): string => {
  const content = Array.isArray(item?.content)
    ? item.content
      .filter((v: unknown) => typeof v === "string" || typeof v === "number")
      .join("\n")
    : item?.content
  const candidates = [content, item?.text, item?.transcript, item?.transcription]
  if (includeAnalysis) candidates.push(item?.summary, item?.analysis_content ?? undefined)
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate
  }
  return ""
}

/** Get each article or transcript in order. Preserve its original whitespace for display and copy. */
export const getExtractedSourceText = (data: unknown): string => {
  const payload = extractCompletedIngestJobTerminalData(data)
  if (typeof payload === "string") return payload.trim() ? payload : ""
  return extractProcessingItems(payload).map(item => {
    const text = resolveContent(item, false)
    if (text.trim() || !Array.isArray(item?.segments)) return text
    return item.segments.map(segment => segment?.text)
      .filter((part): part is string => typeof part === "string" && Boolean(part.trim()))
      .join("\n")
  }).filter(text => text.trim()).join("\n\n")
}
