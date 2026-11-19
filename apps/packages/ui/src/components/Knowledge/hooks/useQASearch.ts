import React from "react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  type RagSettings,
  buildRagSearchRequest
} from "@/services/rag/unified-rag"
import {
  formatRagResult,
  type RagCopyFormat,
  type RagPinnedResult
} from "@/utils/rag-format"
import {
  type RagResult,
  toPinnedResult,
  getResultText,
  getResultTitle,
  getResultUrl,
  getResultSource,
  getResultType,
  extractMediaId
} from "./useKnowledgeSearch"

/**
 * A single document returned by the RAG pipeline.
 */
export type QADocument = {
  id?: string | number
  content?: string
  text?: string
  chunk?: string
  metadata?: Record<string, unknown>
  score?: number
  relevance?: number
  media_id?: number
}

/**
 * Citation returned by the RAG pipeline.
 */
export type QACitation = {
  source?: string
  title?: string
  url?: string
  text?: string
  chunk_index?: number
  score?: number
}

/**
 * Normalized response from the RAG search endpoint.
 */
export type QASearchResponse = {
  generatedAnswer: string | null
  documents: QADocument[]
  citations: QACitation[]
  academicCitations: string[]
  timings: Record<string, number>
  totalTime: number
  cacheHit: boolean
  feedbackId: string | null
  errors: string[]
  query: string
  expandedQueries: string[]
}

export type UseQASearchReturn = {
  // State
  loading: boolean
  response: QASearchResponse | null
  timedOut: boolean
  hasAttemptedSearch: boolean
  queryError: string | null
  /** An answer is being generated for the shown sources (CC-10). */
  answerLoading: boolean
  answerError: string | null

  // Actions
  /** Retrieve sources only; the answer is generated on request (CC-10). */
  runQASearch: (opts?: { applyFirst?: boolean }) => Promise<void>
  /** Generate an answer for the last search, keeping its sources on screen. */
  generateAnswer: () => Promise<void>

  // Result actions
  copyAnswer: () => Promise<void>
  insertAnswer: () => void
  insertChunk: (doc: QADocument) => void
  copyChunk: (doc: QADocument, format?: RagCopyFormat) => Promise<void>
  pinChunk: (doc: QADocument) => void
}

type UseQASearchOptions = {
  resolvedQuery: string
  draftSettings: RagSettings
  applySettings: () => void
  onInsert: (text: string) => void
  pinnedResults: RagPinnedResult[]
  onPin: (item: RagResult) => void
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

const toNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value.trim())
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

const isTimeoutError = (error: unknown) => {
  if (error instanceof Error && error.name === "AbortError") return true
  const message =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : ""
  return (
    message.toLowerCase().includes("timeout") ||
    message.toLowerCase().includes("timed out")
  )
}

/**
 * Normalize the raw RAG API response into a structured QASearchResponse.
 */
const normalizeQAResponse = (
  raw: unknown,
  queryText: string
): QASearchResponse => {
  const obj = isRecord(raw) ? raw : {}

  const generatedAnswer = (() => {
    if (typeof obj.generated_answer === "string") return obj.generated_answer
    if (typeof obj.answer === "string") return obj.answer
    return null
  })()

  const documentsFromPayload = (() => {
    const variants = [obj.documents, obj.results, obj.docs]
    const firstNonEmpty = variants.find(
      (value): value is unknown[] => Array.isArray(value) && value.length > 0
    )
    if (firstNonEmpty) return firstNonEmpty
    const firstArray = variants.find(Array.isArray)
    return Array.isArray(firstArray) ? firstArray : []
  })()

  const documents: QADocument[] = documentsFromPayload
    .filter(isRecord)
    .map((doc) => ({
        id: doc.id as string | number | undefined,
        content:
          typeof doc.content === "string" ? doc.content : undefined,
        text: typeof doc.text === "string" ? doc.text : undefined,
        chunk: typeof doc.chunk === "string" ? doc.chunk : undefined,
        metadata: isRecord(doc.metadata)
          ? (doc.metadata as Record<string, unknown>)
          : undefined,
        score: toNumber(doc.score),
        relevance: toNumber(doc.relevance),
        media_id: toNumber(doc.media_id)
      }))

  const citations: QACitation[] = Array.isArray(obj.citations)
    ? (obj.citations as unknown[]).filter(isRecord).map((c) => ({
        source: typeof c.source === "string" ? c.source : undefined,
        title: typeof c.title === "string" ? c.title : undefined,
        url: typeof c.url === "string" ? c.url : undefined,
        text: typeof c.text === "string" ? c.text : undefined,
        chunk_index:
          toNumber(c.chunk_index),
        score: toNumber(c.score)
      }))
    : []

  const academicCitations = Array.isArray(obj.academic_citations)
    ? (obj.academic_citations as unknown[]).filter(
        (v): v is string => typeof v === "string"
      )
    : []

  const timings = isRecord(obj.timings)
    ? Object.fromEntries(Object.entries(obj.timings).flatMap(([k, v]) => {
        const parsed = toNumber(v)
        return parsed === undefined ? [] : [[k, parsed]]
      }))
    : {}

  const totalTime = toNumber(obj.total_time) ?? toNumber(obj.totalTime) ?? 0

  const cacheHit = obj.cache_hit === true || obj.cacheHit === true

  const feedbackId = (() => {
    if (typeof obj.feedback_id === "string") return obj.feedback_id
    if (typeof obj.feedbackId === "string") return obj.feedbackId
    return null
  })()

  const errors = Array.isArray(obj.errors)
    ? (obj.errors as unknown[]).filter(
        (v): v is string => typeof v === "string"
      )
    : []

  const query = typeof obj.query === "string" ? obj.query : queryText

  const expandedQueries = Array.isArray(obj.expanded_queries)
    ? (obj.expanded_queries as unknown[]).filter(
        (v): v is string => typeof v === "string"
      )
    : Array.isArray(obj.expandedQueries)
      ? (obj.expandedQueries as unknown[]).filter(
        (v): v is string => typeof v === "string"
      )
      : []

  return {
    generatedAnswer,
    documents,
    citations,
    academicCitations,
    timings,
    totalTime,
    cacheHit,
    feedbackId,
    errors,
    query,
    expandedQueries
  }
}

/**
 * Convert a QADocument to a RagResult for reuse with shared helpers.
 */
export const qaDocumentToRagResult = (doc: QADocument): RagResult => ({
  content: doc.content || doc.text || doc.chunk || "",
  metadata: {
    ...doc.metadata,
    ...(doc.id !== undefined ? { id: doc.id } : {}),
    ...(doc.media_id !== undefined ? { media_id: doc.media_id } : {})
  },
  score: doc.score,
  relevance: doc.relevance
})

/**
 * Hook for QA Search — calls the full RAG pipeline and returns structured results.
 */
export function useQASearch({
  resolvedQuery,
  draftSettings,
  applySettings,
  onInsert,
  pinnedResults,
  onPin
}: UseQASearchOptions): UseQASearchReturn {
  const { t } = useTranslation(["sidepanel", "common"])

  const [loading, setLoading] = React.useState(false)
  const [response, setResponse] = React.useState<QASearchResponse | null>(null)
  const [timedOut, setTimedOut] = React.useState(false)
  const [hasAttemptedSearch, setHasAttemptedSearch] = React.useState(false)
  const [queryError, setQueryError] = React.useState<string | null>(null)
  const [answerLoading, setAnswerLoading] = React.useState(false)
  const [answerError, setAnswerError] = React.useState<string | null>(null)
  // The query and settings behind the shown sources, so Generate answer
  // answers the same search even after the user edits the box.
  const lastSearchRef = React.useRef<{
    query: string
    settings: RagSettings
  } | null>(null)
  const searchRunRef = React.useRef(0)

  const runQASearch = React.useCallback(
    async (opts?: { applyFirst?: boolean }) => {
      if (opts?.applyFirst) {
        applySettings()
      }

      const queryText = (resolvedQuery || "").trim()
      if (!queryText) {
        setQueryError(
          t("sidepanel:rag.queryRequired", "Enter a query to search.") as string
        )
        return
      }

      setQueryError(null)
      if (!hasAttemptedSearch) {
        setHasAttemptedSearch(true)
      }

      const runId = ++searchRunRef.current
      setLoading(true)
      setTimedOut(false)
      setResponse(null)
      setAnswerLoading(false)
      setAnswerError(null)
      lastSearchRef.current = { query: queryText, settings: draftSettings }

      try {
        await tldwClient.initialize()
        // Sources first: the answer is opt-in (CC-10, decision [2]), whatever
        // the shared Knowledge settings default to.
        const { query, options, timeoutMs } = buildRagSearchRequest({
          ...draftSettings,
          query: queryText,
          enable_generation: false
        })
        const rawResponse = await tldwClient.ragSearch(query, {
          ...options,
          timeoutMs
        })
        if (runId !== searchRunRef.current) return
        const normalized = normalizeQAResponse(rawResponse, queryText)
        setResponse(normalized)
        setTimedOut(false)
      } catch (e) {
        if (runId !== searchRunRef.current) return
        setResponse(null)
        setTimedOut(isTimeoutError(e))
      } finally {
        if (runId === searchRunRef.current) setLoading(false)
      }
    },
    [applySettings, draftSettings, hasAttemptedSearch, resolvedQuery, t]
  )

  const generateAnswer = React.useCallback(async () => {
    const lastSearch = lastSearchRef.current
    if (!lastSearch) return
    const runId = searchRunRef.current
    setAnswerLoading(true)
    setAnswerError(null)
    try {
      await tldwClient.initialize()
      const { query, options, timeoutMs } = buildRagSearchRequest({
        ...lastSearch.settings,
        query: lastSearch.query,
        enable_generation: true
      })
      const rawResponse = await tldwClient.ragSearch(query, {
        ...options,
        timeoutMs
      })
      if (runId !== searchRunRef.current) return
      const normalized = normalizeQAResponse(rawResponse, lastSearch.query)
      // The answer was written from this response's sources, so show those.
      setResponse(normalized)
      if (!normalized.generatedAnswer) {
        setAnswerError(
          t(
            "sidepanel:qaSearch.noAnswerGenerated",
            "No answer was generated from these sources."
          ) as string
        )
      }
    } catch {
      if (runId !== searchRunRef.current) return
      setAnswerError(
        t(
          "sidepanel:qaSearch.generateAnswerFailed",
          "Couldn't generate an answer. Try again, or check the model in Settings."
        ) as string
      )
    } finally {
      if (runId === searchRunRef.current) setAnswerLoading(false)
    }
  }, [t])

  const copyAnswer = React.useCallback(async () => {
    if (!response?.generatedAnswer) return
    if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      return
    }
    try {
      await navigator.clipboard.writeText(response.generatedAnswer)
    } catch (error) {
      console.error("Failed to copy answer to clipboard:", error)
    }
  }, [response])

  const insertAnswer = React.useCallback(() => {
    if (!response?.generatedAnswer) return
    onInsert(response.generatedAnswer)
  }, [onInsert, response])

  const insertChunk = React.useCallback(
    (doc: QADocument) => {
      const ragResult = qaDocumentToRagResult(doc)
      const pinned = toPinnedResult(ragResult)
      onInsert(formatRagResult(pinned, "markdown"))
    },
    [onInsert]
  )

  const copyChunk = React.useCallback(
    async (doc: QADocument, format: RagCopyFormat = "markdown") => {
      const ragResult = qaDocumentToRagResult(doc)
      const pinned = toPinnedResult(ragResult)
      if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
        return
      }
      try {
        await navigator.clipboard.writeText(formatRagResult(pinned, format))
      } catch (error) {
        console.error("Failed to copy chunk to clipboard:", error)
      }
    },
    []
  )

  const pinChunk = React.useCallback(
    (doc: QADocument) => {
      const ragResult = qaDocumentToRagResult(doc)
      onPin(ragResult)
    },
    [onPin]
  )

  return {
    loading,
    response,
    timedOut,
    hasAttemptedSearch,
    queryError,
    answerLoading,
    answerError,
    runQASearch,
    generateAnswer,
    copyAnswer,
    insertAnswer,
    insertChunk,
    copyChunk,
    pinChunk
  }
}
