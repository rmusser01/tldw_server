import React from "react"
import { useTranslation } from "react-i18next"
import { useStorage } from "@plasmohq/storage/hook"
import { shallow } from "zustand/shallow"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { type RagSettings } from "@/services/rag/unified-rag"
import {
  formatRagResult,
  type RagCopyFormat,
  type RagPinnedResult
} from "@/utils/rag-format"
import { useStoreMessageOption } from "@/store/option"

/**
 * RAG search result type
 */
export type RagResult = {
  content?: string
  text?: string
  chunk?: string
  metadata?: Record<string, unknown>
  score?: number
  relevance?: number
}

/**
 * Batch result grouping
 */
export type BatchResultGroup = {
  query: string
  results: RagResult[]
}

/**
 * Sort mode for results
 */
export type SortMode = "relevance" | "date" | "type"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

const getMetadataValue = (
  metadata: RagResult["metadata"],
  key: string
): unknown => (isRecord(metadata) ? metadata[key] : undefined)

const getMetadataString = (metadata: RagResult["metadata"], keys: string[]) => {
  for (const key of keys) {
    const value = getMetadataValue(metadata, key)
    if (typeof value === "string" && value.trim().length > 0) {
      return value
    }
  }
  return ""
}

const getMetadataPrimitive = (
  metadata: RagResult["metadata"],
  keys: string[]
) => {
  for (const key of keys) {
    const value = getMetadataValue(metadata, key)
    if (typeof value === "string") {
      const trimmed = value.trim()
      if (trimmed.length > 0) return trimmed
    } else if (typeof value === "number") {
      return value
    }
  }
  return undefined
}

const getFirstNonEmptyString = (...values: unknown[]): string => {
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) {
      return value
    }
  }
  return ""
}

// Helper functions for result extraction
export const getResultText = (item: RagResult) =>
  item.content || item.text || item.chunk || ""

export const getResultTitle = (item: RagResult) =>
  getMetadataString(item.metadata, ["title", "source", "url"])

export const getResultUrl = (item: RagResult) =>
  getMetadataString(item.metadata, ["url", "source"])

export const getResultType = (item: RagResult) =>
  getMetadataString(item.metadata, ["type"])

export const getResultDate = (item: RagResult) =>
  getMetadataPrimitive(item.metadata, ["created_at", "date", "added_at"])

export const getResultId = (item: RagResult) =>
  getMetadataPrimitive(item.metadata, ["id"])

export const getResultSource = (item: RagResult) =>
  getMetadataString(item.metadata, ["source"])

export const getResultChunkIndex = (item: RagResult) =>
  getMetadataPrimitive(item.metadata, [
    "chunk_index",
    "chunkIndex",
    "index",
    "offset"
  ])

export const getResultScore = (item: RagResult) =>
  typeof item.score === "number"
    ? item.score
    : typeof item.relevance === "number"
      ? item.relevance
      : undefined

const toPositiveInt = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number(value.trim())
    if (Number.isInteger(parsed) && parsed > 0) return parsed
  }
  return null
}

export const extractMediaId = (item: RagResult): number | null => {
  const fromMediaId = toPositiveInt(getMetadataValue(item.metadata, "media_id"))
  if (fromMediaId !== null) return fromMediaId
  return toPositiveInt(getMetadataValue(item.metadata, "id"))
}

const toOptionalNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  return undefined
}

const getFirstString = (
  record: Record<string, unknown>,
  keys: string[]
): string => {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim()
    }
  }
  return ""
}

export const normalizeMediaSearchResults = (payload: unknown): RagResult[] => {
  const obj = isRecord(payload) ? payload : {}
  const rawItems = Array.isArray(obj.items)
    ? obj.items
    : Array.isArray(obj.results)
      ? obj.results
      : Array.isArray(obj.media)
        ? obj.media
        : []

  const normalized: RagResult[] = []
  for (const rawItem of rawItems) {
    if (!isRecord(rawItem)) continue
    const mediaId = toPositiveInt(rawItem.media_id) ?? toPositiveInt(rawItem.id)
    const title =
      getFirstString(rawItem, ["title", "name", "filename"]) ||
      (mediaId ? `Media ${mediaId}` : "Untitled media")
    const type = getFirstString(rawItem, ["type", "media_type"]) || "media"
    const itemUrl =
      getFirstString(rawItem, ["url", "source"]) ||
      (mediaId ? `/api/v1/media/${mediaId}` : "")
    const snippet =
      getFirstString(rawItem, ["snippet", "summary", "description", "content"]) ||
      `Library item: ${title}`
    const metadata: Record<string, unknown> = {
      title,
      type,
      source: title,
      url: itemUrl,
      origin: "media-library",
      created_at: getFirstString(rawItem, ["created_at", "date", "added_at"]) || undefined
    }
    if (mediaId !== null) {
      metadata.id = mediaId
      metadata.media_id = mediaId
    }

    normalized.push({
      content: snippet,
      metadata,
      score: toOptionalNumber(rawItem.score),
      relevance: toOptionalNumber(rawItem.relevance)
    })
  }

  return normalized
}

const getErrorMessage = (error: unknown) => {
  if (typeof error === "string") return error
  if (error instanceof Error) return error.message
  if (typeof error === "object" && error !== null) {
    const candidate = (error as { message?: unknown }).message
    if (typeof candidate === "string") return candidate
  }
  return ""
}

const isTimeoutError = (error: unknown) => {
  if (error instanceof Error && error.name === "AbortError") return true
  const message = getErrorMessage(error).toLowerCase()
  return message.includes("timeout") || message.includes("timed out")
}

const hashString = (value: string) => {
  let hash = 0
  for (let i = 0; i < value.length; i++) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0
  }
  return hash.toString(36)
}

const buildPinnedResultId = (item: RagResult, text: string) => {
  const seedParts = [
    getResultId(item),
    getResultUrl(item),
    getResultSource(item),
    getResultTitle(item),
    getResultType(item),
    getResultDate(item),
    text ? text.slice(0, 4096) : ""
  ]
    .map((value) => (value == null ? "" : String(value).trim()))
    .filter(Boolean)

  if (seedParts.length === 0) {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return `rag-${crypto.randomUUID()}`
    }
    return `rag-${Math.random().toString(36).slice(2)}`
  }

  return `rag-${hashString(seedParts.join("|"))}`
}

/**
 * Convert a RAG result to a pinned result format
 */
export const toPinnedResult = (item: RagResult): RagPinnedResult => {
  const text = getResultText(item)
  const snippet = text.slice(0, 800)
  const title = getResultTitle(item)
  const url = getResultUrl(item)
  const contextOrigin =
    getMetadataValue(item.metadata, "origin") === "media-library"
      ? "media-library"
      : undefined
  return {
    id: buildPinnedResultId(item, text),
    title: title || undefined,
    source: getResultSource(item) || undefined,
    url: url || undefined,
    snippet,
    type: getResultType(item) || undefined,
    mediaId: extractMediaId(item) ?? undefined,
    contextOrigin
  }
}

export const extractContentFromMediaDetail = (detail: unknown): string => {
  if (typeof detail === "string") return detail
  if (!isRecord(detail)) return ""

  const contentValue = detail.content
  if (isRecord(contentValue)) {
    const nestedContent = getFirstNonEmptyString(
      contentValue.text,
      contentValue.content,
      contentValue.raw_text,
      contentValue.rawText,
      contentValue.transcript,
      contentValue.summary
    )
    if (nestedContent) return nestedContent
  } else if (typeof contentValue === "string" && contentValue.trim().length > 0) {
    return contentValue
  }

  const fromRoot = getFirstNonEmptyString(
    detail.text,
    detail.transcript,
    detail.raw_text,
    detail.rawText,
    detail.raw_content,
    detail.rawContent,
    detail.summary
  )
  if (fromRoot) return fromRoot

  const latestVersion = isRecord(detail.latest_version)
    ? detail.latest_version
    : isRecord(detail.latestVersion)
      ? detail.latestVersion
      : null
  if (latestVersion) {
    const fromLatest = getFirstNonEmptyString(
      latestVersion.content,
      latestVersion.text,
      latestVersion.transcript,
      latestVersion.raw_text,
      latestVersion.rawText,
      latestVersion.summary
    )
    if (fromLatest) return fromLatest
  }

  const data = isRecord(detail.data) ? detail.data : null
  if (data) {
    const fromData = getFirstNonEmptyString(
      data.content,
      data.text,
      data.transcript,
      data.raw_text,
      data.rawText,
      data.summary
    )
    if (fromData) return fromData
  }

  return ""
}

const fetchFullMediaTextById = async (mediaId: number): Promise<string | null> => {
  try {
    await tldwClient.initialize()
    const detail = await tldwClient.getMediaDetails(mediaId, {
      include_content: true,
      include_versions: false,
      include_version_content: false
    })
    const fullText = extractContentFromMediaDetail(detail)
    return fullText || null
  } catch {
    return null
  }
}

export const PINNED_MEDIA_CONTEXT_CHAR_CAP = 20_000
export const PINNED_MEDIA_CONTEXT_TRUNCATION_NOTICE =
  "[Truncated to 20,000 characters]"

const limitPinnedSnippetForContext = (
  pinned: RagPinnedResult
): RagPinnedResult => {
  if (pinned.snippet.length <= PINNED_MEDIA_CONTEXT_CHAR_CAP) return pinned
  const notice = `${PINNED_MEDIA_CONTEXT_TRUNCATION_NOTICE}\n\n`
  const sliceLength = Math.max(
    0,
    PINNED_MEDIA_CONTEXT_CHAR_CAP - notice.length
  )
  return {
    ...pinned,
    snippet: notice + pinned.snippet.slice(0, sliceLength)
  }
}

export const withFullMediaTextIfAvailable = async (
  pinned: RagPinnedResult
): Promise<RagPinnedResult> => {
  if (!pinned.mediaId || pinned.contextOrigin !== "media-library") return pinned
  const fullText = await fetchFullMediaTextById(pinned.mediaId)
  if (!fullText) return pinned
  return {
    ...pinned,
    snippet: fullText
  }
}

/**
 * Return type for useKnowledgeSearch hook
 */
export type UseKnowledgeSearchReturn = {
  // State
  loading: boolean
  results: RagResult[]
  batchResults: BatchResultGroup[]
  sortMode: SortMode
  timedOut: boolean
  hasAttemptedSearch: boolean
  queryError: string | null
  previewItem: RagPinnedResult | null
  ragHintSeen: boolean

  // Pinned results
  pinnedResults: RagPinnedResult[]

  // Actions
  runSearch: (opts?: { applyFirst?: boolean }) => Promise<void>
  setSortMode: (mode: SortMode) => void
  setPreviewItem: (item: RagPinnedResult | null) => void
  sortResults: (items: RagResult[]) => RagResult[]

  // Result actions
  handleInsert: (item: RagResult) => void
  handleAsk: (item: RagResult) => void
  handleOpen: (item: RagResult) => void
  handlePin: (item: RagResult) => void
  handleUnpin: (id: string) => void
  handleClearPins: () => void
  copyResult: (item: RagResult, format: RagCopyFormat) => Promise<void>
}

type UseKnowledgeSearchOptions = {
  resolvedQuery: string
  draftSettings: RagSettings
  applySettings: () => void
  onInsert: (text: string) => void
  onAsk: (text: string, options?: { ignorePinnedResults?: boolean }) => void
}

/**
 * Hook for managing RAG search execution and results
 */
export function useKnowledgeSearch({
  resolvedQuery,
  draftSettings,
  applySettings,
  onInsert,
  onAsk
}: UseKnowledgeSearchOptions): UseKnowledgeSearchReturn {
  const { t } = useTranslation(["sidepanel", "common"])

  // Search state
  const [loading, setLoading] = React.useState(false)
  const [results, setResults] = React.useState<RagResult[]>([])
  const [batchResults, setBatchResults] = React.useState<BatchResultGroup[]>([])
  const [sortMode, setSortMode] = React.useState<SortMode>("relevance")
  const [timedOut, setTimedOut] = React.useState(false)
  const [hasAttemptedSearch, setHasAttemptedSearch] = React.useState(false)
  const [queryError, setQueryError] = React.useState<string | null>(null)
  const [previewItem, setPreviewItem] = React.useState<RagPinnedResult | null>(null)

  // Persisted hint state
  const [ragHintSeen, setRagHintSeen] = useStorage<boolean>(
    "ragSearchHintSeen",
    false
  )

  // Pinned results from store
  const { ragPinnedResults, setRagPinnedResults, setRagMediaIds } = useStoreMessageOption(
    (state) => ({
      ragPinnedResults: state.ragPinnedResults,
      setRagPinnedResults: state.setRagPinnedResults,
      setRagMediaIds: state.setRagMediaIds
    }),
    shallow
  )

  const pinnedResults = ragPinnedResults || []
  const nextPinTokenRef = React.useRef(0)
  const pendingPinTokensRef = React.useRef(new Map<string, number>())
  const collectPinnedMediaIds = React.useCallback((items: RagPinnedResult[]) => {
    const ids = items
      .map((item) => item.mediaId)
      .filter((id): id is number => typeof id === "number" && Number.isInteger(id) && id > 0)
    return Array.from(new Set(ids))
  }, [])

  // Sort results by the selected mode
  const sortResults = React.useCallback(
    (items: RagResult[]) => {
      if (sortMode === "type") {
        return [...items].sort((a, b) =>
          String(getResultType(a)).localeCompare(String(getResultType(b)))
        )
      }
      if (sortMode === "date") {
        return [...items].sort((a, b) => {
          const dateA = new Date(getResultDate(a) || 0).getTime()
          const dateB = new Date(getResultDate(b) || 0).getTime()
          return dateB - dateA
        })
      }
      // Default: relevance
      return [...items].sort((a, b) => {
        const scoreA = getResultScore(a) ?? 0
        const scoreB = getResultScore(b) ?? 0
        return scoreB - scoreA
      })
    },
    [sortMode]
  )

  // Execute search
  const runSearch = React.useCallback(
    async (opts?: { applyFirst?: boolean }) => {
      if (opts?.applyFirst) {
        applySettings()
      }

      const query = (resolvedQuery || "").trim()

      if (!query) {
        setQueryError(
          t("sidepanel:rag.queryRequired", "Enter a query to search.") as string
        )
        return
      }
      const selectedSources = Array.isArray(draftSettings.sources)
        ? draftSettings.sources
        : []
      const canSearchMedia =
        selectedSources.length === 0 || selectedSources.includes("media_db")
      if (!canSearchMedia) {
        setQueryError(
          t(
            "sidepanel:rag.mediaSourceRequired",
            "Select Media source to search your library."
          ) as string
        )
        setResults([])
        setBatchResults([])
        return
      }

      setQueryError(null)
      if (!hasAttemptedSearch) {
        setHasAttemptedSearch(true)
        setRagHintSeen(true)
      }

      setLoading(true)
      setTimedOut(false)
      setResults([])
      setBatchResults([])

      try {
        await tldwClient.initialize()
        const mediaResponse = await tldwClient.searchMedia(
          {
            query,
            fields: ["title", "content"],
            sort_by: "relevance"
          },
          { page: 1, results_per_page: 50 }
        )
        const mediaResults = normalizeMediaSearchResults(mediaResponse)
        setBatchResults([])
        setResults(mediaResults)
        setTimedOut(false)
      } catch (e) {
        setResults([])
        setBatchResults([])
        setTimedOut(isTimeoutError(e))
      } finally {
        setLoading(false)
      }
    },
    [
      applySettings,
      draftSettings,
      hasAttemptedSearch,
      resolvedQuery,
      setRagHintSeen,
      t
    ]
  )

  // Result actions
  const copyResult = React.useCallback(
    async (item: RagResult, format: RagCopyFormat) => {
      const pinned = toPinnedResult(item)
      if (typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
        return
      }
      try {
        const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
        await navigator.clipboard.writeText(
          formatRagResult(resolvedPinned, format)
        )
      } catch (error) {
        console.error("Failed to copy knowledge result to clipboard:", error)
      }
    },
    []
  )

  const handleInsert = React.useCallback(
    (item: RagResult) => {
      void (async () => {
        const pinned = toPinnedResult(item)
        const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
        onInsert(formatRagResult(resolvedPinned, "markdown"))
      })()
    },
    [onInsert]
  )

  const handleAsk = React.useCallback(
    (item: RagResult) => {
      void (async () => {
        const pinned = toPinnedResult(item)
        const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
        // Note: Modal.confirm would need to be handled at the component level
        // For now, we directly call onAsk
        onAsk(formatRagResult(resolvedPinned, "markdown"), {
          ignorePinnedResults: true
        })
      })()
    },
    [onAsk]
  )

  const handleOpen = React.useCallback((item: RagResult) => {
    const url = getResultUrl(item)
    if (!url) return
    window.open(String(url), "_blank")
  }, [])

  const handlePin = React.useCallback(
    (item: RagResult) => {
      void (async () => {
        const pinned = toPinnedResult(item)
        const currentPinnedBefore =
          useStoreMessageOption.getState().ragPinnedResults || []
        if (currentPinnedBefore.some((result) => result.id === pinned.id)) {
          return
        }
        const pinToken = nextPinTokenRef.current + 1
        nextPinTokenRef.current = pinToken
        pendingPinTokensRef.current.set(pinned.id, pinToken)
        const resolvedPinned = limitPinnedSnippetForContext(
          await withFullMediaTextIfAvailable(pinned)
        )
        if (pendingPinTokensRef.current.get(pinned.id) !== pinToken) return
        const currentPinned =
          useStoreMessageOption.getState().ragPinnedResults || []
        if (currentPinned.some((result) => result.id === resolvedPinned.id)) {
          pendingPinTokensRef.current.delete(pinned.id)
          return
        }
        const nextPinned = [...currentPinned, resolvedPinned]
        pendingPinTokensRef.current.delete(pinned.id)
        setRagPinnedResults(nextPinned)
        const mediaIds = collectPinnedMediaIds(nextPinned)
        setRagMediaIds(mediaIds.length > 0 ? mediaIds : null)
      })()
    },
    [collectPinnedMediaIds, setRagMediaIds, setRagPinnedResults]
  )

  const handleUnpin = React.useCallback(
    (id: string) => {
      pendingPinTokensRef.current.delete(id)
      const currentPinned =
        useStoreMessageOption.getState().ragPinnedResults || []
      const nextPinned = currentPinned.filter((item) => item.id !== id)
      setRagPinnedResults(nextPinned)
      const mediaIds = collectPinnedMediaIds(nextPinned)
      setRagMediaIds(mediaIds.length > 0 ? mediaIds : null)
    },
    [collectPinnedMediaIds, setRagMediaIds, setRagPinnedResults]
  )

  const handleClearPins = React.useCallback(() => {
    pendingPinTokensRef.current.clear()
    setRagPinnedResults([])
    setRagMediaIds(null)
  }, [setRagMediaIds, setRagPinnedResults])

  return {
    // State
    loading,
    results,
    batchResults,
    sortMode,
    timedOut,
    hasAttemptedSearch,
    queryError,
    previewItem,
    ragHintSeen,

    // Pinned results
    pinnedResults,

    // Actions
    runSearch,
    setSortMode,
    setPreviewItem,
    sortResults,

    // Result actions
    handleInsert,
    handleAsk,
    handleOpen,
    handlePin,
    handleUnpin,
    handleClearPins,
    copyResult
  }
}
