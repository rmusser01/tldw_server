import React from "react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { RagSettings } from "@/services/rag/unified-rag"
import {
  formatRagResult,
  type RagCopyFormat,
  type RagPinnedResult
} from "@/utils/rag-format"
import {
  type RagResult,
  type SortMode,
  normalizeMediaSearchResults,
  toPinnedResult,
  withFullMediaTextIfAvailable,
  getResultType,
  getResultDate,
  getResultScore,
  getResultUrl,
  extractMediaId
} from "./useKnowledgeSearch"

/**
 * Media type options for file search filtering
 */
export const FILE_SEARCH_MEDIA_TYPES = [
  "video",
  "audio",
  "pdf",
  "article",
  "note",
  "document",
  "epub",
  "html",
  "xml"
] as const

export type FileSearchMediaType = (typeof FILE_SEARCH_MEDIA_TYPES)[number]

export type UseFileSearchReturn = {
  // State
  loading: boolean
  results: RagResult[]
  sortMode: SortMode
  timedOut: boolean
  hasAttemptedSearch: boolean
  queryError: string | null

  // Filters
  mediaTypes: FileSearchMediaType[]
  setMediaTypes: (types: FileSearchMediaType[]) => void

  // Attached tracking
  attachedMediaIds: Set<number>

  // Actions
  runSearch: (opts?: { applyFirst?: boolean }) => Promise<void>
  setSortMode: (mode: SortMode) => void
  sortResults: (items: RagResult[]) => RagResult[]

  // Result actions
  handleAttach: (item: RagResult) => void
  handlePreview: (item: RagResult) => void
  handleOpen: (item: RagResult) => void
  handlePin: (item: RagResult) => void
  copyResult: (item: RagResult, format: RagCopyFormat) => Promise<void>
}

type UseFileSearchOptions = {
  resolvedQuery: string
  draftSettings: RagSettings
  applySettings: () => void
  onInsert: (text: string) => void
  pinnedResults: RagPinnedResult[]
  onPin: (item: RagResult) => void
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
 * Hook for file/media search — searches the media library for discovery and attachment.
 * Refactored from the original useKnowledgeSearch to focus on media item discovery.
 */
export function useFileSearch({
  resolvedQuery,
  draftSettings,
  applySettings,
  onInsert,
  pinnedResults,
  onPin
}: UseFileSearchOptions): UseFileSearchReturn {
  const { t } = useTranslation(["sidepanel", "common"])

  // Search state
  const [loading, setLoading] = React.useState(false)
  const [results, setResults] = React.useState<RagResult[]>([])
  const [sortMode, setSortMode] = React.useState<SortMode>("relevance")
  const [timedOut, setTimedOut] = React.useState(false)
  const [hasAttemptedSearch, setHasAttemptedSearch] = React.useState(false)
  const [queryError, setQueryError] = React.useState<string | null>(null)

  // Filter state
  const [mediaTypes, setMediaTypes] = React.useState<FileSearchMediaType[]>([])

  // Track which media items have been attached this session
  const [attachedMediaIds, setAttachedMediaIds] = React.useState<Set<number>>(
    new Set()
  )

  // Sort results
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

      setQueryError(null)
      if (!hasAttemptedSearch) {
        setHasAttemptedSearch(true)
      }

      setLoading(true)
      setTimedOut(false)
      setResults([])

      try {
        await tldwClient.initialize()
        const payload: Record<string, unknown> = {
          query,
          fields: ["title", "content"],
          sort_by: "relevance"
        }
        if (mediaTypes.length > 0) {
          payload.media_types = mediaTypes
        }
        const mediaResponse = await tldwClient.searchMedia(
          payload as Parameters<typeof tldwClient.searchMedia>[0],
          { page: 1, results_per_page: 50 }
        )
        const mediaResults = normalizeMediaSearchResults(mediaResponse)
        setResults(mediaResults)
        setTimedOut(false)
      } catch (e) {
        setResults([])
        setTimedOut(isTimeoutError(e))
      } finally {
        setLoading(false)
      }
    },
    [applySettings, hasAttemptedSearch, mediaTypes, resolvedQuery, t]
  )

  // Attach action — fetches full text and inserts into chat
  const handleAttach = React.useCallback(
    (item: RagResult) => {
      void (async () => {
        const pinned = toPinnedResult(item)
        const resolved = await withFullMediaTextIfAvailable(pinned)
        onInsert(formatRagResult(resolved, "markdown"))
        // Track as attached
        const mediaId = extractMediaId(item)
        if (mediaId !== null) {
          setAttachedMediaIds((prev) => new Set(prev).add(mediaId))
        }
      })()
    },
    [onInsert]
  )

  const handlePreview = React.useCallback((_item: RagResult) => {
    // Preview is handled at the KnowledgePanel level via callback
  }, [])

  const handleOpen = React.useCallback((item: RagResult) => {
    const url = getResultUrl(item)
    if (!url) return
    window.open(String(url), "_blank")
  }, [])

  const handlePin = React.useCallback(
    (item: RagResult) => {
      onPin(item)
    },
    [onPin]
  )

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
        console.error("Failed to copy result to clipboard:", error)
      }
    },
    []
  )

  return {
    loading,
    results,
    sortMode,
    timedOut,
    hasAttemptedSearch,
    queryError,
    mediaTypes,
    setMediaTypes,
    attachedMediaIds,
    runSearch,
    setSortMode,
    sortResults,
    handleAttach,
    handlePreview,
    handleOpen,
    handlePin,
    copyResult
  }
}
