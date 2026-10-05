import { IDLE_CONTENT_FILTER_PROGRESS } from "@/components/Review/content-filtering-progress"
import { rankKeywordSuggestions } from "@/components/Review/filter-chip-priority"
import { type MediaMultiBatchExportItem,
  buildBatchExportArtifact,
  parseBatchKeywords
} from "@/components/Review/media-multi-batch-actions"
import { DEFAULT_SORT_BY, type MediaDetail,
  type MediaItem,
  type MediaReviewActions,
  type MediaReviewState,
  UNDO_DURATION_SECONDS,
  getContent,
  getErrorStatusCode,
  idsEqual,
  includesId
} from "@/components/Review/media-review-types"
import { buildMediaTrashHandoffSearch } from "@/components/Review/mediaPermalink"
import { buildMediaSearchPayload } from "@/components/Review/mediaSearchRequest"
import { useHomeMilestoneScope } from "@/hooks/useHomeMilestoneScope"
import { bgRequest } from "@/services/background-proxy"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { getNoteKeywords, searchNoteKeywords } from "@/services/note-keywords"
import { clearSetting } from "@/services/settings/registry"
import {
  LAST_MEDIA_ID_SETTING
} from "@/services/settings/ui-settings"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import {
  buildMediaChatHandoffRoute,
  createMediaChatHandoff, removeMediaChatHandoff } from "@/services/tldw/media-chat-handoff"
import { quickIngestAuthority } from "@/services/tldw/quick-ingest-authority"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { downloadBlob } from "@/utils/download-blob"
import { extractMediaDetailAnalysis } from "@/utils/media-detail-content"
import {
  Button,
  Modal } from "antd"
import React from "react"

export function useMediaReviewActions(s: MediaReviewState): MediaReviewActions & { _fetchList: () => Promise<MediaItem[]> } {
  // A retained callback belongs to this rendered owner generation, even if the
  // same owner is verified again after an account transition or remount.
  let authorityRevision: number | null = null
  try {
    authorityRevision = quickIngestAuthority.capture({
      sessionBound: false
    }).authorityRevision
  } catch {
    /* Unverified views cannot start actions. */
  }
  const actionGeneration = React.useMemo(
    () => ({
      authorityKey: s.authorityKey,
      authorityRevision,
      operation: null as ReturnType<typeof quickIngestAuthority.capture> | null,
      lifetime: null as AbortController | null
    }),
    [s.authorityKey, authorityRevision]
  )
  React.useLayoutEffect(() => {
    const lifetime = new AbortController()
    actionGeneration.lifetime = lifetime
    try {
      actionGeneration.operation = quickIngestAuthority.capture({
        sessionBound: false
      })
    } catch {
      actionGeneration.operation = null
    }
    return () => lifetime.abort()
  }, [actionGeneration])
  const captureOperation = React.useCallback(() => {
    const { lifetime, operation, authorityKey } = actionGeneration
    if (
      !authorityKey ||
      !lifetime ||
      lifetime.signal.aborted ||
      operation?.authorityKey !== authorityKey ||
      operation.authorityRevision !== actionGeneration.authorityRevision ||
      !operation.isCurrent()
    )
      return null
    return {
      ...operation,
      isCurrent: () => operation.isCurrent() && !lifetime.signal.aborted
    }
  }, [actionGeneration])

  const detailRequests = React.useRef(new Map<string, Promise<MediaDetail>>())
  const ownerScope = useHomeMilestoneScope()
  const handoffBoundaryRevision = React.useRef(0)
  React.useLayoutEffect(() => watchChatAccountChanges(invalidated => {
    if (invalidated) handoffBoundaryRevision.current += 1
  }), [])
  const handoffLifetime = React.useRef<AbortController | null>(null)
  React.useLayoutEffect(() => {
    const lifetime = new AbortController()
    handoffLifetime.current = lifetime
    return () => lifetime.abort()
  }, [ownerScope, s.selectedIds])
  const {
    t, navigate, message,
    authorityKey, readingActive, setReadingActive, readingIds,
    readingWindowStart, setReadingWindowStart, previewNavigationIds, setPreviewNavigationIds,
    selectedMetadata, isMobileViewport, setMobileTab,
    query, page, pageSize, types, keywordTokens, includeContent, sortBy, dateRange,
    selectedIds, setSelectedIds, focusedId, setFocusedId,
    previewedId, setPreviewedId,
    details, setDetails, setTotal, setAvailableTypes, availableTypes,
    setContentLoading, setContentFilterProgress, contentFilterRunRef,
    setDetailLoading, setFailedIds, detailLoading,
    openAllLimit, allResults, viewerItems,
    viewMode, viewerVirtualizer,
    batchKeywordsDraft, setBatchKeywordsDraft,
    batchExportFormat, setBatchActionLoading, setBatchTrashHandoffIds,
    setCompareLeftText, setCompareRightText,
    setCompareLeftLabel, setCompareRightLabel, setCompareDiffOpen,
    visibleIds,
    setContentExpandedIds, setAnalysisExpandedIds,
    viewerRef, listParentRef,
    lastClickedRef, pendingRestoreFocusIdRef, ensureDetailRef,
    cardRefs, prefersReducedMotion,
    setKeywordOptions,
    data,
    pendingInitialMediaId, setPendingInitialMediaId,
    isOnline
  } = s

  // Reading, comparison, content filtering and export share the same request ceiling.
  const fetchDetail = React.useCallback(
    async (
      id: string | number,
      operation = captureOperation()
    ): Promise<MediaDetail> => {
      if (!operation?.isCurrent()) throw new Error("Review owner is unavailable")
      const key = `${authorityKey}:${id}`
      while (
        detailRequests.current.size >= openAllLimit &&
        !detailRequests.current.has(key)
      ) {
        await Promise.race(
          [...detailRequests.current.values()].map((promise) =>
            promise.catch(() => null)
          )
        )
        if (!operation.isCurrent()) throw new Error("Review owner changed")
      }
      const existing = detailRequests.current.get(key)
      if (existing) return existing
      const pending = bgRequest<MediaDetail>({
        path: `/api/v1/media/${id}?include_content=true&include_versions=false` as any,
        method: "GET" as any,
        abortSignal: operation.signal,
        ...requestScopeFields(operation.requestScope)
      })
        .then((detail) => {
          if (!operation.isCurrent()) throw new Error("Review owner changed")
          return detail
        })
        .finally(() => {
          detailRequests.current.delete(key)
        })
      detailRequests.current.set(key, pending)
      return pending
    },
    [authorityKey, openAllLimit, captureOperation]
  )

  const cancelContentFiltering = React.useCallback(() => {
    if (!captureOperation()) return
    contentFilterRunRef.current += 1
    setContentLoading(false)
    setContentFilterProgress(IDLE_CONTENT_FILTER_PROGRESS)
  }, [captureOperation, contentFilterRunRef, setContentLoading, setContentFilterProgress])

  const runContentFiltering = React.useCallback(async (items: MediaItem[], operation = captureOperation()): Promise<MediaItem[]> => {
    if (!operation?.isCurrent()) return []
    const hasQuery = query.trim().length > 0
    const tokens = keywordTokens.map((k) => k.toLowerCase())
    if (!includeContent || (!hasQuery && tokens.length === 0)) {
      setContentLoading(false)
      setContentFilterProgress(IDLE_CONTENT_FILTER_PROGRESS)
      return items
    }

    const runId = contentFilterRunRef.current + 1
    contentFilterRunRef.current = runId
    setContentLoading(true)
    setContentFilterProgress({
      running: items.length > 0,
      completed: 0,
      total: items.length
    })

    if (items.length === 0) {
      setContentLoading(false)
      setContentFilterProgress(IDLE_CONTENT_FILTER_PROGRESS)
      return items
    }

    const queryLower = query.toLowerCase()
    const enriched: Array<{ m: MediaItem; content: string }> = []

    for (let idx = 0; idx < items.length; idx += 1) {
      if (!operation.isCurrent() || contentFilterRunRef.current !== runId) return []
      const m = items[idx]
      let d = details[m.id]
      if (!d) {
        try {
          d = await fetchDetail(m.id, operation)
          if (!operation.isCurrent() || contentFilterRunRef.current !== runId) return []
          setDetails((prev) => (prev[m.id] ? prev : { ...prev, [m.id]: d! }))
        } catch {
          if (!operation.isCurrent()) return []
          // Failed detail fetch should not terminate full list filtering.
        }
      }
      enriched.push({ m, content: d ? getContent(d) : '' })
      if (contentFilterRunRef.current === runId) {
        setContentFilterProgress({
          running: true,
          completed: idx + 1,
          total: items.length
        })
      }
    }

    if (!operation.isCurrent() || contentFilterRunRef.current !== runId) return []

    const filtered = enriched.filter(({ m, content }) => {
      const hay = `${m.title || ''} ${m.snippet || ''} ${content}`.toLowerCase()
      if (hasQuery && !hay.includes(queryLower)) return false
      if (tokens.length > 0 && !tokens.every((token) => hay.includes(token))) return false
      return true
    }).map(({ m }) => m)

    if (contentFilterRunRef.current === runId) {
      setContentLoading(false)
      setContentFilterProgress({
        running: false,
        completed: items.length,
        total: items.length
      })
    }
    return filtered
  }, [captureOperation, fetchDetail, details, includeContent, keywordTokens, query, contentFilterRunRef, setContentLoading, setContentFilterProgress, setDetails])

  const mapMediaItems = React.useCallback(
    (items: any[]): MediaItem[] =>
      items.map((m: any) => ({
      id: m?.id ?? m?.media_id ?? m?.pk ?? m?.uuid,
      title: m?.title || m?.filename || `Media ${m?.id}`,
      snippet: m?.snippet || m?.summary || "",
      type: String(m?.type || m?.media_type || "").toLowerCase(),
      created_at: m?.created_at
    })),
    []
  )

  const ensureDetail = React.useCallback(
    async (
      id: string | number,
      isRetry = false,
      operation = captureOperation()
    ) => {
      if (!operation?.isCurrent()) return
      if ((!isRetry && details[id]) || detailLoading[id]) return
      setDetailLoading((prev) => ({ ...prev, [id]: true }))
      if (isRetry) {
        setFailedIds((prev) => {
          const next = new Set(prev)
          next.delete(id)
          return next
        })
      }
      try {
        const d = await fetchDetail(id, operation)
        if (!operation.isCurrent()) return
        const base = Array.isArray(data)
          ? data.find((x) => idsEqual(x.id, id))
          : undefined
        const enriched = {
          ...d,
          id,
          title: d?.title ?? base?.title,
          type: d?.type ?? base?.type,
          created_at: d?.created_at ?? base?.created_at
        }
        setDetails((prev) => ({ ...prev, [id]: enriched }))
        setFailedIds((prev) => {
          const next = new Set(prev)
          next.delete(id)
          return next
        })
      } catch (error) {
        if (!operation.isCurrent()) return
        setFailedIds((prev) => new Set(prev).add(id))
        const statusCode = getErrorStatusCode(error)
        if (statusCode === 404 || statusCode === 410) {
          setSelectedIds((prev) => {
            const next = prev.filter(
              (candidateId) => String(candidateId) !== String(id)
            )
            return next.length === prev.length ? prev : next
          })
          setFocusedId((prev) =>
            prev != null && String(prev) === String(id) ? null : prev
          )
        }
      } finally {
        if (operation.isCurrent())
          setDetailLoading((prev) => {
            const next = { ...prev }
            delete next[id]
            return next
          })
      }
    },
    [
      captureOperation,
      fetchDetail,
      data,
      detailLoading,
      details,
      setDetailLoading,
      setDetails,
      setFailedIds,
      setSelectedIds,
      setFocusedId
    ]
  )

  // Keep ref in sync
  React.useEffect(() => {
    ensureDetailRef.current = ensureDetail
  }, [ensureDetail, ensureDetailRef])

  const retryFetch = React.useCallback((id: string | number) => {
    const operation = captureOperation()
    if (!operation) return
    setDetails((prev) => {
      const next = { ...prev }
      delete next[id]
      return next
    })
    void ensureDetail(id, true, operation)
  }, [captureOperation, ensureDetail, setDetails])

  const clearSelectionWithGuard = React.useCallback(() => {
    if (selectedIds.length === 0) return
    const operation = captureOperation()
    if (!operation) return
    const selectionToRestore = [...selectedIds]
    const focusToRestore = focusedId
    const activeElement = document.activeElement as HTMLElement | null
    const activeResultRow = activeElement?.closest<HTMLElement>("[data-media-id][role='button']")
    const activeResultRowId = activeResultRow?.dataset.mediaId ?? null
    const restoreFocusId =
      activeResultRowId ??
      focusToRestore ??
      selectionToRestore[0] ??
      null

    setSelectedIds([])
    setFocusedId(null)
    pendingRestoreFocusIdRef.current = restoreFocusId

    message.info(
      <span>
        {t('mediaPage.selectionClearedCount', 'Selection cleared ({{count}} items).', { count: selectionToRestore.length })}{" "}
        <Button
          type="link"
          size="small"
          className="!p-0"
          onClick={() => {
            if (!operation?.isCurrent()) return
            setSelectedIds(selectionToRestore)
            const restoredFocusId = focusToRestore ?? selectionToRestore[0] ?? null
            setFocusedId(restoredFocusId)
            pendingRestoreFocusIdRef.current = restoreFocusId
            window.setTimeout(() => {
              if (!operation?.isCurrent()) return
              const focusId = pendingRestoreFocusIdRef.current
              if (focusId == null) return
              const container = listParentRef.current
              if (!container) return
              const selectorValue =
                typeof CSS !== "undefined" && typeof CSS.escape === "function"
                  ? CSS.escape(String(focusId))
                  : String(focusId).replace(/["\\]/g, "\\$&")
              const row = container.querySelector<HTMLElement>(
                `[data-media-id="${selectorValue}"][role="button"]`
              )
              if (row) row.focus()
            }, 140)
            message.success(t('mediaPage.selectionRestored', 'Selection restored'))
          }}
        >
          {t('mediaPage.undo', 'Undo')}
        </Button>
      </span>,
      UNDO_DURATION_SECONDS
    )
  }, [captureOperation, selectedIds, focusedId, t, message, setSelectedIds, setFocusedId, listParentRef, pendingRestoreFocusIdRef])

  const focusViewerSoon = React.useCallback((operation: NonNullable<ReturnType<typeof captureOperation>>) => {
    if (!operation.isCurrent()) return
    if (typeof window === "undefined") {
      viewerRef.current?.focus()
      return
    }

    // Browsers can re-focus the clicked result row after the click handler runs.
    window.setTimeout(() => {
      if (!operation.isCurrent()) return
      viewerRef.current?.focus()
    }, 0)
  }, [viewerRef])

  const previewItem = React.useCallback((id: string | number, preserveContext = false) => {
    const operation = captureOperation()
    if (!operation) return
    setReadingActive(false)
    if (isMobileViewport) setMobileTab(2)
    if (!preserveContext) setPreviewNavigationIds(allResults.map(item => item.id))
    setPreviewedId(id)
    focusViewerSoon(operation)
    void ensureDetail(id, false, operation)
  }, [allResults, setPreviewNavigationIds, captureOperation, setReadingActive, isMobileViewport, setMobileTab, setPreviewedId, focusViewerSoon, ensureDetail])

  const startSelectedReview = React.useCallback((id?: string | number) => {
    const operation = captureOperation()
    if (!operation || selectedIds.length === 0) return
    const target = id != null && includesId(selectedIds, id) ? id : selectedIds[0]
    const index = selectedIds.findIndex(candidate => idsEqual(candidate, target))
    setReadingWindowStart(Math.floor(index / openAllLimit) * openAllLimit)
    setFocusedId(target)
    setReadingActive(true)
    if (isMobileViewport) setMobileTab(2)
    focusViewerSoon(operation)
  }, [captureOperation, selectedIds, openAllLimit, setFocusedId, setReadingWindowStart, setReadingActive, isMobileViewport, setMobileTab, focusViewerSoon])

  const returnToPreview = React.useCallback(() => {
    const operation = captureOperation()
    if (!operation) return
    setReadingActive(false)
    if (previewedId != null) void ensureDetail(previewedId, false, operation)
  }, [captureOperation, setReadingActive, previewedId, ensureDetail])

  const changeReadingWindow = React.useCallback((delta: number) => {
    if (!captureOperation()) return
    const lastStart = Math.max(0, Math.floor((selectedIds.length - 1) / openAllLimit) * openAllLimit)
    const next = Math.max(0, Math.min(lastStart, readingWindowStart + delta * openAllLimit))
    startSelectedReview(selectedIds[next])
  }, [captureOperation, selectedIds, openAllLimit, readingWindowStart, startSelectedReview])

  const toggleSelect = React.useCallback(
    async (id: string | number, event?: React.MouseEvent) => {
      if (!captureOperation()) return
      if (event?.shiftKey && lastClickedRef.current != null) {
        const lastIdx = allResults.findIndex((r) =>
          idsEqual(r.id, lastClickedRef.current!)
        )
        const currIdx = allResults.findIndex((r) => idsEqual(r.id, id))
        if (lastIdx !== -1 && currIdx !== -1) {
          const [start, end] = lastIdx < currIdx ? [lastIdx, currIdx] : [currIdx, lastIdx]
          const rangeIds = allResults.slice(start, end + 1).map((r) => r.id)
          setSelectedIds((prev) => [...prev, ...rangeIds.filter(rid => !includesId(prev, rid))])
          lastClickedRef.current = id
          return
        }
      }
      lastClickedRef.current = id
      setSelectedIds((prev) =>
        includesId(prev, id) ? prev.filter(x => !idsEqual(x, id)) : [...prev, id]
      )
    },
    [captureOperation, allResults, setSelectedIds, lastClickedRef]
  )

  // Checkbox selection is metadata only. Fetch just the active reading window.
  React.useEffect(() => {
    if (!readingActive || !captureOperation()) return
    readingIds.forEach(id => { void ensureDetailRef.current(id) })
  }, [captureOperation, readingIds, readingActive, authorityKey, ensureDetailRef])

  React.useEffect(() => {
    if (!readingActive || !captureOperation()) return
    const index = selectedIds.findIndex(id => focusedId != null && idsEqual(id, focusedId))
    if (index < 0) {
      setFocusedId(readingIds[0] ?? null)
      return
    }
    const start = Math.floor(index / openAllLimit) * openAllLimit
    if (start !== readingWindowStart) setReadingWindowStart(start)
  }, [captureOperation, readingActive, selectedIds, focusedId, readingIds, readingWindowStart, setReadingWindowStart, openAllLimit, setFocusedId])

  const addVisibleToSelection = React.useCallback(() => {
    if (!captureOperation()) return
    setSelectedIds(prev => [...prev, ...allResults.map(item => item.id).filter(id => !includesId(prev, id))])
  }, [captureOperation, allResults, setSelectedIds])

  const replaceSelectionWithVisible = React.useCallback(() => {
    const operation = captureOperation()
    if (!operation || allResults.length === 0) return
    const previousSelection = [...selectedIds]
    setSelectedIds(allResults.map(item => item.id))
    setReadingWindowStart(0)
    setFocusedId(allResults[0]?.id ?? null)
    message.info(
      <span>
        {t('mediaPage.selectionReplaced', 'Selection replaced with current visible items.')}{" "}
        <Button type="link" size="small" className="!p-0" onClick={() => {
          if (!operation.isCurrent()) return
          setSelectedIds(previousSelection)
          message.success(t('mediaPage.selectionRestored', 'Selection restored'))
        }}>
          {t("mediaPage.undo", "Undo")}
        </Button>
      </span>,
      UNDO_DURATION_SECONDS
    )
  }, [captureOperation, setReadingWindowStart, allResults, selectedIds, message, t, setSelectedIds, setFocusedId])

  const removeFromSelection = React.useCallback(
    (id: string | number) => {
      if (!captureOperation()) return
      setSelectedIds((prev) => {
        const next = prev.filter((candidate) => !idsEqual(candidate, id))
        if (next.length !== prev.length) {
          setFocusedId((current) => {
            if (current == null) return next[0] ?? null
            return idsEqual(current, id) ? (next[0] ?? null) : current
          })
        }
        return next
      })
    },
    [captureOperation, setSelectedIds, setFocusedId]
  )

  const scrollToCard = React.useCallback(
    (id: string | number) => {
      if (!captureOperation()) return
      const anchor = cardRefs.current[String(id)]
      if (anchor) {
        anchor.scrollIntoView({
          behavior: prefersReducedMotion ? "auto" : "smooth",
          block: "start"
        })
        return
      }
      if (viewMode !== "all") {
        const idx = viewerItems.findIndex((m) => idsEqual(m.id, id))
        if (idx >= 0) viewerVirtualizer.scrollToIndex(idx, { align: "start" })
      }
    },
    [captureOperation, viewMode, viewerItems, viewerVirtualizer, prefersReducedMotion, cardRefs]
  )

  const goRelative = React.useCallback((delta: number) => {
    if (!captureOperation()) return
    const ids = readingActive ? selectedIds : previewNavigationIds
    if (!ids.length) return
    const activeId = readingActive ? focusedId : previewedId
    const currentIndex = ids.findIndex(id => activeId != null && idsEqual(id, activeId))
    const nextIndex = Math.max(0, Math.min(ids.length - 1, Math.max(0, currentIndex) + delta))
    if (readingActive) startSelectedReview(ids[nextIndex])
    else previewItem(ids[nextIndex], true)
  }, [captureOperation, readingActive, previewNavigationIds, selectedIds, focusedId, previewedId, startSelectedReview, previewItem])

  // Pending initial media restoration
  React.useEffect(() => {
    if (!pendingInitialMediaId || !captureOperation()) return
    if (!Array.isArray(allResults) || allResults.length === 0) return
    const match = allResults.find((m) => String(m.id) === pendingInitialMediaId)
    if (!match) return
    setSelectedIds([match.id])
    setFocusedId(match.id)
    void ensureDetail(match.id)
    scrollToCard(match.id)
    setPendingInitialMediaId(null)
    void clearSetting(LAST_MEDIA_ID_SETTING)
  }, [captureOperation, pendingInitialMediaId, allResults, ensureDetail, scrollToCard, setSelectedIds, setFocusedId, setPendingInitialMediaId])

  // Content filtering effects
  React.useEffect(() => {
    if (!includeContent) {
      cancelContentFiltering()
    }
  }, [includeContent, cancelContentFiltering])

  React.useEffect(() => {
    return () => { cancelContentFiltering() }
  }, [cancelContentFiltering])

  // Keyword suggestions
  const loadKeywordSuggestions = React.useCallback(async (q?: string) => {
    try {
      if (q && q.trim().length > 0) {
        const arr = await searchNoteKeywords(q, 10)
        setKeywordOptions(rankKeywordSuggestions(arr, q))
      } else {
        const arr = await getNoteKeywords(200)
        setKeywordOptions(rankKeywordSuggestions(arr, ""))
      }
    } catch {
      // Keyword load failed - feature will use empty suggestions
    }
  }, [setKeywordOptions])

  React.useEffect(() => { if (isOnline) void loadKeywordSuggestions() }, [loadKeywordSuggestions, isOnline])

  const resolveDetailForCompare = React.useCallback(async (id: string | number, operation = captureOperation()): Promise<MediaDetail | null> => {
    if (!operation?.isCurrent()) return null

    const existing = details[id]
    if (existing) return existing
    try {
      const fetched = await fetchDetail(id, operation)
      if (!operation.isCurrent()) return null
      const base = allResults.find((item) => item.id === id)
      const enriched = {
        ...fetched,
        id,
        title: (fetched as any)?.title ?? base?.title,
        type: (fetched as any)?.type ?? base?.type,
        created_at: (fetched as any)?.created_at ?? base?.created_at
      } as MediaDetail
      setDetails((prev) => ({ ...prev, [id]: enriched }))
      return enriched
    } catch {
      return null
    }
  }, [captureOperation, fetchDetail, allResults, details, setDetails])

  const handleCompareContent = React.useCallback(async () => {
    const operation = captureOperation()
    if (!operation) return

    if (selectedIds.length !== 2) return
    const [leftId, rightId] = selectedIds
    const leftDetail = await resolveDetailForCompare(leftId, operation)
    if (!operation.isCurrent()) return
    const rightDetail = await resolveDetailForCompare(rightId, operation)

    if (!operation.isCurrent()) return
    if (!leftDetail || !rightDetail) {
      message.error(
        t('mediaPage.compareContentLoadFailed', 'Could not load both items for comparison. Retry and try again.')
      )
      return
    }

    const leftContent = getContent(leftDetail).trim()
    const rightContent = getContent(rightDetail).trim()
    if (!leftContent || !rightContent) {
      message.error(
        t('mediaPage.compareContentMissing', 'One or both selected items have no content to compare.')
      )
      return
    }

    setCompareLeftText(leftContent)
    setCompareRightText(rightContent)
    setCompareLeftLabel(leftDetail.title || `${t('mediaPage.media', 'Media')} ${leftId}`)
    setCompareRightLabel(rightDetail.title || `${t('mediaPage.media', 'Media')} ${rightId}`)
    setCompareDiffOpen(true)
  }, [captureOperation, message, resolveDetailForCompare, selectedIds, t, setCompareLeftText, setCompareRightText, setCompareLeftLabel, setCompareRightLabel, setCompareDiffOpen])

  const handleChatAboutSelection = React.useCallback(async () => {
    const operation = captureOperation()
    if (!operation) return
    const boundaryRevision = handoffBoundaryRevision.current
    const lifetime = handoffLifetime.current
    if (selectedIds.length === 0 || !ownerScope || !lifetime || lifetime.signal.aborted) return

    const numericIds = Array.from(
      new Set(
        selectedIds
          .map((id) => Number(id))
          .filter((id) => Number.isFinite(id) && id > 0)
          .map((id) => Math.trunc(id))
      )
    )

    if (numericIds.length === 0) {
      message.warning(
        t('mediaPage.chatSelectionInvalid', 'Selected items are unavailable for media-scoped chat.')
      )
      return
    }

    const primaryId = String(numericIds[0])

    const payload = {
      mediaId: primaryId,
      mediaIds: numericIds,
      ownerScope,
      mode: 'rag_media' as const
    }

    try {
      const token = await createMediaChatHandoff(payload)
      if (!operation.isCurrent() || boundaryRevision !== handoffBoundaryRevision.current || lifetime.signal.aborted) {
        await removeMediaChatHandoff(token)
        return
      }
      navigate(buildMediaChatHandoffRoute(token))
    } catch {
      if (!operation.isCurrent() || boundaryRevision !== handoffBoundaryRevision.current || lifetime.signal.aborted) return
      message.error(t('mediaPage.chatPrepareFailed', 'Could not prepare this source for Chat. Please try again.'))
      return
    }

    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('tldw:focus-composer'))
      }
    } catch {
      // ignore
    }

    message.success(
      t('mediaPage.chatSelectionOpened', {
        defaultValue: 'Opened media-scoped RAG chat for {{count}} selected items.',
        count: numericIds.length
      })
    )
  }, [captureOperation, message, navigate, ownerScope, selectedIds, t])

  const getSelectedNumericIds = React.useCallback(() => {
    return Array.from(
      new Set(
        selectedIds
          .map((id) => Number(id))
          .filter((id) => Number.isFinite(id) && id > 0)
          .map((id) => Math.trunc(id))
      )
    )
  }, [selectedIds])

  const openTrashFromBatch = React.useCallback(
    (deletedIds: Array<string | number>, operation = captureOperation()) => {
      if (!operation?.isCurrent()) return
      setBatchTrashHandoffIds([])
      navigate(`/media-trash${buildMediaTrashHandoffSearch(deletedIds)}`)
    },
    [captureOperation, navigate, setBatchTrashHandoffIds]
  )

  const handleBatchAddTags = React.useCallback(async () => {
    const operation = captureOperation()
    if (!operation) return

    if (selectedIds.length === 0) {
      message.warning(t("mediaPage.batchRequiresSelection", "Select at least one item."))
      return
    }

    const keywords = parseBatchKeywords(batchKeywordsDraft)
    if (keywords.length === 0) {
      message.warning(t("mediaPage.batchKeywordsMissing", "Enter one or more tags first."))
      return
    }

    const mediaIds = getSelectedNumericIds()
    if (mediaIds.length === 0) {
      message.warning(
        t("mediaPage.batchNoNumericMediaIds", "Selected items are unavailable for this action.")
      )
      return
    }

    setBatchActionLoading("keywords")
    try {
      const result = await tldwClient.bulkUpdateMediaKeywords({
        media_ids: mediaIds,
        keywords,
        mode: "add"
      }, {requestScope: operation.requestScope, signal: operation.signal})
      if (!operation.isCurrent()) return
      const updated = Number(result?.updated ?? 0)
      const failed = Number(result?.failed ?? 0)

      setBatchKeywordsDraft("")
      if (failed > 0) {
        message.warning(
          t("mediaPage.batchKeywordsPartial", "Updated keywords for {{updated}} item(s); {{failed}} failed.", { updated, failed })
        )
        return
      }

      message.success(
        t("mediaPage.batchKeywordsSuccess", "Updated keywords for {{count}} item(s).", {
          count: updated || mediaIds.length
        })
      )
    } catch {
      if (!operation.isCurrent()) return
      message.error(t("mediaPage.batchKeywordsFailed", "Failed to update selected item tags."))
    } finally {
      if (operation.isCurrent()) setBatchActionLoading(null)
    }
  }, [captureOperation, batchKeywordsDraft, getSelectedNumericIds, message, selectedIds.length, t, setBatchActionLoading, setBatchKeywordsDraft])

  const confirmBatchTrash = React.useCallback(async (): Promise<boolean> => {
    const confirmFn = (Modal as any)?.confirm
    if (typeof confirmFn !== "function") return true

    return await new Promise<boolean>((resolve) => {
      confirmFn({
        title: t("mediaPage.batchTrashConfirmTitle", "Move selected items to trash?"),
        content: t("mediaPage.batchTrashConfirmBody", "You can restore them later from trash."),
        okText: t("mediaPage.batchTrashAction", "Move to trash"),
        cancelText: t("mediaPage.cancel", "Cancel"),
        onOk: () => resolve(true),
        onCancel: () => resolve(false)
      })
    })
  }, [t])

  const handleBatchMoveToTrash = React.useCallback(async () => {
    const operation = captureOperation()
    if (!operation) return

    if (selectedIds.length === 0) {
      message.warning(t("mediaPage.batchRequiresSelection", "Select at least one item."))
      return
    }

    const confirmed = await confirmBatchTrash()
    if (!confirmed || !operation.isCurrent()) return

    const idsToDelete = [...selectedIds]
    setBatchActionLoading("trash")
    try {
      const settled = await Promise.allSettled(
        idsToDelete.map((id) => tldwClient.deleteMedia(id, {requestScope: operation.requestScope, signal: operation.signal}))
      )
      if (!operation.isCurrent()) return
      const deletedIds: Array<string | number> = []
      let failedCount = 0
      settled.forEach((entry, index) => {
        if (entry.status === "fulfilled") {
          deletedIds.push(idsToDelete[index])
        } else {
          failedCount += 1
        }
      })

      if (deletedIds.length === 0) {
        setBatchTrashHandoffIds([])
        message.error(t("mediaPage.batchTrashFailed", "Failed to move selected items to trash."))
        return
      }

      setSelectedIds((prev) => {
        const next = prev.filter(
          (candidate) => !deletedIds.some((deletedId) => idsEqual(deletedId, candidate))
        )
        setFocusedId((current) => {
          if (current == null) return next[0] ?? null
          return next.some((candidate) => idsEqual(candidate, current))
            ? current
            : (next[0] ?? null)
        })
        return next
      })
      setDetails((prev) => {
        const next = { ...prev }
        deletedIds.forEach((id) => { delete next[id] })
        return next
      })
      setBatchTrashHandoffIds(deletedIds)

      const toastContent = (
        <span>
          {failedCount > 0
            ? t("mediaPage.batchTrashPartial", "Moved {{deleted}} item(s) to trash; {{failed}} failed.", { deleted: deletedIds.length, failed: failedCount })
            : t("mediaPage.batchTrashSuccess", "Moved {{count}} item(s) to trash.", { count: deletedIds.length })}{" "}
          <Button
            type="link"
            size="small"
            className="!p-0"
            onClick={() => openTrashFromBatch(deletedIds, operation)}
          >
            {t("mediaPage.openTrash", "Open trash")}
          </Button>
        </span>
      )

      if (failedCount > 0) {
        message.warning(toastContent, UNDO_DURATION_SECONDS)
      } else {
        message.success(toastContent, UNDO_DURATION_SECONDS)
      }
    } catch {
      if (!operation.isCurrent()) return
      setBatchTrashHandoffIds([])
      message.error(t("mediaPage.batchTrashFailed", "Failed to move selected items to trash."))
    } finally {
      if (operation.isCurrent()) setBatchActionLoading(null)
    }
  }, [captureOperation, confirmBatchTrash, message, openTrashFromBatch, selectedIds, t, setBatchActionLoading, setSelectedIds, setFocusedId, setDetails, setBatchTrashHandoffIds])

  const handleBatchExport = React.useCallback(async () => {
    const operation = captureOperation()
    if (!operation) return
    if (selectedIds.length === 0) {
      message.warning(t("mediaPage.batchRequiresSelection", "Select at least one item."))
      return
    }

    setBatchActionLoading("export")
    try {
      const currentResults: MediaItem[] = Array.isArray(data) ? data : []
      const exportItems: MediaMultiBatchExportItem[] = []
      for (const id of selectedIds) {
        if (!operation.isCurrent()) return
        const row = selectedMetadata[String(id)] ?? currentResults.find((candidate) => idsEqual(candidate.id, id))
        // Fetch missing export content one at a time; do not populate the reading window.
        const detail = details[id] ?? (await fetchDetail(id, operation))
        if (!operation.isCurrent()) return
        const analysisText = extractMediaDetailAnalysis(detail)
        exportItems.push({
          id,
          title: detail?.title || row?.title || `${t("mediaPage.media", "Media")} ${id}`,
          snippet: row?.snippet || "",
          type: String(detail?.type || row?.type || "") || null,
          created_at: detail?.created_at || row?.created_at || null,
          keywords: Array.isArray((row as any)?.keywords) ? ((row as any).keywords as string[]) : [],
          content: detail ? getContent(detail) : "",
          analysis: analysisText
        })
      }

      const artifact = buildBatchExportArtifact(exportItems, batchExportFormat)
      const blob = new Blob([artifact.content], { type: artifact.mimeType })
      downloadBlob(blob, `media-multi-export-${Date.now()}.${artifact.extension}`)
      message.success(
        t("mediaPage.batchExportReady", "Exported {{count}} selected item(s).", { count: exportItems.length })
      )
    } catch {
      if (!operation.isCurrent()) return
      message.error(t("mediaPage.batchExportFailed", "Failed to export selection."))
    } finally {
      if (operation.isCurrent()) setBatchActionLoading(null)
    }
  }, [captureOperation, fetchDetail, selectedMetadata, batchExportFormat, data, details, message, selectedIds, t, setBatchActionLoading])

  const handleBatchReprocess = React.useCallback(async () => {
    const operation = captureOperation()
    if (!operation) return

    const mediaIds = getSelectedNumericIds()
    if (mediaIds.length === 0) {
      message.warning(
        t("mediaPage.batchNoNumericMediaIds", "Selected items are unavailable for this action.")
      )
      return
    }

    setBatchActionLoading("reprocess")
    try {
      const settled = await Promise.allSettled(
        mediaIds.map((id) =>
          tldwClient.reprocessMedia(id, { perform_chunking: true, generate_embeddings: true }, {requestScope: operation.requestScope, signal: operation.signal})
        )
      )
      if (!operation.isCurrent()) return
      const successCount = settled.filter((entry) => entry.status === "fulfilled").length
      const failedCount = mediaIds.length - successCount

      if (failedCount > 0) {
        message.warning(
          t("mediaPage.batchReprocessPartial", "Queued reprocess for {{success}} item(s); {{failed}} failed.", { success: successCount, failed: failedCount })
        )
        return
      }

      message.success(
        t("mediaPage.batchReprocessSuccess", "Queued reprocess for {{count}} item(s).", { count: successCount })
      )
    } catch {
      if (!operation.isCurrent()) return
      message.error(t("mediaPage.batchReprocessFailed", "Failed to queue reprocess for selected items."))
    } finally {
      if (operation.isCurrent()) setBatchActionLoading(null)
    }
  }, [captureOperation, getSelectedNumericIds, message, t, setBatchActionLoading])

  const expandAllContent = React.useCallback(() => {
    if (!captureOperation()) return
    setContentExpandedIds(new Set(visibleIds.map((id) => String(id))))
  }, [captureOperation, visibleIds, setContentExpandedIds])
  const collapseAllContent = React.useCallback(() => {
    if (!captureOperation()) return
    setContentExpandedIds(new Set())
  }, [captureOperation, setContentExpandedIds])
  const expandAllAnalysis = React.useCallback(() => {
    if (!captureOperation()) return
    setAnalysisExpandedIds(new Set(visibleIds.map((id) => String(id))))
  }, [captureOperation, visibleIds, setAnalysisExpandedIds])
  const collapseAllAnalysis = React.useCallback(() => {
    if (!captureOperation()) return
    setAnalysisExpandedIds(new Set())
  }, [captureOperation, setAnalysisExpandedIds])

  // fetchList function for the query
  const fetchList = React.useCallback(async (): Promise<MediaItem[]> => {
    const operation = captureOperation()
    if (!operation) throw createServicePromptScopeChangedError()
    const hasQuery = query.trim().length > 0
    const hasDateRange = Boolean(dateRange.startDate || dateRange.endDate)
    const shouldUseSearchEndpoint =
      hasQuery || types.length > 0 || keywordTokens.length > 0 || sortBy !== DEFAULT_SORT_BY || hasDateRange

    if (shouldUseSearchEndpoint) {
      const body = buildMediaSearchPayload({
        query,
        mediaTypes: types,
        includeKeywords: keywordTokens,
        excludeKeywords: [],
        sortBy,
        dateRange
      })
      const res = await bgRequest<any>({
        path: `/api/v1/media/search?page=${page}&results_per_page=${pageSize}` as any,
        method: "POST" as any,
        headers: { "Content-Type": "application/json" },
        body,
        abortSignal: operation.signal,
        ...requestScopeFields(operation.requestScope)
      })
      if (!operation.isCurrent()) throw createServicePromptScopeChangedError()
      const items = Array.isArray(res?.items)
        ? res.items
        : Array.isArray(res?.results) ? res.results : []
      const pagination = res?.pagination
      setTotal(Number(pagination?.total_items || items.length || 0))
      const mapped = mapMediaItems(items)
      const typeSet = new Set(availableTypes)
      for (const it of mapped) if (it.type) typeSet.add(it.type)
      setAvailableTypes(Array.from(typeSet))
      return runContentFiltering(mapped, operation)
    }

    const params = new URLSearchParams({
      page: String(page),
      results_per_page: String(pageSize)
    })
    if (sortBy !== DEFAULT_SORT_BY) params.set("sort_by", sortBy)
    if (dateRange.startDate) params.set("start_date", dateRange.startDate)
    if (dateRange.endDate) params.set("end_date", dateRange.endDate)
    const res = await bgRequest<any>({
      path: `/api/v1/media/?${params.toString()}` as any,
      method: "GET" as any,
      abortSignal: operation.signal,
      ...requestScopeFields(operation.requestScope)
    })
    if (!operation.isCurrent()) throw createServicePromptScopeChangedError()
    const items = Array.isArray(res?.items) ? res.items : []
    const pagination = res?.pagination
    setTotal(Number(pagination?.total_items || items.length || 0))
    const mapped = mapMediaItems(items)
    const typeSet = new Set(availableTypes)
    for (const it of mapped) if (it.type) typeSet.add(it.type)
    setAvailableTypes(Array.from(typeSet))
    return runContentFiltering(mapped, operation)
  }, [captureOperation, query, page, pageSize, types, keywordTokens, sortBy, dateRange, availableTypes, mapMediaItems, runContentFiltering, setTotal, setAvailableTypes])

  return {
    startSelectedReview, returnToPreview, changeReadingWindow,
    previewItem,
    toggleSelect,
    ensureDetail,
    retryFetch,
    removeFromSelection,
    clearSelectionWithGuard,
    addVisibleToSelection,
    replaceSelectionWithVisible,
    goRelative,
    scrollToCard,
    runContentFiltering,
    cancelContentFiltering,
    mapMediaItems,
    loadKeywordSuggestions,
    handleBatchAddTags,
    handleBatchMoveToTrash,
    handleBatchExport,
    handleBatchReprocess,
    handleChatAboutSelection,
    expandAllContent,
    collapseAllContent,
    expandAllAnalysis,
    collapseAllAnalysis,
    getSelectedNumericIds,
    openTrashFromBatch,
    confirmBatchTrash,
    handleCompareContent,
    resolveDetailForCompare,
    // Expose fetchList for query binding
    _fetchList: fetchList
  }
}
