import React from "react"
import { Button, Modal } from "antd"
import { bgRequest } from "@/services/background-proxy"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { getNoteKeywords, searchNoteKeywords } from "@/services/note-keywords"
import { setSetting, clearSetting } from "@/services/settings/registry"
import {
  DISCUSS_MEDIA_PROMPT_SETTING,
  LAST_MEDIA_ID_SETTING
} from "@/services/settings/ui-settings"
import { rankKeywordSuggestions } from "@/components/Review/filter-chip-priority"
import { IDLE_CONTENT_FILTER_PROGRESS } from "@/components/Review/content-filtering-progress"
import {
  buildBatchExportArtifact,
  parseBatchKeywords,
  type MediaMultiBatchExportItem
} from "@/components/Review/media-multi-batch-actions"
import { buildMediaTrashHandoffSearch } from "@/components/Review/mediaPermalink"
import { buildMediaSearchPayload } from "@/components/Review/mediaSearchRequest"
import { downloadBlob } from "@/utils/download-blob"
import { extractMediaDetailAnalysis } from "@/utils/media-detail-content"
import {
  type MediaItem,
  type MediaDetail,
  type MediaReviewState,
  type MediaReviewActions,
  getContent,
  idsEqual,
  includesId,
  getErrorStatusCode,
  DEFAULT_SORT_BY,
  UNDO_DURATION_SECONDS
} from "@/components/Review/media-review-types"

export function useMediaReviewActions(s: MediaReviewState): MediaReviewActions & { _fetchList: () => Promise<MediaItem[]> } {
  const {
    t, navigate, message,
    query, page, pageSize, types, keywordTokens, includeContent, sortBy, dateRange,
    selectedIds, setSelectedIds, focusedId, setFocusedId,
    previewedId, previewIndex, setPreviewedId,
    details, setDetails, setTotal, setAvailableTypes, availableTypes,
    setContentLoading, setContentFilterProgress, contentFilterRunRef,
    setDetailLoading, setFailedIds, detailLoading,
    openAllLimit, allResults, focusIndex, viewerItems,
    viewMode, viewerVirtualizer,
    batchKeywordsDraft, setBatchKeywordsDraft,
    batchExportFormat, setBatchActionLoading, setBatchTrashHandoffIds,
    setCompareLeftText, setCompareRightText,
    setCompareLeftLabel, setCompareRightLabel, setCompareDiffOpen,
    visibleIds,
    setContentExpandedIds, setAnalysisExpandedIds,
    searchInputRef, viewerRef, listParentRef,
    lastClickedRef, pendingRestoreFocusIdRef, ensureDetailRef,
    cardRefs, prefersReducedMotion,
    setChatMode, setSelectedKnowledge, setRagMediaIds,
    setKeywordOptions,
    data, refetch,
    setFocusedId: _setFocusedId,
    pendingInitialMediaId, setPendingInitialMediaId,
    isOnline
  } = s

  const cancelContentFiltering = React.useCallback(() => {
    contentFilterRunRef.current += 1
    setContentLoading(false)
    setContentFilterProgress(IDLE_CONTENT_FILTER_PROGRESS)
  }, [contentFilterRunRef, setContentLoading, setContentFilterProgress])

  const runContentFiltering = React.useCallback(async (items: MediaItem[]): Promise<MediaItem[]> => {
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
      if (contentFilterRunRef.current !== runId) return []
      const m = items[idx]
      let d = details[m.id]
      if (!d) {
        try {
          d = await bgRequest<MediaDetail>({
            path: `/api/v1/media/${m.id}?include_content=true&include_versions=false` as any,
            method: 'GET' as any
          })
          if (contentFilterRunRef.current !== runId) return []
          setDetails((prev) => (prev[m.id] ? prev : { ...prev, [m.id]: d! }))
        } catch {
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

    if (contentFilterRunRef.current !== runId) return []

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
  }, [details, includeContent, keywordTokens, query, contentFilterRunRef, setContentLoading, setContentFilterProgress, setDetails])

  const mapMediaItems = React.useCallback((items: any[]): MediaItem[] => (
    items.map((m: any) => ({
      id: m?.id ?? m?.media_id ?? m?.pk ?? m?.uuid,
      title: m?.title || m?.filename || `Media ${m?.id}`,
      snippet: m?.snippet || m?.summary || "",
      type: String(m?.type || m?.media_type || "").toLowerCase(),
      created_at: m?.created_at
    }))
  ), [])

  const ensureDetail = React.useCallback(async (id: string | number, isRetry = false) => {
    if (details[id] || detailLoading[id]) return
    setDetailLoading((prev) => ({ ...prev, [id]: true }))
    if (isRetry) {
      setFailedIds((prev) => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
    }
    try {
      const d = await bgRequest<MediaDetail>({
        path: `/api/v1/media/${id}?include_content=true&include_versions=false` as any,
        method: 'GET' as any
      })
      const base = Array.isArray(data) ? (data as MediaItem[]).find((x) => idsEqual(x.id, id)) : undefined
      const enriched = { ...d, id, title: (d as any)?.title ?? base?.title, type: (d as any)?.type ?? base?.type, created_at: (d as any)?.created_at ?? base?.created_at } as any
      setDetails((prev) => ({ ...prev, [id]: enriched }))
      setFailedIds((prev) => {
        const next = new Set(prev)
        next.delete(id)
        return next
      })
    } catch (error) {
      setFailedIds((prev) => new Set(prev).add(id))
      const statusCode = getErrorStatusCode(error)
      if (statusCode === 404 || statusCode === 410) {
        setSelectedIds((prev) => {
          const next = prev.filter((candidateId) => String(candidateId) !== String(id))
          return next.length === prev.length ? prev : next
        })
        setFocusedId((prev) => (prev != null && String(prev) === String(id) ? null : prev))
      }
    } finally {
      setDetailLoading((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
    }
  }, [data, detailLoading, details, setDetailLoading, setDetails, setFailedIds, setSelectedIds, setFocusedId])

  // Keep ref in sync
  React.useEffect(() => {
    ensureDetailRef.current = ensureDetail
  }, [ensureDetail, ensureDetailRef])

  const retryFetch = React.useCallback((id: string | number) => {
    setDetails((prev) => {
      const next = { ...prev }
      delete next[id]
      return next
    })
    void ensureDetail(id, true)
  }, [ensureDetail, setDetails])

  const clearSelectionWithGuard = React.useCallback(() => {
    if (selectedIds.length === 0) return
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
        {t('mediaPage.selectionClearedCount', 'Selection cleared ({{count}} items).', { count: selectionToRestore.length })}
        {' '}
        <Button
          type="link"
          size="small"
          className="!p-0"
          onClick={() => {
            setSelectedIds(selectionToRestore)
            const restoredFocusId = focusToRestore ?? selectionToRestore[0] ?? null
            setFocusedId(restoredFocusId)
            pendingRestoreFocusIdRef.current = restoreFocusId
            window.setTimeout(() => {
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
  }, [selectedIds, focusedId, t, message, setSelectedIds, setFocusedId, listParentRef, pendingRestoreFocusIdRef])

  const focusViewerSoon = React.useCallback(() => {
    if (typeof window === "undefined") {
      viewerRef.current?.focus()
      return
    }

    // Browsers can re-focus the clicked result row after the click handler runs.
    window.setTimeout(() => {
      viewerRef.current?.focus()
    }, 0)
  }, [viewerRef])

  const previewItem = React.useCallback((id: string | number) => {
    setPreviewedId(id)
    setFocusedId(id)
    focusViewerSoon()
    void ensureDetail(id)
  }, [setPreviewedId, setFocusedId, focusViewerSoon, ensureDetail])

  const toggleSelect = React.useCallback(async (id: string | number, event?: React.MouseEvent) => {
    if (event?.shiftKey && lastClickedRef.current != null && Array.isArray(data)) {
      const lastIdx = data.findIndex(r => idsEqual(r.id, lastClickedRef.current!))
      const currIdx = data.findIndex(r => idsEqual(r.id, id))
      if (lastIdx !== -1 && currIdx !== -1) {
        const [start, end] = lastIdx < currIdx ? [lastIdx, currIdx] : [currIdx, lastIdx]
        const rangeIds = data.slice(start, end + 1).map(r => r.id)
        const nextSelection = [...selectedIds]
        const remaining = openAllLimit - nextSelection.length
        if (remaining <= 0) {
          message.warning(
            t('mediaPage.selectionLimitReached', {
              defaultValue: 'Selection limit reached ({{limit}} items)',
              limit: openAllLimit
            })
          )
          return
        }
        const newIds = rangeIds.filter((rid) => !includesId(nextSelection, rid))
        let toAdd = newIds
        if (newIds.length > remaining) {
          message.warning(
            t('mediaPage.selectionLimitReached', {
              defaultValue: 'Selection limit reached ({{limit}} items)',
              limit: openAllLimit
            })
          )
          toAdd = newIds.slice(newIds.length - remaining)
        }
        toAdd.forEach((rid) => nextSelection.push(rid))
        setSelectedIds(nextSelection)
        toAdd.forEach((rid) => void ensureDetail(rid))
        setFocusedId(id)
        lastClickedRef.current = id
        setTimeout(() => viewerRef.current?.focus(), 100)
        return
      }
    }

    if (!includesId(selectedIds, id) && selectedIds.length >= openAllLimit) {
      message.warning(
        t('mediaPage.selectionLimitReached', {
          defaultValue: 'Selection limit reached ({{limit}} items)',
          limit: openAllLimit
        })
      )
      return
    }

    lastClickedRef.current = id
    setSelectedIds((prev) => {
      const exists = includesId(prev, id)
      const next = exists ? prev.filter((x) => !idsEqual(x, id)) : [...prev, id]
      if (!exists && viewerRef.current) {
        setTimeout(() => viewerRef.current?.focus(), 100)
      }
      return next
    })
    setFocusedId(id)
    void ensureDetail(id)
  }, [data, selectedIds, openAllLimit, ensureDetail, message, t, setSelectedIds, setFocusedId, lastClickedRef, viewerRef])

  // Auto-fetch details for selected items
  React.useEffect(() => {
    selectedIds.forEach((id) => {
      void ensureDetailRef.current(id)
    })
  }, [selectedIds, ensureDetailRef])

  const addVisibleToSelection = React.useCallback(() => {
    if (allResults.length === 0) return
    if (selectedIds.length >= openAllLimit) {
      message.warning(
        t('mediaPage.selectionLimitReached', {
          defaultValue: 'Selection limit reached ({{limit}} items)',
          limit: openAllLimit
        })
      )
      return
    }

    const visibleSlice = allResults.slice(0, Math.min(allResults.length, openAllLimit))
    const next = [...selectedIds]
    const newlyAdded: Array<string | number> = []

    for (const item of visibleSlice) {
      if (includesId(next, item.id)) continue
      if (next.length >= openAllLimit) break
      next.push(item.id)
      newlyAdded.push(item.id)
    }

    setSelectedIds(next)
    newlyAdded.forEach((id) => void ensureDetail(id))
    if (newlyAdded.length > 0 && focusedId == null) {
      setFocusedId(next[0] ?? null)
    }

    if (allResults.length > openAllLimit || next.length >= openAllLimit) {
      message.info(
        t("mediaPage.openAllCapped", {
          defaultValue: "Showing first {{count}} items to keep things smooth",
          count: openAllLimit
        })
      )
    }
  }, [allResults, ensureDetail, openAllLimit, t, selectedIds, message, focusedId, setSelectedIds, setFocusedId])

  const replaceSelectionWithVisible = React.useCallback(() => {
    if (allResults.length === 0) return
    const previousSelection = [...selectedIds]
    const previousFocus = focusedId
    const visibleSlice = allResults.slice(0, Math.min(allResults.length, openAllLimit))
    const nextIds = visibleSlice.map((m) => m.id)

    setSelectedIds(nextIds)
    nextIds.forEach((id) => void ensureDetail(id))
    setFocusedId(nextIds[0] ?? null)

    message.info(
      <span>
        {t('mediaPage.selectionReplaced', 'Selection replaced with current visible items.')}
        {' '}
        <Button
          type="link"
          size="small"
          className="!p-0"
          onClick={() => {
            setSelectedIds(previousSelection)
            setFocusedId(previousFocus ?? previousSelection[0] ?? null)
            message.success(t('mediaPage.selectionRestored', 'Selection restored'))
          }}
        >
          {t('mediaPage.undo', 'Undo')}
        </Button>
      </span>,
      UNDO_DURATION_SECONDS
    )
  }, [allResults, selectedIds, focusedId, openAllLimit, ensureDetail, message, t, setSelectedIds, setFocusedId])

  const removeFromSelection = React.useCallback((id: string | number) => {
    setSelectedIds((prev) => {
      const next = prev.filter((candidate) => !idsEqual(candidate, id))
      if (next.length !== prev.length) {
        setFocusedId((current) => {
          if (current == null) return next[0] ?? null
          return idsEqual(current, id) ? next[0] ?? null : current
        })
      }
      return next
    })
  }, [setSelectedIds, setFocusedId])

  const scrollToCard = React.useCallback(
    (id: string | number) => {
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
    [viewMode, viewerItems, viewerVirtualizer, prefersReducedMotion, cardRefs]
  )

  const goRelative = React.useCallback(
    (delta: number) => {
      if (allResults.length === 0) return
      const currentIdx =
        selectedIds.length === 0 && previewedId != null && previewIndex >= 0
          ? previewIndex
          : focusIndex >= 0
            ? focusIndex
            : 0
      let next = currentIdx + delta
      if (next < 0) next = 0
      if (next >= allResults.length) next = allResults.length - 1
      const nextId = allResults[next]?.id
      if (nextId != null) {
        setFocusedId(nextId)
        setPreviewedId(nextId)
        void ensureDetail(nextId)
      }
    },
    [allResults, ensureDetail, focusIndex, previewIndex, previewedId, selectedIds.length, setFocusedId, setPreviewedId]
  )

  // Pending initial media restoration
  React.useEffect(() => {
    if (!pendingInitialMediaId) return
    if (!Array.isArray(allResults) || allResults.length === 0) return
    const match = allResults.find((m) => String(m.id) === pendingInitialMediaId)
    if (!match) return
    setSelectedIds([match.id])
    setFocusedId(match.id)
    void ensureDetail(match.id)
    scrollToCard(match.id)
    setPendingInitialMediaId(null)
    void clearSetting(LAST_MEDIA_ID_SETTING)
  }, [pendingInitialMediaId, allResults, ensureDetail, scrollToCard, setSelectedIds, setFocusedId, setPendingInitialMediaId])

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

  const resolveDetailForCompare = React.useCallback(async (id: string | number): Promise<MediaDetail | null> => {
    const existing = details[id]
    if (existing) return existing
    try {
      const fetched = await bgRequest<MediaDetail>({
        path: `/api/v1/media/${id}?include_content=true&include_versions=false` as any,
        method: 'GET' as any
      })
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
  }, [allResults, details, setDetails])

  const handleCompareContent = React.useCallback(async () => {
    if (selectedIds.length !== 2) return
    const [leftId, rightId] = selectedIds
    const leftDetail = await resolveDetailForCompare(leftId)
    const rightDetail = await resolveDetailForCompare(rightId)

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
  }, [message, resolveDetailForCompare, selectedIds, t, setCompareLeftText, setCompareRightText, setCompareLeftLabel, setCompareRightLabel, setCompareDiffOpen])

  const handleChatAboutSelection = React.useCallback(() => {
    if (selectedIds.length === 0) return

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
    setSelectedKnowledge(null as any)
    setRagMediaIds(numericIds)
    setChatMode('rag')

    const payload = {
      mediaId: primaryId,
      mode: 'rag_media' as const
    }

    try {
      void setSetting(DISCUSS_MEDIA_PROMPT_SETTING, payload)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('tldw:discuss-media', {
            detail: { ...payload, mediaIds: numericIds }
          })
        )
      }
    } catch {
      // ignore storage/event errors
    }

    navigate('/')
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
  }, [message, navigate, selectedIds, setChatMode, setRagMediaIds, setSelectedKnowledge, t])

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
    (deletedIds: Array<string | number>) => {
      setBatchTrashHandoffIds([])
      navigate(`/media-trash${buildMediaTrashHandoffSearch(deletedIds)}`)
    },
    [navigate, setBatchTrashHandoffIds]
  )

  const handleBatchAddTags = React.useCallback(async () => {
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
      })
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
      message.error(t("mediaPage.batchKeywordsFailed", "Failed to update selected item tags."))
    } finally {
      setBatchActionLoading(null)
    }
  }, [batchKeywordsDraft, getSelectedNumericIds, message, selectedIds.length, t, setBatchActionLoading, setBatchKeywordsDraft])

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
    if (selectedIds.length === 0) {
      message.warning(t("mediaPage.batchRequiresSelection", "Select at least one item."))
      return
    }

    const confirmed = await confirmBatchTrash()
    if (!confirmed) return

    const idsToDelete = [...selectedIds]
    setBatchActionLoading("trash")
    try {
      const settled = await Promise.allSettled(
        idsToDelete.map((id) => tldwClient.deleteMedia(id))
      )
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
            : next[0] ?? null
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
            : t("mediaPage.batchTrashSuccess", "Moved {{count}} item(s) to trash.", { count: deletedIds.length })}
          {" "}
          <Button
            type="link"
            size="small"
            className="!p-0"
            onClick={() => openTrashFromBatch(deletedIds)}
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
      setBatchTrashHandoffIds([])
      message.error(t("mediaPage.batchTrashFailed", "Failed to move selected items to trash."))
    } finally {
      setBatchActionLoading(null)
    }
  }, [confirmBatchTrash, message, openTrashFromBatch, selectedIds, t, setBatchActionLoading, setSelectedIds, setFocusedId, setDetails, setBatchTrashHandoffIds])

  const handleBatchExport = React.useCallback(() => {
    if (selectedIds.length === 0) {
      message.warning(t("mediaPage.batchRequiresSelection", "Select at least one item."))
      return
    }

    setBatchActionLoading("export")
    try {
      const currentResults: MediaItem[] = Array.isArray(data) ? data : []
      const exportItems: MediaMultiBatchExportItem[] = selectedIds.map((id) => {
        const row = currentResults.find((candidate) => idsEqual(candidate.id, id))
        const detail = details[id]
        const analysisText = extractMediaDetailAnalysis(detail)
        return {
          id,
          title: detail?.title || row?.title || `${t("mediaPage.media", "Media")} ${id}`,
          snippet: row?.snippet || "",
          type: String(detail?.type || row?.type || "") || null,
          created_at: detail?.created_at || row?.created_at || null,
          keywords: Array.isArray((row as any)?.keywords) ? ((row as any).keywords as string[]) : [],
          content: detail ? getContent(detail) : "",
          analysis: analysisText
        }
      })

      const artifact = buildBatchExportArtifact(exportItems, batchExportFormat)
      const blob = new Blob([artifact.content], { type: artifact.mimeType })
      downloadBlob(blob, `media-multi-export-${Date.now()}.${artifact.extension}`)
      message.success(
        t("mediaPage.batchExportReady", "Exported {{count}} selected item(s).", { count: exportItems.length })
      )
    } catch {
      message.error(t("mediaPage.batchExportFailed", "Failed to export selection."))
    } finally {
      setBatchActionLoading(null)
    }
  }, [batchExportFormat, data, details, message, selectedIds, t, setBatchActionLoading])

  const handleBatchReprocess = React.useCallback(async () => {
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
          tldwClient.reprocessMedia(id, { perform_chunking: true, generate_embeddings: true })
        )
      )
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
      message.error(t("mediaPage.batchReprocessFailed", "Failed to queue reprocess for selected items."))
    } finally {
      setBatchActionLoading(null)
    }
  }, [getSelectedNumericIds, message, t, setBatchActionLoading])

  const expandAllContent = React.useCallback(() => {
    setContentExpandedIds(new Set(visibleIds.map((id) => String(id))))
  }, [visibleIds, setContentExpandedIds])
  const collapseAllContent = React.useCallback(() => setContentExpandedIds(new Set()), [setContentExpandedIds])
  const expandAllAnalysis = React.useCallback(() => {
    setAnalysisExpandedIds(new Set(visibleIds.map((id) => String(id))))
  }, [visibleIds, setAnalysisExpandedIds])
  const collapseAllAnalysis = React.useCallback(() => setAnalysisExpandedIds(new Set()), [setAnalysisExpandedIds])

  // fetchList function for the query
  const fetchList = React.useCallback(async (): Promise<MediaItem[]> => {
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
        body
      })
      const items = Array.isArray(res?.items) ? res.items : (Array.isArray(res?.results) ? res.results : [])
      const pagination = res?.pagination
      setTotal(Number(pagination?.total_items || items.length || 0))
      const mapped = mapMediaItems(items)
      const typeSet = new Set(availableTypes)
      for (const it of mapped) if (it.type) typeSet.add(it.type)
      setAvailableTypes(Array.from(typeSet))
      return runContentFiltering(mapped)
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
      method: "GET" as any
    })
    const items = Array.isArray(res?.items) ? res.items : []
    const pagination = res?.pagination
    setTotal(Number(pagination?.total_items || items.length || 0))
    const mapped = mapMediaItems(items)
    const typeSet = new Set(availableTypes)
    for (const it of mapped) if (it.type) typeSet.add(it.type)
    setAvailableTypes(Array.from(typeSet))
    return runContentFiltering(mapped)
  }, [query, page, pageSize, types, keywordTokens, sortBy, dateRange, availableTypes, mapMediaItems, runContentFiltering, setTotal, setAvailableTypes])

  return {
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
