import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { bgRequest } from '@/services/background-proxy'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { clearSetting, getSetting, setSetting } from '@/services/settings/registry'
import {
  LAST_MEDIA_ID_SETTING,
} from '@/services/settings/ui-settings'
import {
  buildMediaPermalinkSearch,
  getMediaPermalinkIdFromSearch,
  normalizeMediaPermalinkId
} from '@/components/Review/mediaPermalink'
import type { MediaResultItem } from '@/components/Media/types'
import {
  deriveMediaMeta,
  extractKeywordsFromMedia,
  getErrorStatusCode,
  isMediaEndpointMissingError
} from './useMediaSearch'
import { MEDIA_STALE_CHECK_INTERVAL_MS } from '@/components/Review/ViewMediaPage'
import { shouldReportMediaDetailFetchError } from '@/components/Review/mediaDetailError'
import { extractMediaDetailContent } from '@/utils/media-detail-content'

export interface UseMediaNavigationStateDeps {
  t: (key: string, opts?: Record<string, any>) => string
  message: {
    error: (msg: string) => void
    warning: (msg: string) => void
    success: (msg: string) => void
  }
  displayResults: MediaResultItem[]
  refetch: () => Promise<any>
}

export function useMediaNavigationState(deps: UseMediaNavigationStateDeps) {
  const { t, message, displayResults, refetch } = deps
  const navigate = useNavigate()
  const location = useLocation()

  const [selected, setSelected] = useState<MediaResultItem | null>(null)
  const [pendingInitialMediaId, setPendingInitialMediaId] = useState<string | null>(null)
  const [pendingInitialMediaIdSource, setPendingInitialMediaIdSource] = useState<
    'url' | 'setting' | null
  >(null)
  const [selectedContent, setSelectedContent] = useState<string>('')
  const [selectedDetail, setSelectedDetail] = useState<any>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailFetchError, setDetailFetchError] = useState<{
    mediaId: string | number
    message: string
  } | null>(null)
  const [staleSelectionNotice, setStaleSelectionNotice] = useState<string | null>(null)
  const [lastFetchedId, setLastFetchedId] = useState<string | number | null>(null)
  const pendingDetailRequestRef = useRef<{
    mediaId: string
    promise: Promise<any>
  } | null>(null)

  const permalinkMediaId = useMemo(
    () => getMediaPermalinkIdFromSearch(location.search),
    [location.search]
  )

  const selectedIndex = displayResults.findIndex((r) => r.id === selected?.id)
  const hasPrevious = selectedIndex > 0
  const hasNext = selectedIndex >= 0 && selectedIndex < displayResults.length - 1

  const handlePrevious = useCallback(() => {
    if (hasPrevious) {
      setSelected(displayResults[selectedIndex - 1])
    }
  }, [displayResults, hasPrevious, selectedIndex])

  const handleNext = useCallback(() => {
    if (hasNext) {
      setSelected(displayResults[selectedIndex + 1])
    }
  }, [displayResults, hasNext, selectedIndex])

  const fetchSelectedDetails = useCallback(async (item: MediaResultItem) => {
    if (item.kind === 'media') {
      return bgRequest<any>({
        path: `/api/v1/media/${item.id}` as any,
        method: 'GET' as any
      })
    }
    if (item.kind === 'note') {
      return item.raw
    }
    return null
  }, [])

  const resolveDetailFetchErrorMessage = useCallback((error: unknown): string => {
    const statusCode = getErrorStatusCode(error)
    if (statusCode === 404) {
      return t('review:mediaPage.detailUnavailable', {
        defaultValue: 'This item is no longer available. It may have been deleted.'
      })
    }
    return t('review:mediaPage.detailFetchFailed', {
      defaultValue: 'Unable to load this item. Please try again.'
    })
  }, [t])

  const loadSelectedDetails = useCallback(async (item: MediaResultItem) => {
    setDetailLoading(true)
    setDetailFetchError(null)
    setSelectedContent('')
    setSelectedDetail(null)

    try {
      const detail = await fetchSelectedDetails(item)
      const content = extractMediaDetailContent(detail)
      setSelectedContent(String(content || ''))
      setSelectedDetail(detail)
      setLastFetchedId(item.id)

      const keywords = extractKeywordsFromMedia(detail)
      if (keywords.length > 0 && (!item.keywords || item.keywords.length === 0)) {
        setSelected((prev) => {
          if (!prev || prev.id !== item.id) return prev
          return { ...prev, keywords }
        })
      }

      return true
    } catch (error) {
      if (shouldReportMediaDetailFetchError(error)) {
        console.error('Error fetching media details:', error)
      }
      setSelectedContent('')
      setSelectedDetail(null)
      setDetailFetchError({
        mediaId: item.id,
        message: resolveDetailFetchErrorMessage(error)
      })
      return false
    } finally {
      setDetailLoading(false)
    }
  }, [
    fetchSelectedDetails,
    resolveDetailFetchErrorMessage
  ])
  const loadSelectedDetailsRef = useRef(loadSelectedDetails)

  useEffect(() => {
    loadSelectedDetailsRef.current = loadSelectedDetails
  }, [loadSelectedDetails])

  // Auto-clear stale selection notice
  useEffect(() => {
    if (!staleSelectionNotice) return
    const timer = window.setTimeout(() => {
      setStaleSelectionNotice(null)
    }, 8000)
    return () => {
      window.clearTimeout(timer)
    }
  }, [staleSelectionNotice])

  // Hydrate pending initial media from URL
  useEffect(() => {
    if (!permalinkMediaId) return
    if (selected?.kind === 'media' && selected?.id != null) return
    setPendingInitialMediaId(permalinkMediaId)
    setPendingInitialMediaIdSource('url')
  }, [permalinkMediaId, selected?.id, selected?.kind])

  // Hydrate pending initial media from settings
  useEffect(() => {
    if (permalinkMediaId) return
    if (selected?.kind === 'media' && selected?.id != null) return
    let cancelled = false
    ;(async () => {
      const lastMediaId = normalizeMediaPermalinkId(
        await getSetting(LAST_MEDIA_ID_SETTING)
      )
      if (cancelled || !lastMediaId) return
      setPendingInitialMediaId((prev) => prev ?? lastMediaId)
      setPendingInitialMediaIdSource((prev) => prev ?? 'setting')
    })()
    return () => {
      cancelled = true
    }
  }, [permalinkMediaId, selected?.id, selected?.kind])

  // Resolve pending initial media id
  useEffect(() => {
    if (!pendingInitialMediaId) return

    const pendingId = pendingInitialMediaId
    const pendingSource = pendingInitialMediaIdSource
    const matchingResult = displayResults.find(
      (item) => item.kind === 'media' && String(item.id) === pendingId
    )
    const pendingRequestMatches =
      pendingDetailRequestRef.current?.mediaId === pendingId
    if (matchingResult && !pendingRequestMatches) {
      pendingDetailRequestRef.current = null
      setSelected(matchingResult)
      setPendingInitialMediaId(null)
      setPendingInitialMediaIdSource(null)
      if (pendingSource === 'setting') {
        void clearSetting(LAST_MEDIA_ID_SETTING)
      }
      return
    }

    let cancelled = false
    let pendingRequest = pendingDetailRequestRef.current
    if (!pendingRequest || pendingRequest.mediaId !== pendingId) {
      pendingRequest = {
        mediaId: pendingId,
        promise: bgRequest<any>({
          path: `/api/v1/media/${pendingId}` as any,
          method: 'GET' as any
        })
      }
      pendingDetailRequestRef.current = pendingRequest
    }
    ;(async () => {
      try {
        const detail = await pendingRequest.promise
        if (cancelled) return

        const resolvedId = detail?.id ?? detail?.media_id ?? pendingId
        const hydratedSelection: MediaResultItem = {
          kind: 'media',
          id: resolvedId,
          title:
            detail?.title ||
            detail?.source?.title ||
            matchingResult?.title ||
            detail?.filename ||
            `Media ${resolvedId}`,
          snippet: detail?.snippet || detail?.summary || '',
          keywords: extractKeywordsFromMedia(detail),
          meta: deriveMediaMeta(detail),
          raw: detail
        }

        // Publish the fetched marker first. In extension pages, async state
        // updates are not guaranteed to batch; selecting first can trigger the
        // normal selection loader and turn this into a duplicate conditional
        // GET whose bodyless 304 replaces the usable detail response.
        setLastFetchedId(resolvedId)
        setSelected(hydratedSelection)
        setSelectedContent(String(extractMediaDetailContent(detail) || ''))
        setSelectedDetail(detail)
      } catch (error) {
        console.debug('Failed to hydrate permalink media selection', error)
      } finally {
        if (!cancelled) {
          if (pendingDetailRequestRef.current === pendingRequest) {
            pendingDetailRequestRef.current = null
          }
          setPendingInitialMediaId(null)
          setPendingInitialMediaIdSource(null)
          if (pendingSource === 'setting') {
            void clearSetting(LAST_MEDIA_ID_SETTING)
          }
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [
    displayResults,
    pendingInitialMediaId,
    pendingInitialMediaIdSource
  ])

  // Load selected item content
  useEffect(() => {
    const currentSelection = selected
    if (!currentSelection) {
      setSelectedContent('')
      setSelectedDetail(null)
      setLastFetchedId(null)
      setDetailFetchError(null)
      setDetailLoading(false)
      return
    }

    if (currentSelection.id === lastFetchedId) {
      return
    }

    void loadSelectedDetailsRef.current(currentSelection)
  }, [lastFetchedId, selected?.id])

  // Stale selection reconciliation
  useEffect(() => {
    if (!selected || selected.kind !== 'media') return
    if (detailLoading) return

    let cancelled = false
    let inFlight = false
    const selectedId = String(selected.id)
    const selectedValue = selected.id

    const reconcileStaleSelection = async () => {
      if (inFlight || cancelled) return
      inFlight = true
      try {
        await bgRequest<any>({
          path: `/api/v1/media/${selectedId}` as any,
          method: 'GET' as any
        })
      } catch (error) {
        const statusCode = getErrorStatusCode(error)
        if (statusCode !== 404 && statusCode !== 410) {
          return
        }
        if (cancelled) return

        const staleMessage = t('review:mediaPage.staleSelectionRecovered', {
          defaultValue:
            'The selected item is no longer available. Your selection was updated.'
        })
        setStaleSelectionNotice(staleMessage)
        message.warning(staleMessage)

        const currentIndex = displayResults.findIndex(
          (item) => String(item.id) === selectedId
        )
        const refreshed = await refetch()
        const refreshedResults = Array.isArray(refreshed?.data)
          ? (refreshed.data as MediaResultItem[])
          : []
        const remaining = refreshedResults.filter(
          (item) => String(item.id) !== selectedId
        )

        if (remaining.length > 0) {
          const nextIndex =
            currentIndex >= 0
              ? Math.min(currentIndex, remaining.length - 1)
              : 0
          const replacement = remaining[nextIndex]
          setLastFetchedId(null)
          setSelected(replacement)
          setDetailFetchError({
            mediaId: selectedValue,
            message: t('review:mediaPage.detailUnavailable', {
              defaultValue: 'This item is no longer available. It may have been deleted.'
            })
          })
          return
        }

        setSelected(null)
        setSelectedContent('')
        setSelectedDetail(null)
        setLastFetchedId(null)
        setDetailFetchError({
          mediaId: selectedValue,
          message: t('review:mediaPage.detailUnavailable', {
            defaultValue: 'This item is no longer available. It may have been deleted.'
          })
        })
      } finally {
        inFlight = false
      }
    }

    const interval = window.setInterval(() => {
      void reconcileStaleSelection()
    }, MEDIA_STALE_CHECK_INTERVAL_MS)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [
    detailLoading,
    displayResults,
    message,
    refetch,
    selected?.id,
    selected?.kind,
    t
  ])

  // Persist selected media permalink
  const selectedMediaPermalinkId =
    selected?.kind === 'media' && selected?.id != null ? String(selected.id) : null

  useEffect(() => {
    if (!selectedMediaPermalinkId) return
    void setSetting(LAST_MEDIA_ID_SETTING, selectedMediaPermalinkId)
  }, [selectedMediaPermalinkId])

  useEffect(() => {
    if (
      selectedMediaPermalinkId == null &&
      pendingInitialMediaIdSource === 'url' &&
      pendingInitialMediaId
    ) {
      return
    }
    const nextSearch = buildMediaPermalinkSearch(
      location.search,
      selectedMediaPermalinkId
    )
    if (nextSearch === location.search) return
    navigate(
      {
        pathname: location.pathname,
        search: nextSearch,
        hash: location.hash
      },
      { replace: true }
    )
  }, [
    location.hash,
    location.pathname,
    location.search,
    navigate,
    pendingInitialMediaId,
    pendingInitialMediaIdSource,
    selectedMediaPermalinkId
  ])

  const handleRetryDetailFetch = useCallback(() => {
    if (!selected) return
    void loadSelectedDetails(selected)
  }, [loadSelectedDetails, selected])

  const handleRefreshMedia = useCallback(async (
    showNavigationPanel: boolean,
    refetchNavigation: () => void
  ) => {
    if (!selected) return
    const refreshed = await loadSelectedDetails(selected)
    if (refreshed && showNavigationPanel) {
      void refetchNavigation()
    }
  }, [loadSelectedDetails, selected])

  return {
    selected, setSelected,
    selectedContent, setSelectedContent,
    selectedDetail, setSelectedDetail,
    detailLoading,
    detailFetchError,
    staleSelectionNotice,
    lastFetchedId, setLastFetchedId,
    selectedIndex,
    hasPrevious,
    hasNext,
    handlePrevious,
    handleNext,
    loadSelectedDetails,
    handleRetryDetailFetch,
    handleRefreshMedia,
    selectedMediaPermalinkId,
    pendingInitialMediaIdSource,
    pendingInitialMediaId,
  }
}
