import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { MutableRefObject } from 'react'
import { tldwClient } from '@/services/tldw/TldwApiClient'

interface UseMediaReadingProgressArgs {
  mediaId: string | number | null
  mediaKind: 'media' | 'note' | null
  mediaDetail?: any
  contentLength: number
  scrollContainerRef: MutableRefObject<HTMLDivElement | null>
  hasNavigationTarget?: boolean
  debounceMs?: number
}

interface ReadingProgressPayload {
  current_page: number
  total_pages: number
  zoom_level?: number
  view_mode?: 'single' | 'continuous' | 'thumbnails'
  cfi?: string
  percentage?: number
}

interface ReadingProgressResponse {
  media_id: number
  has_progress?: boolean
  current_page?: number
  total_pages?: number
  zoom_level?: number
  view_mode?: 'single' | 'continuous' | 'thumbnails'
  percent_complete?: number
  cfi?: string
  last_read_at?: string
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

const deriveTotalPages = (mediaDetail: any, contentLength: number): number => {
  const candidates = [
    mediaDetail?.metadata?.page_count,
    mediaDetail?.metadata?.num_pages,
    mediaDetail?.metadata?.total_pages,
    mediaDetail?.content?.page_count,
    mediaDetail?.content?.total_pages,
    mediaDetail?.page_count,
    mediaDetail?.total_pages
  ]
  for (const candidate of candidates) {
    const asNumber = toFiniteNumber(candidate)
    if (asNumber && asNumber > 0) return Math.max(1, Math.trunc(asNumber))
  }
  if (contentLength > 0) {
    return Math.max(1, Math.ceil(contentLength / 5000))
  }
  return 1
}

const computeReadingProgress = (
  container: HTMLDivElement,
  totalPages: number
): ReadingProgressPayload => {
  const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight)
  const percentage =
    maxScroll <= 0 ? 0 : clamp((container.scrollTop / maxScroll) * 100, 0, 100)
  const currentPage = clamp(
    Math.floor((percentage / 100) * Math.max(0, totalPages - 1)) + 1,
    1,
    Math.max(1, totalPages)
  )
  const percentageRounded = Number(percentage.toFixed(2))

  return {
    current_page: currentPage,
    total_pages: Math.max(1, totalPages),
    zoom_level: 100,
    view_mode: 'continuous',
    percentage: percentageRounded,
    cfi: `scroll:${percentageRounded}`
  }
}

const buildProgressSignature = (payload: ReadingProgressPayload): string => {
  return [
    payload.current_page,
    payload.total_pages,
    payload.zoom_level ?? 100,
    payload.view_mode ?? 'continuous',
    payload.cfi ?? '',
    payload.percentage ?? 0
  ].join(':')
}

const applyProgressToScroll = (
  container: HTMLDivElement,
  percentage: number
): boolean => {
  const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight)
  if (maxScroll <= 0) return false
  container.scrollTop = Math.round(maxScroll * clamp(percentage / 100, 0, 1))
  return true
}

const parseScrollPercentageFromCfi = (cfi: unknown): number | null => {
  if (typeof cfi !== 'string') return null
  const trimmed = cfi.trim().toLowerCase()
  if (!trimmed.startsWith('scroll:')) return null
  const parsed = Number(trimmed.slice('scroll:'.length))
  if (!Number.isFinite(parsed)) return null
  return clamp(parsed, 0, 100)
}

export function useMediaReadingProgress({
  mediaId,
  mediaKind,
  mediaDetail,
  contentLength,
  scrollContainerRef,
  hasNavigationTarget = false,
  debounceMs = 900
}: UseMediaReadingProgressArgs) {
  const mediaIdKey =
    mediaKind === 'media' && mediaId != null ? String(mediaId) : null
  const totalPages = useMemo(
    () => deriveTotalPages(mediaDetail, contentLength),
    [contentLength, mediaDetail]
  )

  const lastSavedSignatureRef = useRef<string>('')
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const restoreTokenRef = useRef(0)
  const lastMediaIdKeyRef = useRef<string | null>(null)
  const [progressPercent, setProgressPercent] = useState<number | null>(null)

  const saveProgress = useCallback(
    async () => {
      if (!mediaIdKey) return
      const container = scrollContainerRef.current
      if (!container) return

      const payload = computeReadingProgress(container, totalPages)
      setProgressPercent(payload.percentage ?? 0)
      const signature = buildProgressSignature(payload)
      if (signature === lastSavedSignatureRef.current) {
        return
      }

      try {
        await tldwClient.updateReadingProgress(mediaIdKey, payload)
        lastSavedSignatureRef.current = signature
      } catch (error) {
        // Preserve read flow; retries happen naturally on subsequent scrolls.
        console.debug('Failed to persist media reading progress', error)
      }
    },
    [mediaIdKey, scrollContainerRef, totalPages]
  )

  const clearProgress = useCallback(async () => {
    if (!mediaIdKey) return
    try {
      await tldwClient.deleteReadingProgress(mediaIdKey)
      lastSavedSignatureRef.current = ''
      setProgressPercent(0)
    } catch (error) {
      console.debug('Failed to clear media reading progress', error)
    }
  }, [mediaIdKey])

  useEffect(() => {
    if (!mediaIdKey || hasNavigationTarget) return
    restoreTokenRef.current += 1
    const restoreToken = restoreTokenRef.current
    let cancelled = false

    const restore = async () => {
      try {
        const progress = (await tldwClient.getReadingProgress(
          mediaIdKey
        )) as ReadingProgressResponse
        if (cancelled || restoreToken !== restoreTokenRef.current) return
        if (progress.has_progress === false) {
          lastSavedSignatureRef.current = ''
          setProgressPercent(0)
          return
        }

        const percentFromServer = toFiniteNumber(progress.percent_complete)
        const percentage =
          percentFromServer != null
            ? clamp(percentFromServer, 0, 100)
            : parseScrollPercentageFromCfi(progress.cfi)

        if (percentage == null) return
        setProgressPercent(Number(percentage.toFixed(2)))

        let attempts = 0
        const maxAttempts = 12
        const tryApply = () => {
          if (cancelled || restoreToken !== restoreTokenRef.current) return
          const container = scrollContainerRef.current
          if (!container) return
          const applied = applyProgressToScroll(container, percentage)
          if (applied || attempts >= maxAttempts) {
            const payload = computeReadingProgress(container, totalPages)
            lastSavedSignatureRef.current = buildProgressSignature(payload)
            return
          }
          attempts += 1
          requestAnimationFrame(tryApply)
        }
        requestAnimationFrame(tryApply)
      } catch (error) {
        console.debug('Failed to restore media reading progress', error)
      }
    }

    void restore()
    return () => {
      cancelled = true
    }
  }, [hasNavigationTarget, mediaIdKey, scrollContainerRef, totalPages])

  useEffect(() => {
    if (lastMediaIdKeyRef.current !== mediaIdKey) {
      lastSavedSignatureRef.current = ''
      lastMediaIdKeyRef.current = mediaIdKey
      setProgressPercent(mediaIdKey ? 0 : null)
    }
  }, [mediaIdKey])

  useEffect(() => {
    if (!mediaIdKey) return
    const container = scrollContainerRef.current
    if (!container) return

    const onScroll = () => {
      const payload = computeReadingProgress(container, totalPages)
      setProgressPercent(payload.percentage ?? 0)
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
      }
      saveTimerRef.current = setTimeout(() => {
        void saveProgress()
      }, debounceMs)
    }

    container.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      container.removeEventListener('scroll', onScroll)
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
        saveTimerRef.current = null
      }
      void saveProgress()
    }
  }, [debounceMs, mediaIdKey, saveProgress, scrollContainerRef, totalPages])

  return {
    saveProgress,
    clearProgress,
    progressPercent
  }
}
