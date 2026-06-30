import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import {
  describeMediaNavigationTarget,
  type MediaNavigationTargetLike
} from '@/utils/media-navigation-target'
import { applyMediaNavigationTarget } from '@/utils/media-navigation-target-actions'
import { useMediaReadingProgress } from '@/hooks/useMediaReadingProgress'
import { getDesignSystemState } from '@/design-system'
import type { MediaResultItem } from '../types'

const normalizeComparableText = (value: string): string =>
  value.replace(/\s+/g, ' ').trim().toLowerCase()

const findHeadingMatchInContent = (
  root: HTMLElement | null,
  title: string,
  options?: { preferLast?: boolean }
): HTMLElement | null => {
  if (!root) return null
  const normalizedTitle = normalizeComparableText(title)
  if (!normalizedTitle) return null
  const preferLast = Boolean(options?.preferLast)

  const headings = Array.from(root.querySelectorAll('h1,h2,h3,h4,h5,h6'))
  const headingCandidates = preferLast ? [...headings].reverse() : headings
  for (const candidate of headingCandidates) {
    if (!(candidate instanceof HTMLElement)) continue
    const text = normalizeComparableText(candidate.textContent || '')
    if (!text) continue
    if (text === normalizedTitle || text.includes(normalizedTitle)) {
      return candidate
    }
  }

  const allBroaderCandidates = Array.from(
    root.querySelectorAll('p,li,blockquote,div,span')
  )
  const broaderCandidates = preferLast
    ? allBroaderCandidates.slice(-1200).reverse()
    : allBroaderCandidates.slice(0, 1200)
  for (const candidate of broaderCandidates) {
    if (!(candidate instanceof HTMLElement)) continue
    const text = normalizeComparableText(candidate.textContent || '')
    if (!text) continue
    if (text === normalizedTitle || text.includes(normalizedTitle)) {
      return candidate
    }
  }

  return null
}

const focusNavigationMatch = (element: HTMLElement): void => {
  element.scrollIntoView({
    behavior: 'smooth',
    block: 'start',
    inline: 'nearest'
  })
  const priorOutline = element.style.outline
  const priorOutlineOffset = element.style.outlineOffset
  const priorTransition = element.style.transition
  element.style.outline = '2px solid rgba(59, 130, 246, 0.45)'
  element.style.outlineOffset = '2px'
  element.style.transition = priorTransition
    ? `${priorTransition}, outline 0.2s ease`
    : 'outline 0.2s ease'
  window.setTimeout(() => {
    element.style.outline = priorOutline
    element.style.outlineOffset = priorOutlineOffset
    element.style.transition = priorTransition
  }, 1300)
}

const scrollToCharOffset = (
  container: HTMLElement,
  targetStart: number,
  contentLength: number
): boolean => {
  if (!Number.isFinite(targetStart) || targetStart < 0) return false
  if (!Number.isFinite(contentLength) || contentLength <= 0) return false

  const ratio = Math.min(
    1,
    Math.max(0, targetStart / Math.max(1, contentLength - 1))
  )
  const containerMaxScroll = container.scrollHeight - container.clientHeight
  if (Number.isFinite(containerMaxScroll) && containerMaxScroll > 0) {
    const top = Math.round(containerMaxScroll * ratio)
    if (typeof container.scrollTo === 'function') {
      container.scrollTo({ top, behavior: 'smooth' })
    } else {
      container.scrollTop = top
    }
    return true
  }

  if (typeof document === 'undefined') return false
  const docScroller =
    document.scrollingElement instanceof HTMLElement
      ? document.scrollingElement
      : document.documentElement instanceof HTMLElement
        ? document.documentElement
        : null
  if (!docScroller) return false

  const docMaxScroll = docScroller.scrollHeight - docScroller.clientHeight
  if (!Number.isFinite(docMaxScroll) || docMaxScroll <= 0) return false
  const top = Math.round(docMaxScroll * ratio)

  if (typeof window !== 'undefined' && typeof window.scrollTo === 'function') {
    window.scrollTo({ top, behavior: 'smooth' })
  } else if (typeof docScroller.scrollTo === 'function') {
    docScroller.scrollTo({ top, behavior: 'smooth' })
  } else {
    docScroller.scrollTop = top
  }
  return true
}

const scrollToPageNumber = (
  container: HTMLElement,
  pageNumber: number,
  pageCountHint: number
): boolean => {
  if (!Number.isFinite(pageNumber) || pageNumber < 1) return false
  if (!Number.isFinite(pageCountHint) || pageCountHint < 1) return false

  const ratio = Math.min(
    1,
    Math.max(0, (pageNumber - 1) / Math.max(1, pageCountHint - 1))
  )
  const containerMaxScroll = container.scrollHeight - container.clientHeight
  if (Number.isFinite(containerMaxScroll) && containerMaxScroll > 0) {
    const top = Math.round(containerMaxScroll * ratio)
    if (typeof container.scrollTo === 'function') {
      container.scrollTo({ top, behavior: 'smooth' })
    } else {
      container.scrollTop = top
    }
    return true
  }

  if (typeof document === 'undefined') return false
  const docScroller =
    document.scrollingElement instanceof HTMLElement
      ? document.scrollingElement
      : document.documentElement instanceof HTMLElement
        ? document.documentElement
        : null
  if (!docScroller) return false

  const docMaxScroll = docScroller.scrollHeight - docScroller.clientHeight
  if (!Number.isFinite(docMaxScroll) || docMaxScroll <= 0) return false
  const top = Math.round(docMaxScroll * ratio)

  if (typeof window !== 'undefined' && typeof window.scrollTo === 'function') {
    window.scrollTo({ top, behavior: 'smooth' })
  } else if (typeof docScroller.scrollTo === 'function') {
    docScroller.scrollTo({ top, behavior: 'smooth' })
  } else {
    docScroller.scrollTop = top
  }
  return true
}

export interface UseReadingProgressDeps {
  selectedMedia: MediaResultItem | null
  mediaDetail: any
  content: string
  contentScrollContainerRef: React.RefObject<HTMLDivElement | null>
  contentBodyRef: React.RefObject<HTMLDivElement | null>
  navigationTarget: MediaNavigationTargetLike | null
  navigationNodeTitle: string | null
  navigationPageCountHint: number | null
  navigationSelectionNonce: number
  selectedMediaId: string | null
  isDetailLoading?: boolean
  t: (key: string, opts?: Record<string, any>) => string
}

export function useReadingProgress(deps: UseReadingProgressDeps) {
  const {
    selectedMedia,
    mediaDetail,
    content,
    contentScrollContainerRef,
    contentBodyRef,
    navigationTarget,
    navigationNodeTitle,
    navigationPageCountHint,
    navigationSelectionNonce,
    selectedMediaId,
    isDetailLoading = false,
    t
  } = deps

  const [showBackToTop, setShowBackToTop] = useState(false)
  const [contentSelectionAnnouncement, setContentSelectionAnnouncement] = useState('')

  const lastAppliedNavigationTargetKeyRef = useRef<string>('')
  const lastAppliedNavigationTitleKeyRef = useRef<string>('')
  const lastAppliedNavigationPageKeyRef = useRef<string>('')
  const lastContentSelectionAnnouncementKeyRef = useRef<string>('')
  const titleRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pageRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const mediaReadingProgress = useMediaReadingProgress({
    mediaId: selectedMedia?.id ?? null,
    mediaKind: selectedMedia?.kind ?? null,
    mediaDetail,
    contentLength: content.length,
    scrollContainerRef: contentScrollContainerRef,
    hasNavigationTarget: Boolean(navigationTarget)
  })
  const progressPercent = mediaReadingProgress?.progressPercent

  const selectedMediaAnnouncementLabel = useMemo(() => {
    if (!selectedMedia) return ''
    const title = String(selectedMedia.title || '').trim()
    if (title) return title
    const kind = String(selectedMedia.kind || 'media').trim() || 'media'
    return `${kind} ${selectedMedia.id}`
  }, [selectedMedia])

  const navigationTargetDescription = useMemo(
    () => describeMediaNavigationTarget(navigationTarget),
    [navigationTarget]
  )

  const handleBackToTop = useCallback(() => {
    const container = contentScrollContainerRef.current
    if (!container) return
    if (typeof container.scrollTo === 'function') {
      container.scrollTo({ top: 0, behavior: 'smooth' })
    } else {
      container.scrollTop = 0
    }
    setShowBackToTop(false)
  }, [contentScrollContainerRef])

  // Back to top visibility
  useEffect(() => {
    const container = contentScrollContainerRef.current
    if (!container) {
      setShowBackToTop(false)
      return
    }

    const updateVisibility = () => {
      setShowBackToTop(container.scrollTop >= 500)
    }

    updateVisibility()
    container.addEventListener('scroll', updateVisibility, { passive: true })
    return () => {
      container.removeEventListener('scroll', updateVisibility)
    }
  }, [content.length, selectedMedia?.id, contentScrollContainerRef])

  // Content selection announcement for a11y
  useEffect(() => {
    if (!selectedMediaId || !selectedMediaAnnouncementLabel) {
      lastContentSelectionAnnouncementKeyRef.current = ''
      setContentSelectionAnnouncement('')
      return
    }

    const stateLabel = isDetailLoading ? 'loading' : 'ready'
    const announcementKey = `${selectedMediaId}:${stateLabel}`
    if (lastContentSelectionAnnouncementKeyRef.current === announcementKey) return

    lastContentSelectionAnnouncementKeyRef.current = announcementKey
    const statusPrefix = isDetailLoading
      ? t('review:mediaPage.contentAnnouncementLoading', {
          defaultValue: getDesignSystemState('loading').label
        })
      : t('review:mediaPage.contentAnnouncementShowing', { defaultValue: 'Showing' })
    setContentSelectionAnnouncement(`${statusPrefix} ${selectedMediaAnnouncementLabel}`)
  }, [isDetailLoading, selectedMediaAnnouncementLabel, selectedMediaId, t])

  // Apply navigation target
  useEffect(() => {
    if (!selectedMediaId || !navigationTarget) {
      lastAppliedNavigationTargetKeyRef.current = ''
      return
    }
    const targetKey = [
      selectedMediaId,
      navigationTarget.target_type,
      navigationTarget.target_start ?? 'null',
      navigationTarget.target_end ?? 'null',
      navigationTarget.target_href ?? 'null',
      content.length,
      navigationSelectionNonce
    ].join(':')
    if (lastAppliedNavigationTargetKeyRef.current === targetKey) return
    lastAppliedNavigationTargetKeyRef.current = targetKey

    applyMediaNavigationTarget(navigationTarget, {
      root: contentBodyRef.current,
      mediaId: selectedMediaId
    })
  }, [content.length, navigationSelectionNonce, navigationTarget, selectedMediaId, contentBodyRef])

  // Navigation title jump (heading scroll)
  useEffect(() => {
    if (titleRetryTimerRef.current) {
      clearTimeout(titleRetryTimerRef.current)
      titleRetryTimerRef.current = null
    }
    if (!selectedMediaId || !navigationTarget) {
      lastAppliedNavigationTitleKeyRef.current = ''
      return
    }
    if (
      navigationTarget.target_type !== 'char_range' &&
      navigationTarget.target_type !== 'page'
    ) {
      return
    }
    const title = String(navigationNodeTitle || '').trim()
    if (!title) return

    const key = [
      selectedMediaId,
      navigationTarget.target_type,
      title.toLowerCase(),
      content.length,
      navigationTarget.target_start ?? 'null',
      navigationTarget.target_end ?? 'null',
      navigationSelectionNonce
    ].join(':')
    if (lastAppliedNavigationTitleKeyRef.current === key) return

    let attempts = 0
    const attemptNavigationTitleJump = (): boolean => {
      const match = findHeadingMatchInContent(contentBodyRef.current, title, {
        preferLast: navigationTarget.target_type === 'page'
      })
      if (match) {
        lastAppliedNavigationTitleKeyRef.current = key
        focusNavigationMatch(match)
        return true
      }

      if (navigationTarget.target_type === 'page') {
        return false
      }

      const start = navigationTarget.target_start
      if (
        typeof start === 'number' &&
        Number.isFinite(start) &&
        contentScrollContainerRef.current
      ) {
        const didScroll = scrollToCharOffset(
          contentScrollContainerRef.current,
          start,
          content.length
        )
        if (didScroll) {
          lastAppliedNavigationTitleKeyRef.current = key
          return true
        }
      }
      return false
    }

    const runAttempt = () => {
      if (attemptNavigationTitleJump()) return
      if (attempts >= 10) return
      attempts += 1
      titleRetryTimerRef.current = setTimeout(runAttempt, 120)
    }
    runAttempt()

    return () => {
      if (titleRetryTimerRef.current) {
        clearTimeout(titleRetryTimerRef.current)
        titleRetryTimerRef.current = null
      }
    }
  }, [
    content.length,
    navigationNodeTitle,
    navigationSelectionNonce,
    navigationTarget,
    selectedMediaId,
    contentBodyRef,
    contentScrollContainerRef
  ])

  // Navigation page jump
  useEffect(() => {
    if (pageRetryTimerRef.current) {
      clearTimeout(pageRetryTimerRef.current)
      pageRetryTimerRef.current = null
    }
    if (!selectedMediaId || !navigationTarget) {
      lastAppliedNavigationPageKeyRef.current = ''
      return
    }
    if (navigationTarget.target_type !== 'page') return

    const pageStart = navigationTarget.target_start
    if (typeof pageStart !== 'number' || !Number.isFinite(pageStart) || pageStart < 1) {
      return
    }
    if (!contentScrollContainerRef.current) return

    const pageCountHint =
      typeof navigationPageCountHint === 'number' &&
      Number.isFinite(navigationPageCountHint) &&
      navigationPageCountHint > 0
        ? Math.trunc(navigationPageCountHint)
        : Math.max(1, Math.trunc(pageStart))

    const key = [
      selectedMediaId,
      Math.trunc(pageStart),
      pageCountHint,
      content.length,
      navigationSelectionNonce
    ].join(':')
    if (lastAppliedNavigationPageKeyRef.current === key) return

    const container = contentScrollContainerRef.current
    if (!container) return

    let attempts = 0
    const attemptNavigationPageJump = (): boolean => {
      const didScroll = scrollToPageNumber(
        container,
        pageStart,
        pageCountHint
      )
      if (didScroll) {
        lastAppliedNavigationPageKeyRef.current = key
        return true
      }
      return false
    }

    const runAttempt = () => {
      if (attemptNavigationPageJump()) return
      if (attempts >= 10) return
      attempts += 1
      pageRetryTimerRef.current = setTimeout(runAttempt, 120)
    }
    runAttempt()

    return () => {
      if (pageRetryTimerRef.current) {
        clearTimeout(pageRetryTimerRef.current)
        pageRetryTimerRef.current = null
      }
    }
  }, [
    content.length,
    navigationPageCountHint,
    navigationSelectionNonce,
    navigationTarget,
    selectedMediaId,
    contentScrollContainerRef
  ])

  // Cleanup timers
  useEffect(() => {
    return () => {
      if (titleRetryTimerRef.current) {
        clearTimeout(titleRetryTimerRef.current)
      }
      if (pageRetryTimerRef.current) {
        clearTimeout(pageRetryTimerRef.current)
      }
    }
  }, [])

  return {
    showBackToTop,
    contentSelectionAnnouncement,
    progressPercent,
    navigationTargetDescription,
    handleBackToTop
  }
}
