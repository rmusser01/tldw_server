import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import type { MediaResultItem } from '../types'
import { buildReadAlongSegments } from '../read-along/media-read-along-segments'
import type { ReadAlongSegment } from '../read-along/types'

export const LARGE_PLAIN_CONTENT_THRESHOLD_CHARS = 120_000
export const LARGE_PLAIN_CONTENT_CHUNK_CHARS = 32_000
const LARGE_PLAIN_CONTENT_PREFETCH_MARGIN_PX = 640

const normalizeFindQuery = (value: string): string => value.trim().toLowerCase()

export const findInContentOffsets = (text: string, query: string): number[] => {
  if (!text) return []
  const normalizedQuery = normalizeFindQuery(query)
  if (!normalizedQuery) return []

  const haystack = text.toLowerCase()
  const offsets: number[] = []
  let fromIndex = 0

  while (fromIndex < haystack.length) {
    const index = haystack.indexOf(normalizedQuery, fromIndex)
    if (index === -1) break
    offsets.push(index)
    fromIndex = index + Math.max(1, normalizedQuery.length)
  }

  return offsets
}

export const getNextFindMatchIndex = (
  currentIndex: number,
  totalMatches: number,
  direction: 1 | -1
): number => {
  if (!Number.isFinite(totalMatches) || totalMatches <= 0) return -1
  if (!Number.isFinite(currentIndex) || currentIndex < 0 || currentIndex >= totalMatches) {
    return direction === -1 ? totalMatches - 1 : 0
  }
  if (direction === 1) {
    return (currentIndex + 1) % totalMatches
  }
  return (currentIndex - 1 + totalMatches) % totalMatches
}

const parseTimestampToSeconds = (value: string): number | null => {
  const trimmed = value.trim()
  if (!trimmed) return null
  const parts = trimmed.split(':').map((part) => Number(part))
  if (parts.some((part) => !Number.isFinite(part) || part < 0)) return null
  if (parts.length === 2) {
    const [minutes, seconds] = parts
    return Math.floor(minutes * 60 + seconds)
  }
  if (parts.length === 3) {
    const [hours, minutes, seconds] = parts
    return Math.floor(hours * 3600 + minutes * 60 + seconds)
  }
  return null
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

export interface UseTranscriptDisplayDeps {
  displayContent: string
  content: string
  selectedMedia: MediaResultItem | null
  effectiveRenderMode: string
  shouldHideTranscriptTimings: boolean
  hasClickableTranscriptTimestamps: boolean
  activeReadAlongSegmentId?: string | null
  contentScrollContainerRef: React.RefObject<HTMLDivElement | null>
  rootContainerRef: React.RefObject<HTMLDivElement | null>
  mediaPlayerRef: React.RefObject<HTMLMediaElement | null>
  t: (key: string, opts?: Record<string, any>) => string
}

const renderReadAlongWrappedText = (
  text: string,
  segments: ReadAlongSegment[],
  activeReadAlongSegmentId: string | null | undefined
): React.ReactNode => {
  if (!text || segments.length === 0) return text

  const visibleSegments = segments
    .filter(
      (segment) =>
        segment.displayStart != null &&
        segment.displayEnd != null &&
        segment.displayStart >= 0 &&
        segment.displayEnd > segment.displayStart &&
        segment.displayStart < text.length &&
        segment.displayEnd <= text.length
    )
    .sort((left, right) => left.displayStart! - right.displayStart!)

  if (visibleSegments.length === 0) return text

  const parts: React.ReactNode[] = []
  let cursor = 0
  visibleSegments.forEach((segment) => {
    const start = segment.displayStart!
    const end = segment.displayEnd!
    if (start < cursor) return
    if (start > cursor) {
      parts.push(text.slice(cursor, start))
    }
    const isActive = segment.id === activeReadAlongSegmentId
    parts.push(
      <span
        key={`read-along-segment-${segment.id}`}
        data-read-along-segment-id={segment.id}
        data-read-along-active={isActive ? 'true' : undefined}
        className={isActive ? 'rounded bg-primary/20 px-0.5 text-text' : undefined}
      >
        {text.slice(start, end)}
      </span>
    )
    cursor = end
  })

  if (cursor < text.length) {
    parts.push(text.slice(cursor))
  }

  return <>{parts}</>
}

export function useTranscriptDisplay(deps: UseTranscriptDisplayDeps) {
  const {
    displayContent,
    content,
    selectedMedia,
    effectiveRenderMode,
    shouldHideTranscriptTimings,
    hasClickableTranscriptTimestamps,
    activeReadAlongSegmentId,
    contentScrollContainerRef,
    rootContainerRef,
    mediaPlayerRef,
    t
  } = deps

  const [visiblePlainContentChars, setVisiblePlainContentChars] = useState(
    () => content.length
  )
  const [findBarOpen, setFindBarOpen] = useState(false)
  const [findQuery, setFindQuery] = useState('')
  const [findMatchOffsets, setFindMatchOffsets] = useState<number[]>([])
  const [activeFindMatchIndex, setActiveFindMatchIndex] = useState(-1)
  const findInputRef = useRef<HTMLInputElement | null>(null)
  const findMatchElementRefs = useRef<Array<HTMLElement | null>>([])

  const normalizedFindQuery = useMemo(() => normalizeFindQuery(findQuery), [findQuery])

  const readAlongSegments = useMemo(
    () =>
      buildReadAlongSegments({
        mediaId: selectedMedia?.id != null ? String(selectedMedia.id) : 'unknown-media',
        content,
        displayContent,
        renderMode: effectiveRenderMode,
        hideTranscriptTimings: shouldHideTranscriptTimings
      }),
    [
      content,
      displayContent,
      effectiveRenderMode,
      selectedMedia?.id,
      shouldHideTranscriptTimings
    ]
  )

  const shouldRenderTranscriptTimestampChips =
    hasClickableTranscriptTimestamps &&
    !shouldHideTranscriptTimings &&
    !normalizedFindQuery

  const shouldUseChunkedPlainRendering = useMemo(
    () =>
      effectiveRenderMode === 'plain' &&
      !shouldRenderTranscriptTimestampChips &&
      !normalizedFindQuery &&
      displayContent.length > LARGE_PLAIN_CONTENT_THRESHOLD_CHARS,
    [
      displayContent.length,
      effectiveRenderMode,
      normalizedFindQuery,
      shouldRenderTranscriptTimestampChips
    ]
  )

  const visiblePlainContent = useMemo(() => {
    if (!displayContent) return ''
    if (!shouldUseChunkedPlainRendering) return displayContent
    return displayContent.slice(
      0,
      Math.max(0, Math.min(displayContent.length, visiblePlainContentChars))
    )
  }, [displayContent, shouldUseChunkedPlainRendering, visiblePlainContentChars])

  const hasUnrenderedPlainContent =
    shouldUseChunkedPlainRendering && visiblePlainContentChars < displayContent.length

  const loadMorePlainContent = useCallback(() => {
    if (!shouldUseChunkedPlainRendering) return
    setVisiblePlainContentChars((prev) =>
      Math.min(
        displayContent.length,
        Math.max(0, prev) + LARGE_PLAIN_CONTENT_CHUNK_CHARS
      )
    )
  }, [displayContent.length, shouldUseChunkedPlainRendering])

  const findMatchCount = findMatchOffsets.length

  const moveFindMatch = useCallback(
    (direction: 1 | -1) => {
      setActiveFindMatchIndex((prev) =>
        getNextFindMatchIndex(prev, findMatchOffsets.length, direction)
      )
    },
    [findMatchOffsets.length]
  )

  const closeFindBar = useCallback(() => {
    setFindBarOpen(false)
    setFindQuery('')
    setFindMatchOffsets([])
    setActiveFindMatchIndex(-1)
    findMatchElementRefs.current = []
  }, [])

  const highlightedPlainContent = useMemo<React.ReactNode>(() => {
    findMatchElementRefs.current = []
    if (!displayContent) {
      return t('review:mediaPage.noContent', {
        defaultValue: 'No content available'
      })
    }
    const plainText = shouldUseChunkedPlainRendering ? visiblePlainContent : displayContent
    if (!normalizedFindQuery) {
      return renderReadAlongWrappedText(
        plainText,
        readAlongSegments,
        activeReadAlongSegmentId
      )
    }
    if (findMatchOffsets.length === 0) {
      return plainText
    }

    const parts: React.ReactNode[] = []
    const queryLength = normalizedFindQuery.length
    let cursor = 0

    findMatchOffsets.forEach((start, index) => {
      if (start < cursor) return
      if (start > cursor) {
        parts.push(displayContent.slice(cursor, start))
      }
      const end = Math.min(displayContent.length, start + queryLength)
      const isActive = index === activeFindMatchIndex
      parts.push(
        <mark
          key={`find-match-${index}-${start}`}
          ref={(node) => {
            findMatchElementRefs.current[index] = node
          }}
          data-find-match-index={index}
          className={
            isActive
              ? 'rounded bg-primary/30 text-text px-0.5'
              : 'rounded bg-warn/20 text-text px-0.5'
          }
        >
          {displayContent.slice(start, end)}
        </mark>
      )
      cursor = end
    })

    if (cursor < displayContent.length) {
      parts.push(displayContent.slice(cursor))
    }

    return <>{parts}</>
  }, [
    activeFindMatchIndex,
    activeReadAlongSegmentId,
    displayContent,
    findMatchOffsets,
    normalizedFindQuery,
    readAlongSegments,
    shouldUseChunkedPlainRendering,
    t,
    visiblePlainContent
  ])

  // Reset visible chars when content changes
  useEffect(() => {
    if (!displayContent) {
      setVisiblePlainContentChars(0)
      return
    }
    if (!shouldUseChunkedPlainRendering) {
      setVisiblePlainContentChars(displayContent.length)
      return
    }
    setVisiblePlainContentChars(
      Math.min(displayContent.length, LARGE_PLAIN_CONTENT_CHUNK_CHARS)
    )
  }, [displayContent.length, selectedMedia?.id, shouldUseChunkedPlainRendering])

  // Infinite scroll for chunked content
  useEffect(() => {
    if (!hasUnrenderedPlainContent) return
    const container = contentScrollContainerRef.current
    if (!container) return

    const maybeLoadMore = () => {
      if (container.clientHeight <= 0 || container.scrollHeight <= 0) {
        return
      }
      if (
        container.scrollTop + container.clientHeight <
        container.scrollHeight - LARGE_PLAIN_CONTENT_PREFETCH_MARGIN_PX
      ) {
        return
      }
      setVisiblePlainContentChars((prev) =>
        Math.min(
          displayContent.length,
          Math.max(0, prev) + LARGE_PLAIN_CONTENT_CHUNK_CHARS
        )
      )
    }

    if (
      visiblePlainContentChars === LARGE_PLAIN_CONTENT_CHUNK_CHARS &&
      container.scrollTop > 0
    ) {
      maybeLoadMore()
    }
    container.addEventListener('scroll', maybeLoadMore, { passive: true })
    return () => {
      container.removeEventListener('scroll', maybeLoadMore)
    }
  }, [displayContent.length, hasUnrenderedPlainContent, visiblePlainContentChars, contentScrollContainerRef])

  // Update find match offsets
  useEffect(() => {
    const offsets = findInContentOffsets(displayContent, findQuery)
    setFindMatchOffsets(offsets)
    setActiveFindMatchIndex(offsets.length > 0 ? 0 : -1)
  }, [displayContent, findQuery])

  // Reset find bar on media change
  useEffect(() => {
    setFindBarOpen(false)
    setFindQuery('')
    setFindMatchOffsets([])
    setActiveFindMatchIndex(-1)
    findMatchElementRefs.current = []
  }, [selectedMedia?.id])

  // Auto-focus find input
  useEffect(() => {
    if (!findBarOpen) return
    const timer = window.setTimeout(() => {
      findInputRef.current?.focus()
      findInputRef.current?.select()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [findBarOpen])

  // Scroll to active match
  useEffect(() => {
    if (activeFindMatchIndex < 0 || findMatchOffsets.length === 0) return

    const activeNode = findMatchElementRefs.current[activeFindMatchIndex]
    if (activeNode && typeof activeNode.scrollIntoView === 'function') {
      activeNode.scrollIntoView({ behavior: 'smooth', block: 'center' })
      return
    }

    const container = contentScrollContainerRef.current
    const offset = findMatchOffsets[activeFindMatchIndex]
    if (container && Number.isFinite(offset) && displayContent.length > 0) {
      scrollToCharOffset(container, offset, displayContent.length)
    }
  }, [activeFindMatchIndex, displayContent.length, findMatchOffsets, contentScrollContainerRef])

  // Ctrl+F handler
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return
      if (event.key.toLowerCase() !== 'f') return

      const target = event.target as HTMLElement | null
      const isTypingTarget =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        Boolean(target?.isContentEditable)
      if (isTypingTarget) return

      const root = rootContainerRef.current
      if (
        root &&
        target &&
        target !== document.body &&
        !root.contains(target)
      ) {
        return
      }

      event.preventDefault()
      setFindBarOpen(true)
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [rootContainerRef])

  const handleTranscriptTimestampSeek = useCallback((timestamp: string) => {
    const seconds = parseTimestampToSeconds(timestamp)
    if (seconds == null) return
    const player = mediaPlayerRef.current
    if (!player) return
    player.currentTime = seconds
  }, [mediaPlayerRef])

  return {
    // State
    visiblePlainContentChars,
    findBarOpen,
    setFindBarOpen,
    findQuery,
    setFindQuery,
    findMatchOffsets,
    activeFindMatchIndex,
    findInputRef,
    findMatchElementRefs,
    // Computed
    normalizedFindQuery,
    shouldRenderTranscriptTimestampChips,
    shouldUseChunkedPlainRendering,
    visiblePlainContent,
    hasUnrenderedPlainContent,
    readAlongSegments,
    findMatchCount,
    highlightedPlainContent,
    // Callbacks
    loadMorePlainContent,
    moveFindMatch,
    closeFindBar,
    handleTranscriptTimestampSeek
  }
}
