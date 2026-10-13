import React, { useMemo, useCallback, useRef, useEffect } from 'react'
import type { MediaNavigationFormat } from '@/utils/media-navigation-scope'
import { MEDIA_DISPLAY_MODE_FORMAT_TO_LABEL } from '@/utils/media-navigation-scope'
import { resolveMediaRenderMode } from '@/utils/media-render-mode'
import { sanitizeMediaRichHtmlWithStats } from '@/utils/media-rich-html-sanitizer'
import { trackMediaNavigationTelemetry } from '@/utils/media-navigation-telemetry'
import {
  hasLeadingTranscriptTimings,
  stripLeadingTranscriptTimings
} from '@/utils/media-transcript-display'
import type { MediaResultItem } from '../types'
import { stripMediaMetadata } from '@/utils/media-metadata-display'
import type { MediaTextSizePreset } from '@/services/settings/ui-settings'

const PLAIN_TEXT_MEDIA_TYPES = new Set(['audio', 'video', 'transcript', 'subtitle'])
const MARKDOWN_HINTS = [
  /^#{1,6}\s+/m,
  /^\s*([-*+]|\d+\.)\s+/m,
  /^>\s+/m,
  /```/,
  /`[^`]+`/,
  /\[[^\]]+\]\([^)]+\)/,
  /<\/?[a-z][\s\S]*>/i
]

const looksLikeMarkdown = (text: string) =>
  MARKDOWN_HINTS.some((pattern) => pattern.test(text))

const shouldForceHardBreaks = (text: string, mediaType?: string) => {
  const normalizedType = mediaType?.toLowerCase().trim()
  if (!normalizedType || !PLAIN_TEXT_MEDIA_TYPES.has(normalizedType)) return false
  return !looksLikeMarkdown(text)
}

export const TEXT_SIZE_CONTROL_OPTIONS: Array<{
  value: MediaTextSizePreset
  label: string
  className: string
  markdownSize: 'xs' | 'sm' | 'base'
  richClass: string
}> = [
  {
    value: 's',
    label: 'S',
    className: 'text-xs leading-relaxed',
    markdownSize: 'xs',
    richClass: 'prose-xs'
  },
  {
    value: 'm',
    label: 'M',
    className: 'text-sm leading-relaxed',
    markdownSize: 'sm',
    richClass: 'prose-sm'
  },
  {
    value: 'l',
    label: 'L',
    className: 'text-base leading-relaxed',
    markdownSize: 'base',
    richClass: 'prose'
  }
]

export interface UseContentRenderingDeps {
  content: string
  selectedMedia: MediaResultItem | null
  contentDisplayMode: MediaNavigationFormat
  resolvedContentFormat: MediaNavigationFormat | null
  allowRichRendering: boolean
  hideTranscriptTimings: boolean | null
  textSizePreset: string | null
  selectedMediaId: string | null
  shouldShowEmbeddedPlayer: boolean
}

export function useContentRendering(deps: UseContentRenderingDeps) {
  const {
    content,
    selectedMedia,
    contentDisplayMode,
    resolvedContentFormat,
    allowRichRendering,
    hideTranscriptTimings,
    textSizePreset,
    selectedMediaId,
    shouldShowEmbeddedPlayer
  } = deps

  const lastSanitizationTelemetryKeyRef = useRef<string>('')

  const resolvedTextSizePreset: MediaTextSizePreset = useMemo(() => {
    const normalized = String(textSizePreset || '').toLowerCase()
    if (normalized === 's' || normalized === 'l') return normalized
    return 'm'
  }, [textSizePreset])

  const textSizeControl =
    TEXT_SIZE_CONTROL_OPTIONS.find(
      (option) => option.value === resolvedTextSizePreset
    ) || TEXT_SIZE_CONTROL_OPTIONS[1]

  const contentBodyTypographyClass = textSizeControl.className
  const markdownPreviewSize = textSizeControl.markdownSize
  const richTextTypographyClass = textSizeControl.richClass

  const shouldHideTranscriptTimings = hideTranscriptTimings ?? true

  const readingContent = useMemo(
    () => selectedMedia?.kind === 'media' ? stripMediaMetadata(content) : content,
    [content, selectedMedia?.kind]
  )

  const displayContent = useMemo(
    () =>
      shouldHideTranscriptTimings
        ? stripLeadingTranscriptTimings(readingContent)
        : readingContent,
    [readingContent, shouldHideTranscriptTimings]
  )

  const effectiveRenderMode = useMemo(
    () =>
      resolveMediaRenderMode({
        requestedMode: contentDisplayMode,
        resolvedContentFormat,
        allowRichRendering
      }),
    [allowRichRendering, contentDisplayMode, resolvedContentFormat]
  )

  const contentForPreview = useMemo(() => {
    if (!displayContent) return ''
    if (selectedMedia?.kind === 'note') return displayContent
    const normalized = displayContent.replace(/\r\n/g, '\n')
    if (!shouldForceHardBreaks(normalized, selectedMedia?.meta?.type)) {
      return normalized
    }
    return normalized.replace(/\n/g, '  \n')
  }, [displayContent, selectedMedia?.kind, selectedMedia?.meta?.type])

  const transcriptLines = useMemo(
    () => (readingContent ? readingContent.replace(/\r\n/g, '\n').split('\n') : []),
    [readingContent]
  )

  const hasTranscriptTimingLines = useMemo(
    () => hasLeadingTranscriptTimings(readingContent),
    [readingContent]
  )

  const hasClickableTranscriptTimestamps = useMemo(
    () => shouldShowEmbeddedPlayer && hasTranscriptTimingLines,
    [hasTranscriptTimingLines, shouldShowEmbeddedPlayer]
  )

  const richSanitization = useMemo(() => {
    if (effectiveRenderMode !== 'html' || !displayContent) {
      return {
        html: '',
        removed_node_count: 0,
        removed_attribute_count: 0,
        blocked_url_schemes: [] as string[]
      }
    }
    return sanitizeMediaRichHtmlWithStats(displayContent)
  }, [displayContent, effectiveRenderMode])

  const sanitizedRichContent = richSanitization.html

  const displayModeOptions = useMemo(() => {
    const baseModes: MediaNavigationFormat[] = ['auto', 'plain', 'markdown']
    if (allowRichRendering) baseModes.push('html')
    return baseModes.map((mode) => ({
      value: mode,
      label: MEDIA_DISPLAY_MODE_FORMAT_TO_LABEL[mode]
    }))
  }, [allowRichRendering])

  // Track sanitization telemetry
  useEffect(() => {
    if (effectiveRenderMode !== 'html') return
    const removedNodeCount = richSanitization.removed_node_count
    const removedAttributeCount = richSanitization.removed_attribute_count
    const blockedSchemes = richSanitization.blocked_url_schemes
      .map((scheme) => String(scheme || '').trim().toLowerCase())
      .filter(Boolean)

    if (
      removedNodeCount <= 0 &&
      removedAttributeCount <= 0 &&
      blockedSchemes.length === 0
    ) {
      return
    }

    const dedupeKey = [
      selectedMediaId || 'none',
      content.length,
      removedNodeCount,
      removedAttributeCount,
      blockedSchemes.join(',')
    ].join(':')
    if (lastSanitizationTelemetryKeyRef.current === dedupeKey) return
    lastSanitizationTelemetryKeyRef.current = dedupeKey

    void trackMediaNavigationTelemetry({
      type: 'media_rich_sanitization_applied',
      removed_node_count: removedNodeCount,
      removed_attribute_count: removedAttributeCount,
      blocked_url_count: blockedSchemes.length
    })

    const uniqueSchemes = new Set(blockedSchemes)
    for (const scheme of uniqueSchemes) {
      void trackMediaNavigationTelemetry({
        type: 'media_rich_sanitization_blocked_url',
        scheme
      })
    }
  }, [content.length, effectiveRenderMode, richSanitization, selectedMediaId])

  return {
    resolvedTextSizePreset,
    textSizeControl,
    contentBodyTypographyClass,
    markdownPreviewSize,
    richTextTypographyClass,
    shouldHideTranscriptTimings,
    displayContent,
    readingContent,
    effectiveRenderMode,
    contentForPreview,
    transcriptLines,
    hasTranscriptTimingLines,
    hasClickableTranscriptTimestamps,
    richSanitization,
    sanitizedRichContent,
    displayModeOptions
  }
}
