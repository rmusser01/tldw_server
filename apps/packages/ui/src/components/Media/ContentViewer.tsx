import {
  ChevronLeft,
  ChevronRight,
  FileSearch,
  ChevronDown,
  ChevronUp,
  Send,
  Copy,
  Sparkles,
  MoreHorizontal,
  MessageSquare,
  Clock,
  FileText,
  Edit3,
  ExternalLink,
  Expand,
  History,
  Minimize2,
  Loader2,
  Trash2,
  UploadCloud,
  User,
  Download
} from 'lucide-react'
import React, { useState, Suspense, useMemo, useRef, useCallback, useEffect } from 'react'
import { Select, Dropdown, Tooltip, message, Spin } from 'antd'
import { useTranslation } from 'react-i18next'
import type { MenuProps } from 'antd'
import type { MediaResultItem } from './types'
import type { MediaNavigationFormat } from '@/utils/media-navigation-scope'
import { parseLeadingTranscriptTiming } from '@/utils/media-transcript-display'
import { requestQuickIngestOpen } from '@/utils/quick-ingest-open'
import { useSetting } from '@/hooks/useSetting'
import {
  MEDIA_COLLAPSED_SECTIONS_SETTING,
  MEDIA_HIDE_TRANSCRIPT_TIMINGS_SETTING,
  MEDIA_TEXT_SIZE_PRESET_SETTING
} from '@/services/settings/ui-settings'
// estimateReadingTimeMinutes moved to useContentMetadata hook
import type { MediaNavigationTargetLike } from '@/utils/media-navigation-target'

// Hooks
import {
  useContentMetadata,
  processingStatusClass,
  formatProcessingStatus
} from './hooks/useContentMetadata'
import { useContentEditState } from './hooks/useContentEditState'
import { useContentRendering, TEXT_SIZE_CONTROL_OPTIONS } from './hooks/useContentRendering'
import {
  useContentViewerModals,
} from './hooks/useContentViewerModals'
import { useReadingProgress } from './hooks/useReadingProgress'
import {
  useTranscriptDisplay,
  findInContentOffsets,
  getNextFindMatchIndex,
  LARGE_PLAIN_CONTENT_THRESHOLD_CHARS,
  LARGE_PLAIN_CONTENT_CHUNK_CHARS
} from './hooks/useTranscriptDisplay'
import { MediaReadAlongPopover } from './read-along/MediaReadAlongPopover'
import { MediaReadAlongTransport } from './read-along/MediaReadAlongTransport'
import { useContentSelectionActions } from './read-along/useContentSelectionActions'
import { useMediaReadAlongSession } from './read-along/useMediaReadAlongSession'
import type { ReadAlongScope } from './read-along/types'

// Re-export for test compatibility
export {
  findInContentOffsets,
  getNextFindMatchIndex,
  LARGE_PLAIN_CONTENT_THRESHOLD_CHARS,
  LARGE_PLAIN_CONTENT_CHUNK_CHARS
}

// Lazy load ContentEditModal for code splitting
const ContentEditModal = React.lazy(() =>
  import('./ContentEditModal').then((m) => ({ default: m.ContentEditModal }))
)
const LazyAnalysisModal = React.lazy(() =>
  import('./AnalysisModal').then((m) => ({ default: m.AnalysisModal }))
)
const LazyAnalysisEditModal = React.lazy(() =>
  import('./AnalysisEditModal').then((m) => ({ default: m.AnalysisEditModal }))
)
const LazyDeveloperToolsSection = React.lazy(() =>
  import('./DeveloperToolsSection').then((m) => ({
    default: m.DeveloperToolsSection
  }))
)
const LazyDiffViewModal = React.lazy(() =>
  import('./DiffViewModal').then((m) => ({ default: m.DiffViewModal }))
)
const LazyVersionHistoryPanel = React.lazy(() =>
  import('./VersionHistoryPanel').then((m) => ({ default: m.VersionHistoryPanel }))
)
const LazyContentViewerDocumentIntelligenceSection = React.lazy(() =>
  import('./ContentViewerDocumentIntelligenceSection').then((m) => ({
    default: m.ContentViewerDocumentIntelligenceSection
  }))
)
const LazyContentViewerActionModals = React.lazy(() =>
  import('./ContentViewerActionModals').then((m) => ({
    default: m.ContentViewerActionModals
  }))
)
const LazyContentViewerMetadataSectionBody = React.lazy(() =>
  import('./ContentViewerMetadataSectionBody').then((m) => ({
    default: m.ContentViewerMetadataSectionBody
  }))
)
const LazyMarkdownPreview = React.lazy(() =>
  import('@/components/Common/MarkdownPreview').then((m) => ({
    default: m.MarkdownPreview
  }))
)

export const shouldShowMediaDeveloperTools = (
  env: Record<string, unknown> | null | undefined
): boolean => {
  if (!env || typeof env !== 'object') return false
  const mode = String((env as Record<string, unknown>).MODE || '').toLowerCase()
  return Boolean((env as Record<string, unknown>).DEV) || mode === 'development'
}

const buildContentRevisionHash = (value: string): string => {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash.toString(16).padStart(8, '0')
}

const clearDocumentSelection = (): void => {
  if (typeof window === 'undefined' || typeof window.getSelection !== 'function') {
    return
  }
  window.getSelection()?.removeAllRanges()
}

// Metadata helpers moved to useContentMetadata hook

interface ContentViewerProps {
  selectedMedia: MediaResultItem | null
  content: string
  mediaDetail?: any
  contentDisplayMode?: MediaNavigationFormat
  resolvedContentFormat?: MediaNavigationFormat | null
  showContentDisplayModeSelector?: boolean
  allowRichRendering?: boolean
  onContentDisplayModeChange?: (mode: MediaNavigationFormat) => void
  isDetailLoading?: boolean
  onPrevious?: () => void
  onNext?: () => void
  hasPrevious?: boolean
  hasNext?: boolean
  currentIndex?: number
  totalResults?: number
  /** True only when the server has zero media items (no filters/search active). */
  isLibraryEmpty?: boolean
  onChatWithMedia?: () => void
  onChatAboutMedia?: () => void
  onGenerateFlashcardsFromContent?: (payload: {
    text: string
    sourceId?: string
    sourceTitle?: string
  }) => void
  onRefreshMedia?: () => void
  onKeywordsUpdated?: (mediaId: string | number, keywords: string[]) => void
  onCreateNoteWithContent?: (content: string, title: string) => void
  onOpenInMultiReview?: () => void
  onSendAnalysisToChat?: (text: string) => void
  contentRef?: (node: HTMLDivElement | null) => void
  onDeleteItem?: (item: MediaResultItem, detail: any | null) => Promise<void>
  navigationTarget?: MediaNavigationTargetLike | null
  navigationNodeTitle?: string | null
  navigationPageCountHint?: number | null
  navigationSelectionNonce?: number
}


export function ContentViewer({
  selectedMedia,
  content,
  mediaDetail,
  contentDisplayMode = 'auto',
  resolvedContentFormat = null,
  showContentDisplayModeSelector = false,
  allowRichRendering = false,
  onContentDisplayModeChange,
  isDetailLoading = false,
  onPrevious,
  onNext,
  hasPrevious = false,
  hasNext = false,
  currentIndex = 0,
  totalResults = 0,
  isLibraryEmpty: isLibraryEmptyProp = false,
  onChatWithMedia,
  onChatAboutMedia,
  onGenerateFlashcardsFromContent,
  onRefreshMedia,
  onKeywordsUpdated,
  onCreateNoteWithContent,
  onOpenInMultiReview,
  onSendAnalysisToChat,
  contentRef,
  onDeleteItem,
  navigationTarget = null,
  navigationNodeTitle = null,
  navigationPageCountHint = null,
  navigationSelectionNonce = 0
}: ContentViewerProps) {
  const { t } = useTranslation(['review', 'common'])
  const [collapsedSections, setCollapsedSections] = useSetting(
    MEDIA_COLLAPSED_SECTIONS_SETTING
  )
  const [hideTranscriptTimings, setHideTranscriptTimings] = useSetting(
    MEDIA_HIDE_TRANSCRIPT_TIMINGS_SETTING
  )
  const [textSizePreset, setTextSizePreset] = useSetting(
    MEDIA_TEXT_SIZE_PRESET_SETTING
  )

  const rootContainerRef = useRef<HTMLDivElement | null>(null)
  const contentBodyRef = useRef<HTMLDivElement | null>(null)
  const contentScrollContainerRef = useRef<HTMLDivElement | null>(null)
  const readAlongTransportAnchorRef = useRef<DOMRect | null>(null)
  const [versionHistoryMounted, setVersionHistoryMounted] = useState(false)

  const selectedMediaId = selectedMedia?.id != null ? String(selectedMedia.id) : null
  const isNote = selectedMedia?.kind === 'note'

  const setRootContainerRef = useCallback(
    (node: HTMLDivElement | null) => {
      rootContainerRef.current = node
      if (contentRef) {
        contentRef(node)
      }
    },
    [contentRef]
  )

  // --- Hook: Content Edit State ---
  const editState = useContentEditState({
    selectedMedia,
    content,
    mediaDetail,
    selectedMediaId,
    isNote,
    onKeywordsUpdated,
    onRefreshMedia,
    onDeleteItem,
    t
  })

  // --- Hook: Content Rendering ---
  const rendering = useContentRendering({
    content,
    selectedMedia,
    contentDisplayMode,
    resolvedContentFormat,
    allowRichRendering,
    hideTranscriptTimings,
    textSizePreset,
    selectedMediaId,
    shouldShowEmbeddedPlayer: false // placeholder, resolved below
  })
  const markdownFallbackContent =
    rendering.contentForPreview ||
    t('review:mediaPage.noContent', {
      defaultValue: 'No content available'
    })

  // --- Hook: Content Viewer Modals ---
  const modals = useContentViewerModals({
    selectedMedia,
    content,
    mediaDetail,
    selectedMediaId,
    isNote,
    editingKeywords: editState.editingKeywords,
    selectedAnalysis: editState.selectedAnalysis,
    collapsedSections,
    setCollapsedSections,
    contentBodyRef,
    onRefreshMedia,
    t
  })

  const contentSelectionIdentityKey = useMemo(
    () => [
      selectedMediaId || 'none',
      selectedMedia?.kind || 'none',
      rendering.effectiveRenderMode,
      content.length,
      buildContentRevisionHash(content)
    ].join(':'),
    [content, rendering.effectiveRenderMode, selectedMedia?.kind, selectedMediaId]
  )

  const selectionActions = useContentSelectionActions({
    contentBodyRef,
    contentIdentityKey: contentSelectionIdentityKey,
    onApplyAnnotationSelection: modals.captureAnnotationSelection
  })

  useEffect(() => {
    selectionActions.clearSelectionActions()
  }, [contentSelectionIdentityKey, selectionActions.clearSelectionActions])

  const readAlong = useMediaReadAlongSession({
    mediaId: selectedMediaId,
    mediaKind: selectedMedia?.kind || null,
    content,
    displayContent: rendering.displayContent,
    renderMode: rendering.effectiveRenderMode,
    hideTranscriptTimings: rendering.shouldHideTranscriptTimings,
    selection: selectionActions.selectionActionState,
    contentBodyRef,
    contentScrollContainerRef,
    embeddedMediaRef: modals.mediaPlayerRef
  })

  const handleStartReadAlongScope = useCallback(
    (scope: ReadAlongScope) => {
      const currentSelection = selectionActions.selectionActionState
      if (!currentSelection) return

      readAlongTransportAnchorRef.current = currentSelection.anchorRect
      void readAlong.start(scope)
      selectionActions.clearSelectionActions()
      clearDocumentSelection()
    },
    [readAlong, selectionActions]
  )

  const handleStopReadAlong = useCallback(() => {
    readAlong.stop()
    selectionActions.clearSelectionActions()
    clearDocumentSelection()
  }, [readAlong, selectionActions])

  const handleToggleReadAlong = useCallback(() => {
    if (readAlong.state.status === 'paused') {
      readAlong.resume()
      return
    }
    readAlong.pause()
  }, [readAlong])

  const supportedReadAlongScopes = useMemo<ReadAlongScope[]>(() => {
    const selection = selectionActions.selectionActionState
    if (
      selection?.mappingConfidence === 'exact' &&
      (selection.startSegmentId ||
        selection.endSegmentId ||
        (selection.sourceStart != null && selection.sourceEnd != null))
    ) {
      return ['selection', 'from-here', 'current-section', 'full-item']
    }
    return ['selection', 'full-item']
  }, [selectionActions.selectionActionState])

  // Now we have shouldShowEmbeddedPlayer from modals, re-run rendering with correct value
  // Actually, we need to use the modals result. Let's restructure:
  // The rendering hook needs shouldShowEmbeddedPlayer which comes from modals.
  // Since hooks can't be called conditionally, we use the modals value directly.
  // We already called rendering with false, but the only thing that uses shouldShowEmbeddedPlayer
  // in the rendering hook is hasClickableTranscriptTimestamps. Let's fix this by
  // re-computing it here.
  const hasClickableTranscriptTimestamps = useMemo(
    () => modals.shouldShowEmbeddedPlayer && rendering.hasTranscriptTimingLines,
    [modals.shouldShowEmbeddedPlayer, rendering.hasTranscriptTimingLines]
  )

  // --- Hook: Transcript Display ---
  const transcript = useTranscriptDisplay({
    displayContent: rendering.displayContent,
    content,
    selectedMedia,
    effectiveRenderMode: rendering.effectiveRenderMode,
    shouldHideTranscriptTimings: rendering.shouldHideTranscriptTimings,
    hasClickableTranscriptTimestamps,
    activeReadAlongSegmentId: readAlong.activeSegmentId,
    contentScrollContainerRef,
    rootContainerRef,
    mediaPlayerRef: modals.mediaPlayerRef,
    t
  })

  const transcriptReadAlongSegments = useMemo(
    () =>
      transcript.readAlongSegments.filter(
        (segment) => segment.kind === 'transcript-line'
      ),
    [transcript.readAlongSegments]
  )

  useEffect(() => {
    const activeSegmentId = readAlong.activeSegmentId
    if (!activeSegmentId || transcript.normalizedFindQuery) return
    const body = contentBodyRef.current
    if (!body) return

    const activeNode = Array.from(
      body.querySelectorAll<HTMLElement>('[data-read-along-segment-id]')
    ).find((node) => node.dataset.readAlongSegmentId === activeSegmentId)
    if (!activeNode) return

    const container = contentScrollContainerRef.current
    const nodeRect = activeNode.getBoundingClientRect()
    const viewportRect = container?.getBoundingClientRect()
    const isOutside = viewportRect
      ? nodeRect.top < viewportRect.top || nodeRect.bottom > viewportRect.bottom
      : nodeRect.top < 0 ||
        nodeRect.bottom > (typeof window !== 'undefined' ? window.innerHeight : 0)

    if (isOutside && typeof activeNode.scrollIntoView === 'function') {
      activeNode.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [
    readAlong.activeSegmentId,
    transcript.normalizedFindQuery,
    transcript.visiblePlainContentChars
  ])

  // --- Hook: Reading Progress ---
  const readingProgress = useReadingProgress({
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
    isDetailLoading,
    t
  })

  // --- Hook: Content Metadata ---
  const copyTextWithToasts = useCallback(async (
    text: string,
    successKey: string,
    defaultSuccess: string
  ) => {
    if (!text) return
    if (!navigator.clipboard?.writeText) {
      message.error(
        t('mediaPage.copyNotSupported', 'Copy is not supported here')
      )
      return
    }
    try {
      await navigator.clipboard.writeText(text)
      message.success(t(successKey, { defaultValue: defaultSuccess }))
    } catch (err) {
      console.error('Failed to copy text:', err)
      message.error(t('mediaPage.copyFailed', 'Failed to copy'))
    }
  }, [t])

  const handleCopyContent = useCallback(() => {
    if (!rendering.displayContent) return
    copyTextWithToasts(
      rendering.displayContent,
      'mediaPage.contentCopied',
      'Content copied'
    )
  }, [rendering.displayContent, copyTextWithToasts])

  const handleCopyMetadata = useCallback(() => {
    if (!selectedMedia) return
    const metadata = {
      id: selectedMedia.id,
      title: selectedMedia.title,
      type: selectedMedia.meta?.type,
      source: selectedMedia.meta?.source,
      duration: selectedMedia.meta?.duration
    }
    copyTextWithToasts(
      JSON.stringify(metadata, null, 2),
      'mediaPage.metadataCopied',
      'Metadata copied'
    )
  }, [selectedMedia, copyTextWithToasts])

  const contentMetadata = useContentMetadata({
    selectedMedia,
    content,
    mediaDetail,
    isNote,
    editState,
    modals,
    onChatWithMedia,
    onChatAboutMedia,
    onGenerateFlashcardsFromContent,
    onCreateNoteWithContent,
    onOpenInMultiReview,
    handleCopyContent,
    handleCopyMetadata,
    t
  })
  const {
    wordCount,
    charCount,
    paragraphCount,
    ingestedAt,
    lastModifiedAt,
    ingestedLabel,
    lastModifiedLabel,
    readingTimeLabel,
    safeMetadataEntries,
    processingStatusBadges,
    hasDetailedMetadata,
    chatWithLabel,
    actionMenuItems
  } = contentMetadata

  // Derived values
  const showDeveloperTools = useMemo(() => {
    try {
      const runtimeEnv = (import.meta as any)?.env || {}
      return shouldShowMediaDeveloperTools(runtimeEnv)
    } catch {
      return false
    }
  }, [])

  const CONTENT_COLLAPSE_THRESHOLD = 2500
  const shouldShowExpandToggle =
    rendering.displayContent && rendering.displayContent.length > CONTENT_COLLAPSE_THRESHOLD

  const toggleSection = (section: string) => {
    void setCollapsedSections((prev) => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  if (!selectedMedia || editState.isAwaitingSelectionUpdate) {
    const isLibraryEmpty = isLibraryEmptyProp && !editState.isAwaitingSelectionUpdate

    return (
      <div className="flex-1 flex items-center justify-center bg-bg" data-testid="content-viewer-empty">
        <div className="text-center max-w-md px-6">
          <div className="mb-6 flex justify-center">
            <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary/10 to-primary/20 flex items-center justify-center">
              <FileSearch className="w-16 h-16 text-primary" />
            </div>
          </div>
          <h2 className="mb-2 text-xl font-semibold text-text">
            {editState.isAwaitingSelectionUpdate
              ? t('common:deleted', { defaultValue: 'Deleted' })
              : isLibraryEmpty
                ? t('review:mediaPage.emptyLibraryTitle', {
                    defaultValue: 'Your library is empty'
                  })
                : t('review:mediaPage.noSelectionTitle', {
                    defaultValue: 'No media item selected'
                  })}
          </h2>
          <p className="text-text-muted">
            {editState.isAwaitingSelectionUpdate
              ? t('review:mediaPage.loadingContent', {
                  defaultValue: 'Loading content...'
                })
              : isLibraryEmpty
                ? t('review:mediaPage.emptyLibraryDescription', {
                    defaultValue:
                      'Ingest your first content to get started.'
                  })
                : t('review:mediaPage.noSelectionDescription', {
                    defaultValue:
                      'Select a media item from the left sidebar to view its content and analyses here.'
                  })}
          </p>
          {!editState.isAwaitingSelectionUpdate && (
            <>
              {isLibraryEmpty ? (
                <button
                  type="button"
                  onClick={() => {
                    requestQuickIngestOpen()
                  }}
                  className="mt-6 inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-primary/90"
                >
                  <UploadCloud className="h-4 w-4" />
                  {t('review:mediaPage.openQuickIngest', {
                    defaultValue: 'Open Quick Ingest'
                  })}
                </button>
              ) : (
                <>
                  <Tooltip
                    title={t('review:mediaPage.keyboardHint', {
                      defaultValue: 'Tip: Use j/k to navigate items, arrow keys to change pages'
                    })}
                  >
                    <button
                      type="button"
                      className="mt-4 inline-flex items-center gap-1 text-xs text-text-subtle hover:text-text-muted"
                    >
                      {t('review:mediaPage.keyboardShortcuts', {
                        defaultValue: 'Keyboard shortcuts'
                      })}
                    </button>
                  </Tooltip>
                  <div className="mt-3">
                    <button
                      type="button"
                      onClick={() => {
                        requestQuickIngestOpen()
                      }}
                      className="inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
                    >
                      <UploadCloud className="h-4 w-4" />
                      {t('review:mediaPage.openQuickIngest', {
                        defaultValue: 'Open Quick Ingest'
                      })}
                    </button>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    )
  }

  return (
    <div ref={setRootContainerRef} className="relative flex-1 flex flex-col bg-bg">
      <div
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
        data-testid="content-selection-live-region"
      >
        {readingProgress.contentSelectionAnnouncement}
      </div>
      {/* Compact Header */}
      <div className="px-4 py-2 border-b border-border bg-surface">
        <div className="flex flex-col md:flex-row items-center gap-3">
          {/* Left: Navigation */}
          <div className="flex items-center gap-1">
            <Tooltip
              title={t('review:reviewPage.prevItem', { defaultValue: 'Previous' })}
            >
              <button
                onClick={onPrevious}
                disabled={!hasPrevious}
                className="p-1.5 text-text-muted hover:bg-surface2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label={t('review:reviewPage.prevItem', { defaultValue: 'Previous' })}
                title={t('review:reviewPage.prevItem', { defaultValue: 'Previous' })}
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
            </Tooltip>
            <span className="text-xs text-text-muted tabular-nums min-w-[40px] text-center">
              {totalResults > 0 ? `${currentIndex + 1}/${totalResults}` : '0/0'}
            </span>
            <Tooltip
              title={t('review:reviewPage.nextItem', { defaultValue: 'Next' })}
            >
              <button
                onClick={onNext}
                disabled={!hasNext}
                className="p-1.5 text-text-muted hover:bg-surface2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label={t('review:reviewPage.nextItem', { defaultValue: 'Next' })}
                title={t('review:reviewPage.nextItem', { defaultValue: 'Next' })}
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </Tooltip>
          </div>

          {/* Center: Title */}
          <Tooltip title={selectedMedia.title || ''} placement="bottom">
            <h3 className="flex-1 text-sm font-medium text-text truncate text-center px-2 max-w-[300px] md:max-w-none">
              {selectedMedia.title || `${selectedMedia.kind} ${selectedMedia.id}`}
            </h3>
          </Tooltip>

          {/* Right: Chat Button + Actions Dropdown */}
          <div className="flex items-center gap-1">
            {!isNote && onChatWithMedia && (
              <Tooltip
                title={t('review:reviewPage.chatWithMediaTooltipClarified', {
                  defaultValue:
                    'Chat with this media by sending its full content to the composer.'
                })}
              >
                <button
                  onClick={onChatWithMedia}
                  className="p-1.5 text-text-muted hover:bg-surface2 rounded"
                  aria-label={chatWithLabel}
                  title={chatWithLabel}
                >
                  <MessageSquare className="w-4 h-4" />
                </button>
              </Tooltip>
            )}
            {!isNote && (
              <Tooltip title={t('review:mediaPage.analyzeButtonTooltip', { defaultValue: 'Generate AI analysis of this content' })}>
                <button
                  onClick={() => modals.setAnalysisModalOpen(true)}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-primaryStrong bg-primary/10 hover:bg-primary/20 rounded transition-colors"
                  aria-label={t('review:mediaPage.analyzeButton', { defaultValue: 'Analyze' })}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  {t('review:mediaPage.analyzeButton', { defaultValue: 'Analyze' })}
                </button>
              </Tooltip>
            )}
            <Dropdown
              menu={{ items: actionMenuItems }}
              trigger={['click']}
              placement="bottomRight"
            >
              <button
                className="p-1.5 text-text-muted hover:bg-surface2 rounded"
                aria-label={t('review:mediaPage.actionsLabel', {
                  defaultValue: 'Actions'
                })}
                title={t('review:mediaPage.actionsLabel', {
                  defaultValue: 'Actions'
                })}
              >
                <MoreHorizontal className="w-4 h-4" />
              </button>
            </Dropdown>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div
        ref={contentScrollContainerRef}
        className="flex-1 overflow-y-auto p-4"
        data-testid="content-scroll-container"
      >
        {isDetailLoading ? (
          <div
            className="flex flex-col items-center justify-center h-64 text-text-muted"
            role="status"
            aria-live="polite"
          >
            <Loader2 className="w-8 h-8 animate-spin mb-3" />
            <span className="text-sm">
              {t('review:mediaPage.loadingContent', { defaultValue: 'Loading content...' })}
            </span>
          </div>
        ) : (
        <div className="max-w-4xl mx-auto">
          {/* Meta Row */}
          <div
            className="flex items-center gap-3 flex-wrap text-xs text-text-muted mb-3"
            data-testid="media-metadata-bar"
          >
            {selectedMedia.meta?.type && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-surface2 text-text capitalize font-medium">
                {selectedMedia.meta.type}
              </span>
            )}
            {selectedMedia.meta?.source && (
              <span className="truncate max-w-[200px]" title={selectedMedia.meta.source}>
                {selectedMedia.meta.source}
              </span>
            )}
            {selectedMedia.meta?.author && (
              <span
                className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                data-testid="media-author"
                title={selectedMedia.meta.author}
              >
                <User className="w-3 h-3" />
                <span className="truncate max-w-[200px]">{selectedMedia.meta.author}</span>
              </span>
            )}
            {selectedMedia.meta?.published_at && (
              <span
                className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                data-testid="media-published-date"
              >
                {t('review:mediaPage.publishedLabel', { defaultValue: 'Published' })}{' '}
                {(() => {
                  try {
                    const d = new Date(selectedMedia.meta.published_at)
                    return Number.isNaN(d.getTime())
                      ? selectedMedia.meta.published_at
                      : d.toLocaleDateString(undefined, { year: 'numeric', month: 'short' })
                  } catch {
                    return selectedMedia.meta.published_at
                  }
                })()}
              </span>
            )}
            {(() => {
              const rawDuration = selectedMedia.meta?.duration as
                | number
                | string
                | null
                | undefined
              const durationSeconds =
                typeof rawDuration === 'number'
                  ? rawDuration
                  : typeof rawDuration === 'string'
                    ? Number(rawDuration)
                    : null
              const durationLabel = formatDuration(durationSeconds)
              if (!durationLabel) return null
              return (
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {durationLabel}
                </span>
              )
            })()}
            {ingestedLabel && (
              <span
                className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                data-testid="media-ingested-date"
                title={ingestedAt || undefined}
              >
                {t('review:mediaPage.ingestedLabel', { defaultValue: 'Ingested' })}{' '}
                {ingestedLabel}
              </span>
            )}
            {lastModifiedLabel && (
              <span
                className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                data-testid="media-last-modified-date"
                title={lastModifiedAt || undefined}
              >
                {t('review:mediaPage.lastModifiedLabel', { defaultValue: 'Updated' })}{' '}
                {lastModifiedLabel}
              </span>
            )}
            <span className="flex items-center gap-1">
              <FileText className="w-3 h-3" />
              {wordCount.toLocaleString()} {t('review:mediaPage.words', { defaultValue: 'words' })}
            </span>
            {readingTimeLabel && (
              <span
                className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                data-testid="media-reading-time"
              >
                {readingTimeLabel}
              </span>
            )}
            {typeof readingProgress.progressPercent === 'number' && Number.isFinite(readingProgress.progressPercent) && (
              <span
                className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                data-testid="media-reading-progress"
                title={t('review:mediaPage.readingProgressTooltip', {
                  defaultValue: 'Reading progress'
                })}
              >
                {t('review:mediaPage.readingProgressLabel', {
                  defaultValue: '{{percent}}% read',
                  percent: Math.round(readingProgress.progressPercent)
                })}
              </span>
            )}
            {/* Transcription model */}
            {(() => {
              const model =
                mediaDetail?.transcription_model ??
                mediaDetail?.metadata?.transcription_model ??
                mediaDetail?.safe_metadata?.transcription_model ??
                mediaDetail?.processing?.transcription_model ??
                selectedMedia?.meta?.transcription_model
              if (typeof model !== 'string' || !model.trim()) return null
              return (
                <span
                  className="inline-flex items-center gap-1 rounded bg-surface2 px-1.5 py-0.5"
                  data-testid="media-transcription-model"
                  title={t('review:mediaPage.transcriptionModelTooltip', {
                    defaultValue: 'Transcription model used'
                  })}
                >
                  {t('review:mediaPage.transcriptionModelBadge', {
                    defaultValue: 'STT: {{model}}',
                    model: model.trim()
                  })}
                </span>
              )
            })()}
            {processingStatusBadges.map((badge) => (
              <span
                key={badge.key}
                className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 ${processingStatusClass(
                  badge.status
                )}`}
                data-testid={`media-processing-status-${badge.key}`}
                data-status={badge.status}
              >
                {badge.label}: {formatProcessingStatus(badge.status)}
              </span>
            ))}
          </div>

          <div className="mb-4 rounded-lg border border-border bg-surface">
            <button
              type="button"
              onClick={() => modals.setMetadataDetailsExpanded((prev) => !prev)}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-xs font-medium text-text hover:bg-surface2"
              aria-expanded={modals.metadataDetailsExpanded}
              data-testid="metadata-details-toggle"
            >
              <span>
                {t('review:mediaPage.metadataDetailsLabel', {
                  defaultValue: 'Metadata details'
                })}
              </span>
              <span className="text-text-muted">
                {modals.metadataDetailsExpanded
                  ? t('review:mediaPage.hideMetadataDetails', {
                      defaultValue: 'Hide'
                    })
                  : t('review:mediaPage.showMetadataDetails', {
                      defaultValue: hasDetailedMetadata ? 'Show' : 'Open'
                    })}
              </span>
            </button>
            {modals.metadataDetailsExpanded && (
              <div
                className="space-y-3 border-t border-border px-3 py-2 text-xs"
                data-testid="metadata-details-panel"
              >
                <div className="space-y-1">
                  <p className="font-medium text-text">
                    {t('review:mediaPage.processingStatusLabel', {
                      defaultValue: 'Processing status'
                    })}
                  </p>
                  {processingStatusBadges.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {processingStatusBadges.map((badge) => (
                        <span
                          key={`detail-${badge.key}`}
                          className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 ${processingStatusClass(
                            badge.status
                          )}`}
                          data-testid={`metadata-processing-${badge.key}`}
                          data-status={badge.status}
                        >
                          {badge.label}: {formatProcessingStatus(badge.status)}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-text-muted" data-testid="metadata-processing-empty">
                      {t('review:mediaPage.processingStatusEmpty', {
                        defaultValue: 'Processing status is not available for this item.'
                      })}
                    </p>
                  )}
                </div>
                <div className="space-y-1">
                  <p className="font-medium text-text">
                    {t('review:mediaPage.additionalMetadataLabel', {
                      defaultValue: 'Additional metadata'
                    })}
                  </p>
                  {safeMetadataEntries.length > 0 ? (
                    <dl className="grid gap-1">
                      {safeMetadataEntries.map((entry) => (
                        <div
                          key={entry.key}
                          className="grid grid-cols-[minmax(0,150px)_1fr] gap-2 rounded bg-surface2 px-2 py-1"
                          data-testid={`metadata-field-${entry.key}`}
                        >
                          <dt className="text-text-muted">{entry.label}</dt>
                          <dd className="break-words text-text">{entry.value}</dd>
                        </div>
                      ))}
                    </dl>
                  ) : (
                    <p className="text-text-muted" data-testid="metadata-safe-empty">
                      {t('review:mediaPage.additionalMetadataEmpty', {
                        defaultValue: 'No additional metadata is available for this item.'
                      })}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Keywords - Compact */}
          <div className="mb-4">
            <Select
              mode="tags"
              allowClear
              data-testid="media-keywords-select"
              placeholder={
                editState.savingKeywords
                  ? t('review:mediaPage.savingKeywords', { defaultValue: 'Saving...' })
                  : t('review:mediaPage.keywordsPlaceholder', { defaultValue: 'Add keywords...' })
              }
              className="w-full"
              size="small"
              value={editState.editingKeywords}
              onChange={(vals) => {
                editState.handleSaveKeywords(vals as string[])
              }}
              loading={editState.savingKeywords}
              disabled={editState.savingKeywords}
              tokenSeparators={[',']}
              suffixIcon={editState.savingKeywords ? <Spin size="small" /> : undefined}
            />
          </div>

          {/* Main Content */}
          <div className="bg-surface border border-border rounded-lg mb-2 overflow-hidden">
            <div className="flex items-center justify-between px-3 py-2 bg-surface2">
              <button
                onClick={() => toggleSection('content')}
                className="flex items-center gap-2 hover:bg-surface -ml-1 px-1 rounded transition-colors"
                title={t('review:mediaPage.content', { defaultValue: 'Content' })}
              >
                <span className="text-sm font-medium text-text">
                  {t('review:mediaPage.content', { defaultValue: 'Content' })}
                </span>
                {collapsedSections.content ? (
                  <ChevronDown className="w-4 h-4 text-text-subtle" />
                ) : (
                  <ChevronUp className="w-4 h-4 text-text-subtle" />
                )}
              </button>
                <div className="flex items-center gap-1">
                  {readingProgress.navigationTargetDescription ? (
                    <span className="rounded bg-surface px-2 py-0.5 text-[11px] text-text-muted">
                      {readingProgress.navigationTargetDescription}
                    </span>
                ) : null}
                {showContentDisplayModeSelector && onContentDisplayModeChange ? (
                  <Select
                    size="small"
                    value={contentDisplayMode}
                    options={rendering.displayModeOptions}
                    onChange={(value) =>
                      onContentDisplayModeChange(value as MediaNavigationFormat)
                    }
                    className="min-w-[132px]"
                    popupMatchSelectWidth={false}
                    aria-label={t('review:mediaPage.displayMode', {
                      defaultValue: 'Display mode'
                    })}
                  />
                ) : null}
                <div
                  className="inline-flex items-center rounded-md border border-border bg-surface p-0.5"
                  role="group"
                  aria-label={t('review:mediaPage.textSize', {
                    defaultValue: 'Text size'
                  })}
                >
                  {TEXT_SIZE_CONTROL_OPTIONS.map((option) => {
                    const isActive = option.value === rendering.resolvedTextSizePreset
                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => {
                          if (option.value === rendering.resolvedTextSizePreset) return
                          void setTextSizePreset(option.value)
                        }}
                        className={`rounded px-1.5 py-0.5 text-[11px] font-medium transition-colors ${
                          isActive
                            ? 'bg-primary text-white'
                            : 'text-text-muted hover:bg-surface2 hover:text-text'
                        }`}
                        aria-pressed={isActive}
                        aria-label={t('review:mediaPage.textSizeOption', {
                          defaultValue: 'Text size {{size}}',
                          size: option.label
                        })}
                        title={t('review:mediaPage.textSizeOption', {
                          defaultValue: 'Text size {{size}}',
                          size: option.label
                        })}
                      >
                        {option.label}
                      </button>
                    )
                  })}
                </div>
                {rendering.hasTranscriptTimingLines ? (
                  <button
                    type="button"
                    onClick={() => {
                      void setHideTranscriptTimings((prev) => !(prev ?? true))
                    }}
                    className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-muted hover:bg-surface2 hover:text-text"
                    aria-label={
                      rendering.shouldHideTranscriptTimings
                        ? t('review:mediaPage.showTimings', {
                            defaultValue: 'Show timings'
                          })
                        : t('review:mediaPage.hideTimings', {
                            defaultValue: 'Hide timings'
                          })
                    }
                    title={
                      rendering.shouldHideTranscriptTimings
                        ? t('review:mediaPage.showTimings', {
                            defaultValue: 'Show timings'
                          })
                        : t('review:mediaPage.hideTimings', {
                            defaultValue: 'Hide timings'
                          })
                    }
                  >
                    {rendering.shouldHideTranscriptTimings
                      ? t('review:mediaPage.showTimings', {
                          defaultValue: 'Show timings'
                        })
                      : t('review:mediaPage.hideTimings', {
                          defaultValue: 'Hide timings'
                        })}
                  </button>
                ) : null}
                {!isNote && content && (
                  <button
                    onClick={editState.openContentEditModal}
                    className="p-1 text-text-muted hover:text-text transition-colors"
                    title={t('review:mediaPage.editContent', {
                      defaultValue: 'Edit content'
                    })}
                    aria-label={t('review:mediaPage.editContent', {
                      defaultValue: 'Edit content'
                    })}
                  >
                    <Edit3 className="w-4 h-4" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => transcript.setFindBarOpen(true)}
                  className="p-1 text-text-muted hover:text-text transition-colors"
                  title={t('review:mediaPage.findInContent', {
                    defaultValue: 'Find in content'
                  })}
                  aria-label={t('review:mediaPage.findInContent', {
                    defaultValue: 'Find in content'
                  })}
                  data-testid="content-find-toggle"
                >
                  <FileSearch className="w-4 h-4" />
                </button>
                {/* Expand/collapse toggle for long content */}
                {!collapsedSections.content && shouldShowExpandToggle && (
                  <button
                    onClick={() => modals.setContentExpanded((v) => !v)}
                    className="p-1 text-text-muted hover:text-text transition-colors"
                    title={
                      modals.contentExpanded
                        ? t('review:mediaPage.collapse', {
                            defaultValue: 'Collapse'
                          })
                        : t('review:mediaPage.expand', {
                            defaultValue: 'Expand'
                          })
                    }
                  >
                    {modals.contentExpanded ? (
                      <Minimize2 className="w-4 h-4" />
                    ) : (
                      <Expand className="w-4 h-4" />
                    )}
                  </button>
                )}
              </div>
            </div>
            {!collapsedSections.content && (
              <div className="p-3 bg-surface animate-in fade-in slide-in-from-top-1 duration-150">
                {transcript.findBarOpen && (
                  <div
                    className="mb-3 rounded-md border border-border bg-surface2 px-2 py-1.5"
                    data-testid="content-find-bar"
                  >
                    <div className="flex items-center gap-2">
                      <label htmlFor="content-find-input" className="sr-only">
                        {t('review:mediaPage.findInContent', {
                          defaultValue: 'Find in content'
                        })}
                      </label>
                      <input
                        id="content-find-input"
                        ref={transcript.findInputRef}
                        type="text"
                        className="h-7 w-full rounded border border-border bg-surface px-2 text-xs text-text outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                        placeholder={t('review:mediaPage.findPlaceholder', {
                          defaultValue: 'Find in content'
                        })}
                        value={transcript.findQuery}
                        onChange={(event) => transcript.setFindQuery(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === 'Escape') {
                            event.preventDefault()
                            transcript.closeFindBar()
                            return
                          }
                          if (event.key === 'Enter') {
                            event.preventDefault()
                            transcript.moveFindMatch(event.shiftKey ? -1 : 1)
                          }
                        }}
                        data-testid="content-find-input"
                      />
                      <span
                        className="whitespace-nowrap text-[11px] text-text-muted"
                        data-testid="content-find-count"
                      >
                        {transcript.findMatchCount > 0 && transcript.activeFindMatchIndex >= 0
                          ? `${transcript.activeFindMatchIndex + 1}/${transcript.findMatchCount}`
                          : `0/${transcript.findMatchCount}`}
                      </span>
                      <button
                        type="button"
                        onClick={() => transcript.moveFindMatch(-1)}
                        className="inline-flex h-7 w-7 items-center justify-center rounded border border-border bg-surface text-text-muted hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
                        aria-label={t('review:mediaPage.findPrevious', {
                          defaultValue: 'Previous match'
                        })}
                        title={t('review:mediaPage.findPrevious', {
                          defaultValue: 'Previous match'
                        })}
                        disabled={transcript.findMatchCount === 0}
                        data-testid="content-find-prev"
                      >
                        <ChevronUp className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => transcript.moveFindMatch(1)}
                        className="inline-flex h-7 w-7 items-center justify-center rounded border border-border bg-surface text-text-muted hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
                        aria-label={t('review:mediaPage.findNext', {
                          defaultValue: 'Next match'
                        })}
                        title={t('review:mediaPage.findNext', {
                          defaultValue: 'Next match'
                        })}
                        disabled={transcript.findMatchCount === 0}
                        data-testid="content-find-next"
                      >
                        <ChevronDown className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={transcript.closeFindBar}
                        className="inline-flex h-7 items-center rounded border border-border bg-surface px-2 text-[11px] text-text-muted hover:text-text"
                        aria-label={t('common:close', { defaultValue: 'Close' })}
                        title={t('common:close', { defaultValue: 'Close' })}
                        data-testid="content-find-close"
                      >
                        {t('common:close', { defaultValue: 'Close' })}
                      </button>
                    </div>
                  </div>
                )}
                {modals.shouldShowEmbeddedPlayer ? (
                  <div className="mb-3 rounded-md border border-border bg-surface2 p-2">
                    {modals.embeddedMediaLoading ? (
                      <div
                        className="flex items-center gap-2 text-xs text-text-muted"
                        data-testid="embedded-media-loading"
                      >
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        {t('review:mediaPage.loadingMediaPreview', {
                          defaultValue: 'Loading media preview...'
                        })}
                      </div>
                    ) : modals.embeddedMediaUrl ? (
                      modals.mediaType === 'video' ? (
                        <video
                          ref={(node) => {
                            modals.mediaPlayerRef.current = node
                          }}
                          src={modals.embeddedMediaUrl}
                          controls
                          preload="metadata"
                          className="w-full rounded"
                          data-testid="embedded-video-player"
                        />
                      ) : (
                        <audio
                          ref={(node) => {
                            modals.mediaPlayerRef.current = node
                          }}
                          src={modals.embeddedMediaUrl}
                          controls
                          preload="metadata"
                          className="w-full"
                          data-testid="embedded-audio-player"
                        />
                      )
                    ) : modals.embeddedMediaError ? (
                      <p className="m-0 text-xs text-warn">{modals.embeddedMediaError}</p>
                    ) : null}
                  </div>
                ) : null}
                <div
                  ref={contentBodyRef}
                  className={`text-sm text-text leading-relaxed ${
                    !modals.contentExpanded && shouldShowExpandToggle ? 'max-h-64 overflow-hidden relative' : ''
                  }`}
                  onMouseUp={selectionActions.handleContentSelectionEvent}
                  onKeyUp={selectionActions.handleContentSelectionEvent}
                >
                  {selectionActions.selectionActionState ? (
                    <MediaReadAlongPopover
                      anchorRect={selectionActions.selectionActionState.anchorRect}
                      viewportRect={
                        contentScrollContainerRef.current?.getBoundingClientRect() ??
                        rootContainerRef.current?.getBoundingClientRect() ??
                        null
                      }
                      supportedScopes={supportedReadAlongScopes}
                      onReadScope={handleStartReadAlongScope}
                      onAnnotate={selectionActions.applyAnnotationSelection}
                      t={t}
                    />
                  ) : null}
                  <MediaReadAlongTransport
                    state={readAlong.state}
                    anchorRect={readAlongTransportAnchorRef.current}
                    viewportRect={
                      contentScrollContainerRef.current?.getBoundingClientRect() ??
                      rootContainerRef.current?.getBoundingClientRect() ??
                      null
                    }
                    onToggle={handleToggleReadAlong}
                    onStop={handleStopReadAlong}
                    onRetry={readAlong.retry}
                    onSkip={() => readAlong.skip('next')}
                    t={t}
                  />
                  {rendering.effectiveRenderMode === 'plain' ? (
                    transcript.shouldRenderTranscriptTimestampChips ? (
                      <div
                        className={`m-0 space-y-1 whitespace-pre-wrap text-text font-mono ${rendering.contentBodyTypographyClass}`}
                      >
                        {(() => {
                          let readAlongSegmentIndex = 0
                          return rendering.transcriptLines.map((line, lineIndex) => {
                            const parsed = parseLeadingTranscriptTiming(line)
                            if (!parsed) {
                              return (
                                <div key={`line-${lineIndex}`}>
                                  {line.length > 0 ? line : '\u00A0'}
                                </div>
                              )
                            }
                            const timestamp = parsed.timestamp
                            const readAlongSegment =
                              parsed.text.trim().length > 0
                                ? transcriptReadAlongSegments[readAlongSegmentIndex++]
                                : undefined
                            const isActive =
                              readAlongSegment?.id === readAlong.activeSegmentId
                            return (
                              <div key={`line-${lineIndex}`} className="flex flex-wrap items-start gap-2">
                                <button
                                  type="button"
                                  onClick={() => transcript.handleTranscriptTimestampSeek(timestamp)}
                                  className="rounded border border-border bg-surface px-1.5 py-0.5 text-[11px] text-primary hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                                  aria-label={t('review:mediaPage.seekToTimestamp', {
                                    defaultValue: 'Seek to {{timestamp}}',
                                    timestamp
                                  })}
                                >
                                  {timestamp}
                                </button>
                                <span className="flex-1 whitespace-pre-wrap break-words">
                                  {parsed.leadingWhitespace}
                                  {parsed.separator}
                                  {readAlongSegment ? (
                                    <span
                                      data-read-along-segment-id={readAlongSegment.id}
                                      data-read-along-active={isActive ? 'true' : undefined}
                                      className={
                                        isActive
                                          ? 'rounded bg-primary/20 px-0.5 text-text'
                                          : undefined
                                      }
                                    >
                                      {parsed.text}
                                    </span>
                                  ) : (
                                    parsed.text
                                  )}
                                </span>
                              </div>
                            )
                          })
                        })()}
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <pre
                          className={`whitespace-pre-wrap text-text font-mono m-0 ${rendering.contentBodyTypographyClass}`}
                        >
                          {transcript.highlightedPlainContent}
                        </pre>
                        {transcript.hasUnrenderedPlainContent ? (
                          <div
                            className="flex flex-wrap items-center justify-between gap-2 rounded border border-border bg-surface2 px-2 py-1 text-[11px] text-text-muted"
                            data-testid="large-content-window-status"
                            data-visible-chars={transcript.visiblePlainContentChars}
                            data-total-chars={rendering.displayContent.length}
                          >
                            <span data-testid="large-content-window-progress">
                              {t('review:mediaPage.largeContentProgress', {
                                defaultValue: `Showing ${transcript.visiblePlainContentChars}/${rendering.displayContent.length} characters`
                              })}
                            </span>
                            <button
                              type="button"
                              onClick={transcript.loadMorePlainContent}
                              className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-primary hover:bg-surface2"
                              data-testid="large-content-window-load-more"
                            >
                              {t('review:mediaPage.largeContentLoadMore', {
                                defaultValue: 'Load more'
                              })}
                            </button>
                          </div>
                        ) : null}
                      </div>
                    )
                  ) : rendering.effectiveRenderMode === 'html' ? (
                    rendering.displayContent ? (
                      <div
                        className={`${rendering.richTextTypographyClass} break-words dark:prose-invert max-w-none prose-p:leading-relaxed`}
                        role="region"
                        aria-label={t('review:mediaPage.contentRegion', { defaultValue: 'Media content' })}
                        dangerouslySetInnerHTML={{
                          __html: rendering.sanitizedRichContent
                        }}
                      />
                    ) : (
                      <p className="m-0 text-sm text-text-muted">
                        {t('review:mediaPage.noContent', {
                          defaultValue: 'No content available'
                        })}
                      </p>
                    )
                  ) : (
                    <Suspense
                      fallback={
                        <div
                          className={`whitespace-pre-wrap text-text ${rendering.contentBodyTypographyClass}`}
                        >
                          {markdownFallbackContent}
                        </div>
                      }
                    >
                      <LazyMarkdownPreview
                        content={markdownFallbackContent}
                        size={rendering.markdownPreviewSize}
                      />
                    </Suspense>
                  )}
                  {/* Fade overlay when collapsed */}
                  {!modals.contentExpanded && shouldShowExpandToggle && (
                    <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-surface to-transparent" />
                  )}
                </div>
                {/* Show more/less button */}
                {shouldShowExpandToggle && (
                  <button
                    onClick={() => modals.setContentExpanded(v => !v)}
                    className="mt-2 text-xs text-primary hover:underline"
                    title={
                      modals.contentExpanded
                        ? t('review:mediaPage.showLess', { defaultValue: 'Show less' })
                        : t('review:mediaPage.showMore', {
                            defaultValue: `Show more (${Math.round(rendering.displayContent.length / 1000)}k chars)`
                          })
                    }
                  >
                    {modals.contentExpanded
                      ? t('review:mediaPage.showLess', { defaultValue: 'Show less' })
                      : t('review:mediaPage.showMore', {
                          defaultValue: `Show more (${Math.round(rendering.displayContent.length / 1000)}k chars)`
                        })}
                  </button>
                )}
              </div>
            )}
          </div>

          {/* Analysis - only for media, not notes */}
          {!isNote && (
            <div className="bg-surface border border-border rounded-lg mb-2 overflow-hidden">
              <div className="flex items-center justify-between px-3 py-2 bg-surface2">
              <button
                onClick={() => toggleSection('analysis')}
                className="flex items-center gap-2 hover:bg-surface -ml-1 px-1 rounded transition-colors"
                title={t('review:reviewPage.analysisTitle', { defaultValue: 'Analysis' })}
              >
                <span className="text-sm font-medium text-text">
                  {t('review:reviewPage.analysisTitle', { defaultValue: 'Analysis' })}
                </span>
                  {collapsedSections.analysis ? (
                    <ChevronDown className="w-4 h-4 text-text-subtle" />
                  ) : (
                    <ChevronUp className="w-4 h-4 text-text-subtle" />
                  )}
                </button>
                <div className="flex items-center gap-2">
                <button
                  onClick={() => modals.setAnalysisModalOpen(true)}
                  className="px-2 py-1 bg-primary hover:bg-primaryStrong text-white rounded text-xs font-medium flex items-center gap-1 transition-colors"
                  title={t('review:mediaPage.generateAnalysisHint', {
                    defaultValue: 'Generate new analysis'
                  })}
                >
                  <Sparkles className="w-3 h-3" />
                  {t('review:mediaPage.generateAnalysis', { defaultValue: 'Generate' })}
                </button>
                  {editState.existingAnalyses.length > 0 && (
                    <>
                      {/* Send to chat button */}
                      {onSendAnalysisToChat && (
                        <button
                          onClick={() => {
                            if (editState.activeAnalysis) onSendAnalysisToChat(editState.activeAnalysis.text)
                          }}
                          className="p-1 text-text-muted hover:text-text transition-colors"
                          title={t('review:reviewPage.sendAnalysisToChat', {
                            defaultValue: 'Send analysis to chat'
                          })}
                        >
                          <Send className="w-3.5 h-3.5" />
                          </button>
                      )}
                      {/* Copy analysis button */}
                      <button
                        onClick={() =>
                          editState.activeAnalysis &&
                          copyTextWithToasts(
                            editState.activeAnalysis.text,
                            'mediaPage.analysisCopied',
                            'Analysis copied'
                          )
                        }
                        className="p-1 text-text-muted hover:text-text transition-colors"
                        title={t('review:reviewPage.copyAnalysis', { defaultValue: 'Copy analysis' })}
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                      {/* Edit analysis button */}
                      <button
                        onClick={editState.openAnalysisEditModal}
                        className="p-1 text-text-muted hover:text-text transition-colors"
                        title={t('review:mediaPage.editAnalysis', { defaultValue: 'Edit analysis' })}
                      >
                        <Edit3 className="w-3.5 h-3.5" />
                      </button>
                    </>
                  )}
                  {editState.activeAnalysis && editState.analysisIsLong && (
                    <button
                      onClick={() => {
                        if (collapsedSections.analysis) {
                          toggleSection('analysis')
                          editState.setAnalysisExpanded(true)
                          return
                        }
                        editState.setAnalysisExpanded((v) => !v)
                      }}
                      className="p-1 text-text-muted hover:text-text transition-colors"
                      aria-label={
                        collapsedSections.analysis || !editState.analysisExpanded
                          ? `${t('review:reviewPage.expandAnalysis', { defaultValue: 'Expand' })} ${t('review:reviewPage.analysisTitle', { defaultValue: 'Analysis' })}`
                          : `${t('review:reviewPage.collapseAnalysis', { defaultValue: 'Collapse' })} ${t('review:reviewPage.analysisTitle', { defaultValue: 'Analysis' })}`
                      }
                      title={
                        collapsedSections.analysis || !editState.analysisExpanded
                          ? `${t('review:reviewPage.expandAnalysis', { defaultValue: 'Expand' })} ${t('review:reviewPage.analysisTitle', { defaultValue: 'Analysis' })}`
                          : `${t('review:reviewPage.collapseAnalysis', { defaultValue: 'Collapse' })} ${t('review:reviewPage.analysisTitle', { defaultValue: 'Analysis' })}`
                      }
                    >
                      {collapsedSections.analysis || !editState.analysisExpanded ? (
                        <Expand className="w-4 h-4" />
                      ) : (
                        <Minimize2 className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>
              {!collapsedSections.analysis && (
                <div className="p-3 bg-surface animate-in fade-in slide-in-from-top-1 duration-150">
                  {editState.existingAnalyses.length > 0 ? (
                    <div className="space-y-3">
                      {editState.existingAnalyses.length > 1 && (
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-text-muted">
                            {t('review:mediaPage.analysis', { defaultValue: 'Analysis' })}
                          </span>
                          <Select
                            size="small"
                            value={editState.activeAnalysisIndex}
                            onChange={editState.setActiveAnalysisIndex}
                            className="min-w-[220px]"
                            aria-label={t('review:mediaPage.analysis', { defaultValue: 'Analysis' })}
                          >
                            {editState.existingAnalyses.map((analysis, idx) => (
                              <Select.Option key={idx} value={idx}>
                                {analysis.type}
                              </Select.Option>
                            ))}
                          </Select>
                        </div>
                      )}
                      {editState.activeAnalysis && (() => {
                        const trimmedOptimistic = editState.optimisticAnalysis.trim()
                        const isOptimistic =
                          trimmedOptimistic && editState.activeAnalysis!.text.trim() === trimmedOptimistic
                        return (
                          <div className="space-y-1">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-medium text-text-muted uppercase">
                                  {editState.activeAnalysis!.type}
                                </span>
                                {isOptimistic && (
                                  <span className="text-[10px] uppercase tracking-wide bg-surface2 text-text-muted border border-border rounded px-1.5 py-0.5">
                                    {t('review:mediaPage.analysisPending', {
                                      defaultValue: 'Pending save'
                                    })}
                                  </span>
                                )}
                              </div>
                              <button
                                onClick={() =>
                                  copyTextWithToasts(
                                    editState.activeAnalysis!.text,
                                    'mediaPage.analysisCopied',
                                    'Analysis copied'
                                  )
                                }
                                className="p-0.5 text-text-subtle hover:text-text"
                                aria-label={t('review:mediaPage.copyAnalysis', {
                                  defaultValue: 'Copy analysis to clipboard'
                                })}
                                title={t('review:mediaPage.copyAnalysis', {
                                  defaultValue: 'Copy analysis to clipboard'
                                })}
                              >
                                <Copy className="w-3 h-3" />
                              </button>
                            </div>
                            <div className="text-sm text-text whitespace-pre-wrap leading-relaxed">
                            {editState.analysisShown}
                            </div>
                          </div>
                        )
                      })()}
                    </div>
                  ) : (
                    <div className="text-sm text-text-muted text-center py-4">
                      {t('review:reviewPage.noAnalysis', {
                        defaultValue: 'No analysis yet'
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Statistics */}
          <div className="bg-surface border border-border rounded-lg mb-2 overflow-hidden">
            <button
              onClick={() => toggleSection('statistics')}
              className="w-full flex items-center justify-between px-3 py-2 bg-surface2 hover:bg-surface transition-colors"
              title={t('review:mediaPage.statistics', { defaultValue: 'Statistics' })}
            >
              <span className="text-sm font-medium text-text">
                {t('review:mediaPage.statistics', { defaultValue: 'Statistics' })}
              </span>
              {collapsedSections.statistics ? (
                <ChevronDown className="w-4 h-4 text-text-subtle" />
              ) : (
                <ChevronUp className="w-4 h-4 text-text-subtle" />
              )}
            </button>
            {!collapsedSections.statistics && (
              <div className="p-3 bg-surface animate-in fade-in slide-in-from-top-1 duration-150">
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div className="flex flex-col">
                    <span className="text-text-muted text-xs">
                      {t('review:mediaPage.words', { defaultValue: 'Words' })}
                    </span>
                    <span className="text-text font-medium">
                      {wordCount}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-text-muted text-xs">
                      {t('review:mediaPage.characters', { defaultValue: 'Characters' })}
                    </span>
                    <span className="text-text font-medium">
                      {charCount}
                    </span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-text-muted text-xs">
                      {t('review:mediaPage.paragraphs', { defaultValue: 'Paragraphs' })}
                    </span>
                    <span className="text-text font-medium">
                      {paragraphCount}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Metadata */}
          <div className="bg-surface border border-border rounded-lg mb-2 overflow-hidden">
            <button
              onClick={() => toggleSection('metadata')}
              className="w-full flex items-center justify-between px-3 py-2 bg-surface2 hover:bg-surface transition-colors"
              title={t('review:mediaPage.metadata', { defaultValue: 'Metadata' })}
            >
              <span className="text-sm font-medium text-text">
                {t('review:mediaPage.metadata', { defaultValue: 'Metadata' })}
              </span>
              {collapsedSections.metadata ? (
                <ChevronDown className="w-4 h-4 text-text-subtle" />
              ) : (
                <ChevronUp className="w-4 h-4 text-text-subtle" />
              )}
            </button>
            {!collapsedSections.metadata && (
              <Suspense fallback={null}>
                <LazyContentViewerMetadataSectionBody
                  selectedMedia={selectedMedia}
                  t={t}
                />
              </Suspense>
            )}
          </div>

          {/* Document Intelligence */}
          {!isNote && (
            <Suspense
              fallback={
                <div
                  className="bg-surface border border-border rounded-lg mb-2 overflow-hidden"
                  data-testid="media-intelligence-section"
                >
                  <button
                    onClick={() => toggleSection('intelligence')}
                    className="w-full flex items-center justify-between px-3 py-2 bg-surface2 hover:bg-surface transition-colors"
                    title={t('review:mediaPage.documentIntelligence', {
                      defaultValue: 'Document Intelligence'
                    })}
                    data-testid="media-intelligence-toggle"
                  >
                    <span className="text-sm font-medium text-text">
                      {t('review:mediaPage.documentIntelligence', {
                        defaultValue: 'Document Intelligence'
                      })}
                    </span>
                    {modals.intelligenceSectionCollapsed ? (
                      <ChevronDown className="w-4 h-4 text-text-subtle" />
                    ) : (
                      <ChevronUp className="w-4 h-4 text-text-subtle" />
                    )}
                  </button>
                  {!modals.intelligenceSectionCollapsed ? (
                    <div
                      className="space-y-2 p-3 bg-surface animate-in fade-in slide-in-from-top-1 duration-150"
                      data-testid="media-intelligence-panel"
                    >
                      <div
                        className="text-xs text-text-muted"
                        data-testid="media-intelligence-loading"
                      >
                        {t('review:mediaPage.intelligenceLoading', {
                          defaultValue: 'Loading intelligence data...'
                        })}
                      </div>
                    </div>
                  ) : null}
                </div>
              }
            >
              <LazyContentViewerDocumentIntelligenceSection
                modals={modals}
                onToggleCollapsed={() => toggleSection('intelligence')}
                t={t}
              />
            </Suspense>
          )}

          {/* Version History - only for media */}
          {!isNote && (
            <div className="mb-2">
              {versionHistoryMounted ? (
                <Suspense
                  fallback={
                    <div className="rounded-lg border border-border bg-surface overflow-hidden">
                      <div className="w-full flex items-center justify-between px-3 py-2 bg-surface2 text-text">
                        <div className="flex items-center gap-2">
                          <History className="w-4 h-4 text-text-subtle" />
                          <span className="text-sm font-medium text-text">
                            {t('mediaPage.versionHistory', 'Version History')}
                          </span>
                        </div>
                        <Loader2 className="w-4 h-4 animate-spin text-text-subtle" />
                      </div>
                    </div>
                  }
                >
                  <LazyVersionHistoryPanel
                    mediaId={selectedMedia.id}
                    currentContent={content}
                    currentPrompt={editState.derivedPrompt}
                    currentAnalysis={editState.derivedAnalysisContent}
                    defaultExpanded
                    onVersionLoad={(vContent, vAnalysis, vPrompt, vNum) => {
                      if (vAnalysis) {
                        editState.setEditingAnalysisText(vAnalysis)
                        editState.setAnalysisEditModalOpen(true)
                      }
                    }}
                    onRefresh={onRefreshMedia}
                    onShowDiff={modals.handleShowDiff}
                  />
                </Suspense>
              ) : (
                <div className="rounded-lg border border-border bg-surface overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setVersionHistoryMounted(true)}
                    className="w-full flex items-center justify-between px-3 py-2 bg-surface2 hover:bg-surface transition-colors text-text"
                    title={t('mediaPage.versionHistory', 'Version History')}
                    aria-label={t('mediaPage.versionHistory', 'Version History')}
                  >
                    <div className="flex items-center gap-2">
                      <History className="w-4 h-4 text-text-subtle" />
                      <span className="text-sm font-medium text-text">
                        {t('mediaPage.versionHistory', 'Version History')}
                      </span>
                    </div>
                    <ChevronDown className="w-4 h-4 text-text-subtle" />
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Developer Tools */}
          {showDeveloperTools ? (
            <Suspense fallback={null}>
              <LazyDeveloperToolsSection
                data={mediaDetail}
                label={t('review:mediaPage.developerTools', {
                  defaultValue: 'Developer Tools'
                })}
              />
            </Suspense>
          ) : null}
          {selectedMedia && onDeleteItem && (
            <div className="mt-2">
              <button
                type="button"
                onClick={editState.handleDeleteItem}
                disabled={editState.deletingItem}
                className="w-full inline-flex items-center justify-center gap-2 rounded-md border border-danger/30 px-3 py-2 text-sm text-danger hover:bg-danger/10 disabled:opacity-60"
                title={t('review:mediaPage.deleteItem', { defaultValue: 'Delete item' })}
              >
                {editState.deletingItem ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
                {editState.deletingItem
                  ? t('review:mediaPage.deletingItem', { defaultValue: 'Deleting...' })
                  : t('review:mediaPage.deleteItem', { defaultValue: 'Delete item' })}
              </button>
            </div>
          )}
        </div>
        )}
      </div>

      {readingProgress.showBackToTop && (
        <button
          type="button"
          onClick={readingProgress.handleBackToTop}
          className="absolute bottom-4 right-4 z-20 inline-flex items-center gap-1 rounded-full border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text shadow-sm hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          aria-label={t('review:mediaPage.backToTop', {
            defaultValue: 'Back to top'
          })}
          title={t('review:mediaPage.backToTop', {
            defaultValue: 'Back to top'
          })}
        >
          <ChevronUp className="h-3.5 w-3.5" />
          {t('review:mediaPage.backToTop', {
            defaultValue: 'Back to top'
          })}
        </button>
      )}

      {/* Export and schedule-refresh modals */}
      {selectedMedia &&
      !isNote &&
      (modals.exportModalOpen || modals.scheduleRefreshModalOpen) ? (
        <Suspense fallback={null}>
          <LazyContentViewerActionModals modals={modals} t={t} />
        </Suspense>
      ) : null}

      {/* Analysis Generation Modal - only for media */}
      {selectedMedia && !isNote && modals.analysisModalOpen && (
        <Suspense fallback={null}>
          <LazyAnalysisModal
            open={modals.analysisModalOpen}
            onClose={() => modals.setAnalysisModalOpen(false)}
            mediaId={selectedMedia.id}
            mediaContent={content}
            onAnalysisGenerated={(analysisText) => {
              if (analysisText) {
                editState.setOptimisticAnalysis(analysisText)
              }
              if (onRefreshMedia) {
                onRefreshMedia()
              }
            }}
          />
        </Suspense>
      )}

      {/* Analysis Edit Modal */}
      {editState.analysisEditModalOpen ? (
        <Suspense fallback={null}>
          <LazyAnalysisEditModal
            open={editState.analysisEditModalOpen}
            onClose={() => editState.setAnalysisEditModalOpen(false)}
            initialText={editState.editingAnalysisText}
            mediaId={selectedMedia?.id}
            content={content}
            prompt={editState.derivedPrompt}
            onSendToChat={onSendAnalysisToChat}
            onSaveNewVersion={() => {
              if (onRefreshMedia) {
                onRefreshMedia()
              }
            }}
          />
        </Suspense>
      ) : null}

      {/* Content Edit Modal */}
      {selectedMedia && !isNote && editState.contentEditModalOpen ? (
        <Suspense fallback={null}>
          <ContentEditModal
            open={editState.contentEditModalOpen}
            onClose={() => editState.setContentEditModalOpen(false)}
            initialText={editState.editingContentText || content}
            mediaId={selectedMedia.id}
            analysisContent={editState.derivedAnalysisContent}
            prompt={editState.derivedPrompt}
            onSaveNewVersion={() => {
              if (onRefreshMedia) {
                onRefreshMedia()
              }
            }}
          />
        </Suspense>
      ) : null}

      {/* Diff View Modal */}
      {modals.diffModalOpen ? (
        <Suspense fallback={null}>
          <LazyDiffViewModal
            open={modals.diffModalOpen}
            onClose={modals.closeDiffModal}
            leftText={modals.diffLeftText}
            rightText={modals.diffRightText}
            leftLabel={modals.diffLeftLabel}
            rightLabel={modals.diffRightLabel}
            metadataDiff={modals.diffMetadataSummary || undefined}
          />
        </Suspense>
      ) : null}
    </div>
  )
}

function formatDuration(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(Number(seconds))) return null
  const total = Math.max(0, Math.floor(Number(seconds)))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60

  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}
