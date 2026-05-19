import { useState, useCallback } from 'react'
import { CheckCircle2, Circle, Clock, FileText, List, LayoutGrid, Loader2, Star, User, AlertCircle, Upload } from 'lucide-react'
import { Tooltip, Button, Input } from 'antd'
import { useTranslation } from 'react-i18next'
import { formatRelativeTime } from '@/utils/dateFormatters'
import { highlightMatches } from '@/components/Media/highlightMatches'
import {
  persistFirstIngestDismissed,
  readFirstIngestDismissed
} from '@/utils/ftux-storage'

export type ResultsViewMode = 'standard' | 'compact'

interface Result {
  id: string | number
  title?: string
  kind: 'media' | 'note'
  snippet?: string
  keywords?: string[]
  meta?: {
    type?: string
    source?: string | null
    duration?: number | null
    status?: any
    created_at?: string
    author?: string | null
    published_at?: string | null
    transcription_model?: string | null
    word_count?: number | null
    page_count?: number | null
  }
}

/** Normalize processing status string to a display category. */
const normalizeProcessingStatus = (status: any): 'complete' | 'processing' | 'failed' | 'unknown' => {
  if (status == null) return 'unknown'
  const s = String(status).toLowerCase().trim()
  if (s === 'completed' || s === 'succeeded' || s === 'done' || s === 'ready') return 'complete'
  if (s === 'running' || s === 'processing' || s === 'started' || s === 'queued' || s === 'pending' || s === 'in_progress') return 'processing'
  if (s === 'failed' || s === 'error' || s === 'cancelled') return 'failed'
  return 'unknown'
}

/** Format duration seconds to human readable. */
const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

/** Format publication date to short display. */
const formatPublishedDate = (dateStr: string): string => {
  try {
    const d = new Date(dateStr)
    if (Number.isNaN(d.getTime())) return dateStr
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short' })
  } catch {
    return dateStr
  }
}

interface ResultsListProps {
  results: Result[]
  selectedId: string | number | null
  onSelect: (id: string | number) => void
  totalCount: number
  loadedCount: number
  isLoading?: boolean
  hasActiveFilters?: boolean
  searchQuery?: string
  onClearSearch?: () => void
  onClearFilters?: () => void
  onOpenQuickIngest?: (detail?: unknown) => void
  favorites?: Set<string>
  onToggleFavorite?: (id: string) => void
  selectionMode?: boolean
  selectedIds?: Set<string>
  onToggleSelected?: (id: string | number) => void
  readingProgress?: Map<string, number>
  viewMode?: ResultsViewMode
  onViewModeChange?: (mode: ResultsViewMode) => void
  stickyHeader?: boolean
}

export function ResultsList({
  results,
  selectedId,
  onSelect,
  totalCount,
  loadedCount,
  isLoading = false,
  hasActiveFilters = false,
  searchQuery = '',
  onClearSearch,
  onClearFilters,
  onOpenQuickIngest,
  favorites,
  onToggleFavorite,
  selectionMode = false,
  selectedIds,
  onToggleSelected,
  readingProgress,
  viewMode = 'standard',
  onViewModeChange,
  stickyHeader = true
}: ResultsListProps) {
  const { t } = useTranslation(['review'])
  const hasSearchQuery = searchQuery.trim().length > 0
  const isCompact = viewMode === 'compact'

  const [tutorialDismissed, setTutorialDismissed] = useState(() => {
    return readFirstIngestDismissed()
  })
  const [ingestUrl, setIngestUrl] = useState('')

  const handleDismissTutorial = useCallback(() => {
    setTutorialDismissed(true)
    persistFirstIngestDismissed()
  }, [])

  const handleIngestClick = useCallback(() => {
    const trimmed = ingestUrl.trim()
    if (trimmed && onOpenQuickIngest) {
      onOpenQuickIngest({ source: trimmed })
    } else if (onOpenQuickIngest) {
      onOpenQuickIngest()
    }
  }, [ingestUrl, onOpenQuickIngest])

  const buildInspectorTooltip = (result: Result) => {
    const title = result.title || `${result.kind} ${result.id}`
    const lines: Array<{
      label: string
      value: string
      multiline?: boolean
      preserveWhitespace?: boolean
    }> = [
      {
        label: t('mediaPage.titleLabel', 'Title'),
        value: title
      },
      {
        label: t('mediaPage.typeLabel', 'Type'),
        value: result.meta?.type || result.kind
      }
    ]
    if (result.meta?.author) {
      lines.push({
        label: t('mediaPage.authorLabel', 'Author'),
        value: result.meta.author
      })
    }
    if (result.meta?.source) {
      lines.push({
        label: t('mediaPage.source', 'Source'),
        value: result.meta.source
      })
    }
    if (result.meta?.published_at) {
      lines.push({
        label: t('mediaPage.publishedLabel', 'Published'),
        value: formatPublishedDate(result.meta.published_at)
      })
    }
    if (result.meta?.created_at) {
      lines.push({
        label: t('mediaPage.ingested', 'Ingested'),
        value: formatRelativeTime(result.meta.created_at, t, { compact: true })
      })
    }
    if (result.meta?.duration != null && result.meta.duration > 0) {
      lines.push({
        label: t('mediaPage.durationLabel', 'Duration'),
        value: formatDuration(result.meta.duration)
      })
    }
    if (result.meta?.page_count != null) {
      lines.push({
        label: t('mediaPage.pagesLabel', 'Pages'),
        value: String(result.meta.page_count)
      })
    }
    if (result.meta?.transcription_model) {
      lines.push({
        label: t('mediaPage.transcriptionModelLabel', 'Transcription'),
        value: result.meta.transcription_model
      })
    }
    {
      const progress = readingProgress?.get(String(result.id))
      if (progress != null && progress > 0) {
        lines.push({
          label: t('mediaPage.readingProgressLabel', 'Read'),
          value: `${Math.round(progress)}%`
        })
      }
    }
    if (result.snippet) {
      lines.push({
        label: t('mediaPage.previewLabel', 'Preview'),
        value: result.snippet,
        multiline: true,
        preserveWhitespace: true
      })
    }
    if (Array.isArray(result.keywords) && result.keywords.length > 0) {
      lines.push({
        label: t('mediaPage.keywords', 'Keywords'),
        value: result.keywords.join(', '),
        multiline: true
      })
    }

    return (
      <div className="space-y-1 text-xs max-w-xs">
        {lines.map((line, index) => (
          <div
            key={`${line.label}-${index}`}
            className={line.multiline ? "flex flex-col gap-1" : "flex gap-1"}
          >
            <span className="font-medium text-text">{line.label}:</span>
            <span
              className={
                line.multiline
                  ? `text-text-subtle break-words line-clamp-4 ${
                      line.preserveWhitespace ? "whitespace-pre-wrap" : "whitespace-normal"
                    }`
                  : "text-text-subtle break-words"
              }
            >
              {line.value}
            </span>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div data-testid="media-results-list">
      {/* Results Header */}
      <div
        className={`flex items-center justify-between border-b border-border px-4 py-2.5 ${
          stickyHeader
            ? 'sticky top-0 z-10 bg-surface2/70 backdrop-blur-[1px]'
            : 'bg-surface'
        }`}
      >
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-text-muted">
            {t('mediaPage.results', 'Results')}
          </span>
          <span className="text-[11px] font-medium tabular-nums text-text">
            {loadedCount} / {totalCount}
          </span>
        </div>
        {onViewModeChange && (
          <div className="flex items-center gap-0.5">
            <Tooltip title={t('mediaPage.standardView', 'Standard view')}>
              <button
                type="button"
                onClick={() => onViewModeChange('standard')}
                className={`p-1 rounded transition-colors ${viewMode === 'standard' ? 'bg-primary/10 text-primaryStrong' : 'text-text-muted hover:text-text hover:bg-surface'}`}
                aria-label={t('mediaPage.standardView', 'Standard view')}
                aria-pressed={viewMode === 'standard'}
              >
                <LayoutGrid className="w-3.5 h-3.5" />
              </button>
            </Tooltip>
            <Tooltip title={t('mediaPage.compactView', 'Compact view')}>
              <button
                type="button"
                onClick={() => onViewModeChange('compact')}
                className={`p-1 rounded transition-colors ${viewMode === 'compact' ? 'bg-primary/10 text-primaryStrong' : 'text-text-muted hover:text-text hover:bg-surface'}`}
                aria-label={t('mediaPage.compactView', 'Compact view')}
                aria-pressed={viewMode === 'compact'}
              >
                <List className="w-3.5 h-3.5" />
              </button>
            </Tooltip>
          </div>
        )}
      </div>

      {/* Results List */}
      <div className="divide-y divide-border">
        {/* Skeleton loading */}
        {isLoading && results.length === 0 ? (
          <>
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="px-4 py-2.5 animate-pulse">
                <div className="flex items-start gap-2.5">
                  <div className="w-4 h-4 bg-surface2 rounded mt-0.5" />
                  <div className="flex-1">
                    <div className="flex gap-1.5 mb-1">
                      <div className="w-12 h-4 bg-surface2 rounded" />
                      <div className="w-16 h-4 bg-surface2 rounded" />
                    </div>
                    <div className="w-3/4 h-4 bg-surface2 rounded mb-1" />
                    <div className="w-1/2 h-3 bg-surface2 rounded" />
                  </div>
                </div>
              </div>
            ))}
          </>
        ) : results.length === 0 && !isLoading ? (
          <div className="px-4 py-6 text-center">
            {hasActiveFilters ? (
              <>
                <p className="text-text-muted text-sm mb-2">
                  {t('mediaPage.noMatchingResults', 'No results match your filters')}
                </p>
                <p className="text-xs text-text-subtle mb-3">
                  {t('mediaPage.noMatchingResultsHint', 'Try broadening your query or removing filters.')}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  {hasSearchQuery && onClearSearch && (
                    <Button size="small" onClick={onClearSearch}>
                      {t('mediaPage.clearSearch', 'Clear search')}
                    </Button>
                  )}
                  {onClearFilters && (
                    <Button size="small" onClick={onClearFilters}>
                      {t('mediaPage.clearFilters', 'Clear filters')}
                    </Button>
                  )}
                  {onOpenQuickIngest && (
                    <Button size="small" onClick={() => onOpenQuickIngest()}>
                      {t('mediaPage.openQuickIngest', 'Open Quick Ingest')}
                    </Button>
                  )}
                </div>
              </>
            ) : !hasSearchQuery && !tutorialDismissed && onOpenQuickIngest ? (
              <div data-testid="first-ingest-tutorial" className="px-4 py-8 flex flex-col items-center gap-3">
                <Upload className="w-8 h-8 text-primary" />
                <h3 className="text-base font-semibold text-text">
                  {t('mediaPage.ftuxTitle', 'Get started — ingest your first content')}
                </h3>
                <p className="text-xs text-text-subtle max-w-sm">
                  {t('mediaPage.ftuxHint', 'Paste a YouTube URL, or use Quick Ingest to upload PDFs, audio, EPUB, and more.')}
                </p>
                <div className="flex items-center gap-2 w-full max-w-sm">
                  <Input
                    placeholder={t('mediaPage.ftuxUrlPlaceholder', 'Paste a YouTube URL...')}
                    value={ingestUrl}
                    onChange={(e) => setIngestUrl(e.target.value)}
                    onPressEnter={handleIngestClick}
                    size="middle"
                  />
                  <Button type="primary" onClick={handleIngestClick} aria-label={t('mediaPage.ftuxIngestButton', 'Ingest')}>
                    {t('mediaPage.ftuxIngestButton', 'Ingest')}
                  </Button>
                </div>
                <button
                  type="button"
                  onClick={handleDismissTutorial}
                  className="text-xs text-text-muted hover:text-text underline mt-1"
                >
                  {t('mediaPage.ftuxSkip', 'Skip for now')}
                </button>
              </div>
            ) : (
              <>
                <p className="text-text-muted text-sm mb-2">
                  {t('mediaPage.noResults', 'No results found')}
                </p>
                <p className="text-xs text-text-subtle mb-3">
                  {t('mediaPage.noResultsHint', 'Try broader terms, or ingest new content to search.')}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  {hasSearchQuery && onClearSearch && (
                    <Button size="small" onClick={onClearSearch}>
                      {t('mediaPage.clearSearch', 'Clear search')}
                    </Button>
                  )}
                  {onOpenQuickIngest && (
                    <Button size="small" onClick={() => onOpenQuickIngest()}>
                      {t('mediaPage.openQuickIngest', 'Open Quick Ingest')}
                    </Button>
                  )}
                </div>
              </>
            )}
          </div>
        ) : (
          results.map((result) => {
            const relativeDate = result.meta?.created_at
              ? formatRelativeTime(result.meta.created_at, t, { compact: true })
              : null
            const bulkSelected = selectedIds?.has(String(result.id)) === true
            const showSelectedStyle = selectionMode ? bulkSelected : selectedId === result.id
            const processingStatus = normalizeProcessingStatus(result.meta?.status)
            const progress = readingProgress?.get(String(result.id))
            const hasProgress = progress != null && progress > 0
            const isRead = progress != null && progress >= 95

            return (
              <div
              role="button"
              tabIndex={0}
              key={result.id}
              onClick={() => {
                if (selectionMode && onToggleSelected) {
                  onToggleSelected(result.id)
                  return
                }
                onSelect(result.id)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  if (selectionMode && onToggleSelected) {
                    onToggleSelected(result.id)
                    return
                  }
                  onSelect(result.id)
                }
              }}
              aria-label={t('mediaPage.selectResult', 'Select {{type}}: {{title}}', {
                type: result.kind,
                title: result.title || `${result.kind} ${result.id}`
              })}
              aria-selected={showSelectedStyle}
              className={`w-full ${isCompact ? 'py-2' : 'py-3'} text-left hover:bg-surface2/80 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-inset cursor-pointer ${
                showSelectedStyle
                  ? 'bg-surface2/90 border-l-4 border-l-primary px-3.5'
                  : 'px-4'
              }`}
            >
              <div className="flex items-start gap-3">
                {selectionMode && (
                  <input
                    type="checkbox"
                    checked={bulkSelected}
                    onChange={() => onToggleSelected?.(result.id)}
                    onClick={(event) => event.stopPropagation()}
                    className="mt-1 h-4 w-4 rounded border-border bg-surface"
                    aria-label={t('mediaPage.selectResultCheckbox', {
                      defaultValue: 'Select {{title}}',
                      title: result.title || `${result.kind} ${result.id}`
                    })}
                    data-testid={`results-select-${String(result.id)}`}
                  />
                )}
                <div className="mt-0.5 flex flex-col items-center gap-1">
                  <FileText className="w-4 h-4 text-text-subtle" />
                  {!isCompact && onToggleFavorite && (
                    <Tooltip title={favorites?.has(String(result.id)) ? t('mediaPage.unfavorite', 'Remove from favorites') : t('mediaPage.favorite', 'Add to favorites')}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          onToggleFavorite(String(result.id))
                        }}
                        className="p-1.5 hover:bg-surface2 rounded transition-colors"
                        aria-label={favorites?.has(String(result.id)) ? t('mediaPage.unfavorite', 'Remove from favorites') : t('mediaPage.favorite', 'Add to favorites')}
                        title={favorites?.has(String(result.id)) ? t('mediaPage.unfavorite', 'Remove from favorites') : t('mediaPage.favorite', 'Add to favorites')}
                      >
                        <Star className={`w-3.5 h-3.5 ${
                          favorites?.has(String(result.id))
                            ? 'text-warn fill-warn'
                            : 'text-text-subtle'
                        }`} />
                      </button>
                    </Tooltip>
                  )}
                </div>
                <Tooltip placement="right" title={buildInspectorTooltip(result)}>
                  <div className="flex-1 min-w-0">
                    {/* Badge row: kind, type, processing status, date */}
                    <div className="mb-0.5 flex items-center gap-1.5">
                      <span className="inline-flex items-center rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primaryStrong">
                        {result.kind.toUpperCase()}
                      </span>
                      {result.meta?.type && (
                        <span className="inline-flex items-center rounded bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text capitalize">
                          {result.meta.type}
                        </span>
                      )}
                      {/* Processing status indicator */}
                      {processingStatus === 'processing' && (
                        <Tooltip title={t('mediaPage.statusProcessing', 'Processing...')}>
                          <span className="inline-flex items-center">
                            <Clock className="w-3 h-3 text-primary animate-pulse" />
                          </span>
                        </Tooltip>
                      )}
                      {processingStatus === 'failed' && (
                        <Tooltip title={t('mediaPage.statusFailed', 'Processing failed')}>
                          <span className="inline-flex items-center">
                            <AlertCircle className="w-3 h-3 text-danger" />
                          </span>
                        </Tooltip>
                      )}
                      {/* Duration for audio/video */}
                      {result.meta?.duration != null && result.meta.duration > 0 && (
                        <span className="inline-flex items-center rounded bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
                          {formatDuration(result.meta.duration)}
                        </span>
                      )}
                      {/* Page count */}
                      {!isCompact && result.meta?.page_count != null && (
                        <span className="inline-flex items-center rounded bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
                          {t('mediaPage.pageCountBadge', '{{count}}pp', { count: result.meta.page_count })}
                        </span>
                      )}
                      {/* Date: prefer publication date, fallback to ingestion date */}
                      {result.meta?.published_at ? (
                        <Tooltip title={relativeDate ? t('mediaPage.ingestedDateTooltip', 'Ingested {{date}}', { date: relativeDate }) : undefined}>
                          <span className="inline-flex items-center rounded bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
                            {formatPublishedDate(result.meta.published_at)}
                          </span>
                        </Tooltip>
                      ) : relativeDate ? (
                        <span className="inline-flex items-center rounded bg-surface2 px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
                          {relativeDate}
                        </span>
                      ) : null}
                      {/* Reading progress indicator */}
                      {isRead && (
                        <Tooltip title={t('mediaPage.readComplete', 'Read')}>
                          <span className="inline-flex items-center">
                            <CheckCircle2 className="w-3 h-3 text-success" />
                          </span>
                        </Tooltip>
                      )}
                      {hasProgress && !isRead && (
                        <Tooltip title={t('mediaPage.readingProgressBadge', '{{percent}}% read', { percent: Math.round(progress) })}>
                          <span className="inline-flex items-center">
                            <Circle className="w-3 h-3 text-primary" strokeWidth={3} strokeDasharray={`${(progress / 100) * 9.42} 9.42`} />
                          </span>
                        </Tooltip>
                      )}
                    </div>
                    {/* Title */}
                    <div className="truncate text-[13px] leading-5 text-text font-semibold">
                      {result.title || `${result.kind} ${result.id}`}
                    </div>
                    {/* Author line */}
                    {result.meta?.author && (
                      <div className="mt-0.5 flex items-center gap-1">
                        <User className="w-3 h-3 text-text-subtle shrink-0" />
                        <span className="truncate text-[11px] text-text-muted">
                          {result.meta.author}
                        </span>
                      </div>
                    )}
                    {/* Snippet - hidden in compact mode */}
                    {!isCompact && result.snippet && (
                      <div className="mt-0.5 line-clamp-1 text-[11px] text-text-muted">
                        {highlightMatches(result.snippet, searchQuery)}
                      </div>
                    )}
                    {/* Keywords - hidden in compact mode */}
                    {!isCompact && Array.isArray(result.keywords) && result.keywords.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {result.keywords.slice(0, 5).map((keyword, idx) => (
                          <span
                            key={idx}
                            className="inline-flex max-w-[120px] items-center rounded bg-surface2 px-2 py-0.5 text-[10px] font-medium text-text line-clamp-1"
                            title={keyword}
                          >
                            {keyword}
                          </span>
                        ))}
                        {result.keywords.length > 5 && (
                          <Tooltip
                            title={t('mediaPage.moreTags', '+{{count}} more tags', { count: result.keywords.length - 5 })}
                          >
                            <span className="inline-flex items-center px-2 py-0.5 text-[10px] font-medium text-text-muted">
                              +{result.keywords.length - 5}
                            </span>
                          </Tooltip>
                        )}
                      </div>
                    )}
                    {/* Source - hidden in compact mode */}
                    {!isCompact && result.meta?.source && (
                      <div className="mt-0.5 text-[11px] text-text-subtle">
                        {result.meta.source}
                      </div>
                    )}
                  </div>
                </Tooltip>
              </div>
            </div>
            )
          })
        )}
        {/* Loading indicator */}
        {isLoading && (
          <div className="px-4 py-3 text-center text-text-muted flex items-center justify-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">{t('mediaPage.loadingMore', 'Loading more results...')}</span>
          </div>
        )}
      </div>
    </div>
  )
}
