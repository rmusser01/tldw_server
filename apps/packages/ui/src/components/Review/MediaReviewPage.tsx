import React from "react"
import { Input, Button, Tag, Modal, Drawer, Empty } from "antd"
import { ChevronDown, ChevronRight } from "lucide-react"
import { DiffViewModal } from "@/components/Media/DiffViewModal"
import { useMediaReviewState } from "@/components/Review/hooks/useMediaReviewState"
import { useMediaReviewActions } from "@/components/Review/hooks/useMediaReviewActions"
import { useMediaReviewKeyboard } from "@/components/Review/hooks/useMediaReviewKeyboard"
import { MediaReviewBatchBar } from "@/components/Review/MediaReviewBatchBar"
import { MediaReviewFilterSidebar } from "@/components/Review/MediaReviewFilterSidebar"
import { MediaReviewResultsList } from "@/components/Review/MediaReviewResultsList"
import { MediaReviewReadingPane } from "@/components/Review/MediaReviewReadingPane"
import { ResizablePanels } from "@/components/Review/ResizablePanels"
import type { MediaItem, MediaReviewState } from "@/components/Review/media-review-types"
import {
  DEFAULT_SORT_BY
} from "@/components/Review/media-review-types"

export const MediaReviewPage: React.FC = () => {
  const fetchListRef = React.useRef<(() => Promise<MediaItem[]>) | null>(null)
  const state = useMediaReviewState(fetchListRef)
  const actions = useMediaReviewActions(state)

  // Wire up the fetchList ref for the query
  fetchListRef.current = (actions as any)._fetchList

  // Compute collapsedFilterChips (depends on state + refetch for remove callbacks)
  const collapsedFilterChips = React.useMemo(() => {
    const chips: Array<{ key: string; label: string; remove: () => void }> = []
    const {
      t, types, setTypes, keywordTokens, setKeywordTokens,
      includeContent, setIncludeContent, sortBy, setSortBy,
      dateRange, setDateRange, setPage, refetch,
      sortLabelLookup, hasDateRangeFilter, dateRangeLabel
    } = state

    for (const type of types) {
      chips.push({
        key: `type-${type}`,
        label: type,
        remove: () => { setTypes((prev) => prev.filter((c) => c !== type)); setPage(1); refetch() }
      })
    }
    for (const keyword of keywordTokens) {
      chips.push({
        key: `keyword-${keyword}`,
        label: keyword,
        remove: () => { setKeywordTokens((prev) => prev.filter((c) => c !== keyword)); setPage(1); refetch() }
      })
    }
    if (includeContent) {
      chips.push({
        key: "content-scope",
        label: t("mediaPage.contentSearchLabel", "Search full content (slower)"),
        remove: () => { setIncludeContent(false); setPage(1); refetch() }
      })
    }
    if (sortBy !== DEFAULT_SORT_BY) {
      chips.push({
        key: "sort",
        label: t("mediaPage.sortChipLabel", "Sort: {{value}}", { value: sortLabelLookup[sortBy] || sortBy }),
        remove: () => { setSortBy(DEFAULT_SORT_BY); setPage(1); refetch() }
      })
    }
    if (hasDateRangeFilter && dateRangeLabel) {
      chips.push({
        key: "date-range",
        label: t("mediaPage.dateRangeChipLabel", "Date: {{value}}", { value: dateRangeLabel }),
        remove: () => { setDateRange({ startDate: null, endDate: null }); setPage(1); refetch() }
      })
    }
    return chips
  }, [
    state.types, state.keywordTokens, state.includeContent, state.sortBy,
    state.dateRange, state.hasDateRangeFilter, state.dateRangeLabel,
    state.sortLabelLookup, state.t, state.setTypes, state.setKeywordTokens,
    state.setIncludeContent, state.setSortBy, state.setDateRange,
    state.setPage, state.refetch
  ])

  // Augment state with collapsedFilterChips
  const fullState: MediaReviewState = React.useMemo(
    () => ({ ...state, collapsedFilterChips }),
    [state, collapsedFilterChips]
  )

  useMediaReviewKeyboard(fullState, actions)

  const {
    t, selectedIds,
    query, setQuery, setPage, refetch,
    isMobileViewport,
    searchInputRef, filtersCollapsed, setFiltersCollapsed,
    activeFilterCount, selectionStatusText,
    selectedItemsDrawerOpen, setSelectedItemsDrawerOpen,
    batchTrashHandoffIds, setBatchTrashHandoffIds,
    helpModalOpen, setHelpModalOpen,
    compareDiffOpen, setCompareDiffOpen,
    compareLeftText, compareRightText, compareLeftLabel, compareRightLabel,
    details, previewIndex
  } = fullState

  const { openTrashFromBatch, scrollToCard, removeFromSelection } = actions

  // Filter sidebar panel content
  const filterPanel = (
    <div className="flex flex-col h-full p-2 bg-surface border border-border rounded">
      {/* Search */}
      <div className="mb-2">
        <Input
          ref={(node: any) => {
            searchInputRef.current = node?.input ?? node ?? null
          }}
          placeholder={t('mediaPage.searchPlaceholder', 'Search media (title/content)')}
          aria-label={t('mediaPage.searchPlaceholder', 'Search media (title/content)') as string}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onPressEnter={() => { setPage(1); refetch() }}
          className="w-full"
        />
        <div className="flex gap-1 mt-1">
          <Button size="small" type="primary" onClick={() => { setPage(1); refetch() }}>{t('mediaPage.search', 'Search')}</Button>
          <Button size="small" onClick={() => { setQuery(""); setPage(1); refetch() }}>{t('mediaPage.clear', 'Clear')}</Button>
        </div>
      </div>
      {/* Filters */}
      <MediaReviewFilterSidebar state={fullState} actions={actions} />
      {/* Collapsed filter chips */}
      {collapsedFilterChips.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1 text-xs">
          {collapsedFilterChips.map((chip) => (
            <button
              key={chip.key}
              type="button"
              className="inline-flex items-center gap-1 rounded border border-border bg-surface2 px-2 py-1 text-text-muted hover:text-text"
              onClick={chip.remove}
              aria-label={t("mediaPage.removeFilter", "Remove filter {{label}}", { label: chip.label }) as string}
            >
              <span className="truncate max-w-[180px]">{chip.label}</span>
              <span aria-hidden="true">×</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )

  // Results panel content (with batch bar at bottom)
  const resultsPanel = (
    <div className="flex flex-col h-full p-2 bg-surface border border-border rounded">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-xs text-text-muted" data-testid="media-review-selection-count">{t('mediaPage.selectedTotal', '{{count}} selected', {count: selectedIds.length})}</span>
        <span className="text-xs text-text-muted" data-testid="media-multi-selection-status">{t('mediaPage.selectionStatus', 'Selection status: {{status}}', {status: selectionStatusText})}</span>
        <Button size="small" disabled={!selectedIds.length} onClick={() => actions.startSelectedReview()}>{t('mediaPage.reviewSelected', 'Review selected ({{count}})', {count: selectedIds.length})}</Button>
      </div>

      {selectedIds.length > 0 && (
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs text-text-muted whitespace-nowrap">
            {t('mediaPage.selectedAcrossPages', 'Selected across pages: {{count}}', {
              count: selectedIds.length
            })}
          </span>
          <Button
            size="small"
            onClick={() => setSelectedItemsDrawerOpen(true)}
            data-testid="view-selected-items-button"
          >
            {t('mediaPage.viewSelectedItems', 'View selected items')}
          </Button>
        </div>
      )}

      {batchTrashHandoffIds.length > 0 && (
        <div
          className="mb-1 flex flex-wrap items-center gap-2 rounded border border-primary/30 bg-primary/10 px-2 py-1 text-xs text-primary"
          data-testid="media-multi-trash-handoff"
        >
          <span>
            {t("mediaPage.batchTrashHandoff", "Batch delete complete. Review items in trash.")}
          </span>
          <Button
            size="small"
            type="link"
            className="!p-0"
            onClick={() => openTrashFromBatch(batchTrashHandoffIds)}
          >
            {t("mediaPage.openTrash", "Open trash")}
          </Button>
          <Button
            size="small"
            type="text"
            onClick={() => setBatchTrashHandoffIds([])}
          >
            {t("mediaPage.dismiss", "Dismiss")}
          </Button>
        </div>
      )}

      {/* Results list (flex-1 to fill) */}
      <div className="flex-1 min-h-0 flex flex-col">
        <MediaReviewResultsList state={fullState} actions={actions} />
      </div>

      {/* Batch bar at bottom of results panel */}
      <MediaReviewBatchBar state={fullState} actions={actions} />
    </div>
  )

  // Reading pane panel
  const readingPanel = (
    <MediaReviewReadingPane state={fullState} actions={actions} />
  )

  return (
    <div className="flex h-full min-h-0 w-full flex-1 flex-col">
      {/* Aria-live region for selection count announcements */}
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {t('mediaPage.selectedTotal', '{{count}} selected', {count: selectedIds.length})}
      </div>

      {/* Three-panel layout */}
      <div className="flex-1 min-h-0">
        <ResizablePanels
          left={filterPanel}
          center={resultsPanel}
          right={readingPanel}
          collapsed={isMobileViewport}
          mobileTab={fullState.mobileTab}
          onMobileTabChange={fullState.setMobileTab}
          tabLabels={[
            t('mediaPage.filtersTab', 'Filters'),
            t('mediaPage.resultsTab', 'Results'),
            t('mediaPage.contentTab', 'Content')
          ]}
        />
      </div>

      {/* Status bar */}
      <div
        className="flex items-center justify-between gap-4 border-t border-border bg-surface px-3 py-1 text-xs text-text-muted"
        data-testid="media-review-status-bar"
      >
        <div className="flex items-center gap-3">
          <span>
            {selectedIds.length > 0
              ? t('mediaPage.statusSelected', '{{count}} selected', { count: selectedIds.length })
              : t('mediaPage.statusNoneSelected', 'No selection')}
          </span>
          {fullState.readingActive && fullState.focusIndex >= 0 ? (
            <span>
              {t('mediaPage.statusReading', 'Reading item {{current}} of {{total}}', {
                current: fullState.focusIndex + 1,
                total: selectedIds.length
              })}
            </span>
          ) : !fullState.readingActive && previewIndex >= 0 && (
            <span>
              {t('mediaPage.statusPreview', 'Previewing {{current}} of {{total}}', {
                current: previewIndex + 1,
                total: fullState.previewNavigationIds.length
              })}
            </span>
          )}
        </div>
        <span className="text-text-muted/60">
          {t('mediaPage.statusHelpHint', '? for shortcuts')}
        </span>
      </div>

      <Drawer
        title={t('mediaPage.selectedItemsTitle', 'Selected items ({{count}})', { count: selectedIds.length })}
        open={selectedItemsDrawerOpen}
        onClose={() => setSelectedItemsDrawerOpen(false)}
        placement="right"
        size="default"
      >
        {selectedIds.length === 0 ? (
          <Empty description={t('mediaPage.selectedItemsEmpty', 'No selected items yet.')} />
        ) : (
          <div className="space-y-2" data-testid="selected-items-drawer">
            {selectedIds.map((id, idx) => {
              const detail = details[id]
              const row = fullState.selectedMetadata[String(id)]
              const itemTitle = detail?.title || row?.title || `${t('mediaPage.media', 'Media')} ${id}`
              const itemType = detail?.type || row?.type
              const itemDate = detail?.created_at || row?.created_at
              const isLoading = fullState.detailLoading[id]
              const hasFailed = fullState.failedIds.has(id)

              return (
                <div
                  key={String(id)}
                  className="rounded border border-border p-2"
                  data-testid={`selected-item-${String(id)}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="text-[11px] text-text-muted">#{idx + 1}</div>
                      <div className="truncate font-medium">{itemTitle}</div>
                      <div className="mt-1 flex items-center gap-1 text-[11px] text-text-muted">
                        {itemType ? <Tag>{String(itemType).toLowerCase()}</Tag> : null}
                        {itemDate ? <span>{new Date(itemDate).toLocaleString()}</span> : null}
                        {isLoading ? <span>{t('mediaPage.loading', 'Loading...')}</span> : null}
                        {hasFailed ? <span>{t('mediaPage.loadFailed', 'Failed to load content')}</span> : null}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        size="small"
                        onClick={() => {
                          actions.startSelectedReview(id)
                          scrollToCard(id)
                          setSelectedItemsDrawerOpen(false)
                        }}
                      >
                        {t('mediaPage.jumpToItem', 'Jump to item')}
                      </Button>
                      <Button
                        size="small"
                        onClick={() => removeFromSelection(id)}
                      >
                        {t("mediaPage.unstack", "Remove from selection")}
                      </Button>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </Drawer>

      {/* Keyboard shortcuts modal */}
      <Modal
        title={t('mediaPage.keyboardShortcuts', 'Keyboard Shortcuts')}
        open={helpModalOpen}
        onCancel={() => setHelpModalOpen(false)}
        footer={null}
        width={400}
      >
        <div className="space-y-3 text-sm">
          <div className="font-medium text-text-muted mb-2">{t('mediaPage.navigationShortcuts', 'Navigation')}</div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutNextItem', 'Next item')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">j</kbd> / <kbd className="px-2 py-1 bg-surface2 rounded text-xs">↓</kbd></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutPrevItem', 'Previous item')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">k</kbd> / <kbd className="px-2 py-1 bg-surface2 rounded text-xs">↑</kbd></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutToggleExpand', 'Toggle content expand')}</span>
            <kbd className="px-2 py-1 bg-surface2 rounded text-xs">o</kbd>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutFocusSearch', 'Focus search')}</span>
            <kbd className="px-2 py-1 bg-surface2 rounded text-xs">/</kbd>
          </div>
          <div className="font-medium text-text-muted mt-4 mb-2">{t('mediaPage.selectionShortcuts', 'Selection')}</div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutToggleSelect', 'Toggle selection on previewed item')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">x</kbd> / <kbd className="px-2 py-1 bg-surface2 rounded text-xs">Space</kbd></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutSelectAll', 'Select all visible')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">Ctrl</kbd>+<kbd className="px-2 py-1 bg-surface2 rounded text-xs">A</kbd></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutClearSelection', 'Clear selection')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">Esc</kbd> <span className="text-text-muted">×2</span></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutRangeSelect', 'Range selection')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">Shift</kbd>+{t('mediaPage.click', 'Click')}</span>
          </div>
          <div className="font-medium text-text-muted mt-4 mb-2">{t('mediaPage.contentShortcuts', 'Content')}</div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutContentSearch', 'Search in content')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">Ctrl</kbd>+<kbd className="px-2 py-1 bg-surface2 rounded text-xs">F</kbd></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutToggleCompare', 'Toggle comparison split')}</span>
            <span><kbd className="px-2 py-1 bg-surface2 rounded text-xs">Ctrl</kbd>+<kbd className="px-2 py-1 bg-surface2 rounded text-xs">\</kbd></span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border">
            <span>{t('mediaPage.shortcutShowHelp', 'Show this help')}</span>
            <kbd className="px-2 py-1 bg-surface2 rounded text-xs">?</kbd>
          </div>
        </div>
      </Modal>

      <DiffViewModal
        open={compareDiffOpen}
        onClose={() => setCompareDiffOpen(false)}
        leftText={compareLeftText}
        rightText={compareRightText}
        leftLabel={compareLeftLabel}
        rightLabel={compareRightLabel}
      />
    </div>
  )
}

export default MediaReviewPage
