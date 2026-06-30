import React from "react"
import { Button, Tag, Tooltip, Radio, Select, Dropdown, Switch, Spin, Skeleton, Empty, Typography } from "antd"
import { CopyIcon, HelpCircle, Settings2, ChevronLeft, ChevronRight, Layers, LayoutGrid, Focus, Rows3, Check, MessageSquare } from "lucide-react"
import { ChevronDown, ChevronUp } from "lucide-react"
import type { VirtualItem } from "@tanstack/react-virtual"
import { Alert } from "@/components/ui/primitives"
import { clearSetting } from "@/services/settings/registry"
import {
  MEDIA_REVIEW_SELECTION_SETTING,
  MEDIA_REVIEW_FOCUSED_ID_SETTING,
  MEDIA_REVIEW_VIEW_MODE_SETTING,
  MEDIA_REVIEW_ORIENTATION_SETTING,
  MEDIA_REVIEW_FILTERS_COLLAPSED_SETTING
} from "@/services/settings/ui-settings"
import {
  stripLeadingTranscriptTimings
} from "@/utils/media-transcript-display"
import { getContentLayout } from "@/components/Review/card-content-density"
import { ContentRenderer } from "@/components/Review/ContentRenderer"
import { InContentSearch } from "@/components/Review/InContentSearch"
import { SectionNavigator, type ContentSection } from "@/components/Review/SectionNavigator"
import { ComparisonSplit } from "@/components/Review/ComparisonSplit"
import type { MediaReviewState, MediaReviewActions, MediaDetail } from "@/components/Review/media-review-types"
import { getContent, MINIMAP_COLLAPSE_THRESHOLD } from "@/components/Review/media-review-types"
import { scrollSectionIntoView } from "@/components/Review/reading-pane-section-navigation"

interface MediaReviewReadingPaneProps {
  state: MediaReviewState
  actions: MediaReviewActions
}

export const MediaReviewReadingPane: React.FC<MediaReviewReadingPaneProps> = ({ state, actions }) => {
  const [contentSearchQuery, setContentSearchQuery] = React.useState("")
  const [inContentSearchVisible, setInContentSearchVisible] = React.useState(false)
  const [inlineCompareMode, setInlineCompareMode] = React.useState(false)

  const {
    t, message,
    selectedIds, setSelectedIds, focusedId, setFocusedId,
    previewedId, previewedDetail, previewIndex,
    details, detailLoading, failedIds,
    viewMode, viewModeState, setViewModeState, setViewMode,
    viewerItems, focusedDetail, focusIndex, allResults,
    viewerRef, viewerParentRef, stackParentRef, cardRefs,
    viewerVirtualizer, stackVirtualizer,
    helpDismissed, helpDismissedLoading, setHelpDismissed,
    helpModalOpen, setHelpModalOpen,
    isMobileViewport,
    orientation, setOrientation,
    hideTranscriptTimings, setHideTranscriptTimings, shouldHideTranscriptTimings,
    contentExpandedIds, setContentExpandedIds,
    analysisExpandedIds, setAnalysisExpandedIds,
    showEmptyAnalysisIds, setShowEmptyAnalysisIds,
    copiedIds, setCopiedIds,
    autoViewMode, setAutoViewModeSetting,
    autoModeInlineNotice, setAutoModeInlineNotice,
    manualViewModePinned, setManualViewModePinned,
    collapseOthers, setCollapseOthers,
    selectedItemsDrawerOpen, setSelectedItemsDrawerOpen,
    openAllLimit,
    hasTranscriptTimingContentInViewer,
    cardCls, setDetails, setQuery, setTypes, setKeywordTokens
  } = state

  // Determine what to show: previewed item (if no selection), or selected items in viewer mode
  const showPreviewMode = selectedIds.length === 0 && previewedDetail != null
  const effectiveItems = showPreviewMode ? [previewedDetail] : viewerItems
  const navigationIndex = showPreviewMode ? previewIndex : focusIndex

  // Content for search/section navigation (use first effective item's content)
  const primaryContent = React.useMemo(() => {
    const first = effectiveItems[0]
    if (!first) return ""
    const raw = getContent(first) || ""
    return shouldHideTranscriptTimings ? stripLeadingTranscriptTimings(raw) : raw
  }, [effectiveItems, shouldHideTranscriptTimings])

  const {
    goRelative, scrollToCard, ensureDetail,
    removeFromSelection, addVisibleToSelection, replaceSelectionWithVisible,
    handleChatAboutSelection,
    expandAllContent, collapseAllContent, expandAllAnalysis, collapseAllAnalysis,
    retryFetch
  } = actions

  // Exit compare mode if selection drops below 2
  React.useEffect(() => {
    if (inlineCompareMode && selectedIds.length < 2) {
      setInlineCompareMode(false)
    }
  }, [inlineCompareMode, selectedIds.length])

  // Ctrl+F to open in-content search, Ctrl+\ to toggle comparison
  React.useEffect(() => {
    const el = viewerRef?.current
    if (!el) return
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "f") {
        e.preventDefault()
        setInContentSearchVisible(true)
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "\\") {
        e.preventDefault()
        if (selectedIds.length >= 2 && selectedIds.length <= 4) {
          setInlineCompareMode((prev) => {
            if (!prev) selectedIds.forEach((id) => void ensureDetail(id))
            return !prev
          })
        }
      }
    }
    el.addEventListener("keydown", handler)
    return () => el.removeEventListener("keydown", handler)
  }, [viewerRef, selectedIds, ensureDetail])

  const renderCard = (
    d: MediaDetail,
    idx: number,
    opts?: {
      virtualRow?: VirtualItem
      isAllMode?: boolean
    }
  ) => {
    if (!d) return null
    const { virtualRow, isAllMode } = opts || {}
    const key = String(d.id)
    const isFocused = d.id === focusedId
    const rawContent = getContent(d) || ""
    const content = shouldHideTranscriptTimings
      ? stripLeadingTranscriptTimings(rawContent)
      : rawContent
    const analysisText =
      d.summary ||
      (d as any)?.analysis ||
      (d as any)?.analysis_content ||
      (d as any)?.analysisContent ||
      ""
    const hasAnalysis = analysisText.trim().length > 0
    const analysisIsLong = analysisText.length > 1600
    const contentExpanded = contentExpandedIds.has(key)
    const analysisExpanded = analysisExpandedIds.has(key)
    const showEmptyAnalysisPanel = showEmptyAnalysisIds.has(key)
    const contentShown = content
    const analysisShown = !analysisIsLong || analysisExpanded ? analysisText : `${analysisText.slice(0, 1600)}…`
    const contentLayout = getContentLayout(content.length)
    const contentDefaultHeight = `${contentLayout.minHeightEm}em`
    const contentContainerStyle =
      contentExpanded || !contentLayout.capped
        ? { minHeight: contentDefaultHeight }
        : {
            minHeight: contentDefaultHeight,
            maxHeight: contentDefaultHeight,
            overflowY: "auto" as const
          }
    const isLoadingDetail = detailLoading[d.id]
    const hasFailed = failedIds.has(d.id)
    const rawSource = (d as any)?.source || (d as any)?.url || (d as any)?.original_url
    const source =
      rawSource && typeof rawSource === "object"
        ? (rawSource.url || rawSource.title || rawSource.href || "")
        : rawSource
    const transcriptLen = rawContent?.length
      ? Math.round(rawContent.length / 1000)
      : null

    const style =
      virtualRow != null
        ? {
            position: "absolute" as const,
            top: 0,
            left: 0,
            width: "100%",
            transform: `translateY(${virtualRow.start}px)`
          }
        : undefined
    const recentCopy =
      copiedIds.has(`content-${key}`) ||
      copiedIds.has(`analysis-${key}`) ||
      copiedIds.has(`both-${key}`)

    const handleCopyAction = async (mode: "content" | "analysis" | "both") => {
      const copyKey = `${mode}-${key}`
      if (mode === "analysis" && !analysisText) {
        message.info(t("mediaPage.noAnalysisToCopy", "No analysis available to copy"))
        return
      }
      const copyPayload =
        mode === "content"
          ? content
          : mode === "analysis"
            ? analysisText || ""
            : [
                content ? `${t("mediaPage.mediaContent", "Media Content")}:\n${content}` : "",
                analysisText ? `${t("mediaPage.analysis", "Analysis")}:\n${analysisText}` : ""
              ]
                .filter(Boolean)
                .join("\n\n")
      try {
        await navigator.clipboard.writeText(copyPayload)
        setCopiedIds((prev) => new Set(prev).add(copyKey))
        setTimeout(
          () =>
            setCopiedIds((prev) => {
              const next = new Set(prev)
              next.delete(copyKey)
              return next
            }),
          2000
        )
        if (mode === "analysis") {
          message.success(t('mediaPage.analysisCopied', 'Analysis copied'))
        } else if (mode === "both") {
          message.success(t('mediaPage.copyBothCopied', 'Content and analysis copied'))
        } else {
          message.success(t('mediaPage.contentCopied', 'Content copied'))
        }
      } catch {
        message.error(t('mediaPage.copyFailed', 'Copy failed'))
      }
    }

    return (
      <div
        key={key}
        ref={(el) => {
          if (virtualRow) viewerVirtualizer.measureElement(el)
          cardRefs.current[key] = el
        }}
        data-index={virtualRow?.index ?? idx}
        style={style}
        className={`${cardCls} shadow-sm ${isFocused ? 'ring-2 ring-primary ring-offset-2 ring-offset-surface' : ''}`}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="font-semibold leading-tight flex items-center gap-2">
              <span>{d.title || `${t('mediaPage.media', 'Media')} ${d.id}`}</span>
              <ChevronDown className="h-4 w-4 text-text-subtle" />
            </div>
            <div className="text-[11px] text-text-muted flex items-center gap-2 mt-1 flex-wrap">
              {isAllMode && <Tag>{t("mediaPage.stackPosition", "#{{num}}", { num: idx + 1 })}</Tag>}
              {d.type && <Tag>{String(d.type).toLowerCase()}</Tag>}
              {d.created_at && <span>{new Date(d.created_at).toLocaleString()}</span>}
              {(d as any)?.duration && <span>{t("mediaPage.duration", "{{value}}", { value: (d as any).duration })}</span>}
              {source && <span className="truncate max-w-[10rem]">{String(source)}</span>}
              {transcriptLen ? <span>{t("mediaPage.transcriptLength", "{{k}}k chars", { k: transcriptLen })}</span> : null}
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            {viewMode === "spread" && (
              <Tooltip title={t("mediaPage.unstackTooltip", "Remove this item from selection")}>
                <Button size="small" onClick={() => removeFromSelection(d.id)}>
                  {t("mediaPage.unstack", "Remove from selection")}
                </Button>
              </Tooltip>
            )}
            <Dropdown
              menu={{
                items: [
                  {
                    key: "content",
                    label: t("mediaPage.copyContentLabel", "Copy Content"),
                    onClick: () => { void handleCopyAction("content") }
                  },
                  {
                    key: "analysis",
                    label: t("mediaPage.copyAnalysisLabel", "Copy Analysis"),
                    disabled: !hasAnalysis && !isLoadingDetail,
                    onClick: () => { void handleCopyAction("analysis") }
                  },
                  {
                    key: "both",
                    label: t("mediaPage.copyBothLabel", "Copy both"),
                    disabled: !content && !hasAnalysis && !isLoadingDetail,
                    onClick: () => { void handleCopyAction("both") }
                  }
                ]
              }}
              trigger={["click"]}
            >
              <Button
                size="small"
                icon={
                  recentCopy
                    ? ((<Check className="w-4 h-4 text-success" />) as any)
                    : ((<CopyIcon className="w-4 h-4" />) as any)
                }
                data-testid={`media-review-copy-menu-${key}`}
              >
                {t("mediaPage.copyMenuLabel", "Copy")}
                <ChevronDown className="ml-1 h-3 w-3" />
              </Button>
            </Dropdown>
          </div>
        </div>

        {hasFailed && (
          <Alert
            variant="error"
            className="mt-3"
            title={t('mediaPage.loadFailed', 'Failed to load content')}
            action={{
              label: t('mediaPage.retry', 'Retry'),
              onClick: () => retryFetch(d.id)
            }}
          />
        )}

        <div className="mt-3 rounded border border-border p-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Typography.Text type="secondary">{t('mediaPage.mediaContent', 'Media Content')}</Typography.Text>
              {isLoadingDetail && <Spin size="small" />}
            </div>
            <Button
              size="small"
              type="text"
              icon={contentExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              onClick={() => {
                setContentExpandedIds((prev) => {
                  const next = collapseOthers ? new Set<string>() : new Set(prev)
                  if (next.has(key)) next.delete(key)
                  else next.add(key)
                  return next
                })
              }}
            >
              {contentExpanded ? t('mediaPage.collapse', 'Collapse') : t('mediaPage.expand', 'Expand')}
            </Button>
          </div>
          <div
            className="mt-2"
            style={contentContainerStyle}
            data-testid={`media-review-content-body-${key}`}
          >
            {isLoadingDetail ? (
              <Skeleton active paragraph={{ rows: 3 }} title={false} />
            ) : content ? (
              <ContentRenderer
                content={contentShown}
                hideTranscriptTimings={shouldHideTranscriptTimings}
                searchQuery={contentSearchQuery}
              />
            ) : (
              <span className="text-text-muted">{t('mediaPage.noContent', 'No content available')}</span>
            )}
          </div>
        </div>

        {(hasAnalysis || showEmptyAnalysisPanel || isLoadingDetail) ? (
          <div
            className="mt-3 rounded border border-border p-2"
            data-testid={`media-review-analysis-panel-${key}`}
          >
            <div className="flex items-center justify-between">
              <Typography.Text type="secondary">{t("mediaPage.analysis", "Analysis")}</Typography.Text>
              <div className="flex items-center gap-2">
                {!hasAnalysis && !isLoadingDetail && (
                  <Button
                    size="small"
                    type="link"
                    onClick={() => {
                      setShowEmptyAnalysisIds((prev) => {
                        const next = new Set(prev)
                        next.delete(key)
                        return next
                      })
                    }}
                  >
                    {t("mediaPage.hideEmptyAnalysisPanel", "Hide empty panel")}
                  </Button>
                )}
                <Button
                  size="small"
                  type="text"
                  icon={analysisExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  onClick={() => {
                    setAnalysisExpandedIds((prev) => {
                      const next = collapseOthers ? new Set<string>() : new Set(prev)
                      if (next.has(key)) next.delete(key)
                      else next.add(key)
                      return next
                    })
                  }}
                  disabled={!hasAnalysis && !isLoadingDetail}
                >
                  {analysisExpanded ? t('mediaPage.collapse', 'Collapse') : t('mediaPage.expand', 'Expand')}
                </Button>
              </div>
            </div>
            <div className="mt-2 prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap break-words text-sm text-text leading-relaxed">
              {isLoadingDetail ? (
                <Skeleton active paragraph={{ rows: 2 }} title={false} />
              ) : hasAnalysis ? (
                analysisShown
              ) : (
                <span className="text-text-muted">{t("mediaPage.noAnalysis", "No analysis available")}</span>
              )}
            </div>
          </div>
        ) : (
          <div
            className="mt-3 flex items-center justify-between rounded border border-dashed border-border px-3 py-2"
            data-testid={`media-review-analysis-empty-${key}`}
          >
            <span className="text-xs text-text-muted">
              {t("mediaPage.analysisUnavailableCompact", "Analysis not available")}
            </span>
            <Button
              size="small"
              type="link"
              onClick={() => {
                setShowEmptyAnalysisIds((prev) => new Set(prev).add(key))
              }}
            >
              {t("mediaPage.showEmptyAnalysisPanel", "Show panel")}
            </Button>
          </div>
        )}
      </div>
    )
  }

  return (
    <div
      ref={viewerRef}
      tabIndex={-1}
      className="flex-1 border border-border rounded p-2 bg-surface h-full flex flex-col min-w-0 relative focus:outline-none focus:ring-2 focus:ring-primary/30"
    >
      <div className="sticky top-0 z-20 bg-surface pb-2 border-b border-border">
        {/* Row 1: View Controls */}
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <div className="text-sm font-medium text-text">{t('mediaPage.viewer', 'Viewer')}</div>
              <div className="text-xs text-text-muted">
                {viewMode === "spread"
                  ? t("mediaPage.viewerCount", "{{count}} open", { count: viewerItems.length })
                  : viewMode === "list"
                    ? t("mediaPage.viewerSingle", "Single item view")
                    : t("mediaPage.viewerAll", "All items (stacked)")}
              </div>
            </div>
            {isMobileViewport ? (
              <div className="flex items-center gap-2">
                <Tag data-testid="mobile-view-mode-badge">
                  {viewMode === "all"
                    ? t("mediaPage.allMode", "Stack")
                    : t("mediaPage.listMode", "Focus")}
                </Tag>
                {selectedIds.length > 1 && (
                  <Button
                    size="small"
                    onClick={() => {
                      const nextMode = viewMode === "all" ? "list" : "all"
                      setViewModeState(nextMode)
                      if (nextMode === "all") {
                        selectedIds.forEach((id) => void ensureDetail(id))
                      } else {
                        const nextFocused = focusedId ?? selectedIds[0] ?? allResults[0]?.id
                        if (nextFocused != null) {
                          setFocusedId(nextFocused)
                          void ensureDetail(nextFocused)
                        }
                      }
                    }}
                  >
                    {viewMode === "all"
                      ? t("mediaPage.listMode", "Focus")
                      : t("mediaPage.allMode", "Stack")}
                  </Button>
                )}
              </div>
            ) : (
              <Tooltip title={t("mediaPage.spreadModeTooltip", "View selected items side-by-side for comparison")}>
                <span>
                  <Radio.Group
                    value={viewMode}
                    onChange={(e) => {
                      const next = e.target.value as "spread" | "list" | "all"
                      setManualViewModePinned(true)
                      setAutoModeInlineNotice(null)
                      setViewMode(next)
                      if (next === "list") {
                        const id = focusedId ?? selectedIds[0] ?? allResults[0]?.id
                        if (id != null) {
                          setFocusedId(id)
                          void ensureDetail(id)
                        }
                      } else if (next === "all") {
                        const ids = selectedIds.length > 0 ? selectedIds : allResults.slice(0, openAllLimit).map((m) => m.id)
                        setSelectedIds(ids)
                        ids.forEach((id) => void ensureDetail(id))
                      }
                    }}
                    optionType="button"
                    size="small"
                  >
                    <Tooltip title={t("mediaPage.spreadModeTooltip", "View selected items side-by-side for comparison")}>
                      <Radio.Button value="spread">
                        <LayoutGrid className="w-3.5 h-3.5 inline mr-1" />
                        {t("mediaPage.spreadMode", "Compare")}
                        {selectedIds.length > 0 && <span className="ml-1 text-xs opacity-70">({selectedIds.length})</span>}
                      </Radio.Button>
                    </Tooltip>
                    <Tooltip title={t("mediaPage.listModeTooltip", "View one item at a time with navigation")}>
                      <Radio.Button value="list">
                        <Focus className="w-3.5 h-3.5 inline mr-1" />
                        {t("mediaPage.listMode", "Focus")}
                        {selectedIds.length > 0 && (
                          <span className="ml-1 text-xs opacity-70">
                            ({focusedId != null ? selectedIds.indexOf(focusedId) + 1 : 1}/{selectedIds.length})
                          </span>
                        )}
                      </Radio.Button>
                    </Tooltip>
                    <Tooltip title={t("mediaPage.allModeTooltip", "View all selected items in a scrollable list")}>
                      <Radio.Button value="all">
                        <Rows3 className="w-3.5 h-3.5 inline mr-1" />
                        {t("mediaPage.allMode", "Stack")}
                        {selectedIds.length > 0 && <span className="ml-1 text-xs opacity-70">({selectedIds.length})</span>}
                      </Radio.Button>
                    </Tooltip>
                  </Radio.Group>
                </span>
              </Tooltip>
            )}
            {viewMode === "list" && (
              <Select
                size="small"
                className="min-w-[12rem]"
                placeholder={t("mediaPage.pickItem", "Pick an item")}
                value={focusedId ?? undefined}
                onChange={(val) => {
                  setFocusedId(val as any)
                  void ensureDetail(val as any)
                }}
                options={allResults.map((m, idx) => ({
                  label: `${idx + 1}. ${m.title || `Media ${m.id}`}`,
                  value: m.id
                }))}
              />
            )}
            <div className="h-5 w-px bg-border mx-1" />
            <Radio.Group
              size="small"
              value={orientation}
              onChange={(e) => { void setOrientation(e.target.value) }}
              options={[
                { label: t('mediaPage.vertical', 'Vertical'), value: 'vertical' },
                { label: t('mediaPage.horizontal', 'Horizontal'), value: 'horizontal' }
              ]}
              optionType="button"
            />
            <Dropdown
              menu={{
                items: [
                  {
                    key: 'autoViewMode',
                    label: (
                      <div className="flex items-center justify-between gap-4">
                        <span>{t("mediaPage.autoViewMode", "Auto-select view mode")}</span>
                        <Switch
                          size="small"
                          checked={autoViewMode}
                          onChange={(v) => {
                            void setAutoViewModeSetting(v)
                            if (v) setManualViewModePinned(false)
                            setAutoModeInlineNotice(null)
                          }}
                        />
                      </div>
                    )
                  },
                  {
                    key: 'collapseOthers',
                    label: (
                      <div className="flex items-center justify-between gap-4">
                        <span>{t("mediaPage.collapseOthers", "Collapse others on expand")}</span>
                        <Switch size="small" checked={collapseOthers} onChange={setCollapseOthers} />
                      </div>
                    )
                  },
                  { type: 'divider' as const },
                  {
                    key: 'openAll',
                    label: `${t("mediaPage.openAll", "Add visible to selection")} (${Math.min(allResults.length, openAllLimit)})`,
                    onClick: addVisibleToSelection,
                    disabled: allResults.length === 0
                  },
                  {
                    key: 'replaceWithVisible',
                    label: `${t("mediaPage.replaceWithVisible", "Replace selection with visible")} (${Math.min(allResults.length, openAllLimit)})`,
                    onClick: replaceSelectionWithVisible,
                    disabled: allResults.length === 0
                  },
                  {
                    key: 'viewSelectedItems',
                    label: t('mediaPage.viewSelectedItems', 'View selected items'),
                    onClick: () => setSelectedItemsDrawerOpen(true),
                    disabled: selectedIds.length === 0
                  },
                  { type: 'divider' as const },
                  {
                    key: 'showGuide',
                    label: t("mediaPage.showGuide", "Show getting started guide"),
                    onClick: () => setHelpDismissed(false)
                  },
                  { type: 'divider' as const },
                  {
                    key: 'clearSession',
                    label: t("mediaPage.clearSession", "Clear review session"),
                    danger: true,
                    onClick: async () => {
                      await clearSetting(MEDIA_REVIEW_SELECTION_SETTING)
                      await clearSetting(MEDIA_REVIEW_FOCUSED_ID_SETTING)
                      await clearSetting(MEDIA_REVIEW_VIEW_MODE_SETTING)
                      await clearSetting(MEDIA_REVIEW_ORIENTATION_SETTING)
                      await clearSetting(MEDIA_REVIEW_FILTERS_COLLAPSED_SETTING)
                      setSelectedIds([])
                      setFocusedId(null)
                      setDetails({})
                      setQuery("")
                      setTypes([])
                      setKeywordTokens([])
                      message.success(t("mediaPage.sessionCleared", "Review session cleared"))
                    }
                  }
                ]
              }}
              trigger={['click']}
            >
              <Button
                size="small"
                icon={<Settings2 className="w-3.5 h-3.5" />}
                aria-haspopup="menu"
                className="min-h-[44px] min-w-[44px]"
              >
                {t("mediaPage.viewerOptions", "Options")}
                <ChevronDown className="w-3 h-3 ml-1" />
              </Button>
            </Dropdown>
          </div>
        </div>
        {autoModeInlineNotice && (
          <div className="mb-2 rounded border border-primary/30 bg-primary/10 px-2 py-1 text-xs text-primary">
            {autoModeInlineNotice}
          </div>
        )}
        {/* Row 2: Navigation & Content Actions */}
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Tooltip title={t("mediaPage.prevItemTooltip", "Previous item (←)")}>
              <Button
                size="small"
                onClick={() => goRelative(-1)}
                disabled={navigationIndex <= 0}
                icon={<ChevronLeft className="w-4 h-4" />}
              >
                {t("mediaPage.prevItem", "Prev")}
              </Button>
            </Tooltip>
            <span className="text-xs text-text-muted min-w-[5rem] text-center">
              {navigationIndex >= 0
                ? t("mediaPage.itemPosition", "Item {{current}} of {{total}}", { current: navigationIndex + 1, total: allResults.length })
                : t("mediaPage.noItemSelected", "No item selected")}
            </span>
            <Tooltip title={t("mediaPage.nextItemTooltip", "Next item (→)")}>
              <Button
                size="small"
                onClick={() => goRelative(1)}
                disabled={navigationIndex < 0 || navigationIndex >= allResults.length - 1}
                icon={<ChevronRight className="w-4 h-4" />}
                iconPlacement="end"
              >
                {t("mediaPage.nextItem", "Next")}
              </Button>
            </Tooltip>
          </div>
          <div className="flex items-center gap-2">
            {hasTranscriptTimingContentInViewer && (
              <Button
                size="small"
                onClick={() => void setHideTranscriptTimings((prev) => !(prev ?? true))}
              >
                {shouldHideTranscriptTimings
                  ? t("mediaPage.showTimings", "Show timings")
                  : t("mediaPage.hideTimings", "Hide timings")}
              </Button>
            )}
            <Dropdown
              menu={{
                items: [
                  { key: 'expandContent', label: t("mediaPage.expandAllContent", "Expand all content"), onClick: expandAllContent },
                  { key: 'collapseContent', label: t("mediaPage.collapseAllContent", "Collapse all content"), onClick: collapseAllContent },
                  { type: 'divider' as const },
                  { key: 'expandAnalysis', label: t("mediaPage.expandAllAnalysis", "Expand all analysis"), onClick: expandAllAnalysis },
                  { key: 'collapseAnalysis', label: t("mediaPage.collapseAllAnalysis", "Collapse all analysis"), onClick: collapseAllAnalysis }
                ]
              }}
              trigger={['click']}
            >
              <Button size="small" icon={<Layers className="w-3.5 h-3.5" />}>
                {t("mediaPage.expandAllDropdown", "Expand/Collapse")}
                <ChevronDown className="w-3 h-3 ml-1" />
              </Button>
            </Dropdown>
            {selectedIds.length >= 2 && selectedIds.length <= 4 && (
              <Button
                data-compare-content-trigger="true"
                size="small"
                type={inlineCompareMode ? "primary" : "default"}
                onClick={() => {
                  if (inlineCompareMode) {
                    setInlineCompareMode(false)
                  } else {
                    // Ensure all selected items have details loaded
                    selectedIds.forEach((id) => void ensureDetail(id))
                    setInlineCompareMode(true)
                  }
                }}
              >
                {inlineCompareMode
                  ? t('mediaPage.exitCompare', 'Exit compare')
                  : t('mediaPage.compareContent', 'Compare content')}
              </Button>
            )}
            {selectedIds.length > 0 && (
              <Tooltip
                title={t('mediaPage.chatSelectionTooltip', {
                  defaultValue: 'Start a media-scoped RAG chat using the selected items.'
                })}
              >
                <Button
                  size="small"
                  icon={<MessageSquare className="w-3.5 h-3.5" />}
                  onClick={handleChatAboutSelection}
                >
                  {t('mediaPage.chatSelectionAction', {
                    defaultValue: 'Chat about selection ({{count}})',
                    count: selectedIds.length
                  })}
                </Button>
              </Tooltip>
            )}
            <Button
              size="small"
              shape="circle"
              type="text"
              onClick={() => setHelpModalOpen(true)}
              aria-label={
                t("mediaPage.viewerHelpLabel", "Multi-Item Review keyboard shortcuts") as string
              }
              className="text-text-subtle hover:text-text min-h-[44px] min-w-[44px]"
            >
              ?
            </Button>
          </div>
        </div>
        {/* In-content search + section navigator */}
        {effectiveItems.length > 0 && (
          <div className="flex items-center gap-2 mt-2">
            <SectionNavigator
              content={primaryContent}
              onNavigate={(section) => {
                // Scroll the content area to the section offset
                const contentEl = viewerRef?.current?.querySelector("[data-testid^='media-review-content-body-']")
                if (!contentEl) return
                scrollSectionIntoView(contentEl, section)
              }}
              t={t}
            />
            {!inContentSearchVisible && (
              <Button
                size="small"
                type="text"
                onClick={() => setInContentSearchVisible(true)}
                data-testid="open-content-search"
              >
                {t("mediaPage.searchInContent", "Search in content...")}
              </Button>
            )}
          </div>
        )}
        <InContentSearch
          content={primaryContent}
          onQueryChange={setContentSearchQuery}
          visible={inContentSearchVisible}
          onClose={() => setInContentSearchVisible(false)}
          t={t}
        />
        {selectedIds.length > 0 && (
          <div
            data-testid="media-review-open-items"
            className="mt-2 flex w-full min-w-0 flex-wrap items-start gap-2 text-xs text-text-muted"
          >
            <span className="font-medium shrink-0">{t("mediaPage.openMiniMap", "Open items")}</span>
            <span className="min-w-0 break-words">
              {t("mediaPage.viewerFlowHint", "Search/filter in sidebar, inspect in viewer, jump using Open items.")}
            </span>
            {selectedIds.length <= MINIMAP_COLLAPSE_THRESHOLD ? (
              <div className="flex w-full min-w-0 flex-wrap items-start gap-2">
                {selectedIds.map((id, idx) => {
                  const d = details[id]
                  const isLoading = detailLoading[id]
                  const hasFailed = failedIds.has(id)
                  const itemLabel = `${idx + 1}. ${d?.title || `${t('mediaPage.media', 'Media')} ${id}`}${d?.type ? ` (${String(d.type)})` : ""}`
                  return (
                    <Button
                      key={String(id)}
                      size="small"
                      type={focusedId === id ? "primary" : "default"}
                      danger={hasFailed}
                      onClick={() => {
                        setFocusedId(id)
                        scrollToCard(id)
                      }}
                      title={itemLabel}
                      className={`${isLoading ? "animate-pulse" : ""} min-h-[44px] min-w-[44px] max-w-[42ch] !h-auto whitespace-normal break-words text-left leading-5`}
                    >
                      {isLoading && <Spin size="small" className="mr-1" />}
                      <span className="whitespace-normal break-words">{itemLabel}</span>
                    </Button>
                  )
                })}
              </div>
            ) : (
              <Dropdown
                menu={{
                  items: selectedIds.map((id, idx) => {
                    const d = details[id]
                    const isLoading = detailLoading[id]
                    const hasFailed = failedIds.has(id)
                    return {
                      key: String(id),
                      label: (
                        <span className={isLoading ? "animate-pulse" : ""}>
                          {isLoading && <Spin size="small" className="mr-1" />}
                          {idx + 1}. {d?.title || `${t('mediaPage.media', 'Media')} ${id}`} {d?.type ? `(${String(d.type)})` : ""}
                        </span>
                      ),
                      danger: hasFailed,
                      onClick: () => {
                        setFocusedId(id)
                        scrollToCard(id)
                      }
                    }
                  }),
                  selectedKeys: focusedId != null ? [String(focusedId)] : []
                }}
                trigger={['click']}
              >
                <Button size="small" className="min-h-[44px] min-w-[44px]">
                  {t("mediaPage.jumpToItem", "Jump to item")} ({selectedIds.length})
                  <ChevronDown className="w-3 h-3 ml-1" />
                </Button>
              </Dropdown>
            )}
          </div>
        )}
      </div>
      <div className="flex flex-1 min-h-0 gap-3">
        <div className="flex-1 flex flex-col min-h-0">
          {inlineCompareMode && selectedIds.length >= 2 ? (
            <ComparisonSplit
              items={selectedIds
                .slice(0, 4)
                .map((id) => details[id])
                .filter(Boolean) as MediaDetail[]}
              hideTranscriptTimings={shouldHideTranscriptTimings}
              onClose={() => setInlineCompareMode(false)}
              t={t}
            />
          ) : effectiveItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full p-8 text-center">
              {!helpDismissedLoading && !helpDismissed ? (
                <div className="max-w-md">
                  <HelpCircle className="w-10 h-10 mx-auto mb-4 text-primary" />
                  <h3 className="text-lg font-medium text-text mb-3">
                    {t('mediaPage.firstUseTitle', 'Quick Guide: Multi-Item Review')}
                  </h3>
                  <ol className="text-left text-sm text-text-muted space-y-3 mb-6">
                    <li className="flex gap-2">
                      <span className="font-semibold text-primary">1.</span>
                      <span>
                        <strong>{t('mediaPage.firstUseStep1', 'Select items')}</strong> — {t('mediaPage.firstUseStep1Desc', 'Click items in the left panel to add them to your viewer.')}
                      </span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold text-primary">2.</span>
                      <span>
                        <strong>{t('mediaPage.firstUseStep2', 'Choose a view')}</strong> — {t('mediaPage.firstUseStep2Desc', 'Use "Compare" for side-by-side, "Focus" for one at a time, or "Stack" to see all.')}
                      </span>
                    </li>
                    <li className="flex gap-2">
                      <span className="font-semibold text-primary">3.</span>
                      <span>
                        <strong>{t('mediaPage.firstUseStep3', 'Navigate')}</strong> — {t('mediaPage.firstUseStep3Desc', 'Use Prev/Next buttons or keyboard (Tab + Enter) to move through items.')}
                      </span>
                    </li>
                  </ol>
                  <Button type="primary" onClick={() => setHelpDismissed(true)}>
                    {t('mediaPage.gotIt', 'Got it')}
                  </Button>
                </div>
              ) : (
                <div className="text-text-muted">
                  <Empty
                    description={t('mediaPage.selectItemsHint', 'Select items on the left to view here.')}
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                  />
                </div>
              )}
            </div>
          ) : showPreviewMode && previewedDetail ? (
            <div
              ref={viewerParentRef}
              className="relative flex-1 min-h-0 overflow-auto"
            >
              {renderCard(previewedDetail, 0)}
            </div>
          ) : (
            <>
              <div
                ref={viewMode === "all" ? stackParentRef : viewerParentRef}
                className="relative flex-1 min-h-0 overflow-auto"
              >
                {viewMode === "all" ? (
                  <div
                    data-testid="media-review-stack-virtualized"
                    style={{
                      height: `${stackVirtualizer.getTotalSize()}px`,
                      position: "relative"
                    }}
                  >
                    {stackVirtualizer.getVirtualItems().map((virtualRow: VirtualItem) => {
                      const d = viewerItems[virtualRow.index]
                      if (!d) return null
                      return renderCard(d, virtualRow.index, {
                        virtualRow,
                        isAllMode: true
                      })
                    })}
                  </div>
                ) : (
                  <div
                    style={{
                      height: `${viewerVirtualizer.getTotalSize()}px`,
                      position: "relative"
                    }}
                  >
                    {viewerVirtualizer.getVirtualItems().map((virtualRow: VirtualItem) => {
                      const d = viewMode === "spread" ? viewerItems[virtualRow.index] : viewMode === "list" ? focusedDetail : viewerItems[virtualRow.index]
                      if (!d) return null
                      return renderCard(d, virtualRow.index, { virtualRow })
                    })}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
