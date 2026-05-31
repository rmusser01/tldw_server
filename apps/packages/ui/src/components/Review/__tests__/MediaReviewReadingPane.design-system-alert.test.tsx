import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MediaReviewReadingPane } from "../MediaReviewReadingPane"
import type { MediaDetail, MediaReviewActions, MediaReviewState } from "../media-review-types"

vi.mock("@/components/Review/ContentRenderer", () => ({
  ContentRenderer: ({ content }: { content: string }) => (
    <div data-testid="mock-content-renderer">{content}</div>
  )
}))

vi.mock("@/components/Review/InContentSearch", () => ({
  InContentSearch: () => null
}))

vi.mock("@/components/Review/SectionNavigator", () => ({
  SectionNavigator: () => null
}))

vi.mock("@/components/Review/ComparisonSplit", () => ({
  ComparisonSplit: () => null
}))

vi.mock("@/services/settings/registry", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/settings/registry")>()),
  clearSetting: vi.fn()
}))

const interpolate = (template: string, opts?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(opts?.[key] ?? ""))

const t = (
  key: string,
  defaultValueOrOpts?: string | Record<string, unknown>,
  opts?: Record<string, unknown>
) => {
  if (typeof defaultValueOrOpts === "string") {
    return interpolate(defaultValueOrOpts, opts)
  }
  if (typeof defaultValueOrOpts?.defaultValue === "string") {
    return interpolate(defaultValueOrOpts.defaultValue, defaultValueOrOpts)
  }
  return key
}

const makeVirtualizer = () => ({
  getTotalSize: () => 240,
  getVirtualItems: () => [
    {
      index: 0,
      key: 0,
      start: 0,
      size: 240,
      end: 240
    }
  ],
  measureElement: vi.fn()
})

const makeDetail = (): MediaDetail => ({
  id: 42,
  title: "Failed media",
  type: "video",
  created_at: "2026-05-01T00:00:00Z",
  content: "Transcript body"
})

const makeState = (): MediaReviewState => {
  const detail = makeDetail()
  const viewerVirtualizer = makeVirtualizer()
  const stackVirtualizer = makeVirtualizer()

  return {
    t,
    message: {
      info: vi.fn(),
      warning: vi.fn(),
      success: vi.fn(),
      error: vi.fn()
    },
    selectedIds: [detail.id],
    setSelectedIds: vi.fn(),
    focusedId: detail.id,
    setFocusedId: vi.fn(),
    previewedId: null,
    setPreviewedId: vi.fn(),
    previewedDetail: null,
    previewIndex: -1,
    details: { [detail.id]: detail },
    setDetails: vi.fn(),
    detailLoading: {},
    setDetailLoading: vi.fn(),
    failedIds: new Set([detail.id]),
    setFailedIds: vi.fn(),
    viewMode: "spread",
    viewModeState: "spread",
    setViewModeState: vi.fn(),
    setViewMode: vi.fn(),
    viewerItems: [detail],
    focusedDetail: detail,
    focusIndex: 0,
    allResults: [detail],
    viewerRef: React.createRef<HTMLDivElement>(),
    viewerParentRef: React.createRef<HTMLDivElement>(),
    stackParentRef: React.createRef<HTMLDivElement>(),
    cardRefs: { current: {} },
    viewerVirtualizer,
    stackVirtualizer,
    helpDismissed: true,
    helpDismissedLoading: false,
    setHelpDismissed: vi.fn(),
    helpModalOpen: false,
    setHelpModalOpen: vi.fn(),
    isMobileViewport: false,
    orientation: "vertical",
    setOrientation: vi.fn(),
    hideTranscriptTimings: true,
    setHideTranscriptTimings: vi.fn(),
    shouldHideTranscriptTimings: false,
    contentExpandedIds: new Set(),
    setContentExpandedIds: vi.fn(),
    analysisExpandedIds: new Set(),
    setAnalysisExpandedIds: vi.fn(),
    showEmptyAnalysisIds: new Set(),
    setShowEmptyAnalysisIds: vi.fn(),
    copiedIds: new Set(),
    setCopiedIds: vi.fn(),
    autoViewMode: false,
    autoViewModeSetting: false,
    setAutoViewModeSetting: vi.fn(),
    autoModeInlineNotice: null,
    setAutoModeInlineNotice: vi.fn(),
    manualViewModePinned: false,
    setManualViewModePinned: vi.fn(),
    collapseOthers: false,
    setCollapseOthers: vi.fn(),
    selectedItemsDrawerOpen: false,
    setSelectedItemsDrawerOpen: vi.fn(),
    openAllLimit: 25,
    hasTranscriptTimingContentInViewer: false,
    cardCls: "rounded border border-border bg-surface p-3",
    setQuery: vi.fn(),
    setTypes: vi.fn(),
    setKeywordTokens: vi.fn()
  } as unknown as MediaReviewState
}

const makeActions = (retryFetch = vi.fn()): MediaReviewActions =>
  ({
    previewItem: vi.fn(),
    toggleSelect: vi.fn(),
    ensureDetail: vi.fn(),
    retryFetch,
    removeFromSelection: vi.fn(),
    clearSelectionWithGuard: vi.fn(),
    addVisibleToSelection: vi.fn(),
    replaceSelectionWithVisible: vi.fn(),
    goRelative: vi.fn(),
    scrollToCard: vi.fn(),
    runContentFiltering: vi.fn(),
    cancelContentFiltering: vi.fn(),
    mapMediaItems: vi.fn(),
    loadKeywordSuggestions: vi.fn(),
    handleBatchAddTags: vi.fn(),
    handleBatchMoveToTrash: vi.fn(),
    handleBatchExport: vi.fn(),
    handleBatchReprocess: vi.fn(),
    handleCompareContent: vi.fn(),
    handleChatAboutSelection: vi.fn(),
    expandAllContent: vi.fn(),
    collapseAllContent: vi.fn(),
    expandAllAnalysis: vi.fn(),
    collapseAllAnalysis: vi.fn(),
    getSelectedNumericIds: vi.fn(),
    openTrashFromBatch: vi.fn(),
    confirmBatchTrash: vi.fn(),
    resolveDetailForCompare: vi.fn()
  }) as unknown as MediaReviewActions

describe("MediaReviewReadingPane product-state alerts", () => {
  it("renders failed content through the design-system Alert and keeps retry behavior", () => {
    const retryFetch = vi.fn()

    render(
      <MediaReviewReadingPane
        state={makeState()}
        actions={makeActions(retryFetch)}
      />
    )

    const failedTitle = screen.getByText("Failed to load content")
    expect(failedTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(retryFetch).toHaveBeenCalledWith(42)
  })
})
