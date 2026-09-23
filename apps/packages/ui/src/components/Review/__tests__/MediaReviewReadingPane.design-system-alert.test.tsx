import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MediaReviewReadingPane } from "../MediaReviewReadingPane"
import type { MediaDetail, MediaReviewActions, MediaReviewState } from "../media-review-types"
import englishReview from "@/assets/locale/en/review.json"

vi.mock("@/components/Review/InContentSearch", () => ({
  InContentSearch: () => null
}))

vi.mock("@/components/Review/SectionNavigator", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/Review/SectionNavigator")>()),
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

const makeState = (
  detail: MediaDetail = makeDetail(),
  failedIds: Set<string | number> = new Set([detail.id])
): MediaReviewState => {
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
    failedIds,
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
  it.each(["fallback", "English"])("hides selection removal from an unselected preview with the %s label", (locale) => {
    const detail = makeDetail()
    const actions = makeActions()
    const state = {
      ...makeState(detail, new Set()),
      t: ((...args: Parameters<typeof t>) => locale === "English" && args[0] === "mediaPage.unstack"
        ? englishReview.mediaPage.unstack
        : t(...args)) as MediaReviewState["t"],
      selectedIds: [],
      viewerItems: [],
      previewedId: detail.id,
      previewedDetail: detail,
      previewIndex: 0
    }
    render(<MediaReviewReadingPane state={state} actions={actions} />)

    expect(screen.getByText("Transcript body")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^(Unstack|Remove from selection)$/ })).not.toBeInTheDocument()
    expect(actions.removeFromSelection).not.toHaveBeenCalled()
  })

  it("does not offer removal for a rendered card whose ID is outside the selection", () => {
    const state = { ...makeState(makeDetail(), new Set()), selectedIds: [99] }
    render(<MediaReviewReadingPane state={state} actions={makeActions()} />)

    expect(screen.getByText("Transcript body")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Remove from selection" })).not.toBeInTheDocument()
  })

  it.each([42, "42"])("keeps removal available for matching selected ID %s", (selectedId) => {
    const detail = makeDetail()
    const actions = makeActions()
    const state = { ...makeState(detail, new Set()), selectedIds: [selectedId] }
    render(<MediaReviewReadingPane state={state} actions={actions} />)

    fireEvent.click(screen.getByRole("button", { name: "Remove from selection" }))

    expect(actions.removeFromSelection).toHaveBeenCalledExactlyOnceWith(detail.id)
  })

  it("counts a visible preview when no items are selected", () => {
    const detail = makeDetail()
    const state = {
      ...makeState(detail, new Set()),
      selectedIds: [],
      viewerItems: [],
      previewedId: detail.id,
      previewedDetail: detail,
      previewIndex: 0
    }
    const view = render(<MediaReviewReadingPane state={state} actions={makeActions()} />)

    expect(screen.getByText("1 open")).toBeInTheDocument()

    view.rerender(<MediaReviewReadingPane state={{ ...state, previewedId: null, previewedDetail: null }} actions={makeActions()} />)
    expect(screen.getByText("0 open")).toBeInTheDocument()

    view.rerender(<MediaReviewReadingPane state={makeState(detail, new Set())} actions={makeActions()} />)
    expect(screen.getByText("1 open")).toBeInTheDocument()
  })

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

  it("shows analysis persisted under the media detail processing field", () => {
    const detail = {
      ...makeDetail(),
      processing: {
        analysis: "Persisted analysis from the real backend"
      }
    } as MediaDetail

    render(
      <MediaReviewReadingPane
        state={makeState(detail, new Set())}
        actions={makeActions()}
      />
    )

    expect(
      screen.getByText("Persisted analysis from the real backend")
    ).toBeInTheDocument()
    expect(screen.queryByText("Analysis not available")).not.toBeInTheDocument()
  })

  it("renders analysis formatting and copies the complete original Markdown", async () => {
    const analysis = "## Garden analysis\n\nThe **public** garden.\n\n- Seven beds"
    const detail = { ...makeDetail(), processing: { analysis } } as MediaDetail
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) }
    })
    render(<MediaReviewReadingPane state={makeState(detail, new Set())} actions={makeActions()} />)

    const panel = within(screen.getByTestId("media-review-analysis-panel-42"))
    await waitFor(() => expect(panel.getByRole("heading", { level: 2, name: "Garden analysis" })).toBeInTheDocument())
    expect(panel.getByText("public").tagName).toBe("STRONG")
    expect(panel.getByRole("listitem")).toHaveTextContent("Seven beds")
    fireEvent.click(screen.getByRole("button", { name: /Copy/ }))
    fireEvent.click(await screen.findByRole("menuitem", { name: "Copy Analysis" }))
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith(analysis))
  })

  it("renders analysis links safely while preserving ordinary prose", async () => {
    const analysis = "Plain explanation.\n\n[Safe](https://example.com/) [Unsafe](javascript:alert(1))\n\n<script>alert(1)</script><img src=x onerror=alert(1)>"
    const detail = { ...makeDetail(), processing: { analysis } } as MediaDetail
    render(<MediaReviewReadingPane state={makeState(detail, new Set())} actions={makeActions()} />)

    const panel = screen.getByTestId("media-review-analysis-panel-42")
    expect(await within(panel).findByRole("link", { name: "Safe" })).toHaveAttribute("href", "https://example.com/")
    expect(within(panel).getByText("Plain explanation.")).toBeInTheDocument()
    expect(panel.querySelector('a[href^="javascript:"], script, img[onerror]')).toBeNull()
  })
})
