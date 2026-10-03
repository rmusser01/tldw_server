import type { VirtualItem } from "@tanstack/react-virtual"
import type { MediaDateRange, MediaSortBy } from "@/components/Review/mediaSearchRequest"
import type { ContentFilterProgress } from "@/components/Review/content-filtering-progress"
import type { MediaMultiBatchExportFormat } from "@/components/Review/media-multi-batch-actions"
import { extractMediaDetailContent } from "@/utils/media-detail-content"

// ── Domain types ────────────────────────────────────────────────
export type MediaItem = {
  id: string | number
  title?: string
  snippet?: string
  type?: string
  created_at?: string
}

export type MediaDetail = {
  id: string | number
  title?: string
  type?: string
  created_at?: string
  content?: string | Record<string, unknown>
  text?: string
  raw_text?: string
  summary?: string
  latest_version?: { content?: string | Record<string, unknown> }
}

// ── Utility helpers ─────────────────────────────────────────────
export const getContent = (d: MediaDetail): string => {
  return extractMediaDetailContent(d)
}

export const idsEqual = (a: string | number, b: string | number): boolean =>
  String(a) === String(b)

export const includesId = (ids: Array<string | number>, candidate: string | number): boolean =>
  ids.some((id) => idsEqual(id, candidate))

export const getErrorStatusCode = (error: unknown): number | null => {
  if (!error || typeof error !== "object") return null
  const candidate = error as Record<string, unknown>
  const rawStatus =
    candidate.status ??
    (candidate.response &&
    typeof candidate.response === "object" &&
    (candidate.response as Record<string, unknown>).status != null
      ? (candidate.response as Record<string, unknown>).status
      : null) ??
    candidate.statusCode
  const parsed = Number(rawStatus)
  return Number.isFinite(parsed) ? parsed : null
}

// ── Constants ───────────────────────────────────────────────────
export const MINIMAP_COLLAPSE_THRESHOLD = 8
export const SELECTION_WARNING_THRESHOLD = 25
export const UNDO_DURATION_SECONDS = 15
export const MOBILE_REVIEW_MEDIA_QUERY = '(max-width: 1023px)'
export const MEDIA_CONTENT_DEFAULT_ROWS = 10
export const MEDIA_CONTENT_DEFAULT_MIN_HEIGHT_EM = MEDIA_CONTENT_DEFAULT_ROWS * 1.625
export const DEFAULT_SORT_BY: MediaSortBy = "relevance"
export const RESULTS_ROW_ESTIMATE_SIZE = 84

export const getIsMobileReviewViewport = (): boolean => {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false
  }
  return window.matchMedia(MOBILE_REVIEW_MEDIA_QUERY).matches
}

// ── State shape returned by useMediaReviewState ─────────────────
export interface MediaReviewState {
  // i18n & routing
  t: (key: string, defaultValueOrOpts?: string | Record<string, unknown>, opts?: Record<string, unknown>) => string
  navigate: (path: string) => void
  message: {
    info: (content: React.ReactNode, duration?: number) => void
    warning: (content: React.ReactNode, duration?: number) => void
    success: (content: React.ReactNode, duration?: number) => void
    error: (content: React.ReactNode, duration?: number) => void
  }

  // Search & filtering
  query: string
  setQuery: React.Dispatch<React.SetStateAction<string>>
  page: number
  setPage: React.Dispatch<React.SetStateAction<number>>
  pageSize: number
  setPageSize: React.Dispatch<React.SetStateAction<number>>
  total: number
  setTotal: React.Dispatch<React.SetStateAction<number>>
  types: string[]
  setTypes: React.Dispatch<React.SetStateAction<string[]>>
  keywordTokens: string[]
  setKeywordTokens: React.Dispatch<React.SetStateAction<string[]>>
  keywordOptions: string[]
  setKeywordOptions: React.Dispatch<React.SetStateAction<string[]>>
  includeContent: boolean
  setIncludeContent: React.Dispatch<React.SetStateAction<boolean>>
  sortBy: MediaSortBy
  setSortBy: React.Dispatch<React.SetStateAction<MediaSortBy>>
  dateRange: MediaDateRange
  setDateRange: React.Dispatch<React.SetStateAction<MediaDateRange>>
  availableTypes: string[]
  setAvailableTypes: React.Dispatch<React.SetStateAction<string[]>>

  authorityKey: string | null
  readingActive: boolean
  setReadingActive: React.Dispatch<React.SetStateAction<boolean>>
  readingWindowStart: number
  setReadingWindowStart: React.Dispatch<React.SetStateAction<number>>
  readingIds: Array<string | number>
  selectedMetadata: Record<string, MediaItem>
  mobileTab: 0 | 1 | 2
  setMobileTab: React.Dispatch<React.SetStateAction<0 | 1 | 2>>
  navigationTotal: number

  // Selection & focus
  selectedIds: Array<string | number>
  setSelectedIds: React.Dispatch<React.SetStateAction<Array<string | number>>>
  focusedId: string | number | null
  setFocusedId: React.Dispatch<React.SetStateAction<string | number | null>>
  previewNavigationIds: Array<string | number>
  setPreviewNavigationIds: React.Dispatch<React.SetStateAction<Array<string | number>>>
  previewedId: string | number | null
  setPreviewedId: React.Dispatch<React.SetStateAction<string | number | null>>
  selectionRestored: boolean
  setSelectionRestored: React.Dispatch<React.SetStateAction<boolean>>

  // Content & details
  details: Record<string | number, MediaDetail>
  setDetails: React.Dispatch<React.SetStateAction<Record<string | number, MediaDetail>>>
  contentLoading: boolean
  setContentLoading: React.Dispatch<React.SetStateAction<boolean>>
  contentFilterProgress: ContentFilterProgress
  setContentFilterProgress: React.Dispatch<React.SetStateAction<ContentFilterProgress>>
  contentExpandedIds: Set<string>
  setContentExpandedIds: React.Dispatch<React.SetStateAction<Set<string>>>
  analysisExpandedIds: Set<string>
  setAnalysisExpandedIds: React.Dispatch<React.SetStateAction<Set<string>>>
  showEmptyAnalysisIds: Set<string>
  setShowEmptyAnalysisIds: React.Dispatch<React.SetStateAction<Set<string>>>
  detailLoading: Record<string | number, boolean>
  setDetailLoading: React.Dispatch<React.SetStateAction<Record<string | number, boolean>>>
  failedIds: Set<string | number>
  setFailedIds: React.Dispatch<React.SetStateAction<Set<string | number>>>
  openAllLimit: number

  // View mode & layout
  orientation: string | undefined
  setOrientation: (value: string | ((prev: string | undefined) => string | undefined)) => Promise<void>
  hideTranscriptTimings: boolean | undefined
  setHideTranscriptTimings: (value: boolean | ((prev: boolean | undefined) => boolean | undefined)) => Promise<void>
  shouldHideTranscriptTimings: boolean
  isMobileViewport: boolean
  setIsMobileViewport: React.Dispatch<React.SetStateAction<boolean>>
  sidebarHidden: boolean
  setSidebarHidden: React.Dispatch<React.SetStateAction<boolean>>
  viewMode: "spread" | "list" | "all"
  viewModeState: "spread" | "list" | "all"
  setViewModeState: React.Dispatch<React.SetStateAction<"spread" | "list" | "all">>
  setViewMode: (mode: "spread" | "list" | "all") => void
  filtersCollapsed: boolean | undefined
  setFiltersCollapsed: (value: boolean | ((prev: boolean | undefined) => boolean | undefined)) => Promise<void>
  autoViewMode: boolean
  autoViewModeSetting: boolean | undefined
  setAutoViewModeSetting: (value: boolean | ((prev: boolean | undefined) => boolean | undefined)) => Promise<void>
  manualViewModePinned: boolean
  setManualViewModePinned: React.Dispatch<React.SetStateAction<boolean>>
  autoModeInlineNotice: string | null
  setAutoModeInlineNotice: React.Dispatch<React.SetStateAction<string | null>>
  collapseOthers: boolean
  setCollapseOthers: React.Dispatch<React.SetStateAction<boolean>>
  pendingInitialMediaId: string | null
  setPendingInitialMediaId: React.Dispatch<React.SetStateAction<string | null>>

  // Batch operations
  batchKeywordsDraft: string
  setBatchKeywordsDraft: React.Dispatch<React.SetStateAction<string>>
  batchExportFormat: MediaMultiBatchExportFormat
  setBatchExportFormat: React.Dispatch<React.SetStateAction<MediaMultiBatchExportFormat>>
  batchActionLoading: null | "keywords" | "trash" | "export" | "reprocess"
  setBatchActionLoading: React.Dispatch<React.SetStateAction<null | "keywords" | "trash" | "export" | "reprocess">>
  batchTrashHandoffIds: Array<string | number>
  setBatchTrashHandoffIds: React.Dispatch<React.SetStateAction<Array<string | number>>>

  // Compare diff
  compareDiffOpen: boolean
  setCompareDiffOpen: React.Dispatch<React.SetStateAction<boolean>>
  compareLeftText: string
  setCompareLeftText: React.Dispatch<React.SetStateAction<string>>
  compareRightText: string
  setCompareRightText: React.Dispatch<React.SetStateAction<string>>
  compareLeftLabel: string
  setCompareLeftLabel: React.Dispatch<React.SetStateAction<string>>
  compareRightLabel: string
  setCompareRightLabel: React.Dispatch<React.SetStateAction<string>>

  // Help & modals
  helpDismissed: boolean
  setHelpDismissed: (val: boolean) => void
  helpDismissedLoading: boolean
  helpModalOpen: boolean
  setHelpModalOpen: React.Dispatch<React.SetStateAction<boolean>>
  selectedItemsDrawerOpen: boolean
  setSelectedItemsDrawerOpen: React.Dispatch<React.SetStateAction<boolean>>
  copiedIds: Set<string>
  setCopiedIds: React.Dispatch<React.SetStateAction<Set<string>>>

  // Refs
  searchInputRef: React.MutableRefObject<HTMLInputElement | null>
  listParentRef: React.RefObject<HTMLDivElement | null>
  viewerRef: React.RefObject<HTMLDivElement | null>
  viewerParentRef: React.RefObject<HTMLDivElement | null>
  stackParentRef: React.RefObject<HTMLDivElement | null>
  cardRefs: React.MutableRefObject<Record<string, HTMLElement | null>>
  lastClickedRef: React.MutableRefObject<string | number | null>
  lastEscapePressRef: React.MutableRefObject<number>
  pendingRestoreFocusIdRef: React.MutableRefObject<string | number | null>
  contentFilterRunRef: React.MutableRefObject<number>
  prevAutoViewModeRef: React.MutableRefObject<string | null>
  ensureDetailRef: React.MutableRefObject<(id: string | number, isRetry?: boolean) => Promise<void>>

  // Derived values
  allResults: MediaItem[]
  hasResults: boolean
  viewerItems: MediaDetail[]
  hasTranscriptTimingContentInViewer: boolean
  visibleIds: Array<string | number>
  focusedDetail: MediaDetail | null
  focusIndex: number
  previewedDetail: MediaDetail | null
  previewIndex: number
  cardCls: string
  prefersReducedMotion: boolean
  isOnline: boolean

  // Selection status
  selectionStatusLevel: "limit" | "warning" | "safe"
  selectionStatusText: string
  activeFilterCount: number
  contentProgressLabel: string
  hasDateRangeFilter: boolean
  sortOptions: Array<{ value: MediaSortBy; label: string }>
  sortLabelLookup: Record<string, string>
  dateRangeLabel: string | null
  collapsedFilterChips: Array<{ key: string; label: string; remove: () => void }>

  // Chat integration
  setChatMode: (mode: string) => void
  setSelectedKnowledge: (val: any) => void
  setRagMediaIds: (ids: number[]) => void

  // Query
  data: MediaItem[] | undefined
  isFetching: boolean
  refetch: () => void

  // Virtualizers
  listVirtualizer: any
  viewerVirtualizer: any
  stackVirtualizer: any
}

// ── Actions shape returned by useMediaReviewActions ─────────────
export interface MediaReviewActions {
  startSelectedReview: (id?: string | number) => void
  returnToPreview: () => void
  changeReadingWindow: (delta: number) => void
  previewItem: (id: string | number, preserveContext?: boolean) => void
  toggleSelect: (id: string | number, event?: React.MouseEvent) => Promise<void>
  ensureDetail: (id: string | number, isRetry?: boolean) => Promise<void>
  retryFetch: (id: string | number) => void
  removeFromSelection: (id: string | number) => void
  clearSelectionWithGuard: () => void
  addVisibleToSelection: () => void
  replaceSelectionWithVisible: () => void
  goRelative: (delta: number) => void
  scrollToCard: (id: string | number) => void
  runContentFiltering: (items: MediaItem[]) => Promise<MediaItem[]>
  cancelContentFiltering: () => void
  mapMediaItems: (items: any[]) => MediaItem[]
  loadKeywordSuggestions: (q?: string) => Promise<void>
  handleBatchAddTags: () => Promise<void>
  handleBatchMoveToTrash: () => Promise<void>
  handleBatchExport: () => void
  handleBatchReprocess: () => Promise<void>
  handleCompareContent: () => Promise<void>
  handleChatAboutSelection: () => void
  expandAllContent: () => void
  collapseAllContent: () => void
  expandAllAnalysis: () => void
  collapseAllAnalysis: () => void
  getSelectedNumericIds: () => number[]
  openTrashFromBatch: (deletedIds: Array<string | number>) => void
  confirmBatchTrash: () => Promise<boolean>
  resolveDetailForCompare: (id: string | number) => Promise<MediaDetail | null>
}

// ── RenderCard options ──────────────────────────────────────────
export interface RenderCardOpts {
  virtualRow?: VirtualItem
  isAllMode?: boolean
}
