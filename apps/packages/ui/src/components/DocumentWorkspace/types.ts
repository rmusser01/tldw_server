/**
 * Types for Document Workspace feature
 * A research paper reader with AI-powered insights and document-scoped chat
 */

export type DocumentType = "pdf" | "epub"

export type SidebarTab = "insights" | "pages" | "toc" | "info" | "references"

export type RightPanelTab = "chat" | "annotations" | "citations" | "quiz"

export type ViewMode = "single" | "continuous" | "thumbnails"

export type EpubTheme = "light" | "dark" | "sepia"

export type EpubScrollMode = "paginated" | "continuous"

export type EpubSpreadMode = "none" | "auto" | "always"

export type EpubFontFamily = "inter" | "georgia" | "merriweather" | "system"

export type InsightDetailLevel = "brief" | "standard" | "detailed"

export type AnnotationColor = "yellow" | "green" | "blue" | "pink"

export type AnnotationType = "highlight" | "page_note"

export interface SearchResult {
  page: number
  text: string
  matchIndex: number
  itemIndex: number
}

/**
 * Per-document viewer state that persists when switching between tabs.
 * This state is stored locally in memory; server-side persistence
 * happens via the reading progress hooks.
 */
export interface DocumentViewerState {
  currentPage: number
  totalPages: number
  zoomLevel: number
  viewMode: ViewMode
  currentCfi: string | null
  currentPercentage: number
  currentChapterTitle: string | null
  searchQuery: string
  searchResults: SearchResult[]
  activeSearchIndex: number
}

export interface OpenDocument {
  id: number
  title: string
  type: DocumentType
  url?: string
  /** Per-document viewer state, saved when switching away */
  viewerState?: DocumentViewerState
}

export interface Annotation {
  id: string
  documentId: number
  /** Page number for PDF, CFI string for EPUB */
  location: number | string
  text: string
  color: AnnotationColor
  note?: string
  /** Type of annotation - highlight (selected text) or page_note (standalone note) */
  annotationType?: AnnotationType
  /** For EPUB: chapter title where annotation was made */
  chapterTitle?: string
  /** For EPUB: reading percentage (0-100) when annotation was made */
  percentage?: number
  createdAt: Date
  updatedAt: Date
  /** Server's last updated timestamp for conflict detection */
  serverUpdatedAt?: string
}

export type AnnotationSyncStatus = "synced" | "pending" | "error"
export type WorkspaceHealthStatus = "unknown" | "ok" | "error"

export interface InsightSection {
  key: string
  title: string
  content: string
}

export interface DocumentInsights {
  documentId: number
  sections: InsightSection[]
  generatedAt: Date
}

export interface Reference {
  id: string
  authors: string[]
  title: string
  year?: number
  venue?: string
  doi?: string
  arxivId?: string
  url?: string
  citationCount?: number
  isOpenAccess?: boolean
}

export interface TocItem {
  title: string
  page: number
  level: number
  children?: TocItem[]
  /** For EPUB: the href to navigate to */
  href?: string
}

export interface DocumentOutline {
  documentId: number
  items: TocItem[]
}

export interface Figure {
  id: string
  page: number
  imageUrl: string
  caption?: string
}

export interface DocumentMetadata {
  id: number
  title: string
  authors?: string[]
  creator?: string
  producer?: string
  fileName?: string
  abstract?: string
  keywords?: string[]
  pageCount?: number
  createdDate?: Date
  modifiedDate?: Date
  fileSize?: number
  type: DocumentType
}

// EPUB-specific location info
export interface EpubLocation {
  cfi: string
  percentage: number
  chapterIndex?: number
  chapterTitle?: string
}

// Store state interface
export interface DocumentWorkspaceState {
  // Active document
  activeDocumentId: number | null
  activeDocumentType: DocumentType | null

  // Open documents (tabs)
  openDocuments: OpenDocument[]

  // Recently closed documents (for undo)
  recentlyClosed: ClosedDocument[]

  // Layout
  leftSidebarCollapsed: boolean
  rightPanelCollapsed: boolean
  activeSidebarTab: SidebarTab
  activeRightTab: RightPanelTab

  // Viewer (PDF uses page numbers, EPUB uses CFI + percentage)
  currentPage: number
  totalPages: number
  zoomLevel: number
  viewMode: ViewMode

  // EPUB-specific state
  currentCfi: string | null
  currentPercentage: number
  currentChapterTitle: string | null
  epubTheme: EpubTheme
  epubScrollMode: EpubScrollMode
  epubSpreadMode: EpubSpreadMode
  epubFontSize: number
  epubFontFamily: EpubFontFamily
  epubLineHeight: number
  insightDetailLevel: InsightDetailLevel

  // Search
  searchOpen: boolean
  searchQuery: string
  searchResults: SearchResult[]
  activeSearchIndex: number
  searchMatchCase: boolean
  searchWordBoundary: boolean

  // Annotations (local cache)
  annotations: Annotation[]
  pendingAnnotations: Annotation[]
  annotationSyncStatus: AnnotationSyncStatus

  // Server health (document workspace tables)
  annotationsHealth: WorkspaceHealthStatus
  progressHealth: WorkspaceHealthStatus
}

// Store actions interface
export interface ClosedDocument {
  doc: OpenDocument
  closedAt: number
}

export interface DocumentWorkspaceActions {
  // Document management
  openDocument: (doc: OpenDocument) => void
  closeDocument: (id: number) => void
  setActiveDocument: (id: number | null) => void
  undoCloseDocument: () => OpenDocument | null

  // Layout
  setLeftSidebarCollapsed: (collapsed: boolean) => void
  setRightPanelCollapsed: (collapsed: boolean) => void
  setActiveSidebarTab: (tab: SidebarTab) => void
  setActiveRightTab: (tab: RightPanelTab) => void

  // Viewer
  setCurrentPage: (page: number) => void
  setTotalPages: (total: number) => void
  setZoomLevel: (zoom: number) => void
  setViewMode: (mode: ViewMode) => void
  goToNextPage: () => void
  goToPreviousPage: () => void

  // EPUB-specific
  setCurrentCfi: (cfi: string | null) => void
  setCurrentPercentage: (percentage: number) => void
  setCurrentChapterTitle: (title: string | null) => void
  setEpubTheme: (theme: EpubTheme) => void
  setEpubScrollMode: (mode: EpubScrollMode) => void
  setEpubSpreadMode: (mode: EpubSpreadMode) => void
  setEpubFontSize: (size: number) => void
  setEpubFontFamily: (family: EpubFontFamily) => void
  setEpubLineHeight: (height: number) => void
  setInsightDetailLevel: (level: InsightDetailLevel) => void

  // Search
  setSearchOpen: (open: boolean) => void
  setSearchQuery: (query: string) => void
  setSearchResults: (results: SearchResult[]) => void
  setActiveSearchIndex: (index: number) => void
  clearSearch: () => void
  setSearchMatchCase: (matchCase: boolean) => void
  setSearchWordBoundary: (wordBoundary: boolean) => void

  // Annotations
  addAnnotation: (annotation: Omit<Annotation, "id" | "createdAt" | "updatedAt">) => void
  updateAnnotation: (id: string, updates: Partial<Annotation>) => void
  removeAnnotation: (id: string) => void
  setAnnotations: (annotations: Annotation[]) => void
  setAnnotationSyncStatus: (status: AnnotationSyncStatus) => void
  setAnnotationsHealth: (status: WorkspaceHealthStatus) => void
  setProgressHealth: (status: WorkspaceHealthStatus) => void

  // Reset
  reset: () => void
}

export type DocumentWorkspaceStore = DocumentWorkspaceState & DocumentWorkspaceActions

// Default values
export const DEFAULT_ZOOM_LEVEL = 100
export const MIN_ZOOM_LEVEL = 25
export const MAX_ZOOM_LEVEL = 400
export const ZOOM_STEP = 25

export const DEFAULT_DOCUMENT_WORKSPACE_STATE: DocumentWorkspaceState = {
  activeDocumentId: null,
  activeDocumentType: null,
  openDocuments: [],
  recentlyClosed: [],
  leftSidebarCollapsed: false,
  rightPanelCollapsed: false,
  activeSidebarTab: "insights",
  activeRightTab: "chat",
  currentPage: 1,
  totalPages: 0,
  zoomLevel: DEFAULT_ZOOM_LEVEL,
  viewMode: "single",
  currentCfi: null,
  currentPercentage: 0,
  currentChapterTitle: null,
  epubTheme: "light",
  epubScrollMode: "paginated",
  epubSpreadMode: "none",
  epubFontSize: 100,
  epubFontFamily: "inter",
  epubLineHeight: 1.6,
  insightDetailLevel: "standard",
  searchOpen: false,
  searchQuery: "",
  searchResults: [],
  activeSearchIndex: 0,
  searchMatchCase: false,
  searchWordBoundary: false,
  annotations: [],
  pendingAnnotations: [],
  annotationSyncStatus: "synced",
  annotationsHealth: "unknown",
  progressHealth: "unknown"
}
