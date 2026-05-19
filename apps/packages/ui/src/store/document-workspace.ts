import { create } from "zustand"
import type {
  DocumentWorkspaceStore,
  OpenDocument,
  ClosedDocument,
  DocumentViewerState,
  SidebarTab,
  RightPanelTab,
  ViewMode,
  Annotation,
  AnnotationSyncStatus,
  WorkspaceHealthStatus,
  SearchResult,
  EpubTheme,
  EpubScrollMode,
  EpubSpreadMode,
  EpubFontFamily,
  InsightDetailLevel
} from "@/components/DocumentWorkspace/types"
import {
  DEFAULT_ZOOM_LEVEL,
  MIN_ZOOM_LEVEL,
  MAX_ZOOM_LEVEL
} from "@/components/DocumentWorkspace/types"

const generateId = () =>
  `ann_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`

/**
 * Creates a default viewer state for a new document
 */
const createDefaultViewerState = (): DocumentViewerState => ({
  currentPage: 1,
  totalPages: 0,
  zoomLevel: DEFAULT_ZOOM_LEVEL,
  viewMode: "single",
  currentCfi: null,
  currentPercentage: 0,
  currentChapterTitle: null,
  searchQuery: "",
  searchResults: [],
  activeSearchIndex: 0
})

/**
 * Captures the current viewer state from the store
 */
const captureViewerState = (state: DocumentWorkspaceStore): DocumentViewerState => ({
  currentPage: state.currentPage,
  totalPages: state.totalPages,
  zoomLevel: state.zoomLevel,
  viewMode: state.viewMode,
  currentCfi: state.currentCfi,
  currentPercentage: state.currentPercentage,
  currentChapterTitle: state.currentChapterTitle,
  searchQuery: state.searchQuery,
  searchResults: state.searchResults,
  activeSearchIndex: state.activeSearchIndex
})

export const useDocumentWorkspaceStore = create<DocumentWorkspaceStore>(
  (set, get) => ({
    // Initial state
    activeDocumentId: null,
    activeDocumentType: null,
    openDocuments: [],
    recentlyClosed: [],
    leftSidebarCollapsed: false,
    rightPanelCollapsed: false,
    activeSidebarTab: "toc",
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
    progressHealth: "unknown",

    // Document management
    openDocument: (doc: OpenDocument) => {
      set((state) => {
        // Save current document's viewer state before switching
        let updatedOpenDocs = state.openDocuments
        if (state.activeDocumentId !== null && state.activeDocumentId !== doc.id) {
          updatedOpenDocs = state.openDocuments.map((d) =>
            d.id === state.activeDocumentId
              ? { ...d, viewerState: captureViewerState(state) }
              : d
          )
        }

        const existingDoc = updatedOpenDocs.find((d) => d.id === doc.id)
        if (existingDoc) {
          // Restore existing document's viewer state
          const viewerState = existingDoc.viewerState ?? createDefaultViewerState()
          const mergedDoc: OpenDocument = {
            ...existingDoc,
            ...doc,
            viewerState
          }
          const mergedOpenDocs = updatedOpenDocs.map((d) =>
            d.id === doc.id ? mergedDoc : d
          )
          return {
            openDocuments: mergedOpenDocs,
            activeDocumentId: doc.id,
            activeDocumentType: mergedDoc.type,
            currentPage: viewerState.currentPage,
            totalPages: viewerState.totalPages,
            zoomLevel: viewerState.zoomLevel,
            viewMode: viewerState.viewMode,
            currentCfi: viewerState.currentCfi,
            currentPercentage: viewerState.currentPercentage,
            currentChapterTitle: viewerState.currentChapterTitle,
            searchQuery: viewerState.searchQuery,
            searchResults: viewerState.searchResults,
            activeSearchIndex: viewerState.activeSearchIndex
          }
        }

        // Add new document with default viewer state
        const defaultState = createDefaultViewerState()
        return {
          openDocuments: [...updatedOpenDocs, { ...doc, viewerState: defaultState }],
          activeDocumentId: doc.id,
          activeDocumentType: doc.type,
          currentPage: defaultState.currentPage,
          totalPages: defaultState.totalPages,
          zoomLevel: defaultState.zoomLevel,
          viewMode: defaultState.viewMode,
          currentCfi: defaultState.currentCfi,
          currentPercentage: defaultState.currentPercentage,
          currentChapterTitle: defaultState.currentChapterTitle,
          searchQuery: defaultState.searchQuery,
          searchResults: defaultState.searchResults,
          activeSearchIndex: defaultState.activeSearchIndex
        }
      })
    },

    closeDocument: (id: number) => {
      set((state) => {
        const closedDoc = state.openDocuments.find((d) => d.id === id)
        const newOpenDocs = state.openDocuments.filter((d) => d.id !== id)
        const wasActive = state.activeDocumentId === id

        // Save the closed document for potential undo (keep max 3, drop oldest)
        const updatedClosed = closedDoc
          ? [...state.recentlyClosed, { doc: { ...closedDoc, viewerState: closedDoc.id === state.activeDocumentId ? captureViewerState(state) : closedDoc.viewerState }, closedAt: Date.now() }].slice(-3)
          : state.recentlyClosed

        if (!wasActive) {
          return { openDocuments: newOpenDocs, recentlyClosed: updatedClosed }
        }

        // If closing active document, activate the next available and restore its state
        const newActive = newOpenDocs.length > 0 ? newOpenDocs[0] : null
        if (!newActive) {
          const defaultState = createDefaultViewerState()
          return {
            openDocuments: newOpenDocs,
            recentlyClosed: updatedClosed,
            activeDocumentId: null,
            activeDocumentType: null,
            currentPage: defaultState.currentPage,
            totalPages: defaultState.totalPages,
            zoomLevel: defaultState.zoomLevel,
            viewMode: defaultState.viewMode,
            currentCfi: defaultState.currentCfi,
            currentPercentage: defaultState.currentPercentage,
            currentChapterTitle: defaultState.currentChapterTitle,
            searchQuery: defaultState.searchQuery,
            searchResults: defaultState.searchResults,
            activeSearchIndex: defaultState.activeSearchIndex
          }
        }

        // Restore the new active document's viewer state
        const viewerState = newActive.viewerState ?? createDefaultViewerState()
        return {
          openDocuments: newOpenDocs,
          recentlyClosed: updatedClosed,
          activeDocumentId: newActive.id,
          activeDocumentType: newActive.type,
          currentPage: viewerState.currentPage,
          totalPages: viewerState.totalPages,
          zoomLevel: viewerState.zoomLevel,
          viewMode: viewerState.viewMode,
          currentCfi: viewerState.currentCfi,
          currentPercentage: viewerState.currentPercentage,
          currentChapterTitle: viewerState.currentChapterTitle,
          searchQuery: viewerState.searchQuery,
          searchResults: viewerState.searchResults,
          activeSearchIndex: viewerState.activeSearchIndex
        }
      })
    },

    undoCloseDocument: () => {
      const state = get()
      const last = state.recentlyClosed[state.recentlyClosed.length - 1]
      if (!last) return null
      // Remove from recentlyClosed and re-open
      set({ recentlyClosed: state.recentlyClosed.slice(0, -1) })
      // Re-open the document (openDocument handles restoring viewer state)
      get().openDocument(last.doc)
      return last.doc
    },

    setActiveDocument: (id: number | null) => {
      set((state) => {
        // Save current document's viewer state before switching
        let updatedOpenDocs = state.openDocuments
        if (state.activeDocumentId !== null && state.activeDocumentId !== id) {
          updatedOpenDocs = state.openDocuments.map((d) =>
            d.id === state.activeDocumentId
              ? { ...d, viewerState: captureViewerState(state) }
              : d
          )
        }

        if (id === null) {
          const defaultState = createDefaultViewerState()
          return {
            openDocuments: updatedOpenDocs,
            activeDocumentId: null,
            activeDocumentType: null,
            currentPage: defaultState.currentPage,
            totalPages: defaultState.totalPages,
            zoomLevel: defaultState.zoomLevel,
            viewMode: defaultState.viewMode,
            currentCfi: defaultState.currentCfi,
            currentPercentage: defaultState.currentPercentage,
            currentChapterTitle: defaultState.currentChapterTitle,
            searchQuery: defaultState.searchQuery,
            searchResults: defaultState.searchResults,
            activeSearchIndex: defaultState.activeSearchIndex
          }
        }

        const doc = updatedOpenDocs.find((d) => d.id === id)
        if (!doc) return { openDocuments: updatedOpenDocs }

        // Restore the target document's viewer state
        const viewerState = doc.viewerState ?? createDefaultViewerState()
        return {
          openDocuments: updatedOpenDocs,
          activeDocumentId: id,
          activeDocumentType: doc.type,
          currentPage: viewerState.currentPage,
          totalPages: viewerState.totalPages,
          zoomLevel: viewerState.zoomLevel,
          viewMode: viewerState.viewMode,
          currentCfi: viewerState.currentCfi,
          currentPercentage: viewerState.currentPercentage,
          currentChapterTitle: viewerState.currentChapterTitle,
          searchQuery: viewerState.searchQuery,
          searchResults: viewerState.searchResults,
          activeSearchIndex: viewerState.activeSearchIndex
        }
      })
    },

    // Layout
    setLeftSidebarCollapsed: (collapsed: boolean) => {
      set({ leftSidebarCollapsed: collapsed })
    },

    setRightPanelCollapsed: (collapsed: boolean) => {
      set({ rightPanelCollapsed: collapsed })
    },

    setActiveSidebarTab: (tab: SidebarTab) => {
      set({ activeSidebarTab: tab })
    },

    setActiveRightTab: (tab: RightPanelTab) => {
      set({ activeRightTab: tab })
    },

    // Viewer
    setCurrentPage: (page: number) => {
      const { totalPages } = get()
      const clampedPage = Math.max(1, Math.min(page, totalPages || page))
      set({ currentPage: clampedPage })
    },

    setTotalPages: (total: number) => {
      set({ totalPages: total })
    },

    setZoomLevel: (zoom: number) => {
      const clampedZoom = Math.max(MIN_ZOOM_LEVEL, Math.min(zoom, MAX_ZOOM_LEVEL))
      set({ zoomLevel: clampedZoom })
    },

    setViewMode: (mode: ViewMode) => {
      set({ viewMode: mode })
    },

    goToNextPage: () => {
      const { currentPage, totalPages } = get()
      if (currentPage < totalPages) {
        set({ currentPage: currentPage + 1 })
      }
    },

    goToPreviousPage: () => {
      const { currentPage } = get()
      if (currentPage > 1) {
        set({ currentPage: currentPage - 1 })
      }
    },

    // EPUB-specific
    setCurrentCfi: (cfi: string | null) => {
      set({ currentCfi: cfi })
    },

    setCurrentPercentage: (percentage: number) => {
      set({ currentPercentage: percentage })
    },

    setCurrentChapterTitle: (title: string | null) => {
      set({ currentChapterTitle: title })
    },

    setEpubTheme: (theme: EpubTheme) => {
      set({ epubTheme: theme })
      // Persist to localStorage
      try {
        localStorage.setItem("epub-theme", theme)
      } catch (e) {
        // Ignore storage errors
      }
    },

    setEpubScrollMode: (mode: EpubScrollMode) => {
      set({ epubScrollMode: mode })
      // Persist to localStorage
      try {
        localStorage.setItem("epub-scroll-mode", mode)
      } catch (e) {
        // Ignore storage errors
      }
    },

    setEpubSpreadMode: (mode: EpubSpreadMode) => {
      set({ epubSpreadMode: mode })
      try {
        localStorage.setItem("epub-spread-mode", mode)
      } catch (e) {
        // Ignore storage errors
      }
    },

    setEpubFontSize: (size: number) => {
      const clamped = Math.max(50, Math.min(200, size))
      set({ epubFontSize: clamped })
      try {
        localStorage.setItem("epub-font-size", String(clamped))
      } catch (e) {
        // Ignore storage errors
      }
    },

    setEpubFontFamily: (family: EpubFontFamily) => {
      set({ epubFontFamily: family })
      try {
        localStorage.setItem("epub-font-family", family)
      } catch (e) {
        // Ignore storage errors
      }
    },

    setEpubLineHeight: (height: number) => {
      const clamped = Math.max(1.0, Math.min(2.5, height))
      set({ epubLineHeight: clamped })
      try {
        localStorage.setItem("epub-line-height", String(clamped))
      } catch (e) {
        // Ignore storage errors
      }
    },

    setInsightDetailLevel: (level: InsightDetailLevel) => {
      set({ insightDetailLevel: level })
      // Persist to localStorage
      try {
        localStorage.setItem("insight-detail-level", level)
      } catch (e) {
        // Ignore storage errors
      }
    },

    // Search
    setSearchOpen: (open: boolean) => {
      set({ searchOpen: open })
    },

    setSearchQuery: (query: string) => {
      set({ searchQuery: query })
    },

    setSearchResults: (results: SearchResult[]) => {
      set({ searchResults: results, activeSearchIndex: 0 })
    },

    setActiveSearchIndex: (index: number) => {
      const { searchResults } = get()
      if (index >= 0 && index < searchResults.length) {
        set({ activeSearchIndex: index })
      }
    },

    clearSearch: () => {
      set({
        searchQuery: "",
        searchResults: [],
        activeSearchIndex: 0
      })
    },

    setSearchMatchCase: (matchCase: boolean) => {
      set({ searchMatchCase: matchCase })
    },

    setSearchWordBoundary: (wordBoundary: boolean) => {
      set({ searchWordBoundary: wordBoundary })
    },

    // Annotations
    addAnnotation: (annotation) => {
      const now = new Date()
      const newAnnotation: Annotation = {
        ...annotation,
        id: generateId(),
        createdAt: now,
        updatedAt: now
      }
      set((state) => ({
        annotations: [...state.annotations, newAnnotation],
        pendingAnnotations: [...state.pendingAnnotations, newAnnotation],
        annotationSyncStatus: "pending"
      }))
    },

    updateAnnotation: (id: string, updates: Partial<Annotation>) => {
      set((state) => ({
        annotations: state.annotations.map((ann) =>
          ann.id === id ? { ...ann, ...updates, updatedAt: new Date() } : ann
        ),
        annotationSyncStatus: "pending"
      }))
    },

    removeAnnotation: (id: string) => {
      set((state) => ({
        annotations: state.annotations.filter((ann) => ann.id !== id),
        annotationSyncStatus: "pending"
      }))
    },

    setAnnotations: (annotations: Annotation[]) => {
      set({ annotations, pendingAnnotations: [], annotationSyncStatus: "synced" })
    },

    setAnnotationSyncStatus: (status: AnnotationSyncStatus) => {
      set({ annotationSyncStatus: status })
      if (status === "synced") {
        set({ pendingAnnotations: [] })
      }
    },
    setAnnotationsHealth: (status: WorkspaceHealthStatus) => {
      set({ annotationsHealth: status })
    },
    setProgressHealth: (status: WorkspaceHealthStatus) => {
      set({ progressHealth: status })
    },

    // Reset
    reset: () => {
      set({
        activeDocumentId: null,
        activeDocumentType: null,
        openDocuments: [],
        recentlyClosed: [],
        leftSidebarCollapsed: false,
        rightPanelCollapsed: false,
        activeSidebarTab: "toc",
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
      })
    }
  })
)

// Initialize settings from localStorage on module load
if (typeof window !== "undefined") {
  try {
    const savedTheme = localStorage.getItem("epub-theme")
    const savedScrollMode = localStorage.getItem("epub-scroll-mode")
    const savedDetailLevel = localStorage.getItem("insight-detail-level")
    if (savedTheme && ["light", "dark", "sepia"].includes(savedTheme)) {
      useDocumentWorkspaceStore.setState({ epubTheme: savedTheme as EpubTheme })
    }
    if (savedScrollMode && ["paginated", "continuous"].includes(savedScrollMode)) {
      useDocumentWorkspaceStore.setState({ epubScrollMode: savedScrollMode as EpubScrollMode })
    }
    if (savedDetailLevel && ["brief", "standard", "detailed"].includes(savedDetailLevel)) {
      useDocumentWorkspaceStore.setState({ insightDetailLevel: savedDetailLevel as InsightDetailLevel })
    }
    const savedSpreadMode = localStorage.getItem("epub-spread-mode")
    if (savedSpreadMode && ["none", "auto", "always"].includes(savedSpreadMode)) {
      useDocumentWorkspaceStore.setState({ epubSpreadMode: savedSpreadMode as EpubSpreadMode })
    }
    const savedFontSize = localStorage.getItem("epub-font-size")
    if (savedFontSize) {
      const size = parseInt(savedFontSize, 10)
      if (size >= 50 && size <= 200) {
        useDocumentWorkspaceStore.setState({ epubFontSize: size })
      }
    }
    const savedFontFamily = localStorage.getItem("epub-font-family")
    if (savedFontFamily && ["inter", "georgia", "merriweather", "system"].includes(savedFontFamily)) {
      useDocumentWorkspaceStore.setState({ epubFontFamily: savedFontFamily as EpubFontFamily })
    }
    const savedLineHeight = localStorage.getItem("epub-line-height")
    if (savedLineHeight) {
      const height = parseFloat(savedLineHeight)
      if (height >= 1.0 && height <= 2.5) {
        useDocumentWorkspaceStore.setState({ epubLineHeight: height })
      }
    }
  } catch (e) {
    // Ignore storage errors
  }
}
