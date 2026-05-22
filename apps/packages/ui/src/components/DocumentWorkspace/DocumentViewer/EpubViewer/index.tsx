import React, { useEffect, useRef, useCallback, useState } from "react"
import { useTranslation } from "react-i18next"
import { Spin } from "antd"
import type { Book, Rendition, NavItem } from "epubjs"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { TextSelectionPopover } from "../TextSelectionPopover"
import { EpubSearch } from "./EpubSearch"
import { EPUB_THEMES, FONT_FAMILY_CSS } from "@/hooks/document-workspace/useEpubSettings"
import type { EpubLocation } from "@/hooks/document-workspace/useEpubReader"
import type { TocItem, Annotation, AnnotationColor, EpubTheme, EpubScrollMode, EpubSpreadMode, EpubFontFamily } from "../../types"

// Color mapping for EPUB highlights
const HIGHLIGHT_COLORS: Record<AnnotationColor, string> = {
  yellow: "rgba(254, 240, 138, 0.4)",
  green: "rgba(187, 247, 208, 0.4)",
  blue: "rgba(191, 219, 254, 0.4)",
  pink: "rgba(251, 207, 232, 0.4)"
}

type RelocatedLocation = {
  start?: {
    cfi?: string
  }
}

type RelocatedSetters = {
  setCurrentCfi: (cfi: string) => void
  setCurrentPercentage: (percentage: number) => void
  setCurrentPage: (page: number) => void
}

const applyRelocatedLocation = (
  book: Book,
  location: RelocatedLocation | null | undefined,
  setters: RelocatedSetters
): { cfi: string; percentage: number; locationIndex: number } | null => {
  const cfi = location?.start?.cfi
  if (!cfi) return null

  const percentage = book.locations.percentageFromCfi(cfi) || 0
  const locationIndex = Number(book.locations.locationFromCfi(cfi) ?? 0)

  setters.setCurrentCfi(cfi)
  setters.setCurrentPercentage(percentage * 100)
  setters.setCurrentPage(locationIndex + 1) // 1-indexed for UI consistency

  return { cfi, percentage, locationIndex }
}

/**
 * Convert epub.js NavItem to our TocItem format
 */
function convertNavToTocItems(nav: NavItem[], level: number = 0): TocItem[] {
  return nav.map((item, idx) => ({
    title: item.label.trim(),
    page: idx + 1,
    level,
    href: item.href,
    children: item.subitems ? convertNavToTocItems(item.subitems, level + 1) : undefined
  }))
}

/**
 * Build theme body overrides from typography settings
 */
const buildTypographyOverrides = (
  fontFamily: EpubFontFamily,
  fontSize: number,
  lineHeight: number
): Record<string, string> => ({
  "font-family": FONT_FAMILY_CSS[fontFamily],
  "font-size": fontSize === 100 ? "1em" : `${fontSize}%`,
  "line-height": String(lineHeight),
  "padding": "20px"
})

interface EpubViewerProps {
  url: string
  documentId: number
  onLoadSuccess?: (data: { chapterCount: number; toc: NavItem[] }) => void
  onLoadError?: (error: Error) => void
  onLocationChange?: (location: EpubLocation) => void
}

/**
 * EPUB viewer component using epub.js.
 *
 * Renders EPUB documents with chapter navigation, text selection,
 * and progress tracking via CFI (Canonical Fragment Identifier).
 */
export const EpubViewer: React.FC<EpubViewerProps> = ({
  url,
  documentId,
  onLoadSuccess,
  onLoadError,
  onLocationChange
}) => {
  const { t } = useTranslation(["option", "common"])
  const containerRef = useRef<HTMLDivElement>(null)
  const bookRef = useRef<Book | null>(null)
  const renditionRef = useRef<Rendition | null>(null)

  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [toc, setToc] = useState<NavItem[]>([])

  // Text selection state for popover
  const [selection, setSelection] = useState<{
    text: string
    cfi: string
    rect: DOMRect
  } | null>(null)

  // Store access
  const currentPage = useDocumentWorkspaceStore((s) => s.currentPage)
  const setCurrentPage = useDocumentWorkspaceStore((s) => s.setCurrentPage)
  const setTotalPages = useDocumentWorkspaceStore((s) => s.setTotalPages)
  const currentCfi = useDocumentWorkspaceStore((s) => s.currentCfi)
  const setCurrentCfi = useDocumentWorkspaceStore((s) => s.setCurrentCfi)
  const setCurrentPercentage = useDocumentWorkspaceStore((s) => s.setCurrentPercentage)
  const setCurrentChapterTitle = useDocumentWorkspaceStore((s) => s.setCurrentChapterTitle)
  const annotations = useDocumentWorkspaceStore((s) => s.annotations)
  const epubTheme = useDocumentWorkspaceStore((s) => s.epubTheme)
  const epubScrollMode = useDocumentWorkspaceStore((s) => s.epubScrollMode)
  const epubSpreadMode = useDocumentWorkspaceStore((s) => s.epubSpreadMode)
  const epubFontSize = useDocumentWorkspaceStore((s) => s.epubFontSize)
  const epubFontFamily = useDocumentWorkspaceStore((s) => s.epubFontFamily)
  const epubLineHeight = useDocumentWorkspaceStore((s) => s.epubLineHeight)

  // Dispatch loading event when starting
  useEffect(() => {
    if (url) {
      window.dispatchEvent(
        new CustomEvent("epub-loading", {
          detail: { documentId }
        })
      )
    }
  }, [url, documentId])

  // Initialize epub.js book
  useEffect(() => {
    if (!url || !containerRef.current) return

    let mounted = true

    const initEpub = async () => {
      setIsLoading(true)
      setError(null)

      try {
        // Dynamically import epub.js for SSR compatibility
        const ePubModule = await import("epubjs")
        const ePub = ePubModule.default

        // Clean up previous book
        if (bookRef.current) {
          bookRef.current.destroy()
        }

        const book = ePub(url)
        bookRef.current = book

        await book.ready

        if (!mounted || !containerRef.current) {
          book.destroy()
          return
        }

        // Get initial settings from store
        const initialState = useDocumentWorkspaceStore.getState()
        const initialScrollMode = initialState.epubScrollMode
        const initialTheme = initialState.epubTheme
        const initialSpreadMode = initialState.epubSpreadMode
        const initialFontSize = initialState.epubFontSize
        const initialFontFamily = initialState.epubFontFamily
        const initialLineHeight = initialState.epubLineHeight

        // Create rendition with current scroll mode and spread
        const rendition = book.renderTo(containerRef.current, {
          width: "100%",
          height: "100%",
          spread: initialSpreadMode,
          flow: initialScrollMode === "continuous" ? "scrolled" : "paginated"
        })

        renditionRef.current = rendition

        // Load navigation
        await book.loaded.navigation

        // Generate locations for percentage tracking (150 chars per location)
        await book.locations.generate(150)

        // Set total "pages" (actually locations for EPUB)
        const totalLocations = book.locations.length()
        setTotalPages(totalLocations)

        // Extract TOC
        const navigation = book.navigation
        setToc(navigation.toc)

        // Dispatch TOC ready event for TableOfContentsTab
        const tocItems = convertNavToTocItems(navigation.toc)
        window.dispatchEvent(
          new CustomEvent("epub-outline-ready", {
            detail: { documentId, items: tocItems }
          })
        )

        // Display the book - start from saved position if available
        if (currentCfi) {
          await rendition.display(currentCfi)
        } else {
          await rendition.display()
        }

        if (!mounted) {
          rendition.destroy()
          book.destroy()
          return
        }

        // Set up location change handler
        rendition.on("relocated", (location: any) => {
          if (!mounted) return

          const relocated = applyRelocatedLocation(book, location, {
            setCurrentCfi,
            setCurrentPercentage,
            setCurrentPage
          })
          if (!relocated) return
          const { cfi, percentage } = relocated

          // Find chapter info
          let chapterTitle: string | undefined
          let chapterIndex: number | undefined

          const spine = book.spine as any
          const href = location?.start?.href
          const chapter = href ? spine?.get?.(href) : null
          if (chapter) {
            chapterIndex = chapter.index
          }

          // Find chapter title from TOC
          const findTocItem = (items: NavItem[]): NavItem | undefined => {
            for (const item of items) {
              if (item.href && location?.start?.href?.includes(item.href.split("#")[0])) {
                return item
              }
              if (item.subitems) {
                const found = findTocItem(item.subitems)
                if (found) return found
              }
            }
            return undefined
          }
          const tocItem = findTocItem(navigation.toc)
          if (tocItem) {
            chapterTitle = tocItem.label.trim()
          }

          // Update chapter title in store for annotation creation
          setCurrentChapterTitle(chapterTitle ?? null)

          onLocationChange?.({
            cfi,
            percentage: percentage * 100,
            chapterIndex,
            chapterTitle
          })
        })

        // Set up text selection handler
        rendition.on("selected", (cfiRange: string, contents: any) => {
          if (!mounted) return

          try {
            const range = rendition.getRange(cfiRange)
            if (!range) return

            const text = range.toString().trim()
            if (text.length === 0) return

            // Get the bounding rect relative to viewport
            const rect = range.getBoundingClientRect()

            setSelection({
              text,
              cfi: cfiRange,
              rect
            })
          } catch (e) {
            console.error("Selection error:", e)
          }
        })

        // Clear selection when clicking elsewhere
        rendition.on("click", () => {
          setSelection(null)
        })

        // Register all themes with typography overrides
        const typographyOverrides = buildTypographyOverrides(initialFontFamily, initialFontSize, initialLineHeight)
        Object.entries(EPUB_THEMES).forEach(([name, styles]) => {
          rendition.themes.register(name, {
            ...styles,
            body: {
              ...styles.body,
              ...typographyOverrides
            }
          })
        })

        // Apply current theme
        rendition.themes.select(initialTheme)

        setIsLoading(false)
        const spineCount = (book.spine as any)?.length ?? 0
        onLoadSuccess?.({
          chapterCount: spineCount,
          toc: navigation.toc
        })
      } catch (err) {
        if (mounted) {
          const error = err instanceof Error ? err : new Error("Failed to load EPUB")
          setError(error.message)
          setIsLoading(false)
          onLoadError?.(error)
        }
      }
    }

    initEpub()

    return () => {
      mounted = false
      if (renditionRef.current) {
        renditionRef.current.destroy()
        renditionRef.current = null
      }
      if (bookRef.current) {
        bookRef.current.destroy()
        bookRef.current = null
      }
    }
  }, [url]) // Only re-init on URL change

  // Listen for navigation events from TOC (href-based)
  useEffect(() => {
    const handleNavigate = (e: CustomEvent<{ href: string; documentId: number }>) => {
      if (e.detail.documentId !== documentId) return

      const rendition = renditionRef.current
      if (rendition) {
        rendition.display(e.detail.href)
      }
    }

    window.addEventListener("epub-navigate", handleNavigate as EventListener)

    return () => {
      window.removeEventListener("epub-navigate", handleNavigate as EventListener)
    }
  }, [documentId])

  // Listen for CFI navigation events (from annotations panel)
  useEffect(() => {
    const handleNavigateCfi = (e: CustomEvent<{ cfi: string; documentId: number }>) => {
      if (e.detail.documentId !== documentId) return

      const rendition = renditionRef.current
      if (rendition) {
        rendition.display(e.detail.cfi)
      }
    }

    window.addEventListener("epub-navigate-cfi", handleNavigateCfi as EventListener)

    return () => {
      window.removeEventListener("epub-navigate-cfi", handleNavigateCfi as EventListener)
    }
  }, [documentId])

  // Render highlights from annotations
  // NOTE: epubScrollMode is included as a dependency because switching scroll modes
  // destroys and recreates the rendition, so highlights need to be re-applied
  useEffect(() => {
    const rendition = renditionRef.current
    if (!rendition || isLoading) return

    // Clear existing highlights
    // Note: epub.js doesn't have a clear all method, so we track added highlights
    const highlightIds: string[] = []

    // Add highlights for EPUB annotations (those with CFI locations)
    annotations
      .filter((ann): ann is Annotation & { location: string } =>
        typeof ann.location === "string" &&
        ann.documentId === documentId &&
        ann.annotationType === "highlight"
      )
      .forEach((ann) => {
        try {
          rendition.annotations.highlight(
            ann.location,
            { id: ann.id },
            undefined, // onClick callback - optional
            `highlight-${ann.color}`,
            { fill: HIGHLIGHT_COLORS[ann.color], "fill-opacity": "1" }
          )
          highlightIds.push(ann.id)
        } catch (e) {
          // Highlight CFI may not be valid for current view
          console.debug("Could not render highlight:", ann.id, e)
        }
      })

    // Cleanup function to remove highlights
    return () => {
      highlightIds.forEach((id) => {
        try {
          rendition.annotations.remove(annotations.find(a => a.id === id)?.location as string, "highlight")
        } catch (e) {
          // Ignore cleanup errors
        }
      })
    }
  }, [annotations, documentId, isLoading, epubScrollMode])

  // Handle navigation via store's currentPage (which maps to location index)
  useEffect(() => {
    const book = bookRef.current
    const rendition = renditionRef.current

    if (!book || !rendition || isLoading) return

    // Convert page number to CFI and navigate
    const cfi = book.locations.cfiFromLocation(currentPage - 1)
    if (cfi) {
      rendition.display(cfi)
    }
  }, [currentPage, isLoading])

  // Keyboard navigation
  useEffect(() => {
    const rendition = renditionRef.current
    if (!rendition) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if in input
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return
      }

      switch (e.key) {
        case "ArrowRight":
        case "PageDown":
          e.preventDefault()
          rendition.next()
          break
        case "ArrowLeft":
        case "PageUp":
          e.preventDefault()
          rendition.prev()
          break
      }
    }

    // Also handle rendition keyboard events
    rendition.on("keydown", handleKeyDown)

    return () => {
      rendition.off("keydown", handleKeyDown)
    }
  }, [isLoading])

  // Handle theme changes
  useEffect(() => {
    const rendition = renditionRef.current
    if (!rendition || isLoading) return

    rendition.themes.select(epubTheme)
  }, [epubTheme, isLoading])

  // Handle layout changes that require re-rendering the book
  // (scroll mode, spread mode, font size, font family, line height)
  useEffect(() => {
    const book = bookRef.current
    const rendition = renditionRef.current
    if (!book || !rendition || isLoading || !containerRef.current) return

    // Get current location before destroying rendition
    const currentLocation = rendition.currentLocation()
    const currentCfiFromStore = useDocumentWorkspaceStore.getState().currentCfi

    // Destroy current rendition
    rendition.destroy()

    // Create new rendition with updated settings
    const newRendition = book.renderTo(containerRef.current, {
      width: "100%",
      height: "100%",
      spread: epubSpreadMode,
      flow: epubScrollMode === "continuous" ? "scrolled" : "paginated"
    })

    renditionRef.current = newRendition

    // Re-register themes with current typography
    const typographyOverrides = buildTypographyOverrides(epubFontFamily, epubFontSize, epubLineHeight)
    Object.entries(EPUB_THEMES).forEach(([name, styles]) => {
      newRendition.themes.register(name, {
        ...styles,
        body: {
          ...styles.body,
          ...typographyOverrides
        }
      })
    })

    // Apply current theme
    newRendition.themes.select(epubTheme)

    // Display at previous location
    const targetCfi = currentCfiFromStore || ((currentLocation as any)?.start?.cfi)
    if (targetCfi) {
      newRendition.display(targetCfi)
    } else {
      newRendition.display()
    }

    // Re-setup event handlers
    newRendition.on("relocated", (location: any) => {
      const store = useDocumentWorkspaceStore.getState()
      applyRelocatedLocation(book, location, {
        setCurrentCfi: store.setCurrentCfi,
        setCurrentPercentage: store.setCurrentPercentage,
        setCurrentPage: store.setCurrentPage
      })
    })

    newRendition.on("selected", (cfiRange: string) => {
      try {
        const range = newRendition.getRange(cfiRange)
        if (!range) return

        const text = range.toString().trim()
        if (text.length === 0) return

        const rect = range.getBoundingClientRect()
        setSelection({ text, cfi: cfiRange, rect })
      } catch (e) {
        console.error("Selection error:", e)
      }
    })

    newRendition.on("click", () => {
      setSelection(null)
    })
  }, [epubScrollMode, epubSpreadMode, epubFontSize, epubFontFamily, epubLineHeight])

  const clearSelection = useCallback(() => {
    setSelection(null)
  }, [])

  if (!url) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <DesignSystemAlert
          variant="warning"
          title={t("option:documentWorkspace.noUrl", "No document URL")}
        >
          {t(
            "option:documentWorkspace.selectDocument",
            "Please select a document to view"
          )}
        </DesignSystemAlert>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <DesignSystemAlert
          variant="error"
          title={t("option:documentWorkspace.loadError", "Failed to load EPUB")}
        >
          {error}
        </DesignSystemAlert>
      </div>
    )
  }

  return (
    <div className="relative h-full w-full">
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface/80">
          <div className="flex flex-col items-center gap-2">
            <Spin size="large" />
            <div className="text-sm text-text-muted">
              {t("option:documentWorkspace.loading", "Loading...")}
            </div>
          </div>
        </div>
      )}

      {/* Search overlay */}
      <EpubSearch bookRef={bookRef} renditionRef={renditionRef} />

      {/* Text Selection Popover */}
      {selection && selection.text.length > 0 && (
        <TextSelectionPopover
          text={selection.text}
          position={{
            x: selection.rect.left + selection.rect.width / 2 - 80,
            y: selection.rect.bottom + 8
          }}
          onClose={clearSelection}
          epubCfi={selection.cfi}
        />
      )}

      {/* EPUB container */}
      <div
        ref={containerRef}
        className="h-full w-full"
        style={{
          // epub.js needs explicit dimensions
          minHeight: "400px"
        }}
      />
    </div>
  )
}

export default EpubViewer
