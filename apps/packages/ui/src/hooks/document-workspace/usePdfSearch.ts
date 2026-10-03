import { useState, useCallback, useEffect, useRef } from "react"
import type { DocumentProps } from "react-pdf"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"

export interface SearchResult {
  page: number
  text: string
  matchIndex: number
  itemIndex: number
}

interface TextItem {
  str: string
  transform: number[]
}

export type PdfDocumentProxy = Parameters<
  NonNullable<DocumentProps["onLoadSuccess"]>
>[0]

interface SearchIndex {
  pageTexts: Map<number, string>
  pageItems: Map<number, TextItem[]>
}

/**
 * Escape special regex characters in a string.
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

/**
 * Build a regex for the given query, respecting case sensitivity and word boundary options.
 */
function buildSearchRegex(query: string, matchCase: boolean, wordBoundary: boolean): RegExp {
  let pattern = escapeRegex(query)
  if (wordBoundary) {
    pattern = `\\b${pattern}\\b`
  }
  const flags = matchCase ? "g" : "gi"
  return new RegExp(pattern, flags)
}

/**
 * Find all matches of a query in text, respecting case sensitivity and word boundary options.
 */
function findMatches(
  text: string,
  query: string,
  matchCase: boolean,
  wordBoundary: boolean
): number[] {
  if (!query) return []

  const regex = buildSearchRegex(query, matchCase, wordBoundary)
  const positions: number[] = []

  let match: RegExpExecArray | null
  while ((match = regex.exec(text)) !== null) {
    positions.push(match.index)
    // Prevent infinite loop on zero-length matches
    if (match.index === regex.lastIndex) {
      regex.lastIndex++
    }
  }

  return positions
}

/**
 * Hook for PDF in-document text search.
 *
 * Provides search functionality including:
 * - Building a searchable text index from the PDF
 * - Finding matches across all pages
 * - Navigating between results
 * - Highlighting matching text spans
 */
export function usePdfSearch(
  pdfDocumentRef: React.RefObject<PdfDocumentProxy | null>
) {
  const [searchIndex, setSearchIndex] = useState<SearchIndex | null>(null)
  const [isIndexing, setIsIndexing] = useState(false)
  const [debouncedQuery, setDebouncedQuery] = useState("")
  const highlightedElementsRef = useRef<HTMLElement[]>([])
  const pendingScrollPageRef = useRef<number | null>(null)

  const searchOpen = useDocumentWorkspaceStore((s) => s.searchOpen)
  const searchQuery = useDocumentWorkspaceStore((s) => s.searchQuery)
  const searchResults = useDocumentWorkspaceStore((s) => s.searchResults)
  const activeSearchIndex = useDocumentWorkspaceStore((s) => s.activeSearchIndex)
  const setSearchOpen = useDocumentWorkspaceStore((s) => s.setSearchOpen)
  const setSearchQuery = useDocumentWorkspaceStore((s) => s.setSearchQuery)
  const setSearchResults = useDocumentWorkspaceStore((s) => s.setSearchResults)
  const setActiveSearchIndex = useDocumentWorkspaceStore((s) => s.setActiveSearchIndex)
  const clearSearch = useDocumentWorkspaceStore((s) => s.clearSearch)
  const setCurrentPage = useDocumentWorkspaceStore((s) => s.setCurrentPage)
  const searchMatchCase = useDocumentWorkspaceStore((s) => s.searchMatchCase)
  const searchWordBoundary = useDocumentWorkspaceStore((s) => s.searchWordBoundary)

  /**
   * Build search index from PDF document.
   * Extracts text content from all pages.
   */
  const buildSearchIndex = useCallback(async () => {
    const pdfDocument = pdfDocumentRef.current
    if (!pdfDocument) return

    setIsIndexing(true)

    try {
      const pageTexts = new Map<number, string>()
      const pageItems = new Map<number, TextItem[]>()

      for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {
        const page = await pdfDocument.getPage(pageNum)
        const textContent = await page.getTextContent()

        const rawItems = textContent.items as Array<{
          str?: string
          transform?: number[]
        }>
        const items: TextItem[] = []
        rawItems.forEach((item) => {
          if (typeof item.str === "string") {
            items.push({
              str: item.str,
              transform: Array.isArray(item.transform) ? item.transform : []
            })
          }
        })

        pageItems.set(pageNum, items)
        pageTexts.set(pageNum, items.map((item) => item.str).join(" "))
      }

      setSearchIndex({ pageTexts, pageItems })
    } catch (error) {
      console.error("Error building search index:", error)
    } finally {
      setIsIndexing(false)
    }
  }, [pdfDocumentRef])

  /**
   * Search for query across all pages.
   */
  const search = useCallback(
    (query: string) => {
      if (!searchIndex || !query.trim()) {
        setSearchResults([])
        return
      }

      const results: SearchResult[] = []

      searchIndex.pageTexts.forEach((pageText, page) => {
        const positions = findMatches(pageText, query, searchMatchCase, searchWordBoundary)

        positions.forEach((pos, matchIndex) => {
          // Find which item this match belongs to
          const items = searchIndex.pageItems.get(page) || []
          let currentPos = 0
          let itemIndex = 0

          for (let i = 0; i < items.length; i++) {
            const itemLength = items[i].str.length + 1 // +1 for space
            if (currentPos + itemLength > pos) {
              itemIndex = i
              break
            }
            currentPos += itemLength
          }

          results.push({
            page,
            text: pageText.slice(Math.max(0, pos - 20), pos + query.length + 20),
            matchIndex,
            itemIndex
          })
        })
      })

      // Sort by page number
      results.sort((a, b) => a.page - b.page || a.matchIndex - b.matchIndex)

      setSearchResults(results)
    },
    [searchIndex, setSearchResults, searchMatchCase, searchWordBoundary]
  )

  /**
   * Highlight matching text spans in the PDF text layer.
   */
  const highlightMatches = useCallback(
    (query: string, activeIndex: number = -1, scrollToActive = true) => {
      // Clear previous highlights
      highlightedElementsRef.current.forEach((el) => {
        el.classList.remove("pdf-search-match", "pdf-search-match-active")
      })
      highlightedElementsRef.current = []

      if (!query.trim()) return

      const regex = buildSearchRegex(query, searchMatchCase, searchWordBoundary)

      // Find all text spans in the PDF text layer
      const textLayers = document.querySelectorAll(".react-pdf__Page__textContent")

      textLayers.forEach((layer) => {
        const spans = layer.querySelectorAll("span")

        spans.forEach((span) => {
          const text = span.textContent || ""
          // Reset lastIndex for each span since we reuse the regex
          regex.lastIndex = 0
          if (regex.test(text)) {
            span.classList.add("pdf-search-match")
            highlightedElementsRef.current.push(span as HTMLElement)
          }
        })
      })

      // Highlight active result
      if (activeIndex >= 0 && activeIndex < searchResults.length) {
        const activeResult = searchResults[activeIndex]
        // Find the specific element for the active result on its page
        const pageContainer = document.querySelector(
          `[data-page-number="${activeResult.page}"]`
        )
        if (pageContainer) {
          const textLayer = pageContainer.querySelector(".react-pdf__Page__textContent")
          if (textLayer) {
            const spans = textLayer.querySelectorAll("span.pdf-search-match")
            // Mark the first match on this page as active (simplified logic)
            const activeSpan = spans[0]
            if (activeSpan) {
              activeSpan.classList.add("pdf-search-match-active")
              if (scrollToActive) {
                pendingScrollPageRef.current = null
                activeSpan.scrollIntoView({ behavior: "smooth", block: "center" })
              }
            }
          }
        }
      }
    },
    [searchResults, searchMatchCase, searchWordBoundary]
  )

  /** Navigate immediately if the layer is ready, otherwise on its render event. */
  const navigateToResult = useCallback(
    (index: number) => {
      if (index < 0 || index >= searchResults.length) return
      const result = searchResults[index]
      pendingScrollPageRef.current = result.page
      setActiveSearchIndex(index)
      setCurrentPage(result.page)
      highlightMatches(searchQuery, index)
    },
    [searchResults, setActiveSearchIndex, setCurrentPage, highlightMatches, searchQuery]
  )

  const goToNextResult = useCallback(() => {
    if (searchResults.length === 0) return
    navigateToResult((activeSearchIndex + 1) % searchResults.length)
  }, [activeSearchIndex, searchResults.length, navigateToResult])

  const goToPreviousResult = useCallback(() => {
    if (searchResults.length === 0) return
    navigateToResult(activeSearchIndex === 0 ? searchResults.length - 1 : activeSearchIndex - 1)
  }, [activeSearchIndex, searchResults.length, navigateToResult])

  /** Clear all highlights. */
  const clearHighlights = useCallback(() => {
    highlightedElementsRef.current.forEach((el) => {
      el.classList.remove("pdf-search-match", "pdf-search-match-active")
    })
    highlightedElementsRef.current = []
  }, [])

  // Virtual pages create their text layers after navigation/search effects.
  // Reapply highlights when rendering finishes, without jumping on user scroll.
  useEffect(() => {
    if (!searchOpen) return
    const onTextLayerReady = (event: Event) => {
      const page = Number((event.target as HTMLElement).dataset.pageNumber)
      highlightMatches(searchQuery, activeSearchIndex, pendingScrollPageRef.current === page)
    }
    document.addEventListener("pdf-text-layer-rendered", onTextLayerReady)
    return () => {
      document.removeEventListener("pdf-text-layer-rendered", onTextLayerReady)
      clearHighlights()
    }
  }, [searchOpen, searchQuery, activeSearchIndex, highlightMatches, clearHighlights])

  /**
   * Open search overlay.
   */
  const openSearch = useCallback(() => {
    setSearchOpen(true)
    // Build index if not already built
    if (!searchIndex && !isIndexing) {
      buildSearchIndex()
    }
  }, [setSearchOpen, searchIndex, isIndexing, buildSearchIndex])

  /**
   * Close search overlay and clear results.
   */
  const closeSearch = useCallback(() => {
    setSearchOpen(false)
    clearSearch()
    clearHighlights()
  }, [setSearchOpen, clearSearch, clearHighlights])

  // Build search index when PDF document changes
  useEffect(() => {
    if (pdfDocumentRef.current && !searchIndex && !isIndexing) {
      buildSearchIndex()
    }
  }, [pdfDocumentRef.current, searchIndex, isIndexing, buildSearchIndex])

  // Debounce search query to prevent jank on large PDFs
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery)
    }, 300)
    return () => clearTimeout(timer)
  }, [searchQuery])

  // Perform search when debounced query changes
  useEffect(() => {
    if (searchOpen && debouncedQuery) {
      search(debouncedQuery)
    }
  }, [searchOpen, debouncedQuery, search])

  // Update highlights when results change
  useEffect(() => {
    if (searchOpen && searchResults.length > 0) {
      highlightMatches(searchQuery, activeSearchIndex)
    }
  }, [searchOpen, searchResults, activeSearchIndex, searchQuery, highlightMatches])

  // Clear highlights when search closes
  useEffect(() => {
    if (!searchOpen) {
      clearHighlights()
    }
  }, [searchOpen, clearHighlights])

  return {
    // State
    searchOpen,
    searchQuery,
    searchResults,
    activeSearchIndex,
    isIndexing,

    // Actions
    setSearchQuery,
    openSearch,
    closeSearch,
    goToNextResult,
    goToPreviousResult,
    navigateToResult,
    highlightMatches,
    clearHighlights
  }
}
