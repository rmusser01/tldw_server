import { useState, useCallback, useEffect, useReducer, useRef } from "react"
import { smoothScrollBehavior } from "@/utils/reduced-motion"
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

type PdfDoc = NonNullable<PdfDocumentProxy>

interface IndexEntry {
  doc: PdfDoc
  status: "building" | "ready" | "failed"
  index?: SearchIndex
}

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
  // Index state is keyed to the document proxy it was built from, so a new
  // document never sees (or is overwritten by) another document's index.
  const [entry, setEntry] = useState<IndexEntry | null>(null)
  const entryRef = useRef<IndexEntry | null>(null)
  // The ref is assigned by react-pdf without a re-render, so a changed proxy is
  // also noticed from the text-layer event below (forceRender).
  const [, forceRender] = useReducer((n: number) => n + 1, 0)
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const seenActiveIdRef = useRef(activeDocumentId)
  const staleDocRef = useRef<PdfDocumentProxy | null>(null)
  if (seenActiveIdRef.current !== activeDocumentId) {
    // Tab switch: the proxy still in the ref belongs to the previous document
    // until react-pdf assigns the new one.
    seenActiveIdRef.current = activeDocumentId
    staleDocRef.current = pdfDocumentRef.current
  }
  const rawDoc = pdfDocumentRef.current
  const pdfDocument = rawDoc && rawDoc !== staleDocRef.current ? rawDoc : null
  const renderedDocRef = useRef<PdfDocumentProxy | null>(null)
  renderedDocRef.current = rawDoc
  const current = entry && entry.doc === pdfDocument ? entry : null
  const searchIndex = current?.index ?? null
  const isIndexing = current?.status === "building"
  const indexFailed = current?.status === "failed"
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
  const buildSearchIndex = useCallback(async (pdfDocument: PdfDoc) => {
    const publish = (next: IndexEntry) => {
      // A build for a document that is no longer the one being indexed is dropped.
      if (entryRef.current?.doc !== pdfDocument) return
      entryRef.current = next
      setEntry(next)
    }
    const building: IndexEntry = { doc: pdfDocument, status: "building" }
    entryRef.current = building
    setEntry(building)

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

      publish({ doc: pdfDocument, status: "ready", index: { pageTexts, pageItems } })
    } catch (error) {
      console.error("Error building search index:", error)
      publish({ doc: pdfDocument, status: "failed" })
    }
  }, [])

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
                activeSpan.scrollIntoView({ behavior: smoothScrollBehavior(), block: "center" })
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
      if (pdfDocumentRef.current !== renderedDocRef.current) forceRender()
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
  }, [setSearchOpen])

  /**
   * Close search overlay and clear results.
   */
  const closeSearch = useCallback(() => {
    setSearchOpen(false)
    clearSearch()
    clearHighlights()
  }, [setSearchOpen, clearSearch, clearHighlights])

  // Drop a closed document's index (and orphan its in-flight build) when the proxy changes.
  useEffect(() => {
    if (entryRef.current && entryRef.current.doc !== pdfDocument) {
      entryRef.current = null
      setEntry(null)
    }
  }, [pdfDocument])

  // Build lazily when search is open. Deps deliberately exclude build state: a
  // failed build is retried only when search is reopened or the document changes.
  useEffect(() => {
    if (!searchOpen || !pdfDocument) return
    const e = entryRef.current
    if (e?.doc === pdfDocument && e.status !== "failed") return
    void buildSearchIndex(pdfDocument)
  }, [searchOpen, pdfDocument, buildSearchIndex])

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
    indexFailed,

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
