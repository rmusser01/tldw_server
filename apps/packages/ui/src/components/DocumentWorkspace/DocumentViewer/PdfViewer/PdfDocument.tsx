import React, { useCallback, useState, useRef, useEffect, useLayoutEffect } from "react"
import { Document, pdfjs } from "react-pdf"
import type { DocumentProps } from "react-pdf"
import { Spin, Alert } from "antd"
import { PdfPage } from "./PdfPage"
import { TextSelectionPopover } from "../TextSelectionPopover"
import { useTextSelection } from "@/hooks/document-workspace/useTextSelection"
import type { PdfDocumentProxy } from "@/hooks/document-workspace/usePdfSearch"
import { getBrowserRuntime, isExtensionRuntime } from "@/utils/browser-runtime"
import type { ViewMode } from "../../types"

// Configure PDF.js worker
// For Next.js: The worker is copied to public/ during postinstall (scripts/copy-pdf-worker.mjs)
// For browser extension: Uses bundled worker via runtime.getURL (with CDN fallback)
// For development: Uses CDN for simplicity
function getPdfWorkerSrc(): string {
  // CDN fallback URL
  const cdnUrl = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

  // SSR check
  if (typeof window === "undefined") {
    return cdnUrl
  }

  // In extension runtime, use the packaged worker file from the extension bundle.
  const runtime = getBrowserRuntime()
  const inExtensionRuntime = isExtensionRuntime(runtime)
  if (inExtensionRuntime) {
    return runtime?.getURL ? runtime.getURL("pdf.worker.min.mjs") : cdnUrl
  }

  // In Next.js production builds, use the local worker from public/
  // The file is copied by scripts/copy-pdf-worker.mjs during postinstall
  if (process.env.NODE_ENV === "production") {
    return "/pdf.worker.min.mjs"
  }

  // Development and other environments: use CDN
  return cdnUrl
}

pdfjs.GlobalWorkerOptions.workerSrc = getPdfWorkerSrc()

interface PdfDocumentProps {
  url?: string
  documentId: number
  currentPage: number
  zoomLevel: number
  viewMode: ViewMode
  onLoadSuccess: (numPages: number) => void
  onLoadError: (error: Error) => void
  onPageChange: (page: number) => void
  pdfDocumentRef?: React.MutableRefObject<PdfDocumentProxy | null>
}

export const PdfDocument: React.FC<PdfDocumentProps> = ({
  url,
  documentId,
  currentPage,
  zoomLevel,
  viewMode,
  onLoadSuccess,
  onLoadError,
  onPageChange,
  pdfDocumentRef
}) => {
  const [numPages, setNumPages] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [pdfInstance, setPdfInstance] = useState<PdfDocumentProxy | null>(null)
  const [pageMetrics, setPageMetrics] = useState<{ height: number; width: number }>({
    height: 0,
    width: 0
  })
  const latestPageHeightRef = useRef(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const isUserScrollingRef = useRef(false)
  const scrollTimeoutRef = useRef<number | null>(null)
  const scrollRafRef = useRef<number | null>(null)
  const wheelAccumulatorRef = useRef(0)
  const wheelResetRef = useRef<number | null>(null)
  const [pageHeights, setPageHeights] = useState<number[]>([])
  const [pageOffsets, setPageOffsets] = useState<number[]>([])

  // Text selection for popover actions
  const { selection, clearSelection } = useTextSelection(containerRef)

  const updatePageMetrics = useCallback((metrics: { height: number; width: number }) => {
    latestPageHeightRef.current = metrics.height
    setPageMetrics(metrics)
  }, [])

  const handleDocumentLoadSuccess = useCallback<NonNullable<DocumentProps["onLoadSuccess"]>>(
    (pdf) => {
      setNumPages(pdf.numPages)
      setLoading(false)
      setError(null)
      setPdfInstance(pdf)
      onLoadSuccess(pdf.numPages)
      // Store reference for search functionality
      if (pdfDocumentRef) {
        pdfDocumentRef.current = pdf
      }
    },
    [onLoadSuccess, pdfDocumentRef]
  )

  const handleDocumentLoadError = useCallback(
    (error: Error) => {
      setLoading(false)
      setError(error.message || "Failed to load PDF")
      setPdfInstance(null)
      if (pdfDocumentRef) {
        pdfDocumentRef.current = null
      }
      onLoadError(error)
    },
    [onLoadError, pdfDocumentRef]
  )

  // Scroll to current page in continuous mode
  useEffect(() => {
    if (viewMode === "continuous" && numPages > 0) {
      const pageElement = pageRefs.current.get(currentPage)
      if (pageElement) {
        pageElement.scrollIntoView({ behavior: "smooth", block: "start" })
      }
    }
  }, [currentPage, viewMode, numPages])

  // Handle page click in thumbnail mode
  const handleThumbnailClick = useCallback(
    (pageNumber: number) => {
      onPageChange(pageNumber)
    },
    [onPageChange]
  )

  // Intersection observer for continuous scroll page tracking
  useEffect(() => {
    if (viewMode !== "continuous" || numPages === 0) return

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const pageNumber = parseInt(
              entry.target.getAttribute("data-page-number") || "1",
              10
            )
            onPageChange(pageNumber)
          }
        })
      },
      {
        root: containerRef.current,
        threshold: 0.5
      }
    )

    pageRefs.current.forEach((element) => {
      observer.observe(element)
    })

    return () => {
      observer.disconnect()
    }
  }, [viewMode, numPages, onPageChange])

  const setPageRef = useCallback(
    (pageNumber: number, element: HTMLDivElement | null) => {
      if (element) {
        pageRefs.current.set(pageNumber, element)
      } else {
        pageRefs.current.delete(pageNumber)
      }
    },
    []
  )

  // Fallback measurement based on rendered DOM height (more reliable in extension).
  // Note: `pdfInstance.getPage()` provides initial metrics; ResizeObserver corrects to
  // the actual rendered size once the page is in the DOM.
  useLayoutEffect(() => {
    if (viewMode !== "single") return
    const pageElement = pageRefs.current.get(currentPage)
    if (!pageElement) return
    const rect = pageElement.getBoundingClientRect()
    const latestHeight = latestPageHeightRef.current
    if (rect.height > 0 && Math.abs(rect.height - latestHeight) > 1) {
      updatePageMetrics({ height: rect.height, width: rect.width })
    }

    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const { height, width } = entry.contentRect
      if (height > 0 && Math.abs(height - latestPageHeightRef.current) > 1) {
        updatePageMetrics({ height, width })
      }
    })
    observer.observe(pageElement)
    return () => observer.disconnect()
  }, [viewMode, currentPage, zoomLevel, loading, updatePageMetrics])

  const pageGap = 16

  // Compute per-page dimensions for virtual single-page scrolling.
  useEffect(() => {
    if (!pdfInstance) {
      setPageHeights([])
      setPageOffsets([])
      return
    }
    let cancelled = false
    const s = zoomLevel / 100
    // Reset immediately so stale values from a previous scale aren't used
    setPageHeights([])
    setPageOffsets([])
    const computeAllMetrics = async () => {
      try {
        const heights: number[] = []
        const offsets: number[] = []
        let offset = 0
        let firstHeight = 0
        let firstWidth = 0
        const batchSize = 10
        for (let start = 1; start <= pdfInstance.numPages; start += batchSize) {
          if (cancelled) return
          const end = Math.min(start + batchSize - 1, pdfInstance.numPages)
          const pages = await Promise.all(
            Array.from({ length: end - start + 1 }, (_, index) => pdfInstance.getPage(start + index))
          )
          if (cancelled) return

          for (let index = 0; index < pages.length; index++) {
            if (cancelled) return
            const pageNumber = start + index
            const viewport = pages[index].getViewport({ scale: s })
            if (pageNumber === 1) {
              firstHeight = viewport.height
              firstWidth = viewport.width
            }
            heights.push(viewport.height)
            offsets.push(offset)
            offset += viewport.height + pageGap
          }
        }
        if (!cancelled) {
          setPageHeights(heights)
          setPageOffsets(offsets)
          updatePageMetrics({ height: firstHeight, width: firstWidth })
        }
      } catch (error) {
        if (!cancelled) {
          console.warn('[PdfDocument] Failed to compute page metrics', error)
          setPageHeights([])
          setPageOffsets([])
          updatePageMetrics({ height: 0, width: 0 })
        }
      }
    }
    void computeAllMetrics()
    return () => {
      cancelled = true
    }
  }, [pdfInstance, zoomLevel, updatePageMetrics])

  const scale = zoomLevel / 100
  const fallbackPageHeight = 1100 * scale
  const fallbackPageWidth = 800 * scale
  const basePageHeight = pageMetrics.height > 0 ? pageMetrics.height : fallbackPageHeight
  const basePageWidth = pageMetrics.width > 0 ? pageMetrics.width : fallbackPageWidth
  const virtualPageHeight = basePageHeight + pageGap
  const totalPageCount =
    numPages || pdfDocumentRef?.current?.numPages || 0

  // Disable virtual scroll when per-page heights vary too much (mixed-size PDFs)
  const perPageReady = pageOffsets.length === totalPageCount && totalPageCount > 0
  const heightVarianceHigh = pageHeights.length > 1 && (() => {
    const mean = pageHeights.reduce((s, h) => s + h, 0) / pageHeights.length
    if (mean === 0) return false
    const maxDev = pageHeights.reduce((m, h) => Math.max(m, Math.abs(h - mean)), 0)
    return maxDev / mean > 0.3
  })()
  const virtualScrollEnabled =
    viewMode === "single" && perPageReady && !heightVarianceHigh

  const totalVirtualHeight = perPageReady
    ? pageOffsets[pageOffsets.length - 1] + pageHeights[pageHeights.length - 1] + pageGap
    : virtualPageHeight * totalPageCount

  // Binary search: find 1-based page whose offset region contains scrollTop
  const findPageAtOffset = useCallback(
    (scrollTop: number): number => {
      if (pageOffsets.length === 0) return 1
      let lo = 0
      let hi = pageOffsets.length - 1
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1
        if (pageOffsets[mid] <= scrollTop) lo = mid
        else hi = mid - 1
      }
      return Math.min(totalPageCount, Math.max(1, lo + 1))
    },
    [pageOffsets, totalPageCount]
  )

  const handleVirtualScroll = useCallback(() => {
    if (!virtualScrollEnabled || !containerRef.current) return
    const container = containerRef.current

    isUserScrollingRef.current = true
    if (scrollTimeoutRef.current) {
      window.clearTimeout(scrollTimeoutRef.current)
    }
    scrollTimeoutRef.current = window.setTimeout(() => {
      isUserScrollingRef.current = false
    }, 120)

    if (scrollRafRef.current) {
      cancelAnimationFrame(scrollRafRef.current)
    }
    scrollRafRef.current = requestAnimationFrame(() => {
      const nextPage = findPageAtOffset(container.scrollTop)
      if (nextPage !== currentPage) {
        onPageChange(nextPage)
      }
    })
  }, [virtualScrollEnabled, currentPage, onPageChange, findPageAtOffset])

  useEffect(() => {
    if (!virtualScrollEnabled || !containerRef.current) return
    const container = containerRef.current
    const targetTop = pageOffsets[currentPage - 1] ?? 0
    if (isUserScrollingRef.current) return
    if (Math.abs(container.scrollTop - targetTop) > 4) {
      container.scrollTop = targetTop
    }
  }, [virtualScrollEnabled, currentPage, pageOffsets])

  useEffect(() => {
    // Fallback path only when virtual scrolling is disabled (e.g., page count
    // not yet resolved). This keeps wheel paging from overriding normal scroll.
    if (viewMode !== "single" || virtualScrollEnabled) return
    const container = containerRef.current
    if (!container) return

    const handleWheel = (event: WheelEvent) => {
      if (totalPageCount <= 1) return

      wheelAccumulatorRef.current += event.deltaY
      if (wheelResetRef.current) {
        window.clearTimeout(wheelResetRef.current)
      }
      wheelResetRef.current = window.setTimeout(() => {
        wheelAccumulatorRef.current = 0
      }, 200)

      const threshold = 120
      if (Math.abs(wheelAccumulatorRef.current) >= threshold) {
        event.preventDefault()
        const direction = wheelAccumulatorRef.current > 0 ? 1 : -1
        const nextPage = Math.min(
          totalPageCount,
          Math.max(1, currentPage + direction)
        )
        if (nextPage !== currentPage) {
          onPageChange(nextPage)
        }
        wheelAccumulatorRef.current = 0
      }
    }

    container.addEventListener("wheel", handleWheel, { passive: false })
    return () => {
      container.removeEventListener("wheel", handleWheel)
      if (wheelResetRef.current) {
        window.clearTimeout(wheelResetRef.current)
        wheelResetRef.current = null
      }
      wheelAccumulatorRef.current = 0
    }
  }, [viewMode, virtualScrollEnabled, totalPageCount, currentPage, onPageChange])

  useEffect(() => {
    return () => {
      if (scrollRafRef.current) {
        cancelAnimationFrame(scrollRafRef.current)
        scrollRafRef.current = null
      }
      if (scrollTimeoutRef.current) {
        window.clearTimeout(scrollTimeoutRef.current)
        scrollTimeoutRef.current = null
      }
      if (wheelResetRef.current) {
        window.clearTimeout(wheelResetRef.current)
        wheelResetRef.current = null
      }
      isUserScrollingRef.current = false
      wheelAccumulatorRef.current = 0
    }
  }, [])

  if (!url) {
    return (
      <div className="flex h-full items-center justify-center p-4">
        <Alert
          type="warning"
          title="No document URL"
          description="Please select a document to view"
          showIcon
        />
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="flex h-full min-h-0 w-full flex-col items-center overflow-x-auto overflow-y-auto py-4 px-2 sm:px-4"
      onScroll={virtualScrollEnabled ? handleVirtualScroll : undefined}
    >
      {/* Text Selection Popover */}
      {selection && selection.text.length > 0 && (
        <TextSelectionPopover
          text={selection.text}
          position={{
            x: selection.rect.left + selection.rect.width / 2 - 80, // Center above selection
            y: selection.rect.bottom + 8 // Below selection
          }}
          onClose={clearSelection}
        />
      )}

      <Document
        file={url}
        onLoadSuccess={handleDocumentLoadSuccess}
        onLoadError={handleDocumentLoadError}
        loading={
          <div className="flex h-64 w-full flex-col items-center justify-center gap-2">
            <Spin size="large" />
            <div className="text-sm text-text-muted">Loading document…</div>
          </div>
        }
        error={
          <Alert
            type="error"
            title="Failed to load PDF"
            description={error || "An error occurred while loading the document"}
            showIcon
          />
        }
      >
        {loading ? null : viewMode === "single" ? (
          virtualScrollEnabled ? (
            // Virtualized single-page scroll mode (render neighbors to avoid blank gaps)
            <div
              className="relative"
              style={{
                height: totalVirtualHeight,
                width: basePageWidth,
                minWidth: basePageWidth,
                margin: "0 auto"
              }}
            >
              {[currentPage - 1, currentPage, currentPage + 1]
                .filter(
                  (pageNumber) =>
                    pageNumber >= 1 && pageNumber <= totalPageCount
                )
                .map((pageNumber) => (
                  <div
                    key={`virtual-page-${pageNumber}`}
                    className="absolute left-1/2 -translate-x-1/2"
                    style={{ top: pageOffsets[pageNumber - 1] ?? 0 }}
                  >
                    <PdfPage
                      pageNumber={pageNumber}
                      scale={scale}
                      onSetRef={(el) => setPageRef(pageNumber, el)}
                    />
                  </div>
                ))}
            </div>
          ) : (
            // Fallback single page mode
            <PdfPage
              pageNumber={currentPage}
              scale={scale}
              onSetRef={(el) => setPageRef(currentPage, el)}
            />
          )
        ) : viewMode === "continuous" ? (
          // Continuous scroll mode
          <div className="flex flex-col items-center gap-4">
            {Array.from({ length: numPages }, (_, index) => (
              <PdfPage
                key={`page-${index + 1}`}
                pageNumber={index + 1}
                scale={scale}
                onSetRef={(el) => setPageRef(index + 1, el)}
              />
            ))}
          </div>
        ) : (
          // Thumbnail grid mode
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {Array.from({ length: numPages }, (_, index) => (
              <button
                key={`thumb-${index + 1}`}
                onClick={() => handleThumbnailClick(index + 1)}
                className={`relative cursor-pointer rounded border-2 transition-all hover:shadow-lg ${
                  currentPage === index + 1
                    ? "border-primary shadow-md"
                    : "border-transparent"
                }`}
              >
                <PdfPage
                  pageNumber={index + 1}
                  scale={0.25}
                  onSetRef={(el) => setPageRef(index + 1, el)}
                  hidePageNote
                />
                <span className="absolute bottom-1 left-1/2 -translate-x-1/2 rounded bg-black/70 px-2 py-0.5 text-xs text-white">
                  {index + 1}
                </span>
              </button>
            ))}
          </div>
        )}
      </Document>
    </div>
  )
}

export default PdfDocument
