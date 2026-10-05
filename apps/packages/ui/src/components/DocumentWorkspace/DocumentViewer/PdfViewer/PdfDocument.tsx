import React, { useCallback, useState, useRef, useEffect, useLayoutEffect } from "react"
import { useVirtualizer } from "@tanstack/react-virtual"
import { Document, pdfjs } from "react-pdf"
import type { DocumentProps } from "react-pdf"
import { Spin } from "antd"
import { useTranslation } from "react-i18next"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
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
  // React Virtual exposes mutable instance methods; skip compiler memoization.
  "use no memo"
  const { t } = useTranslation(["option"])
  const [numPages, setNumPages] = useState<number>(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [pdfInstance, setPdfInstance] = useState<PdfDocumentProxy | null>(null)
  const [documentWidth, setDocumentWidth] = useState(0)
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
  const navigationRef = useRef<{ page: number; top: number } | null>(null)
  const wheelAccumulatorRef = useRef(0)
  const wheelResetRef = useRef<number | null>(null)
  const [pageHeights, setPageHeights] = useState<number[]>([])
  const [pageOffsets, setPageOffsets] = useState<number[]>([])
  const [metricsFailed, setMetricsFailed] = useState(false)

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

  // Handle page click in thumbnail mode
  const handleThumbnailClick = useCallback(
    (pageNumber: number) => {
      onPageChange(pageNumber)
    },
    [onPageChange]
  )

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
    setMetricsFailed(false)
    if (!pdfInstance) {
      setDocumentWidth(0)
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
        let maxWidth = 0
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
            }
            maxWidth = Math.max(maxWidth, viewport.width)
            heights.push(viewport.height)
            offsets.push(offset)
            offset += viewport.height + pageGap
          }
        }
        if (!cancelled) {
          setPageHeights(heights)
          setPageOffsets(offsets)
          setDocumentWidth(maxWidth)
          updatePageMetrics({ height: firstHeight, width: maxWidth })
        }
      } catch (error) {
        if (!cancelled) {
          console.warn('[PdfDocument] Failed to compute page metrics', error)
          setPageHeights([])
          setPageOffsets([])
          setMetricsFailed(true)
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
  const basePageWidth = viewMode === 'single'
    ? pageMetrics.width > 0 ? pageMetrics.width : fallbackPageWidth
    : documentWidth || fallbackPageWidth
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

  // eslint-disable-next-line react-hooks/incompatible-library -- This component opts out of compiler memoization for React Virtual's mutable API.
  const continuousPages = useVirtualizer({
    count: viewMode === 'continuous' ? totalPageCount : 0,
    getScrollElement: () => containerRef.current,
    estimateSize: index => (pageHeights[index] ?? basePageHeight) + pageGap,
    overscan: 2,
    initialRect: { width: 800, height: 800 },
    useFlushSync: false
  })
  useEffect(() => { continuousPages.measure() }, [continuousPages, pageHeights])

  // Binary search: find 1-based page whose offset region contains scrollTop
  const findPageAtOffset = useCallback(
    (scrollTop: number): number => {
      if (pageOffsets.length === 0) return Math.min(totalPageCount, Math.max(1, Math.floor(scrollTop / virtualPageHeight) + 1))
      let lo = 0
      let hi = pageOffsets.length - 1
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1
        if (pageOffsets[mid] <= scrollTop) lo = mid
        else hi = mid - 1
      }
      return Math.min(totalPageCount, Math.max(1, lo + 1))
    },
    [pageOffsets, totalPageCount, virtualPageHeight]
  )

  const findVisiblePage = useCallback((container: HTMLDivElement) => {
    // A short final page cannot reach the top of the viewport at low zoom.
    if (container.scrollTop > 0 && container.scrollTop + container.clientHeight >= container.scrollHeight - 1) return totalPageCount
    return findPageAtOffset(container.scrollTop)
  }, [totalPageCount, findPageAtOffset])

  const scrollToPage = useCallback((page: number) => {
    if (viewMode === 'continuous' && containerRef.current) {
      const container = containerRef.current
      const targetTop = pageOffsets[page - 1] ?? (page - 1) * virtualPageHeight
      navigationRef.current = {
        page,
        top: Math.max(0, Math.min(targetTop, container.scrollHeight - container.clientHeight))
      }
      continuousPages.scrollToIndex(page - 1, { align: 'start' })
    } else pageRefs.current.get(page)?.scrollIntoView()
  }, [viewMode, pageOffsets, virtualPageHeight, continuousPages])

  // React-PDF retains its initial link handler, so delegate to current navigation.
  const pageNavigationRef = useRef({ onPageChange, scrollToPage })
  useLayoutEffect(() => {
    pageNavigationRef.current = { onPageChange, scrollToPage }
  }, [onPageChange, scrollToPage])
  const handleInternalLink = useCallback<NonNullable<DocumentProps['onItemClick']>>(({ pageNumber }) => {
    pageNavigationRef.current.onPageChange(pageNumber)
    pageNavigationRef.current.scrollToPage(pageNumber)
  }, [])

  useEffect(() => {
    if (viewMode !== 'continuous') navigationRef.current = null
    if (viewMode !== 'continuous' || !totalPageCount || !containerRef.current) return
    // Scroll tracking feeds currentPage back to us; explicit navigation also
    // reaches pages not currently mounted, including clamped tail pages.
    if (findVisiblePage(containerRef.current) !== currentPage) scrollToPage(currentPage)
  }, [viewMode, totalPageCount, currentPage, pageHeights, scrollToPage, findVisiblePage])

  const handleVirtualScroll = useCallback(() => {
    if ((!virtualScrollEnabled && viewMode !== "continuous") || !containerRef.current) return
    // Estimated offsets change while metadata loads; keep explicit navigation
    // until real dimensions arrive, or use the estimate if metadata fails.
    if (viewMode === "continuous" && !perPageReady && !metricsFailed) return
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
      const navigation = navigationRef.current
      navigationRef.current = null
      const nextPage = navigation && Math.abs(container.scrollTop - navigation.top) < 4
        ? navigation.page
        : findVisiblePage(container)
      if (nextPage !== currentPage) {
        onPageChange(nextPage)
      }
    })
  }, [virtualScrollEnabled, viewMode, perPageReady, metricsFailed, currentPage, onPageChange, findVisiblePage])

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
        <DesignSystemAlert
          variant="warning"
          title="No document URL"
        >
          {t("option:documentWorkspace.selectDocument", "Please select a document to view")}
        </DesignSystemAlert>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="flex h-full min-h-0 w-full flex-col items-center overflow-x-auto overflow-y-auto py-4 px-2 sm:px-4"
      onScroll={virtualScrollEnabled || viewMode === "continuous" ? handleVirtualScroll : undefined}
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
        onItemClick={handleInternalLink}
        loading={
          <div className="flex h-64 w-full flex-col items-center justify-center gap-2">
            <Spin size="large" />
            <div className="text-sm text-text-muted">Loading document…</div>
          </div>
        }
        error={
          <DesignSystemAlert
            variant="error"
            title="Failed to load PDF"
          >
            {error || t("option:documentWorkspace.genericLoadError", "An error occurred while loading the document")}
          </DesignSystemAlert>
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
          <div className="relative" style={{ height: continuousPages.getTotalSize(), width: basePageWidth, minWidth: basePageWidth }}>
            {continuousPages.getVirtualItems().map(page => (
              <div key={page.key} className="absolute left-1/2 -translate-x-1/2" style={{ top: page.start }}>
                <PdfPage pageNumber={page.index + 1} scale={scale} onSetRef={el => setPageRef(page.index + 1, el)} />
              </div>
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
                  lazy
                  placeholderSize={{ width: basePageWidth / scale * 0.25, height: (pageHeights[index] ?? basePageHeight) / scale * 0.25 }}
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
