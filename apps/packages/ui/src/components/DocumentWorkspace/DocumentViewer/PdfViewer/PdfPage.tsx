import React, { useCallback, useState, useRef, useEffect } from "react"
import { Page } from "react-pdf"
import { Spin } from "antd"
import { PageNoteButton } from "../PageNoteButton"

// Note: react-pdf text/annotation layer styles should be imported at the app level
// In Next.js: add to pages/_app.tsx or use next.config.js transpilePackages

interface PdfPageProps {
  pageNumber: number
  scale: number
  onSetRef?: (element: HTMLDivElement | null) => void
  /** Hide the page note button (e.g., in thumbnail view) */
  hidePageNote?: boolean
  lazy?: boolean
  placeholderSize?: { width: number; height: number }
}

export const PdfPage: React.FC<PdfPageProps> = ({
  pageNumber,
  scale,
  onSetRef,
  hidePageNote = false,
  lazy = false,
  placeholderSize
}) => {
  const pageRef = useRef<HTMLDivElement | null>(null)
  const [visible, setVisible] = useState(!lazy)
  useEffect(() => {
    if (!lazy || !pageRef.current) return
    const observer = new IntersectionObserver(entries => {
      setVisible(entries.some(entry => entry.isIntersecting))
    }, { rootMargin: '400px' })
    observer.observe(pageRef.current)
    return () => observer.disconnect()
  }, [lazy])

  const [loading, setLoading] = useState(true)

  const handleRenderSuccess = useCallback(() => {
    setLoading(false)
  }, [])

  const handleRenderError = useCallback(() => {
    setLoading(false)
  }, [])

  const handleTextLayerReady = useCallback(() => {
    pageRef.current?.dispatchEvent(new Event("pdf-text-layer-rendered", { bubbles: true }))
  }, [])

  return (
    <div
      ref={element => { pageRef.current = element; onSetRef?.(element) }}
      style={visible ? undefined : placeholderSize ?? { width: 800 * scale, height: 1100 * scale }}
      data-page-number={pageNumber}
      className="group relative bg-surface shadow-lg"
    >
      {visible && loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface/80">
          <Spin size="small" />
        </div>
      )}
      {/* Page note button - appears on hover */}
      {visible && !hidePageNote && !loading && (
        <PageNoteButton pageNumber={pageNumber} />
      )}
      {visible && <Page
        pageNumber={pageNumber}
        scale={scale}
        onRenderSuccess={handleRenderSuccess}
        onRenderError={handleRenderError}
        onRenderTextLayerSuccess={handleTextLayerReady}
        loading=""
        renderTextLayer={true}
        renderAnnotationLayer={true}
        className="pdf-page"
      />}
    </div>
  )
}

export default PdfPage
