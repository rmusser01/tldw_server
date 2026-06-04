import { Modal, Tooltip } from "antd"
import {
  CopyCheckIcon,
  CopyIcon,
  DownloadIcon,
  MoveIcon,
  RotateCcwIcon,
  XIcon,
  ZoomInIcon,
  ZoomOutIcon
} from "lucide-react"
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import DOMPurify from "dompurify"

export type MermaidPreviewDialogProps = {
  open: boolean
  source: string
  generatedSvg?: string
  onClose: () => void
}

const MIN_ZOOM = 0.25
const MAX_ZOOM = 4
const ZOOM_STEP = 0.25
const KEYBOARD_PAN_STEP = 24

type PanState = {
  x: number
  y: number
}

type DragState = {
  pointerId: number
  startX: number
  startY: number
  originX: number
  originY: number
}

const clampZoom = (value: number) =>
  Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value))

const downloadSvg = (svg: string) => {
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = `mermaid-diagram-${Date.now()}.svg`
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

export const MermaidPreviewDialog: React.FC<MermaidPreviewDialogProps> = ({
  open,
  source,
  generatedSvg,
  onClose
}) => {
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState<PanState>({ x: 0, y: 0 })
  const [copied, setCopied] = useState(false)
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const dragRef = useRef<DragState | null>(null)
  const sanitizedGeneratedSvg = useMemo(
    () =>
      generatedSvg
        ? DOMPurify.sanitize(generatedSvg, { USE_PROFILES: { svg: true } })
        : undefined,
    [generatedSvg]
  )
  const hasGeneratedSvg = Boolean(sanitizedGeneratedSvg)

  useEffect(() => {
    if (!open) return
    setZoom(1)
    setPan({ x: 0, y: 0 })
    dragRef.current = null
  }, [generatedSvg, open, source])

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
      }
    }
  }, [])

  const zoomLabel = useMemo(() => `${Math.round(zoom * 100)}%`, [zoom])
  const canvasTransform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`

  const handleCopySource = useCallback(async () => {
    const writeText = navigator.clipboard?.writeText
    if (!writeText) {
      setCopied(false)
      return
    }

    try {
      await writeText.call(navigator.clipboard, source)
      setCopied(true)
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
      }
      copyTimeoutRef.current = setTimeout(() => {
        setCopied(false)
        copyTimeoutRef.current = null
      }, 2000)
    } catch {
      setCopied(false)
    }
  }, [source])

  const handleDownloadSvg = useCallback(() => {
    if (!sanitizedGeneratedSvg) return
    downloadSvg(sanitizedGeneratedSvg)
  }, [sanitizedGeneratedSvg])

  const handleZoomIn = useCallback(() => {
    setZoom((current) => clampZoom(current + ZOOM_STEP))
  }, [])

  const handleZoomOut = useCallback(() => {
    setZoom((current) => clampZoom(current - ZOOM_STEP))
  }, [])

  const handleReset = useCallback(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!hasGeneratedSvg) return
      dragRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        originX: pan.x,
        originY: pan.y
      }
      event.currentTarget.setPointerCapture?.(event.pointerId)
    },
    [hasGeneratedSvg, pan.x, pan.y]
  )

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current
      if (!drag || drag.pointerId !== event.pointerId) return

      setPan({
        x: drag.originX + event.clientX - drag.startX,
        y: drag.originY + event.clientY - drag.startY
      })
    },
    []
  )

  const stopDragging = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      const drag = dragRef.current
      if (!drag || drag.pointerId !== event.pointerId) return
      dragRef.current = null
      event.currentTarget.releasePointerCapture?.(event.pointerId)
    },
    []
  )

  const handleViewportKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (!hasGeneratedSvg) return

      const step = event.shiftKey ? KEYBOARD_PAN_STEP * 2 : KEYBOARD_PAN_STEP
      const movementByKey: Record<string, PanState> = {
        ArrowDown: { x: 0, y: step },
        ArrowLeft: { x: -step, y: 0 },
        ArrowRight: { x: step, y: 0 },
        ArrowUp: { x: 0, y: -step }
      }
      const movement = movementByKey[event.key]
      if (!movement) return

      event.preventDefault()
      setPan((current) => ({
        x: current.x + movement.x,
        y: current.y + movement.y
      }))
    },
    [hasGeneratedSvg]
  )

  return (
    <Modal
      title="Mermaid diagram preview"
      open={open}
      footer={null}
      width={960}
      onCancel={onClose}
      className="mermaid-preview-dialog"
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2">
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <MoveIcon className="size-4" />
            <span>Drag to pan</span>
            <span
              aria-label="Mermaid preview zoom level"
              className="rounded border border-border px-2 py-1 font-mono text-text"
            >
              {zoomLabel}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-1">
            <Tooltip title="Zoom out">
              <button
                type="button"
                aria-label="Zoom out"
                disabled={!hasGeneratedSvg || zoom <= MIN_ZOOM}
                onClick={handleZoomOut}
                className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
              >
                <ZoomOutIcon className="size-4" />
              </button>
            </Tooltip>
            <Tooltip title="Zoom in">
              <button
                type="button"
                aria-label="Zoom in"
                disabled={!hasGeneratedSvg || zoom >= MAX_ZOOM}
                onClick={handleZoomIn}
                className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
              >
                <ZoomInIcon className="size-4" />
              </button>
            </Tooltip>
            <Tooltip title="Reset zoom and pan">
              <button
                type="button"
                aria-label="Reset zoom and pan"
                disabled={!hasGeneratedSvg}
                onClick={handleReset}
                className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
              >
                <RotateCcwIcon className="size-4" />
              </button>
            </Tooltip>
            <Tooltip title="Copy Mermaid source">
              <button
                type="button"
                aria-label="Copy Mermaid source"
                onClick={handleCopySource}
                className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text"
              >
                {copied ? (
                  <CopyCheckIcon className="size-4 text-success" />
                ) : (
                  <CopyIcon className="size-4" />
                )}
              </button>
            </Tooltip>
            {hasGeneratedSvg && (
              <Tooltip title="Download Mermaid SVG">
                <button
                  type="button"
                  aria-label="Download Mermaid SVG"
                  onClick={handleDownloadSvg}
                  className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text"
                >
                  <DownloadIcon className="size-4" />
                </button>
              </Tooltip>
            )}
            <Tooltip title="Close Mermaid preview">
              <button
                type="button"
                aria-label="Close Mermaid preview"
                onClick={onClose}
                className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text"
              >
                <XIcon className="size-4" />
              </button>
            </Tooltip>
          </div>
        </div>

        {sanitizedGeneratedSvg ? (
          <div
            aria-label="Mermaid diagram viewport"
            className="h-[70vh] min-h-[320px] cursor-grab overflow-auto rounded-md border border-border bg-surface p-4 active:cursor-grabbing"
            tabIndex={0}
            onKeyDown={handleViewportKeyDown}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={stopDragging}
            onPointerCancel={stopDragging}
          >
            <div
              data-testid="mermaid-preview-canvas"
              className="mx-auto flex min-h-full origin-center items-center justify-center transition-transform"
              style={{ transform: canvasTransform }}
              dangerouslySetInnerHTML={{ __html: sanitizedGeneratedSvg }}
            />
          </div>
        ) : (
          <div className="rounded-md border border-border bg-surface p-3">
            <p className="mb-2 text-sm text-text-muted">
              No rendered SVG is available.
            </p>
            <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap text-xs text-text">
              {source}
            </pre>
          </div>
        )}
      </div>
    </Modal>
  )
}

export default MermaidPreviewDialog
