import { Tooltip } from "antd"
import {
  CopyCheckIcon,
  CopyIcon,
  DownloadIcon,
  EyeIcon,
  WorkflowIcon
} from "lucide-react"
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import DOMPurify from "dompurify"
import Mermaid, { type MermaidRenderState } from "./Mermaid"
import { MermaidPreviewDialog } from "./MermaidPreviewDialog"

export type MermaidDiagramBlockProps = {
  source: string
  blockIndex?: number
}

const downloadSvg = (svg: string, blockIndex?: number) => {
  const blob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  const suffix = typeof blockIndex === "number" ? `-${blockIndex + 1}` : ""
  anchor.download = `mermaid-diagram${suffix}.svg`
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

export const MermaidDiagramBlock: React.FC<MermaidDiagramBlockProps> = ({
  source,
  blockIndex
}) => {
  const [renderStatus, setRenderStatus] =
    useState<MermaidRenderState["status"]>("idle")
  const [generatedSvg, setGeneratedSvg] = useState<string | undefined>()
  const [copied, setCopied] = useState(false)
  const [previewOpen, setPreviewOpen] = useState(false)
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const previousSourceRef = useRef(source)
  const sanitizedGeneratedSvg = useMemo(
    () =>
      generatedSvg
        ? DOMPurify.sanitize(generatedSvg, { USE_PROFILES: { svg: true } })
        : undefined,
    [generatedSvg]
  )
  const hasGeneratedSvg = Boolean(sanitizedGeneratedSvg)
  const isRenderError = renderStatus === "error"

  useEffect(() => {
    if (previousSourceRef.current === source) {
      return
    }
    previousSourceRef.current = source
    setRenderStatus("idle")
    setGeneratedSvg(undefined)
    setPreviewOpen(false)
  }, [source])

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) {
        clearTimeout(copyTimeoutRef.current)
      }
    }
  }, [])

  const handleRenderStateChange = useCallback((state: MermaidRenderState) => {
    setRenderStatus(state.status)
    if (state.status === "success" && state.svg) {
      setGeneratedSvg(state.svg)
    }
    if (state.status === "error") {
      setGeneratedSvg(undefined)
    }
  }, [])

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
    downloadSvg(sanitizedGeneratedSvg, blockIndex)
  }, [blockIndex, sanitizedGeneratedSvg])

  const headerId = useMemo(() => {
    if (typeof blockIndex === "number") {
      return `mermaid-diagram-${blockIndex}`
    }
    return undefined
  }, [blockIndex])

  return (
    <>
      <div className="not-prose">
        <div
          aria-labelledby={headerId}
          className="my-4 overflow-hidden rounded-xl border border-border bg-surface"
        >
          <div className="flex flex-row items-center justify-between gap-3 border-b border-border bg-surface2 px-4 py-2">
            <div className="flex min-w-0 items-center gap-2">
              <WorkflowIcon className="size-4 shrink-0 text-text-muted" />
              <span
                id={headerId}
                className="truncate font-mono text-xs text-text-muted"
              >
                mermaid
              </span>
            </div>
            <div className="flex items-center gap-1">
              <Tooltip title="Open Mermaid preview">
                <button
                  type="button"
                  aria-label="Open Mermaid preview"
                  onClick={() => setPreviewOpen(true)}
                  className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface hover:text-text"
                >
                  <EyeIcon className="size-4" />
                </button>
              </Tooltip>
              {hasGeneratedSvg && (
                <Tooltip title="Download Mermaid SVG">
                  <button
                    type="button"
                    aria-label="Download Mermaid SVG"
                    onClick={handleDownloadSvg}
                    className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface hover:text-text"
                  >
                    <DownloadIcon className="size-4" />
                  </button>
                </Tooltip>
              )}
              <Tooltip title="Copy Mermaid source">
                <button
                  type="button"
                  aria-label="Copy Mermaid source"
                  onClick={handleCopySource}
                  className="inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface hover:text-text"
                >
                  {copied ? (
                    <CopyCheckIcon className="size-4 text-success" />
                  ) : (
                    <CopyIcon className="size-4" />
                  )}
                </button>
              </Tooltip>
            </div>
          </div>

          <div className="p-4">
            <div
              aria-hidden={isRenderError}
              className={isRenderError ? "hidden" : ""}
            >
              <Mermaid
                code={source}
                onRenderStateChange={handleRenderStateChange}
              />
            </div>
            {isRenderError && (
              <div className="rounded-md border border-border bg-surface2 p-3">
                <p className="mb-2 text-sm text-text-muted">
                  Unable to render Mermaid diagram.
                </p>
                <pre className="overflow-auto whitespace-pre-wrap text-xs text-text">
                  {source}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={() => setPreviewOpen(false)}
        open={previewOpen}
        source={source}
      />
    </>
  )
}

export default MermaidDiagramBlock
