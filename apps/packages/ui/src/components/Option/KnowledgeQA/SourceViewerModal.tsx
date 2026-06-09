/**
 * SourceViewerModal - Full source content preview dialog
 */

import React, { useEffect } from "react"
import { ExternalLink, X } from "lucide-react"
import { cn } from "@/libs/utils"
import type { RagResult } from "./types"
import {
  getEvidenceOrigin,
  getEvidenceOriginLabel,
  getResultChunkId,
  getResultEvidenceText,
  getResultSourceId,
  getUnavailableEvidenceMessage,
  getSourceTypeLabel,
} from "./sourceListUtils"

type SourceViewerModalProps = {
  open: boolean
  result: RagResult | null
  index: number | null
  onClose: () => void
  className?: string
}

export function SourceViewerModal({
  open,
  result,
  index,
  onClose,
  className,
}: SourceViewerModalProps) {
  useEffect(() => {
    if (!open) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        onClose()
      }
    }
    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [open, onClose])

  if (!open || !result) return null

  const title = result.metadata?.title || result.metadata?.source || "Source"
  const content = getResultEvidenceText(result)
  const unavailableMessage = getUnavailableEvidenceMessage(result)
  const url = result.metadata?.url
  const sourceType = result.sourceType || result.metadata?.source_type
  const sourceLabel = getSourceTypeLabel(sourceType)
  const evidenceOrigin = getEvidenceOrigin(result)
  const evidenceOriginLabel =
    evidenceOrigin !== "unknown_origin" ? getEvidenceOriginLabel(evidenceOrigin) : null
  const sourceId = getResultSourceId(result)
  const chunkId = getResultChunkId(result)
  const pageNumber = result.metadata?.page_number
  const dialogTitleId = "source-viewer-title"
  const metaParts = [
    sourceLabel,
    pageNumber ? `Page ${pageNumber}` : null,
    evidenceOriginLabel,
    sourceId ? `Source ID ${sourceId}` : null,
    chunkId ? `Chunk ${chunkId}` : null,
  ].filter((value): value is string => Boolean(value))

  return (
    <>
      <div
        className="fixed inset-0 z-50 bg-black/45"
        aria-hidden="true"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={dialogTitleId}
        className={cn(
          "fixed inset-x-4 top-6 z-50 mx-auto max-h-[88vh] w-full max-w-3xl overflow-hidden rounded-xl border border-border bg-surface shadow-xl",
          className
        )}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h3 id={dialogTitleId} className="truncate text-base font-semibold">
              {index ? `Source ${index}: ` : ""}
              {title}
            </h3>
            <p className="mt-1 text-xs text-text-muted">
              {metaParts.join(" • ")}
            </p>
          </div>
          <div className="ml-3 flex items-center gap-2">
            {url && (
              <button
                type="button"
                onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
                className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                Open original
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="rounded-md p-1.5 text-text-muted hover:bg-hover hover:text-text transition-colors"
              aria-label="Close source preview"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="max-h-[72vh] overflow-y-auto px-4 py-3">
          {content ? (
            <pre className="whitespace-pre-wrap text-sm leading-relaxed text-text">
              {content}
            </pre>
          ) : unavailableMessage ? (
            <p className="text-sm text-warn">
              {unavailableMessage}
            </p>
          ) : (
            <p className="text-sm text-text-muted">
              Full source content is unavailable for this result.
            </p>
          )}
        </div>
      </div>
    </>
  )
}
