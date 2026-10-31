/**
 * SourceViewerModal - Full source content preview dialog
 */

import React, { useEffect, useRef } from "react"
import { Modal } from "antd"
import { ExternalLink } from "lucide-react"
import { cn } from "@/libs/utils"
import { openExternalUrl } from "@/utils/safe-external-url"
import type { RagResult } from "./types"
import { containDialogTab } from "./dialogKeyboard"
import { getSourceOpenAction } from "./sourceOpenAction"
import {
  getEvidenceOrigin,
  getEvidenceOriginLabel,
  getResultChunkId,
  getResultEvidenceText,
  getResultSourceId,
  getUnavailableEvidenceMessage,
  getSourceTypeLabel,
  getResultSourceType,
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
  const dialogRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const trigger = document.activeElement as HTMLElement | null
    // SourceList clears the result immediately; restore after the dialog unmounts.
    return () =>
      queueMicrotask(() => {
        if (trigger?.isConnected) trigger.focus({ preventScroll: true })
      })
  }, [open])
  if (!result) return null

  const title = result.metadata?.title || result.metadata?.source || "Source"
  const content = getResultEvidenceText(result)
  const unavailableMessage = getUnavailableEvidenceMessage(result)
  const openAction = getSourceOpenAction(result)
  const sourceType = getResultSourceType(result)
  const sourceLabel = getSourceTypeLabel(sourceType)
  const evidenceOrigin = getEvidenceOrigin(result)
  const evidenceOriginLabel =
    evidenceOrigin !== "unknown_origin" ? getEvidenceOriginLabel(evidenceOrigin) : null
  const sourceId = getResultSourceId(result)
  const chunkId = getResultChunkId(result)
  const pageNumber = result.metadata?.page_number
  const metaParts = [
    sourceLabel,
    pageNumber ? `Page ${pageNumber}` : null,
    evidenceOriginLabel,
    sourceId ? `Source ID ${sourceId}` : null,
    chunkId ? `Chunk ${chunkId}` : null,
  ].filter((value): value is string => Boolean(value))

  return (
    <Modal
      open={open}
      modalRender={(content) => (
        <div ref={dialogRef} onKeyDown={containDialogTab}>
          {content}
        </div>
      )}
      afterOpenChange={(visible) => {
        if (visible)
          dialogRef.current?.querySelector<HTMLButtonElement>("button")?.focus()
      }}
      onCancel={onClose}
      footer={null}
      width={768}
      title={`${index ? `Source ${index}: ` : ""}${title}`}
      closable={{ "aria-label": "Close source preview" }}
      className={cn("source-viewer", className)}
      styles={{
        body: { maxHeight: "calc(100dvh - 12rem)", overflowY: "auto" },
      }}
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="min-w-0">
          <p className="mt-1 text-xs text-text-muted">
            {metaParts.join(" • ")}
          </p>
        </div>
        <div className="ml-3 flex items-center gap-2">
          {openAction && (
            <button
              type="button"
              onClick={() =>
                openExternalUrl(
                  openAction.href,
                  "_blank",
                  "noopener,noreferrer",
                )
              }
              className="inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              {openAction.label}
            </button>
          )}
        </div>
      </div>

      <div className="px-4 py-3">
        {content ? (
          <pre className="whitespace-pre-wrap text-sm leading-relaxed text-text">
            {content}
          </pre>
        ) : unavailableMessage ? (
          <p className="text-sm text-warn">{unavailableMessage}</p>
        ) : (
          <p className="text-sm text-text-muted">
            Full source content is unavailable for this result.
          </p>
        )}
      </div>
    </Modal>
  )
}
