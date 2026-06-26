import React from "react"
import { Button, Tag } from "antd"
import { AlertTriangle, MessageSquare } from "lucide-react"
import type { ManuscriptAnnotationResponse } from "@/services/writing-playground"
import { cn } from "@/libs/utils"

export type WritingAnnotationCardProps = {
  annotation: ManuscriptAnnotationResponse
  active?: boolean
  warning?: boolean
  cardId: string
  describedById: string
  inspectorRowId: string
  onFocus: () => void
  onReviewSuggestedFix?: (annotation: ManuscriptAnnotationResponse) => void
  onCopySuggestedFix?: (annotation: ManuscriptAnnotationResponse) => void
}

export function WritingAnnotationCard({
  annotation,
  active = false,
  warning = false,
  cardId,
  describedById,
  inspectorRowId,
  onFocus,
  onReviewSuggestedFix,
  onCopySuggestedFix
}: WritingAnnotationCardProps) {
  const anchorStateLabel = annotation.anchor_status.replace(/_/g, " ")
  const requiresManualFix = annotation.anchor_status === "needs_review"

  return (
    <article
      id={cardId}
      data-testid="writing-annotation-card"
      className={cn(
        "flex w-full flex-col overflow-hidden rounded-md border px-2 py-2 text-left shadow-sm transition-colors",
        active ? "h-[152px]" : "h-[96px]",
        active
          ? "border-primary bg-primary/10"
          : warning
            ? "border-warning/60 bg-warning/10"
            : "border-border bg-surface hover:border-primary/50"
      )}
    >
      <button
        type="button"
        aria-label={`Focus annotation ${annotation.id}`}
        aria-describedby={describedById}
        aria-controls={inspectorRowId}
        onClick={onFocus}
        className="flex min-w-0 flex-1 flex-col gap-1 overflow-hidden text-left"
      >
        <span className="flex min-w-0 items-center gap-1.5 text-[11px] font-medium text-text">
          {warning ? (
            <AlertTriangle aria-hidden className="h-3.5 w-3.5 shrink-0 text-warning" />
          ) : (
            <MessageSquare aria-hidden className="h-3.5 w-3.5 shrink-0 text-text-muted" />
          )}
          <span className="truncate">{annotation.category}</span>
          <span className="sr-only">{annotation.id}</span>
        </span>
        <span
          id={describedById}
          className={cn(
            "whitespace-normal text-[11px] leading-4 text-text",
            active ? "line-clamp-4" : "line-clamp-2"
          )}
        >
          {annotation.body}
        </span>
        <span className="flex flex-wrap gap-1">
          <Tag className="!m-0 !px-1 text-[10px]">{annotation.status}</Tag>
          <Tag className="!m-0 !px-1 text-[10px]">{annotation.source}</Tag>
          <Tag color={warning ? "warning" : undefined} className="!m-0 !px-1 text-[10px]">
            {anchorStateLabel}
          </Tag>
        </span>
        {active && annotation.followup_note ? (
          <span className="line-clamp-1 text-[10px] leading-4 text-text-muted">
            Follow-up: {annotation.followup_note}
          </span>
        ) : null}
      </button>
      {active && annotation.suggested_fix ? (
        <Button
          size="small"
          type="link"
          className="!h-auto !justify-start !p-0 !text-[10px]"
          onClick={() => {
            if (requiresManualFix) {
              if (onCopySuggestedFix) {
                onCopySuggestedFix(annotation)
                return
              }
              onFocus()
              return
            }
            if (onReviewSuggestedFix) {
              onReviewSuggestedFix(annotation)
              return
            }
            onFocus()
          }}
        >
          {requiresManualFix ? "Copy fix manually" : "Create revision"}
        </Button>
      ) : null}
    </article>
  )
}

export default WritingAnnotationCard
