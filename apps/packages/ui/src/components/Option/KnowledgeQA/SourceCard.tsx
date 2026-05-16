/**
 * SourceCard - Individual source/document display
 */

import React, { useCallback, useMemo } from "react"
import { useTranslation } from "react-i18next"
import {
  FileText,
  Globe,
  MessageSquare,
  ExternalLink,
  Copy,
  Check,
  Quote,
  Pin,
  BookOpen,
  ThumbsUp,
  ThumbsDown,
  MoreHorizontal,
} from "lucide-react"
import { cn } from "@/libs/utils"
import type { RagResult } from "./types"
import {
  detectSourceContentFacet,
  formatChunkPosition,
  getSourceContentFacetLabel,
  formatSourceDate,
  getFreshnessDescriptor,
  getRelevanceDescriptor,
  getSourceTypeLabel,
  type CitationUsageAnchor,
  splitTextByHighlights,
} from "./sourceListUtils"

export type SourceAskTemplate = "detail" | "summary" | "quotes"

type SourceCardProps = {
  result: RagResult
  index: number // 1-based for display
  isCited: boolean
  isFocused: boolean
  onSourceHover: (index: number | null) => void
  onAskAbout: (result: RagResult, template: SourceAskTemplate) => void
  onViewFull: (result: RagResult, index: number) => void
  onSourceFeedback: (result: RagResult, index: number, thumb: "up" | "down") => void
  onRetrySourceFeedback: (result: RagResult, index: number) => void
  onTogglePin: (result: RagResult, index: number) => void
  onJumpToCitation: (citationIndex: number, occurrence?: number) => void
  feedbackThumb: "up" | "down" | null
  feedbackSubmitting: boolean
  feedbackError: string | null
  isPinned: boolean
  highlightTerms: string[]
  citationUsages: CitationUsageAnchor[]
  density?: "default" | "compact"
  className?: string
}

function getSourceIcon(sourceType?: string) {
  switch (sourceType) {
    case "notes":
      return FileText
    case "characters":
      return MessageSquare
    case "chats":
      return MessageSquare
    default:
      return FileText
  }
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength).trimEnd() + "..."
}

function buildCitationText(result: RagResult, fallbackIndex: number): string {
  const title =
    result.metadata?.title || result.metadata?.source || `Source ${fallbackIndex}`
  const pageNumber = result.metadata?.page_number
  const url = result.metadata?.url

  const segments = [title]
  if (typeof pageNumber === "number") {
    segments.push(`(Page ${pageNumber})`)
  }
  if (url) {
    segments.push(`- ${url}`)
  }
  return segments.join(" ")
}

function isInteractiveTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) {
    return false
  }
  return Boolean(
    target.closest(
      'button, a, input, select, textarea, [role="button"], [role="link"], [contenteditable]'
    )
  )
}

export function SourceCard({
  result,
  index,
  isCited,
  isFocused,
  onSourceHover,
  onAskAbout,
  onViewFull,
  onSourceFeedback,
  onRetrySourceFeedback,
  onTogglePin,
  onJumpToCitation,
  feedbackThumb,
  feedbackSubmitting,
  feedbackError,
  isPinned,
  highlightTerms,
  citationUsages,
  density = "default",
  className,
}: SourceCardProps) {
  const { t } = useTranslation("knowledge")
  const [copiedState, setCopiedState] = React.useState<"excerpt" | "citation" | null>(null)
  const [isExpanded, setIsExpanded] = React.useState(false)
  const [overflowOpen, setOverflowOpen] = React.useState(false)
  const overflowRef = React.useRef<HTMLDivElement>(null)
  const copiedStateTimeoutRef = React.useRef<number | null>(null)
  const latestCopyRequestIdRef = React.useRef(0)
  const isMountedRef = React.useRef(true)

  const title = result.metadata?.title || result.metadata?.source || `Source ${index}`
  const content = result.content || result.text || result.chunk || ""
  const compactDensity = density === "compact"
  const excerptLength = compactDensity ? 180 : 300
  const excerpt = isExpanded ? content : truncateText(content, excerptLength)
  const excerptSegments = useMemo(
    () => splitTextByHighlights(excerpt, highlightTerms),
    [excerpt, highlightTerms]
  )
  const canExpand = content.length > excerptLength
  const url = result.metadata?.url
  const score = result.score
  const sourceType = result.metadata?.source_type || "media_db"
  const sourceTypeLabel = getSourceTypeLabel(sourceType)
  const sourceFacetLabel = getSourceContentFacetLabel(detectSourceContentFacet(result))
  const sourceKindLabel = sourceTypeLabel === "Other" ? sourceFacetLabel : sourceTypeLabel
  const chunkPosition = formatChunkPosition(result.metadata?.chunk_id)
  const sourceDate = formatSourceDate(result)
  const freshnessDescriptor = getFreshnessDescriptor(result)
  const relevanceDescriptor = getRelevanceDescriptor(score)
  const compactMetaItems = compactDensity
    ? [sourceKindLabel, chunkPosition, freshnessDescriptor?.label ?? sourceDate].filter(
        (value): value is string => Boolean(value)
      )
    : []

  // Determine whether this source is a document that can be opened in the
  // Document Workspace (PDF, EPUB, or generic "document"/"ebook" media types).
  const contentFacet = detectSourceContentFacet(result)
  const rawMediaType = String(
    (result.metadata as Record<string, unknown> | undefined)?.media_type ?? ""
  ).toLowerCase()
  const isDocumentType =
    contentFacet === "pdf" ||
    rawMediaType.includes("pdf") ||
    rawMediaType.includes("ebook") ||
    rawMediaType === "document"

  // Resolve the numeric media_id from metadata for workspace navigation.
  const resolvedMediaId = (() => {
    const raw =
      (result.metadata as Record<string, unknown> | undefined)?.media_id ??
      result.metadata?.id ??
      result.id
    if (typeof raw === "number" && Number.isFinite(raw)) return Math.round(raw)
    if (typeof raw === "string" && /^\d+$/.test(raw.trim())) {
      return Number.parseInt(raw.trim(), 10)
    }
    return null
  })()

  const canOpenInWorkspace = isDocumentType && resolvedMediaId != null

  const Icon = getSourceIcon(sourceType)

  React.useEffect(
    () => () => {
      isMountedRef.current = false
      latestCopyRequestIdRef.current += 1
      if (copiedStateTimeoutRef.current != null) {
        window.clearTimeout(copiedStateTimeoutRef.current)
        copiedStateTimeoutRef.current = null
      }
    },
    []
  )

  // Close overflow menu on outside click
  React.useEffect(() => {
    if (!overflowOpen) return
    function handleClickOutside(event: MouseEvent) {
      if (
        overflowRef.current &&
        !overflowRef.current.contains(event.target as Node)
      ) {
        setOverflowOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [overflowOpen])

  const scheduleCopiedStateReset = useCallback((requestId: number) => {
    if (copiedStateTimeoutRef.current != null) {
      window.clearTimeout(copiedStateTimeoutRef.current)
    }
    copiedStateTimeoutRef.current = window.setTimeout(() => {
      if (!isMountedRef.current || latestCopyRequestIdRef.current !== requestId) {
        return
      }
      copiedStateTimeoutRef.current = null
      setCopiedState(null)
    }, 2000)
  }, [])

  const handleCopyExcerpt = useCallback(async () => {
    const requestId = latestCopyRequestIdRef.current + 1
    latestCopyRequestIdRef.current = requestId
    try {
      await navigator.clipboard.writeText(excerpt)
      if (!isMountedRef.current || latestCopyRequestIdRef.current !== requestId) {
        return
      }
      setCopiedState("excerpt")
      scheduleCopiedStateReset(requestId)
    } catch (error) {
      console.error("Failed to copy source excerpt:", error)
    }
  }, [excerpt, scheduleCopiedStateReset])

  const handleCopyCitation = useCallback(async () => {
    const requestId = latestCopyRequestIdRef.current + 1
    latestCopyRequestIdRef.current = requestId
    try {
      await navigator.clipboard.writeText(buildCitationText(result, index))
      if (!isMountedRef.current || latestCopyRequestIdRef.current !== requestId) {
        return
      }
      setCopiedState("citation")
      scheduleCopiedStateReset(requestId)
    } catch (error) {
      console.error("Failed to copy source citation:", error)
    }
  }, [index, result, scheduleCopiedStateReset])

  const handleOpenExternal = useCallback(() => {
    if (url) {
      window.open(url, "_blank", "noopener,noreferrer")
    }
  }, [url])

  const jumpToCitationFromCard = useCallback(() => {
    if (!isCited) return
    onJumpToCitation(index)
  }, [index, isCited, onJumpToCitation])

  return (
    <div
      id={`source-card-${index - 1}`}
      role="listitem"
      tabIndex={isCited ? 0 : undefined}
      onMouseEnter={() => onSourceHover(index - 1)}
      onMouseLeave={() => onSourceHover(null)}
      onFocusCapture={() => onSourceHover(index - 1)}
      onBlurCapture={(event) => {
        const nextFocused = event.relatedTarget as Node | null
        if (nextFocused && event.currentTarget.contains(nextFocused)) {
          return
        }
        onSourceHover(null)
      }}
      onClick={(event) => {
        if (!isCited || isInteractiveTarget(event.target)) {
          return
        }
        jumpToCitationFromCard()
      }}
      onKeyDown={(event) => {
        if (!isCited || isInteractiveTarget(event.target)) {
          return
        }
        if (event.key !== "Enter" && event.key !== " ") {
          return
        }
        event.preventDefault()
        jumpToCitationFromCard()
      }}
      aria-label={
        isCited
          ? `Source ${index}. Cited in answer. Press Enter to jump to citation ${index}.`
          : undefined
      }
      className={cn(
        "group rounded-lg border transition-all duration-200",
        isFocused
          ? "border-primary bg-primary/5 ring-2 ring-primary/20"
          : isPinned
            ? "border-primary/40 bg-primary/5 hover:border-primary/60"
          : isCited
            ? "border-border bg-surface hover:border-primary/30"
            : "border-border/70 bg-surface/90 hover:border-border-strong",
        isCited && "cursor-pointer",
        className
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-start gap-2.5 p-3 pb-2 sm:gap-3 sm:pb-2",
          compactDensity ? "sm:p-3" : "sm:p-4"
        )}
      >
        {/* Index badge */}
        <div
          className={cn(
            "flex-shrink-0 h-6 w-6 rounded-md flex items-center justify-center text-xs font-medium sm:h-7 sm:w-7 sm:text-sm",
            isCited ? "bg-primary text-white dark:text-slate-900" : "bg-muted text-text"
          )}
        >
          {index}
        </div>

        {/* Title and metadata */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2">
            <Icon className="w-4 h-4 text-text-muted mt-0.5 flex-shrink-0" />
            <div className="min-w-0 flex-1">
              {compactDensity ? (
                <>
                  <div className="flex items-start justify-between gap-2">
                    <h4
                      className="min-w-0 flex-1 truncate font-medium text-[13px]"
                      title={title}
                    >
                      {title}
                    </h4>
                    {relevanceDescriptor ? (
                      <span
                        data-testid="knowledge-source-compact-relevance"
                        className={cn(
                          "shrink-0 rounded-md border px-1.5 py-0.5 text-[10px] font-medium",
                          relevanceDescriptor.className
                        )}
                        title={`${relevanceDescriptor.label} (${relevanceDescriptor.percent}%)`}
                      >
                        {relevanceDescriptor.percent}% match
                      </span>
                    ) : null}
                  </div>
                  {compactMetaItems.length > 0 ? (
                    <div
                      data-testid="knowledge-source-compact-meta"
                      className="mt-1 flex flex-wrap items-center gap-x-1 text-[11px] leading-4 text-text-muted"
                    >
                      {compactMetaItems.map((item, itemIndex) => (
                        <React.Fragment key={`${item}-${itemIndex}`}>
                          {itemIndex > 0 ? <span aria-hidden="true">{" • "}</span> : null}
                          <span>{item}</span>
                        </React.Fragment>
                      ))}
                    </div>
                  ) : null}
                  {isPinned || isCited ? (
                    <div
                      data-testid="knowledge-source-compact-status"
                      className="mt-1 flex flex-wrap items-center gap-2 text-[11px]"
                    >
                      {isPinned ? (
                        <span className="inline-flex items-center gap-1 text-primary">
                          <Pin className="h-3 w-3" />
                          Pinned
                        </span>
                      ) : null}
                      {isCited ? (
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation()
                            onJumpToCitation(index)
                          }}
                          className="inline-flex items-center gap-1 text-primary hover:text-primaryStrong transition-colors"
                          aria-label={`Jump to citation ${index} in answer`}
                          title={`Jump to citation [${index}] in answer`}
                        >
                          <Quote className="w-3 h-3" />
                          Cited
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </>
              ) : (
                <>
                  <h4 className="font-medium truncate text-sm" title={title}>
                    {title}
                  </h4>
                  <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-text-muted sm:gap-2 sm:text-xs">
                    {relevanceDescriptor && (
                      <span
                        className={cn(
                          "rounded px-1.5 py-0.5",
                          relevanceDescriptor.className
                        )}
                        title={`${relevanceDescriptor.label} (${relevanceDescriptor.percent}%)`}
                      >
                        {relevanceDescriptor.label} ({relevanceDescriptor.percent}%)
                      </span>
                    )}
                    <span className="inline-flex items-center gap-0.5">
                      {sourceType === "web" ? (
                        <Globe className="h-3 w-3" />
                      ) : (
                        <BookOpen className="h-3 w-3" />
                      )}
                      {sourceKindLabel}
                    </span>
                    {chunkPosition ? <span>{chunkPosition}</span> : null}
                    {freshnessDescriptor ? (
                      <span
                        className={cn(
                          "rounded border px-1.5 py-0.5",
                          freshnessDescriptor.className
                        )}
                        title={sourceDate ? `Source date ${sourceDate}` : undefined}
                      >
                        {freshnessDescriptor.label}
                      </span>
                    ) : sourceDate ? (
                      <span>{sourceDate}</span>
                    ) : null}
                    {isPinned && (
                      <span className="inline-flex items-center gap-1 rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 text-primary">
                        <Pin className="h-3 w-3" />
                        Pinned
                      </span>
                    )}
                    {isCited && (
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation()
                          onJumpToCitation(index)
                        }}
                        className="inline-flex items-center gap-1 text-primary hover:text-primaryStrong transition-colors"
                        aria-label={`Jump to citation ${index} in answer`}
                        title={`Jump to citation [${index}] in answer`}
                      >
                        <Quote className="w-3 h-3" />
                        Cited
                      </button>
                    )}
                  </div>
                </>
              )}
              {isCited && citationUsages.length > 0 ? (
                <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-text-muted sm:text-xs">
                  <span>Used in answer:</span>
                  {citationUsages.slice(0, 4).map((usage) => (
                    <button
                      key={`${index}-${usage.sentenceNumber}-${usage.occurrence}`}
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        onJumpToCitation(index, usage.occurrence)
                      }}
                      className="rounded border border-primary/30 bg-primary/10 px-1.5 py-0.5 text-primary hover:bg-primary/15 transition-colors"
                      aria-label={`Jump to citation ${index} in answer sentence ${usage.sentenceNumber}: ${usage.sentencePreview}`}
                      title={`Sentence ${usage.sentenceNumber}: ${usage.sentencePreview}`}
                    >
                      S{usage.sentenceNumber}
                    </button>
                  ))}
                  {citationUsages.length > 4 ? (
                    <span className="text-text-muted">
                      +{citationUsages.length - 4} more
                    </span>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {/* Content excerpt */}
      <div className={cn("px-3 pb-2", compactDensity ? "sm:px-3 sm:pb-2" : "sm:px-4 sm:pb-3")}>
        <p className={cn("text-xs text-text-muted leading-relaxed whitespace-pre-wrap", !compactDensity && "sm:text-sm")}>
          {excerptSegments.map((segment, segmentIndex) =>
            segment.highlight ? (
              <mark
                key={`segment-${segmentIndex}`}
                className="rounded bg-primary/20 px-0.5 text-text"
              >
                {segment.text}
              </mark>
            ) : (
              <React.Fragment key={`segment-${segmentIndex}`}>
                {segment.text}
              </React.Fragment>
            )
          )}
        </p>
        {canExpand && (
          <button
            type="button"
            onClick={() => setIsExpanded((previous) => !previous)}
            className="mt-2 text-xs font-medium text-primary hover:text-primaryStrong transition-colors"
          >
            {isExpanded ? "Show less" : "Show more"}
          </button>
        )}
      </div>

      {/* Actions — primary/secondary split */}
      <div
        className={cn(
          "flex flex-wrap items-center justify-between gap-2 border-t border-border/50 bg-bg-subtle px-3 py-2",
          compactDensity ? "sm:px-3" : "sm:px-4"
        )}
      >
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onViewFull(result, index)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            title="View the full source content"
            aria-label={`View source ${index}`}
          >
            <FileText className="w-3.5 h-3.5" />
            View
          </button>

          <button
            type="button"
            onClick={handleCopyCitation}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            title="Copy citation"
            aria-label={`Copy citation for source ${index}`}
          >
            {copiedState === "citation" ? "Copied!" : "Cite"}
          </button>

          <div ref={overflowRef} className="relative">
            <button
              type="button"
              onClick={() => setOverflowOpen((prev) => !prev)}
              className="flex items-center gap-1.5 px-2 py-1.5 text-xs font-medium rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
              aria-label={`More actions for source ${index}`}
              aria-expanded={overflowOpen}
              aria-haspopup="menu"
            >
              <MoreHorizontal className="w-3.5 h-3.5" />
            </button>
            {overflowOpen && (
              <div
                role="menu"
                className="absolute right-0 top-full z-20 mt-1 min-w-[180px] rounded-md border border-border bg-surface py-1 shadow-lg"
              >
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    onTogglePin(result, index)
                    setOverflowOpen(false)
                  }}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  <Pin className="w-3.5 h-3.5" />
                  {isPinned ? "Unpin" : "Pin"}
                </button>
                <div className="my-1 border-t border-border/50" role="separator" />
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    onAskAbout(result, "detail")
                    setOverflowOpen(false)
                  }}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Ask: Tell me more
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    onAskAbout(result, "summary")
                    setOverflowOpen(false)
                  }}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Ask: Summarize
                </button>
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    onAskAbout(result, "quotes")
                    setOverflowOpen(false)
                  }}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Ask: Key quotes
                </button>
                <div className="my-1 border-t border-border/50" role="separator" />
                <button
                  type="button"
                  role="menuitem"
                  aria-label={
                    copiedState === "excerpt"
                      ? `Copied excerpt from source ${index}`
                      : `Copy excerpt from source ${index}`
                  }
                  onClick={() => {
                    handleCopyExcerpt()
                    setOverflowOpen(false)
                  }}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                >
                  {copiedState === "excerpt" ? (
                    <Check className="w-3.5 h-3.5" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                  {copiedState === "excerpt" ? "Copied excerpt" : "Copy excerpt"}
                </button>
                {url && (
                  <button
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      handleOpenExternal()
                      setOverflowOpen(false)
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                    Open original
                  </button>
                )}
                {canOpenInWorkspace && (
                  <>
                    <div className="my-1 border-t border-border/50" role="separator" />
                    <button
                      type="button"
                      role="menuitem"
                      aria-label={t("sourceCard.openInWorkspace", "Open in Document Workspace")}
                      onClick={() => {
                        window.open(
                          `/document-workspace?open=${resolvedMediaId}`,
                          "_blank",
                          "noopener,noreferrer"
                        )
                        setOverflowOpen(false)
                      }}
                      className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-subtle hover:bg-hover hover:text-text transition-colors"
                    >
                      <BookOpen className="w-3.5 h-3.5" />
                      {t("sourceCard.openInWorkspace", "Open in Document Workspace")}
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feedback — visible on hover/focus */}
      <div
        className={cn(
          "flex flex-wrap items-center gap-2 text-xs opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity",
          compactDensity
            ? "justify-end px-3 pb-2 pt-1"
            : "border-t border-border/50 px-4 py-2"
        )}
      >
        {!compactDensity ? <span className="text-text-muted">Relevant?</span> : null}
        <button
          type="button"
          onClick={() => onSourceFeedback(result, index, "up")}
          disabled={feedbackSubmitting}
          aria-pressed={feedbackThumb === "up"}
          className={cn(
            "inline-flex items-center gap-1 rounded-md border px-2 py-1 transition-colors",
            feedbackThumb === "up"
              ? "border-primary bg-primary/10 text-primary"
              : "border-border text-text-subtle hover:text-text hover:bg-hover",
            feedbackSubmitting && "opacity-60 cursor-not-allowed"
          )}
        >
          <ThumbsUp className="w-3.5 h-3.5" />
          Yes
        </button>
        <button
          type="button"
          onClick={() => onSourceFeedback(result, index, "down")}
          disabled={feedbackSubmitting}
          aria-pressed={feedbackThumb === "down"}
          className={cn(
            "inline-flex items-center gap-1 rounded-md border px-2 py-1 transition-colors",
            feedbackThumb === "down"
              ? "border-primary bg-primary/10 text-primary"
              : "border-border text-text-subtle hover:text-text hover:bg-hover",
            feedbackSubmitting && "opacity-60 cursor-not-allowed"
          )}
        >
          <ThumbsDown className="w-3.5 h-3.5" />
          No
        </button>
        {feedbackError && (
          <button
            type="button"
            onClick={() => onRetrySourceFeedback(result, index)}
            className="text-primary underline hover:opacity-80"
          >
            Retry feedback
          </button>
        )}
      </div>
    </div>
  )
}
