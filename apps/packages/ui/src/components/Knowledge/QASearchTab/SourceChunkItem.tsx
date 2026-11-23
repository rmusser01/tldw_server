import React from "react"
import { Button, Tooltip } from "antd"
import { Check, Copy, Eye, Pin, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { QADocument } from "../hooks/useQASearch"
import { highlightText } from "@/utils/text-highlight"
import { chunkScore, relevanceLevel } from "./relevance-label"

type SourceChunkItemProps = {
  document: QADocument
  index: number
  query?: string
  onCopy: (doc: QADocument) => void
  onInsert: (doc: QADocument) => void
  onPin: (doc: QADocument) => void
  onPreview: (doc: QADocument) => void
  isPinned?: boolean
  /** The strongest score in the same list; relevance is relative to it. */
  topScore?: number | null
}

/**
 * Individual source chunk card displaying retrieved text with a plain
 * relevance label (CC-10); the raw score is in the label's tooltip.
 */
export const SourceChunkItem: React.FC<SourceChunkItemProps> = React.memo(
  ({
    document,
    index,
    query,
    onCopy,
    onInsert,
    onPin,
    onPreview,
    isPinned = false,
    topScore = null
  }) => {
    const { t } = useTranslation(["sidepanel"])
    const [copied, setCopied] = React.useState(false)

    const text =
      document.content || document.text || document.chunk || ""
    const snippet = text.slice(0, 400) + (text.length > 400 ? "..." : "")

    const title =
      (document.metadata as Record<string, unknown>)?.title as string ??
      (document.metadata as Record<string, unknown>)?.source as string ??
      ""

    const type =
      (document.metadata as Record<string, unknown>)?.type as string ?? ""

    const score = chunkScore(document)
    const level = relevanceLevel(score, topScore)
    const relevanceText =
      level === "best"
        ? t("sidepanel:qaSearch.relevance.best", "Best match")
        : level === "good"
          ? t("sidepanel:qaSearch.relevance.good", "Good match")
          : level === "weaker"
            ? t("sidepanel:qaSearch.relevance.weaker", "Weaker match")
            : null
    const relevanceDetail =
      score === null
        ? ""
        : (t(
            "sidepanel:qaSearch.relevance.tooltip",
            "Compared with the other results. Score {{score}}",
            { score: score.toFixed(2) }
          ) as string)

    const handleCopy = () => {
      onCopy(document)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }

    return (
      <div className="rounded-lg border border-border bg-surface p-3 transition-colors hover:border-accent/50">
        {/* Header */}
        <div className="flex items-start justify-between gap-2 mb-1">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <span className="inline-flex h-5 min-w-[20px] items-center justify-center rounded bg-surface3 px-1 text-[10px] font-mono text-text-muted flex-shrink-0">
              {index + 1}
            </span>
            {title && (
              <span className="text-xs font-medium text-text truncate">
                {title}
              </span>
            )}
            {type && (
              <span className="text-[10px] text-text-muted capitalize flex-shrink-0">
                {type}
              </span>
            )}
          </div>
          {relevanceText && score !== null && (
            // The comparison and score are text too, so screen readers get
            // them; the hover title repeats them (CC-10 review).
            <span
              className="text-xs text-text-muted whitespace-nowrap"
              title={relevanceDetail}
            >
              {relevanceText}
              <span className="sr-only">{relevanceDetail}</span>
            </span>
          )}
        </div>

        {/* Content */}
        <p className="text-xs text-text-muted line-clamp-4 my-2">
          {query
            ? highlightText(snippet, query, {
                highlightClassName: "bg-warn/20 text-text rounded px-0.5"
              })
            : snippet}
        </p>

        {/* Actions */}
        <div className="flex items-center gap-1">
          <Tooltip title={t("sidepanel:qaSearch.copyChunk", "Copy chunk")}>
            <Button
              type="text"
              size="small"
              onClick={handleCopy}
              icon={
                copied ? (
                  <Check className="h-3 w-3 text-success" />
                ) : (
                  <Copy className="h-3 w-3" />
                )
              }
              className="text-text-muted hover:text-accent"
            >
              {copied
                ? t("sidepanel:qaSearch.copied", "Copied")
                : t("sidepanel:qaSearch.copy", "Copy")}
            </Button>
          </Tooltip>

          <Tooltip
            title={t("sidepanel:qaSearch.insertChunk", "Insert chunk")}
          >
            <Button
              type="text"
              size="small"
              onClick={() => onInsert(document)}
              icon={<Plus className="h-3 w-3" />}
              className="text-text-muted hover:text-accent"
            >
              {t("sidepanel:rag.actions.insert", "Insert")}
            </Button>
          </Tooltip>

          <Tooltip title={t("sidepanel:rag.actions.preview", "Preview")}>
            <Button
              type="text"
              size="small"
              onClick={() => onPreview(document)}
              icon={<Eye className="h-3 w-3" />}
              className="text-text-muted hover:text-accent"
            >
              {t("sidepanel:rag.actions.preview", "Preview")}
            </Button>
          </Tooltip>

          <Tooltip
            title={
              isPinned
                ? t("sidepanel:rag.actions.pinned", "Already pinned")
                : t("sidepanel:rag.actions.pin", "Pin")
            }
          >
            <span className="inline-block">
              <Button
                type="text"
                size="small"
                onClick={() => onPin(document)}
                disabled={isPinned}
                icon={
                  <Pin
                    className={`h-3 w-3 ${isPinned ? "fill-current" : ""}`}
                  />
                }
                className={
                  isPinned
                    ? "text-accent"
                    : "text-text-muted hover:text-accent"
                }
              >
                {t("sidepanel:rag.actions.pin", "Pin")}
              </Button>
            </span>
          </Tooltip>
        </div>
      </div>
    )
  }
)

SourceChunkItem.displayName = "SourceChunkItem"
