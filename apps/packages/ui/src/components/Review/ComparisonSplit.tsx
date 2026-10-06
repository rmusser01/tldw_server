import React from "react"
import { Button, Switch, Tooltip, Tag } from "antd"
import { X, Link2, Unlink2 } from "lucide-react"
import { ContentRenderer } from "@/components/Review/ContentRenderer"
import { useSyncedScroll } from "@/components/Review/hooks/useSyncedScroll"
import {
  computeDiffSync,
  shouldRequireSampling,
  type DiffLine
} from "@/components/Media/diff-worker-client"
import type { MediaDetail } from "@/components/Review/media-review-types"
import { getContent } from "@/components/Review/media-review-types"

interface ComparisonSplitProps {
  items: MediaDetail[]
  hideTranscriptTimings: boolean
  onClose: () => void
  t: (key: string, fallback: string, opts?: Record<string, unknown>) => string
}

/**
 * Inline split-pane comparison for 2-4 selected items.
 * When exactly 2 items, shows inline diff highlights.
 * Supports synchronized scrolling across panels.
 */
export const ComparisonSplit: React.FC<ComparisonSplitProps> = ({
  items,
  hideTranscriptTimings,
  onClose,
  t
}) => {
  const [syncScroll, setSyncScroll] = React.useState(true)
  const [showDiff, setShowDiff] = React.useState(true)
  const syncBindingKey = items.map((item) => String(item.id)).join("|")
  const { setRef } = useSyncedScroll(syncScroll, syncBindingKey)

  const diffLines = React.useMemo<DiffLine[] | null>(() => {
    if (items.length !== 2 || !showDiff) return null
    const left = getContent(items[0]) || ""
    const right = getContent(items[1]) || ""
    // Skip diff for very large content
    if (shouldRequireSampling(left, right)) return null
    return computeDiffSync(left, right)
  }, [items, showDiff])

  if (items.length < 2) return null

  return (
    <div className="flex flex-col h-full" data-testid="comparison-split">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-2 px-2 py-1 border-b border-border bg-surface2/50">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium">
            {t("mediaPage.comparing", "Comparing {{count}} items", { count: items.length })}
          </span>
          <Tooltip title={t("mediaPage.syncScrollTooltip", "Synchronize scrolling across panels")}>
            <div className="flex items-center gap-1">
              {syncScroll ? <Link2 className="w-3.5 h-3.5" /> : <Unlink2 className="w-3.5 h-3.5" />}
              <Switch
                size="small"
                checked={syncScroll}
                onChange={setSyncScroll}
                data-testid="sync-scroll-toggle"
              />
              <span className="text-xs text-text-muted">
                {t("mediaPage.syncScroll", "Sync scroll")}
              </span>
            </div>
          </Tooltip>
          {items.length === 2 && (
            <Tooltip title={t("mediaPage.showDiffTooltip", "Highlight differences between the two items")}>
              <div className="flex items-center gap-1">
                <Switch
                  size="small"
                  checked={showDiff}
                  onChange={setShowDiff}
                  data-testid="diff-toggle"
                />
                <span className="text-xs text-text-muted">
                  {t("mediaPage.showDiff", "Show diff")}
                </span>
              </div>
            </Tooltip>
          )}
        </div>
        <Button
          size="small"
          type="text"
          icon={<X className="w-4 h-4" />}
          onClick={onClose}
          aria-label={t("mediaPage.closeComparison", "Close comparison") as string}
          data-testid="close-comparison"
        />
      </div>

      {/* Split panels */}
      <div
        className="flex flex-1 min-h-0 divide-x divide-border"
        data-testid="comparison-panels"
      >
        {items.map((item, idx) => (
          <div
            key={String(item.id)}
            className="flex-1 flex flex-col min-w-0"
            data-testid={`comparison-panel-${idx}`}
          >
            {/* Panel header */}
            <div className="px-2 py-1 border-b border-border bg-surface2/30 flex items-center gap-2">
              <Tag>{idx + 1}</Tag>
              <span className="text-sm font-medium truncate">
                {item.title || `${t("mediaPage.media", "Media")} ${item.id}`}
              </span>
              {item.type && (
                <Tag className="text-[10px]">{String(item.type).toLowerCase()}</Tag>
              )}
            </div>

            {/* Panel content */}
            <div
              ref={setRef(idx)}
              className="flex-1 overflow-auto p-3"
              data-testid={`comparison-content-${idx}`}
            >
              {diffLines && items.length === 2 ? (
                <DiffContent
                  lines={diffLines}
                  side={idx === 0 ? "left" : "right"}
                />
              ) : (
                <ContentRenderer
                  content={getContent(item) || ""}
                  hideTranscriptTimings={hideTranscriptTimings}
                />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Renders diff-highlighted content for one side */
function DiffContent({ lines, side }: { lines: DiffLine[]; side: "left" | "right" }) {
  return (
    <div className="text-sm font-mono whitespace-pre-wrap" data-testid={`diff-content-${side}`}>
      {lines.map((line, i) => {
        if (line.type === "same") {
          return (
            <div key={i} className="leading-relaxed">
              {line.text}
            </div>
          )
        }
        if (line.type === "del" && side === "left") {
          return (
            <div
              key={i}
              className="leading-relaxed bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200"
              data-diff-type="del"
            >
              {line.text}
            </div>
          )
        }
        if (line.type === "add" && side === "right") {
          return (
            <div
              key={i}
              className="leading-relaxed bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200"
              data-diff-type="add"
            >
              {line.text}
            </div>
          )
        }
        // Skip: additions on left side, deletions on right side
        return null
      })}
    </div>
  )
}
