import React, { useEffect, useState } from "react"
import { X, PanelRightOpen, FileText, BarChart3 } from "lucide-react"
import { cn } from "@/libs/utils"
import { useDesktop } from "@/hooks/useMediaQuery"
import { SourceList } from "../SourceList"

const LazySearchDetailsPanel = React.lazy(() =>
  import("../SearchDetailsPanel").then((module) => ({
    default: module.SearchDetailsPanel,
  })),
)

type EvidenceRailProps = {
  open: boolean
  tab: "sources" | "details"
  onOpenChange: (open: boolean) => void
  onTabChange: (tab: "sources" | "details") => void
  resultsCount: number
  citationsCount: number
  className?: string
}

function EvidenceRailContent({
  tab,
  onOpenChange,
  onTabChange,
  resultsCount,
  citationsCount,
}: Omit<EvidenceRailProps, "open" | "className">) {
  const [sourceAnnouncement, setSourceAnnouncement] = useState("")

  useEffect(() => {
    setSourceAnnouncement(
      `Evidence updated. ${resultsCount} source${resultsCount === 1 ? "" : "s"} and ${citationsCount} citation${citationsCount === 1 ? "" : "s"}.`
    )
  }, [resultsCount, citationsCount])

  return (
    <div className="flex h-full flex-col">
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {sourceAnnouncement}
      </div>
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <h2 className="text-sm font-semibold">Evidence</h2>
        <span className="text-xs text-text-muted">
          {resultsCount} sources • {citationsCount} citations
        </span>
        <button
          type="button"
          onClick={() => onOpenChange(false)}
          className="ml-auto flex items-center justify-center min-w-8 min-h-8 h-8 w-8 rounded-md text-text-muted hover:bg-hover hover:text-text transition-colors"
          aria-label="Close evidence panel"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="border-b border-border px-3 py-2">
        <div className="inline-flex rounded-md border border-border bg-bg-subtle p-0.5">
          <button
            type="button"
            onClick={() => onTabChange("sources")}
            className={cn(
              "inline-flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors",
              tab === "sources"
                ? "bg-primary text-white"
                : "text-text-subtle hover:bg-hover hover:text-text"
            )}
            aria-pressed={tab === "sources"}
          >
            <FileText className="h-3.5 w-3.5" />
            Sources
          </button>
          <button
            type="button"
            onClick={() => onTabChange("details")}
            className={cn(
              "inline-flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors",
              tab === "details"
                ? "bg-primary text-white"
                : "text-text-subtle hover:bg-hover hover:text-text"
            )}
            aria-pressed={tab === "details"}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            Details
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {tab === "sources" ? (
          resultsCount > 0 ? (
            <SourceList layout="rail" />
          ) : (
            <div className="rounded-lg border border-border bg-muted/20 p-3 text-sm text-text-muted">
              No sources yet. Run a query to inspect retrieval evidence.
            </div>
          )
        ) : (
          <React.Suspense fallback={null}>
            <LazySearchDetailsPanel />
          </React.Suspense>
        )}
      </div>
    </div>
  )
}

export function EvidenceRail({
  open,
  tab,
  onOpenChange,
  onTabChange,
  resultsCount,
  citationsCount,
  className,
}: EvidenceRailProps) {
  const isDesktop = useDesktop()

  if (isDesktop) {
    if (!open) {
      return (
        <aside
          className={cn(
            "hidden w-14 shrink-0 border-l border-border bg-surface/40 lg:flex lg:flex-col lg:items-center lg:py-3",
            className
          )}
          aria-label="Open evidence panel"
        >
          <button
            type="button"
            onClick={() => onOpenChange(true)}
            className="relative rounded-md border border-border bg-surface p-2 text-text-subtle hover:bg-hover hover:text-text transition-colors"
            aria-label={`Open evidence panel (${resultsCount} source${resultsCount === 1 ? "" : "s"})`}
            aria-expanded={false}
            aria-controls="knowledge-evidence-panel"
          >
            <PanelRightOpen className="h-4 w-4" />
            {resultsCount > 0 && (
              <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[10px] font-medium text-white">
                {resultsCount}
              </span>
            )}
          </button>
        </aside>
      )
    }

    return (
      <aside
        id="knowledge-evidence-panel"
        className={cn(
          "hidden w-[400px] shrink-0 border-l border-border bg-surface/40 lg:block xl:w-[420px]",
          "motion-safe:transition-all motion-safe:duration-300 motion-safe:animate-in motion-safe:slide-in-from-right motion-reduce:animate-none motion-reduce:transition-none",
          className
        )}
        aria-label="Evidence panel"
      >
        <EvidenceRailContent
          tab={tab}
          onOpenChange={onOpenChange}
          onTabChange={onTabChange}
          resultsCount={resultsCount}
          citationsCount={citationsCount}
        />
      </aside>
    )
  }

  return (
    <>
      {!open ? (
        <button
          type="button"
          onClick={() => onOpenChange(true)}
          className="fixed bottom-4 right-4 z-40 rounded-full border border-border bg-surface px-3 py-2 text-sm text-text-subtle shadow-md hover:bg-hover hover:text-text transition-colors"
          aria-label="Open evidence panel"
          aria-expanded={false}
          aria-controls="knowledge-evidence-panel-mobile"
        >
          Evidence
        </button>
      ) : null}

      {open ? (
        <div id="knowledge-evidence-panel-mobile" className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-black/45"
            onClick={() => onOpenChange(false)}
            aria-label="Close evidence panel"
          />
          <aside className="absolute right-0 top-0 h-full w-[88vw] max-w-md border-l border-border bg-surface shadow-xl motion-safe:transition-all motion-safe:duration-300 motion-safe:animate-in motion-safe:slide-in-from-right motion-reduce:animate-none motion-reduce:transition-none">
            <EvidenceRailContent
              tab={tab}
              onOpenChange={onOpenChange}
              onTabChange={onTabChange}
              resultsCount={resultsCount}
              citationsCount={citationsCount}
            />
          </aside>
        </div>
      ) : null}
    </>
  )
}
