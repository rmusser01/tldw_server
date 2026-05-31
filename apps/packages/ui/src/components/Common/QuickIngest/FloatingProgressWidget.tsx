import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { useTranslation } from "react-i18next"
import { AlertTriangle, Loader2, Check, ExternalLink, XCircle } from "lucide-react"
import { useIngestWizard } from "./IngestWizardContext"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import type { WizardResultItem } from "./types"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Seconds to show the "Done!" state before auto-dismissing. */
const AUTO_DISMISS_DELAY_MS = 10_000

type WidgetTerminalState = "complete" | "failed" | "cancelled" | "interrupted"

type CompletionSummaryInput = {
  collectionName?: string | null
  completedCount: number
  totalCount: number
  results: WizardResultItem[]
}

const plural = (count: number, singular: string, pluralLabel: string): string =>
  `${count} ${count === 1 ? singular : pluralLabel}`

export const buildFloatingProgressCompletionSummary = ({
  collectionName,
  completedCount,
  totalCount,
  results,
}: CompletionSummaryInput): {
  title: string
  detail: string | null
  readinessHint: string | null
} => {
  const title = collectionName?.trim() || ""
  if (results.length === 0) {
    return {
      title,
      detail:
        totalCount > 0
          ? `${completedCount}/${totalCount} finished`
          : null,
      readinessHint: null,
    }
  }

  let succeeded = 0
  let skipped = 0
  let failed = 0
  let cancelled = 0
  for (const item of results) {
    if (item.outcome === "cancelled") {
      cancelled += 1
    } else if (item.outcome === "skipped") {
      skipped += 1
    } else if (item.status === "error" || item.outcome === "failed" || item.outcome === "submit_failed") {
      failed += 1
    } else {
      succeeded += 1
    }
  }

  const parts = [
    succeeded > 0 ? plural(succeeded, "succeeded", "succeeded") : null,
    skipped > 0 ? plural(skipped, "skipped", "skipped") : null,
    failed > 0 ? plural(failed, "failed", "failed") : null,
    cancelled > 0 ? plural(cancelled, "cancelled", "cancelled") : null,
  ].filter((part): part is string => Boolean(part))

  return {
    title,
    detail: parts.length > 0 ? parts.join(", ") : null,
    readinessHint:
      collectionName && results.length > 1
        ? "Open the wizard for collection readiness and retry options."
        : null,
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const FloatingProgressWidget: React.FC = () => {
  const { t } = useTranslation(["option"])
  const { state, restore } = useIngestWizard()
  const { processingState, isMinimized, results, conferenceBatchMetadata } = state
  const { sessionVisibility, sessionLifecycle, showSession } = useQuickIngestSessionStore((store) => ({
    sessionLifecycle: store.session?.lifecycle,
    sessionVisibility: store.session?.visibility,
    showSession: store.showSession,
  }))
  const [dismissed, setDismissed] = useState(false)
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // Compute counts and overall progress
  const { completedCount, totalCount, overallPercent } = useMemo(() => {
    const items = processingState.perItemProgress
    if (items.length === 0) return { completedCount: 0, totalCount: 0, overallPercent: 0 }

    let completed = 0
    let percentSum = 0
    for (const p of items) {
      if (p.status === "complete" || p.status === "failed" || p.status === "cancelled") {
        completed++
      }
      percentSum += p.progressPercent
    }

    return {
      completedCount: completed,
      totalCount: items.length,
      overallPercent: Math.round(percentSum / items.length),
    }
  }, [processingState.perItemProgress])

  const terminalState = useMemo<WidgetTerminalState | null>(() => {
    const items = processingState.perItemProgress
    const hasItems = items.length > 0
    const hasFailed = items.some((item) => item.status === "failed")
    const hasCancelled = items.some((item) => item.status === "cancelled")
    const allTerminal =
      hasItems &&
      items.every((item) =>
        item.status === "complete" ||
        item.status === "failed" ||
        item.status === "cancelled"
      )
    const allComplete = hasItems && items.every((item) => item.status === "complete")
    const allCancelled = hasItems && items.every((item) => item.status === "cancelled")

    if (sessionLifecycle === "interrupted") return "interrupted"
    if (
      sessionLifecycle === "partial_failure" ||
      processingState.status === "error" ||
      hasFailed
    ) {
      return "failed"
    }
    if (
      sessionLifecycle === "cancelled" ||
      processingState.status === "cancelled" ||
      allCancelled
    ) {
      return "cancelled"
    }
    if (
      sessionLifecycle === "completed" ||
      processingState.status === "complete" ||
      allComplete
    ) {
      return "complete"
    }
    if (allTerminal) {
      return hasCancelled ? "cancelled" : "complete"
    }
    return null
  }, [
    processingState.perItemProgress,
    processingState.status,
    sessionLifecycle,
  ])

  const allDone = terminalState !== null
  const completionSummary = useMemo(
    () =>
      buildFloatingProgressCompletionSummary({
        collectionName: conferenceBatchMetadata?.collectionName,
        completedCount,
        totalCount,
        results,
      }),
    [completedCount, conferenceBatchMetadata?.collectionName, results, totalCount]
  )

  // Auto-dismiss after completion
  useEffect(() => {
    if (!isMinimized) {
      // Reset dismissed state when not minimized
      setDismissed(false)
      return
    }

    if (terminalState && isMinimized && !dismissed) {
      dismissTimerRef.current = setTimeout(() => {
        setDismissed(true)
      }, AUTO_DISMISS_DELAY_MS)
    }

    return () => {
      if (dismissTimerRef.current) {
        clearTimeout(dismissTimerRef.current)
        dismissTimerRef.current = null
      }
    }
  }, [terminalState, isMinimized, dismissed])

  const handleOpen = useCallback(() => {
    setDismissed(false)
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current)
      dismissTimerRef.current = null
    }
    showSession()
    restore()
  }, [restore, showSession])

  // Only render when minimized and not dismissed
  if (!isMinimized || sessionVisibility !== "hidden" || dismissed) return null

  const estimatedText =
    processingState.estimatedRemaining > 0
      ? processingState.estimatedRemaining < 60
        ? `~${Math.ceil(processingState.estimatedRemaining)}s`
        : `~${Math.ceil(processingState.estimatedRemaining / 60)} min`
      : ""

  const terminalPresentation = terminalState
    ? {
        complete: {
          icon: <Check className="h-4 w-4 text-primary" strokeWidth={2.5} aria-hidden="true" />,
          label: qi("widget.done", "Done"),
          barClassName: "bg-primary",
        },
        failed: {
          icon: <XCircle className="h-4 w-4 text-danger" strokeWidth={2.5} aria-hidden="true" />,
          label: qi("widget.failed", "Failed"),
          barClassName: "bg-danger",
        },
        cancelled: {
          icon: <XCircle className="h-4 w-4 text-text-muted" strokeWidth={2.5} aria-hidden="true" />,
          label: qi("widget.cancelled", "Cancelled"),
          barClassName: "bg-text-muted",
        },
        interrupted: {
          icon: <AlertTriangle className="h-4 w-4 text-warn" strokeWidth={2.5} aria-hidden="true" />,
          label: qi("widget.interrupted", "Interrupted"),
          barClassName: "bg-warn",
        },
      }[terminalState]
    : null

  const widget = (
    <div
      className="fixed bottom-4 right-4 z-[9000] w-72 rounded-lg border border-border bg-surface shadow-lg"
      role="status"
      aria-live="polite"
      aria-label={qi("widget.ariaLabel", "Ingest progress")}
    >
      <div className="flex flex-col gap-2 p-3">
        {/* Header line */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-medium text-text">
            {terminalPresentation ? (
              <>
                {terminalPresentation.icon}
                <span>{completionSummary.title || terminalPresentation.label}</span>
              </>
            ) : (
              <>
                <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
                <span>
                  {qi("widget.ingesting", "Ingesting {{done}}/{{total}}", {
                    done: completedCount,
                    total: totalCount,
                  })}
                </span>
                {estimatedText && (
                  <span className="text-xs text-text-muted">{estimatedText}</span>
                )}
              </>
            )}
          </div>
        </div>

        {/* Processing description */}
        {!terminalPresentation && (
          <p className="text-[11px] leading-tight text-text-muted">
            {qi(
              "widget.processingHint",
              "Processing and indexing content..."
            )}
          </p>
        )}

        {allDone && (completionSummary.detail || completionSummary.readinessHint) && (
          <p className="text-[11px] leading-tight text-text-muted">
            {completionSummary.detail}
            {completionSummary.detail && completionSummary.readinessHint ? " " : ""}
            {completionSummary.readinessHint}
          </p>
        )}

        {/* Progress bar + percentage + Open button */}
        <div className="flex items-center gap-2">
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface2">
            <div
              className={`h-full rounded-full transition-all duration-300 ${
                terminalPresentation?.barClassName || "bg-primary"
              }`}
              style={{ width: `${overallPercent}%` }}
            />
          </div>
          <span className="w-8 text-right text-xs tabular-nums text-text-muted">
            {overallPercent}%
          </span>
          <button
            type="button"
            onClick={handleOpen}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-primary transition hover:bg-surface2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-focus"
            aria-label={qi("widget.open", "Open ingest wizard")}
          >
            <ExternalLink className="h-3 w-3" aria-hidden="true" />
            {qi("widget.openLabel", "Open")}
          </button>
        </div>
      </div>
    </div>
  )

  return createPortal(widget, document.body)
}

export default FloatingProgressWidget
