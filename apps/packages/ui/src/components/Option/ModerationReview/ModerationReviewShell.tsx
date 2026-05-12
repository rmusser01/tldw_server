import React from "react"
import { Link } from "react-router-dom"
import {
  AlertTriangle,
  CheckCircle2,
  ClipboardList,
  History,
  ListFilter,
  ShieldCheck,
  SlidersHorizontal
} from "lucide-react"
import { useConnectionUxState } from "@/hooks/useConnectionState"
import { useServerOnline } from "@/hooks/useServerOnline"
import { MODERATION_RULES_PATH } from "@/routes/route-paths"

type ModerationReviewShellProps = {
  compact?: boolean
}

const QUEUE_PLACEHOLDERS = [
  {
    label: "Needs review",
    value: "--",
    description: "Awaiting live queue data"
  },
  {
    label: "Blocked",
    value: "--",
    description: "Decision counts not connected yet"
  },
  {
    label: "Redacted",
    value: "--",
    description: "Audit trail not connected yet"
  }
]

const REVIEW_FILTERS = [
  "All items",
  "Needs decision",
  "Escalated",
  "High severity",
  "Low confidence"
]

const backendStatusCopy = (online: boolean, uxState: string) => {
  if (online) {
    return {
      tone: "ok" as const,
      title: "Server reachable",
      description: "Review queue endpoints will appear here when they are connected."
    }
  }
  if (uxState === "error_auth" || uxState === "configuring_auth") {
    return {
      tone: "warn" as const,
      title: "Credentials needed",
      description: "Connect credentials before moderation review data can load."
    }
  }
  if (uxState === "unconfigured" || uxState === "configuring_url") {
    return {
      tone: "warn" as const,
      title: "Server setup incomplete",
      description: "Finish setup before moderation review data can load."
    }
  }
  return {
    tone: "warn" as const,
    title: "Server unreachable",
    description: "Review queue data is unavailable until the tldw server responds."
  }
}

export const ModerationReviewShell: React.FC<ModerationReviewShellProps> = ({
  compact = false
}) => {
  const online = useServerOnline()
  const { uxState } = useConnectionUxState()
  const backendStatus = backendStatusCopy(online, uxState)
  const StatusIcon = backendStatus.tone === "ok" ? CheckCircle2 : AlertTriangle

  return (
    <section
      className="space-y-6"
      data-testid="moderation-review-shell"
      aria-labelledby="moderation-review-title"
    >
      <div className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-5 shadow-sm sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-3xl">
          <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-border bg-surface2 px-3 py-1 text-xs font-medium text-text-muted">
            <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
            Review queue
          </div>
          <h1
            id="moderation-review-title"
            className="text-2xl font-semibold text-text"
          >
            Moderation Review
          </h1>
          <p className="mt-2 text-sm leading-6 text-text-muted">
            This route is reserved for reviewing flagged items, decisions, and
            escalation history. Live moderation review data is not connected in
            this slice yet.
          </p>
        </div>
        <Link
          to={MODERATION_RULES_PATH}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text transition hover:bg-surface3"
        >
          <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
          Open Content Rules
        </Link>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {QUEUE_PLACEHOLDERS.map((item) => (
          <div
            key={item.label}
            className="rounded-lg border border-border bg-surface p-4"
          >
            <div className="text-sm font-medium text-text-muted">{item.label}</div>
            <div className="mt-2 text-2xl font-semibold text-text">{item.value}</div>
            <div className="mt-1 text-xs text-text-muted">{item.description}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(280px,0.7fr)]">
        <div className="rounded-xl border border-border bg-surface p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-text">
            <ClipboardList className="h-4 w-4" aria-hidden="true" />
            Review worklist
          </div>
          <div className="mt-4 rounded-lg border border-dashed border-border bg-surface2 p-5 text-sm text-text-muted">
            The review queue is not connected yet. The next implementation
            stage should connect flagged items with policy category, severity,
            confidence, source context, and decision actions.
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-xl border border-border bg-surface p-4">
            <div className="text-sm font-semibold text-text">
              Backend contract pending
            </div>
            <p className="mt-1 text-sm text-text-muted">
              Review queue, decision, and audit endpoints are planned for the
              next slice.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4">
            <div className="text-sm font-semibold text-text">
              Reviewer permission pending
            </div>
            <p className="mt-1 text-sm text-text-muted">
              Role and permission copy will be wired once review actions are
              connected.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4">
            <div className="flex items-start gap-3">
              <StatusIcon
                className={
                  backendStatus.tone === "ok"
                    ? "mt-0.5 h-4 w-4 text-green-600"
                    : "mt-0.5 h-4 w-4 text-yellow-600"
                }
                aria-hidden="true"
              />
              <div>
                <div className="text-sm font-semibold text-text">
                  {backendStatus.title}
                </div>
                <p className="mt-1 text-sm text-text-muted">
                  {backendStatus.description}
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-surface p-4">
            <div className="flex items-center gap-2 text-sm font-semibold text-text">
              <ListFilter className="h-4 w-4" aria-hidden="true" />
              Filters planned
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {REVIEW_FILTERS.map((filter) => (
                <span
                  key={filter}
                  className="rounded-full border border-border bg-surface2 px-2.5 py-1 text-xs text-text-muted"
                >
                  {filter}
                </span>
              ))}
            </div>
          </div>

          {!compact && (
            <div className="rounded-xl border border-border bg-surface p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-text">
                <History className="h-4 w-4" aria-hidden="true" />
                Audit trail
              </div>
              <p className="mt-2 text-sm text-text-muted">
                Decision history, reviewer identity, and reversal controls are
                planned for the live review workflow.
              </p>
            </div>
          )}
        </aside>
      </div>
    </section>
  )
}
