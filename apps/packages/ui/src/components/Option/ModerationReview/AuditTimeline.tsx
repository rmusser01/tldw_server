import React from "react"
import { Clock3, History, RotateCcw, ShieldOff } from "lucide-react"

import type { ModerationReviewDecisionHistoryEntry } from "@/services/moderation"
import {
  decisionActionLabel,
  formatReviewDate,
  REVIEW_STATUS_LABELS
} from "./review-utils"

type AuditTimelineProps = {
  decisions: ModerationReviewDecisionHistoryEntry[]
}

const undoCopy = (entry: ModerationReviewDecisionHistoryEntry): string => {
  if (entry.undone_at) {
    return `Undone ${formatReviewDate(entry.undone_at)}`
  }
  if (entry.undo_eligible) {
    return entry.undo_expires_at
      ? `Undo available until ${formatReviewDate(entry.undo_expires_at)}`
      : "Undo available"
  }
  return "Undo unavailable"
}

export const AuditTimeline: React.FC<AuditTimelineProps> = ({ decisions }) => {
  return (
    <section className="rounded-md border border-border bg-surface p-3" aria-labelledby="review-audit-title">
      <div className="flex items-center gap-2">
        <History className="h-4 w-4 text-text-muted" aria-hidden="true" />
        <h3 id="review-audit-title" className="text-sm font-semibold text-text">
          Decision history
        </h3>
      </div>

      {decisions.length === 0 ? (
        <div className="mt-3 text-sm text-text-muted">No decisions recorded yet.</div>
      ) : (
        <ol className="mt-3 space-y-3">
          {decisions.map((entry) => (
            <li key={entry.id} className="rounded-md border border-border bg-surface2 p-3 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="font-medium text-text">
                  {decisionActionLabel(entry.action)} · {REVIEW_STATUS_LABELS[entry.status]}
                </div>
                <div className="text-xs text-text-muted">{formatReviewDate(entry.decided_at)}</div>
              </div>
              <div className="mt-1 text-xs text-text-muted">
                {entry.actor_id} · from {REVIEW_STATUS_LABELS[entry.previous_status]}
              </div>
              {entry.reason && <div className="mt-2 text-text-muted">{entry.reason}</div>}
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-text-muted">
                <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-1">
                  <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
                  {undoCopy(entry)}
                </span>
                {entry.redaction_state === "redacted" && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-red-200 bg-red-50 px-2 py-1 text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
                    <ShieldOff className="h-3.5 w-3.5" aria-hidden="true" />
                    Content redacted
                  </span>
                )}
                {entry.undo_expires_at && !entry.undone_at && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-1">
                    <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
                    Expires {formatReviewDate(entry.undo_expires_at)}
                  </span>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
