import React from "react"
import { FileText, ShieldAlert, ShieldOff } from "lucide-react"

import type { ModerationReviewItem } from "@/services/moderation"
import { AuditTimeline } from "./AuditTimeline"
import {
  formatReviewDate,
  getReviewItemSourceLabel,
  getReviewItemUserLabel,
  getSafeFieldWarnings,
  REVIEW_STATUS_LABELS,
  SEVERITY_LABELS
} from "./review-utils"

type ReviewItemDetailProps = {
  item: ModerationReviewItem | null
  loading?: boolean
}

const JsonPreview: React.FC<{ value: unknown }> = ({ value }) => (
  <pre className="max-h-40 max-w-full overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface2 p-3 text-xs text-text-muted">
    {JSON.stringify(value || {}, null, 2)}
  </pre>
)

export const ReviewItemDetail: React.FC<ReviewItemDetailProps> = ({ item, loading = false }) => {
  if (loading) {
    return (
      <div className="rounded-lg border border-border bg-surface p-4" role="status" aria-live="polite">
        <span className="sr-only">Loading review item detail</span>
        <div className="h-3 w-32 rounded bg-surface3" />
        <div className="mt-3 h-3 w-full rounded bg-surface3" />
        <div className="mt-2 h-3 w-2/3 rounded bg-surface3" />
      </div>
    )
  }

  if (!item) {
    return (
      <div className="rounded-lg border border-border bg-surface p-5 text-sm text-text-muted">
        Select a review item to inspect its context and policy snapshot.
      </div>
    )
  }

  const safeWarnings = getSafeFieldWarnings(item)
  const isContentRedacted = Boolean(item.content_redacted_at) || item.excerpt === "[content redacted]"

  return (
    <article className="space-y-4 rounded-lg border border-border bg-surface p-4" aria-labelledby="review-detail-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="review-detail-title" className="text-lg font-semibold text-text">
            Review item detail
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            {getReviewItemSourceLabel(item)} · {getReviewItemUserLabel(item)}
          </p>
        </div>
        <div className="flex flex-wrap gap-2 text-xs">
          <span className="rounded-full border border-border bg-surface2 px-2 py-1 text-text-muted">
            {REVIEW_STATUS_LABELS[item.status]}
          </span>
          <span className="rounded-full border border-border bg-surface2 px-2 py-1 text-text-muted">
            {item.severity ? SEVERITY_LABELS[item.severity] : "Unknown severity"}
          </span>
          <span className="rounded-full border border-border bg-surface2 px-2 py-1 text-text-muted">
            {item.phase}
          </span>
        </div>
      </div>

      {safeWarnings.length > 0 && (
        <div className="rounded-md border border-yellow-300 bg-yellow-50 p-3 text-sm text-yellow-900 dark:border-yellow-900/50 dark:bg-yellow-950/30 dark:text-yellow-200">
          <div className="flex items-center gap-2 font-semibold">
            <ShieldAlert className="h-4 w-4" aria-hidden="true" />
            Some policy fields are unavailable
          </div>
          <p className="mt-1">Unavailable safe fields: {safeWarnings.join(", ")}.</p>
        </div>
      )}

      {isContentRedacted && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
          <div className="flex items-center gap-2 font-semibold">
            <ShieldOff className="h-4 w-4" aria-hidden="true" />
            Review content redacted
          </div>
          <p className="mt-1">Excerpt, context, and match samples have been replaced with safe placeholders.</p>
        </div>
      )}

      <section>
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-text">
          <FileText className="h-4 w-4" aria-hidden="true" />
          Sanitized excerpt
        </div>
        <blockquote className="rounded-md border border-border bg-surface2 p-3 text-sm leading-6 text-text">
          {item.excerpt}
        </blockquote>
      </section>

      <dl className="grid gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-text-muted">Category</dt>
          <dd className="mt-1 text-text">{item.category || "Uncategorized"}</dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-text-muted">Created</dt>
          <dd className="mt-1 text-text">{formatReviewDate(item.created_at)}</dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-text-muted">Recommended action</dt>
          <dd className="mt-1 text-text">{item.recommended_action ? `Recommended: ${item.recommended_action}` : "None"}</dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-text-muted">Source</dt>
          <dd className="mt-1 text-text">{getReviewItemSourceLabel(item)}</dd>
        </div>
      </dl>

      <section>
        <h3 className="text-sm font-semibold text-text">Matches</h3>
        <div className="mt-2 space-y-2">
          {(item.matches || []).length === 0 && <div className="text-sm text-text-muted">No match metadata available.</div>}
          {(item.matches || []).map((match, index) => (
            <div key={`${match.category || "match"}-${index}`} className="rounded-md border border-border bg-surface2 p-3 text-sm">
              <div className="font-medium text-text">{match.category || "Uncategorized"} · {match.action || "unknown"}</div>
              <div className="mt-1 text-xs text-text-muted">
                {match.pattern_type || "pattern"} {typeof match.confidence === "number" ? `· ${Math.round(match.confidence * 100)}% confidence` : ""}
              </div>
              {match.rule_id && <div className="mt-1 text-xs text-text-muted">Rule: {match.rule_id}</div>}
              {match.sample && <div className="mt-2 text-text-muted">{match.sample}</div>}
            </div>
          ))}
        </div>
      </section>

      <details className="rounded-md border border-border bg-surface p-3">
        <summary className="cursor-pointer text-sm font-semibold text-text">Context and effective policy</summary>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div>
            <div className="mb-1 text-xs font-medium uppercase tracking-wide text-text-muted">Context</div>
            <JsonPreview value={item.context || {}} />
          </div>
          <div>
            <div className="mb-1 text-xs font-medium uppercase tracking-wide text-text-muted">Effective policy</div>
            <JsonPreview value={item.effective_policy || {}} />
          </div>
        </div>
      </details>

      <AuditTimeline decisions={item.decision_history || []} />
    </article>
  )
}
