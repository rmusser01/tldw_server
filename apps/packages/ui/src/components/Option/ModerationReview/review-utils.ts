import type {
  ModerationDecisionAction,
  ModerationReviewItem,
  ModerationReviewSort,
  ModerationReviewStatus,
  ModerationSeverity
} from "@/services/moderation"
import { getDesignSystemState } from "@/design-system"

const BLOCKED_STATE_LABEL = getDesignSystemState("blocked").label

export const REVIEW_STATUS_LABELS: Record<ModerationReviewStatus, string> = {
  needs_review: "Needs review",
  approved: "Approved",
  blocked: BLOCKED_STATE_LABEL,
  redacted: "Redacted",
  dismissed: "Dismissed",
  escalated: "Escalated"
}

export const SEVERITY_LABELS: Record<ModerationSeverity, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical"
}

export const DECISION_ACTION_LABELS: Record<ModerationDecisionAction, string> = {
  approve: "Approve",
  block: "Block",
  redact: "Redact",
  dismiss: "Dismiss",
  escalate: "Escalate"
}

export function decisionActionLabel(action: ModerationDecisionAction): string {
  return DECISION_ACTION_LABELS[action] || action
}

export function decisionRequiresReason(action: ModerationDecisionAction): boolean {
  return action === "block" || action === "redact" || action === "escalate"
}

export function decisionNeedsConfirmation(action: ModerationDecisionAction): boolean {
  return action === "block" || action === "redact" || action === "escalate"
}

export function formatReviewDate(value?: string | null): string {
  if (!value) {
    return "Unknown time"
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  })
}

export function getReviewItemSourceLabel(item: Pick<ModerationReviewItem, "source_type" | "source_id">): string {
  if (item.source_type && item.source_id) {
    return `${item.source_type}: ${item.source_id}`
  }
  return item.source_type || item.source_id || "Unknown source"
}

export function getReviewItemUserLabel(
  item: Pick<ModerationReviewItem, "user_id" | "session_id">
): string {
  if (item.user_id) {
    return item.user_id
  }
  return item.session_id ? `Session ${item.session_id}` : "Unknown user"
}

export function sortReviewItems(
  items: ModerationReviewItem[],
  sort: ModerationReviewSort
): ModerationReviewItem[] {
  return [...items].sort((a, b) => {
    const aTime = new Date(a.created_at).getTime()
    const bTime = new Date(b.created_at).getTime()
    const normalizedA = Number.isNaN(aTime) ? 0 : aTime
    const normalizedB = Number.isNaN(bTime) ? 0 : bTime
    return sort === "oldest" ? normalizedA - normalizedB : normalizedB - normalizedA
  })
}

export function isPermissionDeniedError(error: unknown): boolean {
  const status = (error as { status?: unknown; statusCode?: unknown })?.status
  const statusCode = (error as { status?: unknown; statusCode?: unknown })?.statusCode
  if (status === 401 || status === 403 || statusCode === 401 || statusCode === 403) {
    return true
  }
  const message = error instanceof Error ? error.message : String((error as { message?: unknown })?.message || "")
  return /forbidden|permission|unauthorized/i.test(message)
}

export function isBackendUnsupportedError(error: unknown): boolean {
  const status = (error as { status?: unknown; statusCode?: unknown })?.status
  const statusCode = (error as { status?: unknown; statusCode?: unknown })?.statusCode
  if (status === 404 || statusCode === 404) {
    return true
  }
  const message = error instanceof Error ? error.message : String((error as { message?: unknown })?.message || "")
  return /not found|unsupported/i.test(message)
}

export function getSafeFieldWarnings(item: ModerationReviewItem | null): string[] {
  if (!item) {
    return []
  }
  const safe = item.safe_fields || {}
  const warnings: string[] = []
  for (const key of ["excerpt", "context", "effective_policy", "matches"]) {
    if (safe[key] === false) {
      warnings.push(key)
    }
  }
  return warnings
}
