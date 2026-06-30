import React from "react"
import { AlertCircle, LockKeyhole, ServerOff } from "lucide-react"

import { getDesignSystemState } from "@/design-system"
import { isBackendUnsupportedError, isPermissionDeniedError } from "./review-utils"

type ReviewStatePanelsProps = {
  loading?: boolean
  error?: unknown
  partial?: boolean
  warnings?: string[]
  empty?: boolean
  onRetry?: () => void
}

const errorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message
  }
  return String((error as { message?: unknown })?.message || "The review queue could not be loaded.")
}
const PERMISSION_DENIED_STATE_LABEL = getDesignSystemState("permission_denied").label

export const ReviewStatePanels: React.FC<ReviewStatePanelsProps> = ({
  loading = false,
  error,
  partial = false,
  warnings = [],
  empty = false,
  onRetry
}) => {
  if (loading) {
    return (
      <div className="rounded-lg border border-border bg-surface2 p-4" role="status">
        <div className="h-3 w-40 rounded bg-surface3" />
        <div className="mt-3 h-3 w-full rounded bg-surface3" />
        <div className="mt-2 h-3 w-2/3 rounded bg-surface3" />
      </div>
    )
  }

  if (error) {
    const permission = isPermissionDeniedError(error)
    const unsupported = isBackendUnsupportedError(error)
    const Icon = permission ? LockKeyhole : unsupported ? ServerOff : AlertCircle
    const title = permission
      ? PERMISSION_DENIED_STATE_LABEL
      : unsupported
        ? "Review backend unsupported"
        : "Review queue unavailable"
    const description = permission
      ? "Your account can reach the server but does not have moderation review permission."
      : unsupported
        ? "This server does not expose moderation review endpoints yet."
        : errorMessage(error)
    return (
      <div className="rounded-lg border border-border bg-surface p-4" role="alert">
        <div className="flex gap-3">
          <Icon className="mt-0.5 h-4 w-4 text-yellow-600" aria-hidden="true" />
          <div>
            <div className="text-sm font-semibold text-text">{title}</div>
            <p className="mt-1 text-sm text-text-muted">{description}</p>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="mt-3 rounded-md border border-border bg-surface2 px-3 py-1.5 text-sm font-medium text-text hover:bg-surface3"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (empty) {
    return (
      <div className="rounded-lg border border-border bg-surface2 p-5 text-sm text-text-muted">
        <div className="font-semibold text-text">No review items match these filters</div>
        <p className="mt-1">Try a broader status, category, source, or search filter.</p>
      </div>
    )
  }

  if (partial || warnings.length > 0) {
    return (
      <div className="rounded-lg border border-yellow-300 bg-yellow-50 p-3 text-sm text-yellow-900 dark:border-yellow-900/50 dark:bg-yellow-950/30 dark:text-yellow-200">
        <div className="font-semibold">Partial data</div>
        <ul className="mt-1 list-disc space-y-1 pl-4">
          {(warnings.length ? warnings : ["Some review fields are unavailable."]).map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      </div>
    )
  }

  return null
}
