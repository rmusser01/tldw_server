import React from "react"
import { X } from "lucide-react"

type FirstRunBannerProps = {
  variant: "resume" | "nudge"
  resumeStep?: string | null
  onResume?: () => void
  onDismiss: () => void
}

/**
 * Dismissible banner shown across various surfaces to prompt the user to
 * set up or resume configuring their assistant.
 *
 * **"resume"** variant: amber/warning-toned, shown when the user has an
 * in-progress setup. Includes a "Resume" button and a dismiss control.
 *
 * **"nudge"** variant: subtle/muted, shown as a gentle prompt to begin
 * setup. Includes a "Set up" link and a dismiss control.
 */
export const FirstRunBanner: React.FC<FirstRunBannerProps> = ({
  variant,
  resumeStep: _resumeStep,
  onResume,
  onDismiss
}) => {
  if (variant === "resume") {
    return (
      <div
        data-testid="first-run-banner-resume"
        role="alert"
        className="flex items-center justify-between gap-3 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2"
      >
        <p className="text-sm text-warn">Continue setting up your assistant?</p>
        <div className="flex items-center gap-2">
          {onResume && (
            <button
              type="button"
              data-testid="first-run-banner-resume-btn"
              className="rounded-md bg-warn px-3 py-1 text-xs font-medium text-white transition-colors duration-150 hover:opacity-90"
              onClick={onResume}
            >
              Resume
            </button>
          )}
          <button
            type="button"
            aria-label="Dismiss"
            data-testid="first-run-banner-dismiss"
            className="flex-shrink-0 rounded p-1 text-warn transition-colors duration-150 hover:bg-surface2"
            onClick={onDismiss}
          >
            <X className="size-4" />
          </button>
        </div>
      </div>
    )
  }

  // nudge variant
  return (
    <div
      data-testid="first-run-banner-nudge"
      role="status"
      className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface2 px-3 py-2"
    >
      <p className="text-sm text-text-muted">
        Set up an assistant to get more out of this.
        {onResume && (
          <>
            {" "}
            <button
              type="button"
              data-testid="first-run-banner-setup-link"
              className="text-sm font-medium text-primary underline transition-colors duration-150 hover:text-primaryStrong"
              onClick={onResume}
            >
              Set up
            </button>
          </>
        )}
      </p>
      <button
        type="button"
        aria-label="Dismiss"
        data-testid="first-run-banner-dismiss"
        className="flex-shrink-0 rounded p-1 text-text-muted transition-colors duration-150 hover:bg-surface hover:text-text"
        onClick={onDismiss}
      >
        <X className="size-4" />
      </button>
    </div>
  )
}
