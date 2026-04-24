import React from "react"

import { useFirstRunCheck } from "@/hooks/useFirstRunCheck"

type FirstRunGateProps = {
  children: React.ReactNode
  onStartSetup: () => void
}

const DISMISSED_KEY = "assistant_setup_dismissed"

/**
 * Wrapper component that checks whether the user needs to be shown the
 * first-run setup wizard.
 *
 * - While loading, renders children normally (no flash).
 * - If setup should be shown, renders a full-screen overlay prompting the
 *   user to start the assistant setup or skip for now.
 * - Otherwise renders children.
 */
export const FirstRunGate: React.FC<FirstRunGateProps> = ({
  children,
  onStartSetup
}) => {
  const { shouldShowSetup, loading } = useFirstRunCheck()
  const [dismissed, setDismissed] = React.useState(false)

  // While loading or if the user dismissed the overlay, render children
  if (loading || dismissed || !shouldShowSetup) {
    return <>{children}</>
  }

  const handleSkip = () => {
    try {
      localStorage.setItem(DISMISSED_KEY, "true")
    } catch {
      // localStorage may be unavailable; proceed anyway
    }
    setDismissed(true)
  }

  const handleStart = () => {
    setDismissed(true)
    onStartSetup()
  }

  return (
    <div
      data-testid="first-run-gate-overlay"
      className="fixed inset-0 z-50 flex items-center justify-center bg-bg/80 backdrop-blur-sm"
    >
      <div className="mx-4 flex max-w-md flex-col items-center gap-6 rounded-xl border border-border bg-surface p-8 text-center shadow-lg">
        <div>
          <h2 className="text-lg font-semibold text-text">
            Build Your Assistant
          </h2>
          <p className="mt-2 text-sm text-text-muted">
            Set up a personalized AI assistant to help you get more out of your
            workflow.
          </p>
        </div>
        <div className="flex flex-col items-center gap-3">
          <button
            type="button"
            data-testid="first-run-get-started"
            className="inline-flex items-center justify-center rounded-md bg-primary px-5 py-2 text-sm font-medium text-white transition-colors duration-150 hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
            onClick={handleStart}
          >
            Get Started
          </button>
          <button
            type="button"
            data-testid="first-run-skip"
            className="text-sm text-text-muted underline transition-colors duration-150 hover:text-text"
            onClick={handleSkip}
          >
            Skip for now
          </button>
        </div>
      </div>
    </div>
  )
}
