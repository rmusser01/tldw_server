import React from "react"
import Link from "next/link"

type ServerReconnectBannerProps = {
  /** Retry deadline exhausted: polling stopped, manual/online retry required. */
  exhausted?: boolean
  onRetry: () => void
}

/**
 * Non-blocking banner shown by ServerReadinessGate when health checks fail
 * for an established session (perf remediation W1). Content stays mounted
 * behind it; the user is never locked out of the app by this banner.
 */
export const ServerReconnectBanner: React.FC<
  ServerReconnectBannerProps
> = ({ exhausted = false, onRetry }) => {
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="server-reconnect-banner"
      className="border-b border-error/30 bg-error/10 px-4 py-3 text-sm text-text"
    >
      <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="font-medium text-error">Server connection unavailable</p>
          <p className="text-text-muted">
            {exhausted
              ? "The WebUI could not reach the tldw server. Some content may not load until the connection returns."
              : "Reconnecting to the tldw server. Some content may not load until the connection returns."}
          </p>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex shrink-0 items-center rounded-md border border-error/40 px-3 py-1.5 text-xs font-medium text-error transition-colors hover:bg-error/10 focus:outline-none focus:ring-2 focus:ring-error/40"
          >
            Retry connection
          </button>
          <Link
            href="/settings/health"
            className="inline-flex shrink-0 items-center rounded-md border border-error/40 px-3 py-1.5 text-xs font-medium text-error transition-colors hover:bg-error/10 focus:outline-none focus:ring-2 focus:ring-error/40"
          >
            Health &amp; diagnostics
          </Link>
        </div>
      </div>
    </div>
  )
}

export default ServerReconnectBanner
