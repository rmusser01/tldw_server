import React from "react"
import Link from "next/link"

type ServerHealthWarningBannerProps = {
  degradedChecks?: string[]
}

export const ServerHealthWarningBanner: React.FC<
  ServerHealthWarningBannerProps
> = ({ degradedChecks = [] }) => {
  const affectedChecks = degradedChecks.filter(
    (check) => check.trim().length > 0
  )
  const affectedCopy =
    affectedChecks.length > 0
      ? `${affectedChecks.join(", ")} may be limited.`
      : "some server features may be limited."

  return (
    <div
      role="status"
      aria-live="polite"
      className="border-b border-warn/30 bg-warn/10 px-4 py-3 text-sm text-text"
    >
      <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="font-medium text-warn">Server partially degraded</p>
          <p className="text-text-muted">
            Chat is available, but {affectedCopy}
          </p>
        </div>
        <Link
          href="/settings/health"
          className="inline-flex shrink-0 items-center rounded-md border border-warn/40 px-3 py-1.5 text-xs font-medium text-warn transition-colors hover:bg-warn/10 focus:outline-none focus:ring-2 focus:ring-warn/40"
        >
          Open Health & diagnostics
        </Link>
      </div>
    </div>
  )
}

export default ServerHealthWarningBanner
