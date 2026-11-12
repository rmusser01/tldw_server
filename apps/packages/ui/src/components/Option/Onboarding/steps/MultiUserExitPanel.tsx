import React from "react"
import { ExternalLink, Users } from "lucide-react"

import type { FirstRunMetadata } from "@/types/setup-onboarding"

type MultiUserExitPanelProps = {
  metadata: FirstRunMetadata | null
  onBack?: () => void
  onSignIn?: () => void
  onOpenConnectionSettings?: () => void
  loginPending?: boolean
}

const DOCS_REPO_BASE = "https://github.com/rmusser01/tldw_server/blob/main/"

const resolveDocsHref = (path: string): string => {
  const trimmed = path.trim()
  if (
    /^https?:\/\//i.test(trimmed) ||
    trimmed.startsWith("/") ||
    trimmed.startsWith("#")
  ) {
    return trimmed
  }
  return `${DOCS_REPO_BASE}${trimmed.replace(/^\.?\//, "")}`
}

export function MultiUserExitPanel({
  metadata,
  onBack,
  onSignIn,
  onOpenConnectionSettings,
  loginPending
}: MultiUserExitPanelProps) {
  const guidePath = resolveDocsHref(
    metadata?.multi_user_exit?.guide_path ||
    "Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md"
  )
  const checklistPath = metadata?.multi_user_exit?.checklist_path
    ? resolveDocsHref(metadata.multi_user_exit.checklist_path)
    : null

  return (
    <section aria-labelledby="multi-user-exit-title" className="space-y-5">
      <div className="flex items-center gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <Users className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2 id="multi-user-exit-title" className="text-lg font-semibold text-text">
            Multi-user setup guide
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            {onSignIn ? "Your server is configured for multi-user access." : "Use the operator guide to configure multi-user access."}
          </p>
        </div>
      </div>

      <div className="rounded-md border border-border bg-surface px-4 py-4">
        <p className="text-sm text-text">
          {onSignIn
            ? "Sign in with your server account. If you need an account, ask your administrator. Operators can use the guide to finish deployment configuration."
            : "Follow the multi-user guide for auth mode, database, admin account, and deployment hardening. Then open connection settings to sign in."}
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          <a
            href={guidePath}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text hover:bg-surface3"
          >
            <ExternalLink className="size-4" aria-hidden="true" />
            Open guide
          </a>
          {checklistPath ? (
            <a
              href={checklistPath}
              className="inline-flex items-center gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text hover:bg-surface3"
            >
              <ExternalLink className="size-4" aria-hidden="true" />
              Open deployment checklist
            </a>
          ) : null}
        </div>
      </div>

      {onSignIn ? <button
        type="button"
        onClick={onSignIn}
        disabled={loginPending}
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
      >
        {loginPending ? "Opening login..." : "Sign in"}
      </button> : onOpenConnectionSettings ? <button
        type="button"
        onClick={onOpenConnectionSettings}
        className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground"
      >
        Open connection settings
      </button> : null}
      {onBack ? <button
        type="button"
        onClick={onBack}
        className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
      >
        Back to setup paths
      </button> : null}
    </section>
  )
}
