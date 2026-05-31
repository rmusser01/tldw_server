import React from "react"
import {
  RecoveryCallout,
  type StatePanelDiagnostic
} from "@/components/ui/state"

export type BackendUnavailableRecoveryDetails = {
  title?: React.ReactNode
  message?: React.ReactNode
  fixHint?: React.ReactNode
  subtype?: string
  method?: string
  path?: string
  serverUrl?: string
  status?: number
  rawMessage?: string
  source?: string
  recentRequestError?: unknown
  diagnostics?: unknown
}

type BackendUnavailableRecoveryProps = {
  details?: BackendUnavailableRecoveryDetails
  onRetry: () => void
  onReload: () => void
  onOpenDiagnostics: () => void
  onOpenSettings: () => void
}

const DEFAULT_TITLE = "Can't reach your tldw server right now."
const DEFAULT_MESSAGE =
  "Check that your server is running and accessible. Try again, reload the page, or open Health & diagnostics for more details."

const formatStructuredValue = (value: unknown): React.ReactNode => {
  if (value === null || value === undefined) {
    return null
  }

  if (typeof value === "string") {
    return value
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value)
  }

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

const asPreformatted = (value: unknown): React.ReactNode => {
  const formatted = formatStructuredValue(value)

  return formatted ? (
    <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-5">
      {formatted}
    </pre>
  ) : null
}

const hasDiagnostics = (details?: BackendUnavailableRecoveryDetails): boolean =>
  Boolean(
    details &&
      (details.method ||
        details.path ||
        details.serverUrl ||
        details.status !== undefined ||
        details.rawMessage ||
        details.source ||
        details.recentRequestError ||
        details.diagnostics)
  )

const buildDiagnostics = (
  details?: BackendUnavailableRecoveryDetails
): StatePanelDiagnostic[] | undefined => {
  if (!hasDiagnostics(details)) {
    return undefined
  }

  const diagnostics: StatePanelDiagnostic[] = []

  if (details?.method) {
    diagnostics.push({ label: "Request method", value: details.method })
  }

  if (details?.path) {
    diagnostics.push({ label: "Request path", value: details.path, code: true })
  }

  if (details?.serverUrl) {
    diagnostics.push({
      label: "Configured server URL",
      value: details.serverUrl,
      code: true
    })
  }

  if (details?.status !== undefined) {
    diagnostics.push({ label: "Status", value: String(details.status) })
  }

  if (details?.rawMessage) {
    diagnostics.push({ label: "Raw message", value: details.rawMessage })
  }

  if (details?.source) {
    diagnostics.push({ label: "Source", value: details.source })
  }

  if (details?.diagnostics) {
    diagnostics.push({
      label: "Additional diagnostics",
      value: asPreformatted(details.diagnostics)
    })
  }

  if (details?.recentRequestError) {
    diagnostics.push({
      label: "Recent request error",
      value: asPreformatted(details.recentRequestError)
    })
  }

  return diagnostics
}

export const BackendUnavailableRecovery: React.FC<
  BackendUnavailableRecoveryProps
> = ({
  details,
  onRetry,
  onReload,
  onOpenDiagnostics,
  onOpenSettings
}) => {
  const title = details?.title ?? DEFAULT_TITLE
  const message = details?.message ?? DEFAULT_MESSAGE
  const fixHint = details?.fixHint
  const diagnostics = buildDiagnostics(details)

  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-4 py-10 text-text">
      <RecoveryCallout
        state="unavailable"
        title={title}
        message={message}
        diagnostics={diagnostics}
        primaryAction={{ label: "Try again", onClick: onRetry }}
        secondaryActions={[
          { label: "Reload page", onClick: onReload },
          { label: "Open Health & diagnostics", onClick: onOpenDiagnostics },
          { label: "Open Settings", onClick: onOpenSettings }
        ]}
        className="w-full max-w-3xl"
      >
        {fixHint ? (
          <p
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm leading-6 text-text-muted"
            data-testid="backend-recovery-fix-hint"
          >
            <span className="font-medium text-text">How to fix: </span>
            {fixHint}
          </p>
        ) : null}
      </RecoveryCallout>
    </main>
  )
}

export default BackendUnavailableRecovery
