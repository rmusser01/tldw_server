import { useMemo, useState } from "react"
import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  KeyRound,
  ShieldAlert,
  WifiOff,
} from "lucide-react"

import type { ConnectionState, ConnectionUxState } from "@/types/connection"
import { requestOptionalHostPermission } from "@/utils/extension-permissions"

type DiagnosticStatus = "complete" | "missing" | "waiting" | "blocked" | "review"

type DiagnosticCheck = {
  id: string
  label: string
  status: DiagnosticStatus
  description: string
}

type Props = {
  connection: Pick<
    ConnectionState,
    | "serverUrl"
    | "configStep"
    | "errorKind"
    | "lastError"
    | "lastStatusCode"
    | "isChecking"
  >
  uxState: ConnectionUxState
  retryCountdownSeconds: number
  onOpenSetup: () => void
  onOpenSettings: () => void
  onOpenDiagnostics: () => void
  onRetryConnection: () => void
}

const BROWSER_ACCESS_BLOCK_PATTERNS = [
  /absolute url requests are blocked/i,
  /allowlist/i,
  /cors/i,
  /cross-origin/i,
  /not allowed/i,
]

const statusLabel: Record<DiagnosticStatus, string> = {
  complete: "Ready",
  missing: "Missing",
  waiting: "Waiting",
  blocked: "Blocked",
  review: "Needs review",
}

const statusClassName: Record<DiagnosticStatus, string> = {
  complete: "text-success",
  missing: "text-danger",
  waiting: "text-text-muted",
  blocked: "text-danger",
  review: "text-warning",
}

const getServerOriginLabel = (serverUrl: string | null | undefined): string | null => {
  const trimmed = String(serverUrl || "").trim()
  if (!trimmed) return null

  try {
    const parsed = new URL(trimmed)
    return parsed.origin
  } catch {
    return trimmed
  }
}

const hasExtensionHostPermissionRequest = (): boolean => {
  try {
    return Boolean(
      (globalThis as { chrome?: { permissions?: { request?: unknown } } }).chrome
        ?.permissions?.request
    )
  } catch {
    return false
  }
}

const isBrowserAccessBlocked = (lastError: string | null | undefined): boolean => {
  const text = String(lastError || "").trim()
  if (!text) return false
  return BROWSER_ACCESS_BLOCK_PATTERNS.some((pattern) => pattern.test(text))
}

const StatusIcon = ({ status }: { status: DiagnosticStatus }) => {
  if (status === "complete") {
    return <CheckCircle2 className="h-4 w-4 text-success" aria-hidden />
  }
  if (status === "blocked" || status === "missing") {
    return <AlertCircle className="h-4 w-4 text-danger" aria-hidden />
  }
  if (status === "review") {
    return <ShieldAlert className="h-4 w-4 text-warning" aria-hidden />
  }
  return <CircleDashed className="h-4 w-4 text-text-muted" aria-hidden />
}

export function KnowledgeQASetupDiagnostics({
  connection,
  uxState,
  retryCountdownSeconds,
  onOpenSetup,
  onOpenSettings,
  onOpenDiagnostics,
  onRetryConnection,
}: Props) {
  const [hostPermissionNotice, setHostPermissionNotice] = useState<string | null>(null)
  const serverLabel = getServerOriginLabel(connection.serverUrl)
  const hasServerUrl = Boolean(serverLabel)
  const authMissing =
    uxState === "configuring_auth" ||
    uxState === "error_auth" ||
    connection.configStep === "auth" ||
    connection.errorKind === "auth"
  const unreachable =
    uxState === "error_unreachable" || connection.errorKind === "unreachable"
  const browserBlocked = hasServerUrl && isBrowserAccessBlocked(connection.lastError)
  const canRequestHostAccess = hasServerUrl && hasExtensionHostPermissionRequest()

  const checks = useMemo<DiagnosticCheck[]>(() => {
    const serverUrlCheck: DiagnosticCheck = hasServerUrl
      ? {
          id: "server-url",
          label: "Server URL",
          status: "complete",
          description: "Knowledge QA will search against the configured tldw server.",
        }
      : {
          id: "server-url",
          label: "Server URL",
          status: "missing",
          description: "Add a tldw server URL before Knowledge QA can search your library.",
        }

    const credentialCheck: DiagnosticCheck = !hasServerUrl
      ? {
          id: "credentials",
          label: "Credentials",
          status: "waiting",
          description: "Waiting for a server URL before checking credentials.",
        }
      : authMissing
        ? {
            id: "credentials",
            label: "Credentials",
            status: "missing",
            description: "Add the API key or login token for this tldw server.",
          }
        : {
            id: "credentials",
            label: "Credentials",
            status: "complete",
            description: "Credentials are configured for Knowledge QA requests.",
          }

    const browserAccessCheck: DiagnosticCheck = !hasServerUrl
      ? {
          id: "browser-access",
          label: "Browser access",
          status: "waiting",
          description: "Waiting for a server URL before checking browser access.",
        }
      : browserBlocked
        ? {
            id: "browser-access",
            label: "Browser access",
            status: "blocked",
            description:
              "Allowlist this server origin or grant extension host access before retrying.",
          }
        : unreachable
          ? {
              id: "browser-access",
              label: "Browser access",
              status: "review",
              description:
                "If you are using the extension, grant host access for this server. In WebUI, ensure the backend allows this browser origin.",
            }
          : {
              id: "browser-access",
              label: "Browser access",
              status: "complete",
              description: "Browser requests can target the configured tldw server.",
            }

    const backendCheck: DiagnosticCheck = !hasServerUrl
      ? {
          id: "backend",
          label: "Backend reachability",
          status: "waiting",
          description: "Waiting for a server URL before checking backend health.",
        }
      : authMissing
        ? {
            id: "backend",
            label: "Backend reachability",
            status: "waiting",
            description: "Waiting for credentials before checking backend reachability.",
          }
        : unreachable
          ? {
              id: "backend",
              label: "Backend reachability",
              status: "blocked",
              description:
                "Start the tldw server, confirm the URL is correct, then retry Knowledge QA.",
            }
          : connection.isChecking
            ? {
                id: "backend",
                label: "Backend reachability",
                status: "waiting",
                description: "Checking tldw server health...",
              }
            : {
                id: "backend",
                label: "Backend reachability",
                status: "complete",
                description: "The last server health check completed successfully.",
              }

    return [serverUrlCheck, credentialCheck, browserAccessCheck, backendCheck]
  }, [authMissing, browserBlocked, connection.isChecking, hasServerUrl, unreachable])

  const title =
    uxState === "configuring_auth" || uxState === "error_auth"
      ? "Add your credentials to use Knowledge QA"
      : unreachable
        ? "Can't reach your tldw server right now"
        : "Setup Required"

  const message =
    uxState === "configuring_auth" || uxState === "error_auth"
      ? "Your server URL is saved, but Knowledge QA needs valid credentials before it can load."
      : unreachable
        ? "Your server settings are saved, but Knowledge QA cannot reach the tldw server right now."
        : "Complete the server setup to start searching your documents."

  const handleRequestHostAccess = () => {
    const result = requestOptionalHostPermission(
      connection.serverUrl,
      (granted, origin) => {
        setHostPermissionNotice(
          granted
            ? `Host access granted for ${origin}. Retrying connection...`
            : `Host access was not granted for ${origin}.`
        )
        if (granted) {
          onRetryConnection()
        }
      },
      (error) => setHostPermissionNotice(error.message)
    )

    if (!result.supported) {
      setHostPermissionNotice(
        "Host access must be granted from the browser extension permissions screen."
      )
    }
  }

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-8">
      <section
        className="w-full max-w-2xl text-center"
        data-testid="knowledge-setup-diagnostics"
        aria-labelledby="knowledge-setup-title"
      >
        <WifiOff className="w-16 h-16 mx-auto mb-4 text-text-muted" aria-hidden />
        <h2 id="knowledge-setup-title" className="text-xl font-semibold mb-2">
          {title}
        </h2>
        <p className="mx-auto max-w-lg text-text-muted mb-3">{message}</p>
        {serverLabel && (
          <p className="mb-4 text-xs text-text-subtle">
            {`Configured server: ${serverLabel}`}
          </p>
        )}

        <div className="mx-auto mb-4 grid max-w-xl gap-2 text-left">
          {checks.map((check) => (
            <div
              key={check.id}
              data-testid={`knowledge-setup-check-${check.id}`}
              className="rounded-md border border-border bg-surface px-3 py-2"
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <StatusIcon status={check.status} />
                  <span className="font-medium text-text">{check.label}</span>
                </div>
                <span
                  className={`shrink-0 text-xs font-medium ${statusClassName[check.status]}`}
                >
                  {statusLabel[check.status]}
                </span>
              </div>
              <p className="mt-1 text-sm text-text-muted">{check.description}</p>
            </div>
          ))}
        </div>

        {connection.lastError && (
          <p className="mx-auto mb-4 max-w-xl rounded-md border border-border bg-surface2 px-3 py-2 text-left text-xs text-text-muted">
            {`Last error: ${connection.lastError}`}
            {connection.lastStatusCode != null
              ? ` (status ${connection.lastStatusCode})`
              : ""}
          </p>
        )}

        {hostPermissionNotice && (
          <p className="mb-4 text-xs text-text-muted">{hostPermissionNotice}</p>
        )}

        <div className="flex flex-wrap items-center justify-center gap-2">
          {!hasServerUrl ? (
            <button
              type="button"
              onClick={onOpenSetup}
              className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Finish Setup
            </button>
          ) : authMissing ? (
            <button
              type="button"
              onClick={onOpenSettings}
              className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              Open Settings
            </button>
          ) : (
            <button
              type="button"
              onClick={onRetryConnection}
              disabled={connection.isChecking}
              className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {connection.isChecking ? "Checking connection..." : "Retry connection"}
            </button>
          )}

          {canRequestHostAccess && (
            <button
              type="button"
              onClick={handleRequestHostAccess}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
            >
              <KeyRound className="h-4 w-4" aria-hidden />
              Request host access
            </button>
          )}

          <button
            type="button"
            onClick={onOpenSettings}
            className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
          >
            Server settings
          </button>

          <button
            type="button"
            onClick={onOpenDiagnostics}
            className="px-3 py-1.5 rounded-md border border-border bg-surface text-text-subtle hover:bg-hover hover:text-text transition-colors"
          >
            Health & diagnostics
          </button>
        </div>

        {unreachable && (
          <p className="mt-2 text-xs text-text-muted">
            Retrying automatically in {retryCountdownSeconds}s...
          </p>
        )}
      </section>
    </div>
  )
}
