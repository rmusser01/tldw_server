import React, { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { ExternalLink, ShieldAlert, ShieldCheck } from "lucide-react"
import {
  tldwClient,
  type SandboxWorkspaceDiagnosticsResponse,
  type SandboxWorkspaceDiagnosticsRunSummary
} from "@/services/tldw/TldwApiClient"

type WorkspaceSandboxDiagnosticsPanelProps = {
  workspaceId: string
}

type RequestState =
  | { status: "loading"; data: null; error: null }
  | { status: "ready"; data: SandboxWorkspaceDiagnosticsResponse; error: null }
  | { status: "error"; data: null; error: string }

const stateLabel = (state: string | null | undefined): string => {
  if (!state) return "Unknown"
  return state
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

const errorCopy = (
  message: string,
  t: (key: string, fallback: string) => string
): string => {
  if (/403|forbidden|permission/i.test(message)) {
    return t(
      "playground:workspace.sandboxDiagnosticsForbidden",
      "You do not have permission to view sandbox diagnostics for this workspace."
    )
  }
  return t(
    "playground:workspace.sandboxDiagnosticsUnavailable",
    "Sandbox diagnostics are unavailable right now. Workspace sources and chat are unaffected."
  )
}

const safeDiagnosticsHref = (value: string | null | undefined): string | null => {
  const href = value?.trim()
  if (!href) return null
  if (href.startsWith("/") && !href.startsWith("//")) return href
  return null
}

const RunRow: React.FC<{ run: SandboxWorkspaceDiagnosticsRunSummary }> = ({
  run
}) => (
  <li className="rounded border border-border/70 bg-surface px-2.5 py-2">
    <div className="flex min-w-0 flex-wrap items-center gap-2">
      <span className="font-mono text-[11px] text-text">{run.id}</span>
      <span className="rounded bg-surface2 px-1.5 py-0.5 text-[11px] font-medium text-text-muted">
        {stateLabel(run.phase)}
      </span>
      {run.status_reason_code && (
        <span className="rounded bg-surface2 px-1.5 py-0.5 text-[11px] text-text-subtle">
          {stateLabel(run.status_reason_code)}
        </span>
      )}
    </div>
    <p className="mt-1 truncate text-[11px] text-text-subtle">
      {run.runtime || "runtime unknown"}
      {run.workspace_id ? ` - ${run.workspace_id}` : ""}
    </p>
  </li>
)

export const WorkspaceSandboxDiagnosticsPanel: React.FC<
  WorkspaceSandboxDiagnosticsPanelProps
> = ({ workspaceId }) => {
  const { t } = useTranslation(["playground", "common"])
  const [state, setState] = useState<RequestState>({
    status: "loading",
    data: null,
    error: null
  })

  useEffect(() => {
    let cancelled = false
    setState({ status: "loading", data: null, error: null })
    tldwClient
      .getSandboxWorkspaceDiagnostics(workspaceId, {
        sourceLabel: "research_workspace",
        limit: 10
      })
      .then((data) => {
        if (!cancelled) {
          setState({ status: "ready", data, error: null })
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : String(error)
          setState({ status: "error", data: null, error: message })
        }
      })
    return () => {
      cancelled = true
    }
  }, [workspaceId])

  const runtimeConfigHref =
    state.status === "ready"
      ? safeDiagnosticsHref(state.data.links.runtime_config)
      : null
  const sandboxDiagnosticsLabel = t(
    "playground:workspace.sandboxDiagnostics",
    "Sandbox diagnostics"
  )

  return (
    <section
      data-testid="workspace-sandbox-diagnostics-panel"
      className="rounded-md border border-border/70 bg-surface2/45 p-3 text-xs text-text"
      aria-label={sandboxDiagnosticsLabel}
    >
      <div className="mb-2 flex min-w-0 items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          {state.status === "ready" && state.data.admission.state === "available" ? (
            <ShieldCheck className="h-4 w-4 shrink-0 text-success" aria-hidden="true" />
          ) : (
            <ShieldAlert className="h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
          )}
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-text">
              {sandboxDiagnosticsLabel}
            </h3>
            <p className="truncate text-[11px] text-text-subtle">{workspaceId}</p>
          </div>
        </div>
      </div>

      {state.status === "loading" && (
        <p className="text-[11px] text-text-muted">
          {t(
            "playground:workspace.sandboxDiagnosticsLoading",
            "Loading sandbox diagnostics for this workspace."
          )}
        </p>
      )}

      {state.status === "error" && (
        <p className="text-[11px] text-error">{errorCopy(state.error, t)}</p>
      )}

      {state.status === "ready" && (
        <div className="space-y-3">
          <div className="grid gap-2 sm:grid-cols-2">
            <div className="rounded border border-border/70 bg-surface px-2.5 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-text">
                  {t("playground:workspace.sandboxRuntime", "Runtime")}
                </span>
                <span className="rounded bg-surface2 px-1.5 py-0.5 text-[11px] text-text-muted">
                  {stateLabel(state.data.runtime.state)}
                </span>
              </div>
              <p className="mt-1 text-[11px] leading-4 text-text-muted">
                {state.data.runtime.message}
              </p>
            </div>
            <div className="rounded border border-border/70 bg-surface px-2.5 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-text">
                  {t("playground:workspace.sandboxAdmission", "Admission")}
                </span>
                <span className="rounded bg-surface2 px-1.5 py-0.5 text-[11px] text-text-muted">
                  {stateLabel(state.data.admission.state)}
                </span>
              </div>
              <p className="mt-1 text-[11px] leading-4 text-text-muted">
                {state.data.admission.message}
              </p>
            </div>
          </div>

          <div>
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <span className="font-semibold text-text">
                {t(
                  "playground:workspace.sandboxRecentRuns",
                  "Recent workspace runs"
                )}
              </span>
              {runtimeConfigHref && (
                <a
                  href={runtimeConfigHref}
                  className="inline-flex shrink-0 items-center gap-1 rounded border border-border px-1.5 py-0.5 text-[11px] font-semibold text-text-muted hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current"
                >
                  {t(
                    "playground:workspace.sandboxRuntimeConfig",
                    "Runtime config"
                  )}
                  <ExternalLink className="h-3 w-3" aria-hidden="true" />
                </a>
              )}
            </div>
            {state.data.runs.items.length === 0 ? (
              <p className="rounded border border-border/70 bg-surface px-2.5 py-2 text-[11px] text-text-muted">
                {t(
                  "playground:workspace.sandboxNoRuns",
                  "No sandbox runs are linked to this workspace yet."
                )}
              </p>
            ) : (
              <ul className="grid gap-1.5">
                {state.data.runs.items.map((run) => (
                  <RunRow key={run.id} run={run} />
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
