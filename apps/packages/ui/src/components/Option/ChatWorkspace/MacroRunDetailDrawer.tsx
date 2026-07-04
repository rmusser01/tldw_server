import React from "react"

import {
  getChatMacroRun,
  type ChatMacroRunDetailResponse
} from "@/services/chat-macros"

export interface MacroRunDetailDrawerProps {
  runId: string | null
  open: boolean
  onClose: () => void
}

const branchTitle = (
  branch: ChatMacroRunDetailResponse["branches"][number]
): string => branch.label || branch.output_name || branch.step_id || branch.branch_id

export const MacroRunDetailDrawer = ({
  runId,
  open,
  onClose
}: MacroRunDetailDrawerProps) => {
  const [detail, setDetail] = React.useState<ChatMacroRunDetailResponse | null>(null)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (!open || !runId) return

    let cancelled = false
    setLoading(true)
    setError(null)
    setDetail(null)

    getChatMacroRun(runId)
      .then((response) => {
        if (cancelled) return
        if (!response.ok || !response.data) {
          setError(response.error || `Unable to load macro run (${response.status})`)
          return
        }
        setDetail(response.data)
      })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof Error ? err.message : "Unable to load macro run")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [open, runId])

  if (!open) return null

  return (
    <aside
      className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-l border-border bg-background text-text shadow-xl"
      role="dialog"
      aria-modal="true"
      aria-label="Macro run detail"
    >
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
        <div>
          <h2 className="text-base font-semibold">Macro run detail</h2>
          {runId ? (
            <p className="mt-1 break-all text-xs text-text-muted">Run {runId}</p>
          ) : null}
        </div>
        <button
          type="button"
          className="inline-flex min-h-[32px] items-center rounded-md border border-border px-3 py-1.5 text-sm font-medium text-text transition-colors hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          onClick={onClose}
          aria-label="Close macro run detail"
        >
          Close
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {loading ? (
          <p className="text-sm text-text-muted">Loading macro run</p>
        ) : null}
        {error ? (
          <p className="text-sm font-medium text-danger" role="alert">
            {error}
          </p>
        ) : null}

        {detail ? (
          <div className="space-y-4">
            <section className="rounded-md border border-border bg-surface px-3 py-3">
              <dl className="grid gap-2 text-sm sm:grid-cols-2">
                <div>
                  <dt className="text-xs font-medium text-text-muted">Status</dt>
                  <dd>{detail.run.status}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium text-text-muted">Macro</dt>
                  <dd>/{detail.run.macro_command}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium text-text-muted">Output profile</dt>
                  <dd>{detail.run.output_profile || "default"}</dd>
                </div>
              </dl>
            </section>

            <section>
              <h3 className="text-sm font-semibold">Branches</h3>
              <div className="mt-2 space-y-2">
                {detail.branches.length > 0 ? (
                  detail.branches.map((branch) => (
                    <article
                      key={branch.branch_id}
                      className="rounded-md border border-border bg-surface px-3 py-3"
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <h4 className="text-sm font-semibold">{branchTitle(branch)}</h4>
                        <span className="text-xs text-text-muted">
                          Branch status: {branch.status}
                        </span>
                      </div>
                      {branch.output ? (
                        <p className="mt-2 whitespace-pre-wrap break-words text-sm">
                          {branch.output}
                        </p>
                      ) : null}
                      {branch.error_code || branch.error ? (
                        <p className="mt-2 break-words text-sm text-danger">
                          {branch.error_code || branch.error}
                        </p>
                      ) : null}
                    </article>
                  ))
                ) : (
                  <p className="text-sm text-text-muted">No branch detail yet.</p>
                )}
              </div>
            </section>
          </div>
        ) : null}
      </div>
    </aside>
  )
}
