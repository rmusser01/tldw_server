import type { StagedWorkspaceSource } from "./types"

export type ContextStagingCardProps = {
  sources: StagedWorkspaceSource[]
  isSending?: boolean
  canSend?: boolean
  onClear: () => void
  onInsert: () => void
  onSend: () => void
}

const actionButtonClass =
  "inline-flex min-h-[28px] items-center justify-center rounded-md border border-border px-2.5 py-1 text-xs font-medium text-text transition-colors hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

const primaryButtonClass =
  "inline-flex min-h-[28px] items-center justify-center rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50"

export const ContextStagingCard = ({
  sources,
  isSending = false,
  canSend = true,
  onClear,
  onInsert,
  onSend
}: ContextStagingCardProps) => {
  const hasSources = sources.length > 0
  const sendDisabled = isSending || !canSend || !hasSources

  return (
    <section
      aria-label="Staged context"
      className="rounded-lg border border-border bg-surface p-3 text-sm text-text"
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <h2 className="text-sm font-semibold">
            {isSending ? "Sending staged context" : "Context staged - not sent"}
          </h2>
          <p className="text-xs text-text-muted">
            {isSending
              ? "Sending with staged context"
              : hasSources
                ? `${sources.length} source${sources.length === 1 ? "" : "s"} staged`
                : "No context staged"}
          </p>
        </div>

        {hasSources ? (
          <ul className="space-y-2">
            {sources.map((source) => (
              <li
                key={source.sourceId}
                className="flex min-w-0 flex-col gap-1 rounded-md border border-border bg-surface2/50 px-2 py-1.5"
              >
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                  <span className="min-w-0 break-words font-medium text-text">
                    {source.title}
                  </span>
                  <span className="rounded-full border border-border bg-surface px-2 py-0.5 text-[11px] font-medium text-text-muted">
                    {source.availability}
                  </span>
                </div>
                <div className="flex min-w-0 flex-wrap gap-x-2 gap-y-1 text-xs text-text-muted">
                  <span className="min-w-0 break-words">{source.scopeLabel}</span>
                  <span aria-hidden="true">/</span>
                  <span>{source.type}</span>
                </div>
                {source.statusMessage ? (
                  <p className="min-w-0 break-words text-xs text-text-muted">
                    {source.statusMessage}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="rounded-md border border-dashed border-border bg-surface2/40 px-2 py-1.5 text-xs text-text-muted">
            Stage sources from the workspace before sending with staged context.
          </p>
        )}

        <div className="flex flex-wrap justify-end gap-2">
          <button
            type="button"
            className={actionButtonClass}
            onClick={onClear}
            aria-label="Clear staged context"
          >
            Clear staged context
          </button>
          <button
            type="button"
            className={actionButtonClass}
            onClick={onInsert}
            aria-label="Insert context summary"
          >
            Insert context summary
          </button>
          <button
            type="button"
            className={primaryButtonClass}
            onClick={onSend}
            disabled={sendDisabled}
            aria-label="Send with staged context"
          >
            Send with staged context
          </button>
        </div>
      </div>
    </section>
  )
}
