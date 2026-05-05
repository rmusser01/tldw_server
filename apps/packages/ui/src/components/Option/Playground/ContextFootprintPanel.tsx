import React from "react"

export type ContextFootprintRow = {
  id: string
  label: string
  tokens: number
}

type Props = {
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
  rows: ContextFootprintRow[]
  nonMessageContextPercent: number | null
  showNonMessageContextWarning: boolean
  thresholdPercent: number
  onClearPromptContext: () => void
  onClearPinnedSourceContext: () => void
  onClearHistoryContext: () => void
  onCreateSummaryCheckpoint: () => void
  onReviewCharacterContext: () => void
  onTrimLargestContextContributor: () => void
}

const toText = (value: unknown): string =>
  typeof value === "string" ? value : String(value)

export const ContextFootprintPanel: React.FC<Props> = ({
  t,
  rows,
  nonMessageContextPercent,
  showNonMessageContextWarning,
  thresholdPercent,
  onClearPromptContext,
  onClearPinnedSourceContext,
  onClearHistoryContext,
  onCreateSummaryCheckpoint,
  onReviewCharacterContext,
  onTrimLargestContextContributor
}) => (
  <div className="space-y-2 rounded-md border border-border bg-surface2 p-2">
    <p className="text-xs font-medium text-text-muted">
      {toText(
        t(
          "playground:tokens.contextBreakdownTitle",
          "Context footprint estimate"
        )
      )}
    </p>
    <div className="space-y-1 text-xs text-text-muted">
      {rows
        .filter((entry) => entry.tokens > 0)
        .map((entry) => (
          <div
            key={entry.id}
            className="flex items-center justify-between gap-2 rounded border border-border bg-surface px-2 py-1"
          >
            <span className="truncate">{entry.label}</span>
            <span className="shrink-0 tabular-nums">
              {entry.tokens.toLocaleString()}{" "}
              {toText(t("playground:tokens.tokenUnit", "tokens"))}
            </span>
          </div>
        ))}
      {rows.every((entry) => entry.tokens <= 0) && (
        <p>
          {toText(
            t(
              "playground:tokens.contextBreakdownEmpty",
              "No additional context contributors detected."
            )
          )}
        </p>
      )}
    </div>
    <div className="flex flex-wrap gap-2">
      <button
        type="button"
        onClick={onClearPromptContext}
        className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-subtle hover:bg-surface-hover hover:text-text"
      >
        {toText(t("playground:tokens.clearPromptContext", "Clear prompts"))}
      </button>
      <button
        type="button"
        onClick={onClearPinnedSourceContext}
        className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-subtle hover:bg-surface-hover hover:text-text"
      >
        {toText(
          t("playground:tokens.clearPinnedContext", "Clear pinned sources")
        )}
      </button>
      <button
        type="button"
        onClick={onClearHistoryContext}
        className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-subtle hover:bg-surface-hover hover:text-text"
      >
        {toText(t("playground:tokens.clearHistoryContext", "Clear history"))}
      </button>
      <button
        type="button"
        onClick={onCreateSummaryCheckpoint}
        className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-subtle hover:bg-surface-hover hover:text-text"
      >
        {toText(
          t(
            "playground:tokens.createSummaryCheckpoint",
            "Create checkpoint summary"
          )
        )}
      </button>
      <button
        type="button"
        onClick={onReviewCharacterContext}
        className="rounded border border-border bg-surface px-2 py-0.5 text-[11px] text-text-subtle hover:bg-surface-hover hover:text-text"
      >
        {toText(
          t("playground:tokens.reviewCharacterContext", "Review character")
        )}
      </button>
    </div>
    {typeof nonMessageContextPercent === "number" && (
      <p className="text-[11px] text-text-subtle">
        {toText(
          t("playground:tokens.nonMessageShare", "Non-message context share")
        )}
        :{" "}
        {Math.round(nonMessageContextPercent)}%
      </p>
    )}
    {showNonMessageContextWarning && (
      <div className="flex flex-wrap items-center justify-between gap-2 rounded border border-warn/40 bg-warn/10 px-2 py-1 text-xs text-warn">
        <span>
          {toText(
            t(
              "playground:tokens.contextBreakdownWarning",
              "Non-message context exceeds {{threshold}}% of your context window.",
              { threshold: thresholdPercent } as any
            )
          )}
        </span>
        <button
          type="button"
          onClick={onTrimLargestContextContributor}
          className="rounded border border-warn/40 bg-surface px-2 py-0.5 text-[11px] font-medium text-warn hover:bg-warn/10"
        >
          {toText(
            t(
              "playground:tokens.trimLargestContext",
              "Trim largest contributor"
            )
          )}
        </button>
      </div>
    )}
  </div>
)
