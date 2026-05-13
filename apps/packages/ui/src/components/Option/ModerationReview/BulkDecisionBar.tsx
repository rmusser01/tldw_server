import React from "react"

import type {
  ModerationDecisionAction,
  ModerationReviewBulkDecisionResponse
} from "@/services/moderation"
import {
  decisionActionLabel,
  decisionNeedsConfirmation,
  decisionRequiresReason
} from "./review-utils"

type BulkDecisionBarProps = {
  selectedCount: number
  deciding?: ModerationDecisionAction | null
  result?: ModerationReviewBulkDecisionResponse | null
  onBulkDecision: (action: ModerationDecisionAction, reason?: string) => Promise<void> | void
  onClearSelection: () => void
}

const ACTIONS: ModerationDecisionAction[] = ["approve", "dismiss", "block", "redact", "escalate"]

export const BulkDecisionBar: React.FC<BulkDecisionBarProps> = ({
  selectedCount,
  deciding = null,
  result,
  onBulkDecision,
  onClearSelection
}) => {
  const [reason, setReason] = React.useState("")
  const [validation, setValidation] = React.useState<string | null>(null)
  const disabled = selectedCount === 0 || Boolean(deciding)

  const runBulkDecision = async (action: ModerationDecisionAction) => {
    if (decisionRequiresReason(action) && !reason.trim()) {
      setValidation("Reason required")
      return
    }
    if (decisionNeedsConfirmation(action) && !window.confirm(`${decisionActionLabel(action)} ${selectedCount} review items?`)) {
      return
    }
    setValidation(null)
    await onBulkDecision(action, reason.trim() || undefined)
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-3" data-testid="moderation-bulk-decision-bar">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-text">{selectedCount} selected</div>
          <label className="mt-2 grid gap-1 text-xs font-medium text-text-muted">
            Bulk decision reason
            <textarea
              value={reason}
              onChange={(event) => {
                setReason(event.target.value)
                if (validation) {
                  setValidation(null)
                }
              }}
              rows={2}
              className="min-w-[260px] resize-y rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
              placeholder="Required for block, redact, and escalate"
            />
          </label>
          {validation && <div className="mt-2 text-sm font-medium text-red-600">{validation}</div>}
        </div>
        <div className="flex flex-wrap gap-2">
          {ACTIONS.map((action) => (
            <button
              key={action}
              type="button"
              onClick={() => void runBulkDecision(action)}
              disabled={disabled}
              className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text hover:bg-surface3 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {deciding === action ? "Saving" : `${decisionActionLabel(action)} selected`}
            </button>
          ))}
          <button
            type="button"
            onClick={onClearSelection}
            disabled={selectedCount === 0}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text-muted hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Clear selection
          </button>
        </div>
      </div>

      {result && (
        <div className="mt-3 rounded-md border border-border bg-surface2 p-3 text-sm" role="status">
          <div className="font-medium text-text">
            {result.ok_count} updated · {result.error_count} failed
          </div>
          {result.error_count > 0 && (
            <ul className="mt-2 space-y-1 text-red-700 dark:text-red-300">
              {result.results
                .filter((entry) => !entry.ok)
                .map((entry) => (
                  <li key={entry.item_id}>
                    {entry.item_id}: {entry.error || "failed"}
                  </li>
                ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
