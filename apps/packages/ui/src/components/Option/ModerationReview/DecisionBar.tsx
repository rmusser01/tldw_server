import React from "react"

import type { ModerationDecisionAction } from "@/services/moderation"
import { decisionActionLabel, decisionNeedsConfirmation, decisionRequiresReason } from "./review-utils"

type DecisionBarProps = {
  disabled?: boolean
  deciding?: ModerationDecisionAction | "undo" | null
  onDecision: (action: ModerationDecisionAction, reason?: string) => Promise<void> | void
  undoToken?: string | null
  undoExpiresAt?: string | null
  onUndo?: () => Promise<void> | void
}

const ACTIONS: ModerationDecisionAction[] = ["approve", "block", "redact", "dismiss", "escalate"]

export const DecisionBar: React.FC<DecisionBarProps> = ({
  disabled = false,
  deciding,
  onDecision,
  undoToken,
  undoExpiresAt,
  onUndo
}) => {
  const [reason, setReason] = React.useState("")
  const [validation, setValidation] = React.useState<string | null>(null)
  const undoExpired = React.useMemo(() => {
    if (!undoExpiresAt) {
      return false
    }
    const expires = new Date(undoExpiresAt).getTime()
    return Number.isFinite(expires) && expires < Date.now()
  }, [undoExpiresAt])

  const runDecision = async (action: ModerationDecisionAction) => {
    if (decisionRequiresReason(action) && !reason.trim()) {
      setValidation("Reason required")
      return
    }
    if (decisionNeedsConfirmation(action) && !window.confirm(`${decisionActionLabel(action)} this review item?`)) {
      return
    }
    setValidation(null)
    await onDecision(action, reason.trim() || undefined)
  }

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <label className="grid gap-1 text-xs font-medium text-text-muted">
        Decision reason
        <textarea
          value={reason}
          onChange={(event) => {
            setReason(event.target.value)
            if (validation) {
              setValidation(null)
            }
          }}
          rows={3}
          className="resize-y rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20"
          placeholder="Required for block, redact, and escalate"
        />
      </label>
      {validation && <div className="mt-2 text-sm font-medium text-red-600">{validation}</div>}
      <div className="mt-3 flex flex-wrap gap-2">
        {ACTIONS.map((action) => (
          <button
            key={action}
            type="button"
            onClick={() => void runDecision(action)}
            disabled={disabled || Boolean(deciding)}
            className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text hover:bg-surface3 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {deciding === action ? "Saving" : decisionActionLabel(action)}
          </button>
        ))}
        {undoToken && onUndo && (
          <button
            type="button"
            onClick={() => void onUndo()}
            disabled={Boolean(deciding) || undoExpired}
            title={undoExpired ? "Undo expired" : undoExpiresAt ? `Undo expires ${undoExpiresAt}` : undefined}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-blue-700 hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60 dark:text-blue-300 dark:hover:bg-blue-950/30"
          >
            {deciding === "undo" ? "Undoing" : undoExpired ? "Undo expired" : "Undo decision"}
          </button>
        )}
      </div>
    </div>
  )
}
