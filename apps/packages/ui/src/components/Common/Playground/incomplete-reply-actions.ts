/**
 * Recovery actions for a reply the model cut off at its output limit
 * (UX G06: CM-03). The reply itself is fine up to where it stops, so the
 * actions are to get the rest (Continue), to ask again (Retry), or to pick a
 * model with a larger limit (Switch model).
 */

export type IncompleteReplyAction = {
  id: string
  label: string
  onClick: () => void
  disabledReason?: string | null
}

export const incompleteReplyRecoveryActions = ({
  t,
  disabledReason,
  onRetry,
  onSwitchModel,
  onContinue
}: {
  t: (key: string, fallback: string) => string
  /** Why Retry and Continue cannot run on this reply right now. */
  disabledReason?: string | null
  onRetry: () => void
  onSwitchModel: () => void
  /** Omitted where the reply cannot be continued. */
  onContinue?: () => void
}): IncompleteReplyAction[] => {
  const actions: IncompleteReplyAction[] = []
  if (onContinue)
    actions.push({
      id: "continue",
      label: t("playground:errorRecovery.continueFromPartial", "Continue from partial"),
      disabledReason,
      onClick: onContinue
    })
  actions.push(
    {
      id: "retry",
      label: t("playground:errorRecovery.retrySameModel", "Retry same model"),
      disabledReason,
      onClick: onRetry
    },
    {
      id: "switch",
      label: t("playground:errorRecovery.switchModel", "Switch model"),
      onClick: onSwitchModel
    }
  )
  return actions
}
