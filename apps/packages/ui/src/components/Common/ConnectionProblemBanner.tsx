import React from "react"
import { RecoveryCallout, type StateAction } from "@/components/ui/state"

type ConnectionProblemBannerProps = {
  badgeLabel?: React.ReactNode
  title: React.ReactNode
  description?: React.ReactNode
  examples?: React.ReactNode[]
  primaryActionLabel?: React.ReactNode
  onPrimaryAction?: () => void
  retryActionLabel?: React.ReactNode
  onRetry?: () => void
  retryDisabled?: boolean
  secondaryActionLabel?: React.ReactNode
  onSecondaryAction?: () => void
  primaryDisabled?: boolean
  secondaryDisabled?: boolean
  className?: string
}

const ConnectionProblemBanner: React.FC<ConnectionProblemBannerProps> = ({
  badgeLabel,
  title,
  description,
  examples,
  primaryActionLabel,
  onPrimaryAction,
  retryActionLabel,
  onRetry,
  retryDisabled,
  secondaryActionLabel,
  onSecondaryAction,
  primaryDisabled,
  secondaryDisabled,
  className
}) => {
  const composedTitle = badgeLabel ? (
    <span className="inline-flex items-center gap-2">
      <span className="rounded-full bg-warn/10 px-2 py-0.5 text-[11px] font-medium text-warn">
        {badgeLabel}
      </span>
      <span>{title}</span>
    </span>
  ) : (
    title
  )
  const primaryAction: StateAction | undefined = primaryActionLabel
    ? {
        label: primaryActionLabel,
        onClick: onPrimaryAction,
        disabled: primaryDisabled,
        ariaLabel:
          typeof primaryActionLabel === "string" ? primaryActionLabel : undefined
      }
    : undefined
  const secondaryActions: StateAction[] = []

  if (retryActionLabel && onRetry) {
    secondaryActions.push({
      label: retryActionLabel,
      onClick: onRetry,
      disabled: retryDisabled,
      ariaLabel:
        typeof retryActionLabel === "string" ? retryActionLabel : undefined
    })
  }

  if (secondaryActionLabel) {
    secondaryActions.push({
      label: secondaryActionLabel,
      onClick: onSecondaryAction,
      disabled: secondaryDisabled,
      ariaLabel:
        typeof secondaryActionLabel === "string"
          ? secondaryActionLabel
          : undefined
    })
  }

  return (
    <RecoveryCallout
      state="unavailable"
      title={composedTitle}
      message={description}
      primaryAction={primaryAction}
      secondaryActions={secondaryActions}
      className={className}
    >
      {examples && examples.length > 0 ? (
        <ul className="list-disc space-y-1 pl-4 text-sm text-text-muted">
          {examples.map((example, index) => (
            <li key={index}>{example}</li>
          ))}
        </ul>
      ) : null}
    </RecoveryCallout>
  )
}

export default ConnectionProblemBanner
