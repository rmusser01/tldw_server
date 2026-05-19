import React from "react"
import type { LucideIcon } from "lucide-react"
import { EmptyState } from "@/components/ui/feedback/EmptyState"

type FeatureEmptyStateProps = {
  title: React.ReactNode
  description?: React.ReactNode
  examples?: React.ReactNode[]
  primaryActionLabel?: React.ReactNode
  onPrimaryAction?: () => void
  secondaryActionLabel?: React.ReactNode
  onSecondaryAction?: () => void
  className?: string
  primaryDisabled?: boolean
  secondaryDisabled?: boolean
  /** Optional icon to display above the title for visual interest */
  icon?: LucideIcon
  /** Icon color class (default: text-text-subtle) */
  iconClassName?: string
}

const FeatureEmptyState: React.FC<FeatureEmptyStateProps> = ({
  title,
  description,
  examples,
  primaryActionLabel,
  onPrimaryAction,
  secondaryActionLabel,
  onSecondaryAction,
  className,
  primaryDisabled = false,
  secondaryDisabled = false,
  icon: Icon,
  iconClassName
}) => {
  return (
    <EmptyState
      title={title}
      description={description}
      examples={examples}
      icon={Icon}
      iconClassName={iconClassName}
      size="lg"
      variant="card"
      className={className}
      primaryAction={
        primaryActionLabel
          ? {
              label: primaryActionLabel,
              onClick: onPrimaryAction,
              disabled: primaryDisabled,
              title:
                typeof primaryActionLabel === "string"
                  ? primaryActionLabel
                  : undefined
            }
          : undefined
      }
      secondaryAction={
        secondaryActionLabel
          ? {
              label: secondaryActionLabel,
              onClick: onSecondaryAction,
              disabled: secondaryDisabled,
              title:
                typeof secondaryActionLabel === "string"
                  ? secondaryActionLabel
                  : undefined
            }
          : undefined
      }
    />
  )
}

export default FeatureEmptyState
