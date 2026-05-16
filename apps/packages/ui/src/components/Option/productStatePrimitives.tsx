import type { ReactNode } from "react"

import {
  Alert as DesignSystemAlert,
  Badge as DesignSystemBadge,
  type AlertVariant,
  type BadgeVariant,
} from "@/components/ui/primitives"

type ProductStateAlertProps = {
  type?: AlertVariant
  variant?: AlertVariant
  title?: ReactNode
  message?: ReactNode
  description?: ReactNode
  children?: ReactNode
  showIcon?: boolean
  closable?: boolean
  onClose?: () => void
  className?: string
  "data-testid"?: string
}

type ProductStateBadgeProps = {
  color?: string
  children: ReactNode
  className?: string
  title?: string
}

const badgeVariantByColor: Record<string, BadgeVariant> = {
  blue: "primary",
  cyan: "info",
  default: "secondary",
  geekblue: "primary",
  gold: "warning",
  green: "success",
  magenta: "primary",
  orange: "warning",
  purple: "primary",
  red: "danger",
  volcano: "danger",
}

const resolveBadgeVariant = (color?: string): BadgeVariant =>
  color ? badgeVariantByColor[color] ?? "secondary" : "secondary"

export const ProductStateAlert = ({
  type,
  variant,
  title,
  message,
  description,
  children,
  closable,
  onClose,
  className,
  "data-testid": dataTestId,
}: ProductStateAlertProps) => {
  const resolvedVariant = variant ?? type ?? "info"
  const resolvedTitle = title ?? message
  const resolvedDescription = children ?? description ?? null

  if (resolvedDescription === null || resolvedDescription === "") {
    return (
      <DesignSystemAlert
        variant={resolvedVariant}
        dismissible={closable}
        onDismiss={onClose}
        className={className}
        data-testid={dataTestId}
      >
        {resolvedTitle}
      </DesignSystemAlert>
    )
  }

  return (
    <DesignSystemAlert
      variant={resolvedVariant}
      title={resolvedTitle}
      dismissible={closable}
      onDismiss={onClose}
      className={className}
      data-testid={dataTestId}
    >
      {resolvedDescription}
    </DesignSystemAlert>
  )
}

export const ProductStateBadge = ({
  color,
  children,
  className,
  title,
}: ProductStateBadgeProps) => (
  <DesignSystemBadge
    variant={resolveBadgeVariant(color)}
    className={className}
    title={title}
  >
    {children}
  </DesignSystemBadge>
)
