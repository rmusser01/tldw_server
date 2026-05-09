import React from "react"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"

export interface StatusBadgeProps {
  variant: "demo" | "warning" | "error"
  children: React.ReactNode
}

const VARIANT_STATES: Record<StatusBadgeProps["variant"], DesignSystemStateKey> = {
  demo: "degraded",
  warning: "degraded",
  error: "error",
}

const SEVERITY_BADGE_VARIANTS = {
  success: "success",
  error: "danger",
  warning: "warning",
  info: "info",
  neutral: "secondary",
} satisfies Record<ReturnType<typeof getDesignSystemState>["severity"], BadgeVariant>

const getBadgeVariant = (variant: StatusBadgeProps["variant"]): BadgeVariant => {
  if (variant === "demo") {
    return "demo"
  }

  const state = getDesignSystemState(VARIANT_STATES[variant])
  return SEVERITY_BADGE_VARIANTS[state.severity]
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  variant,
  children
}) => {
  const state = getDesignSystemState(VARIANT_STATES[variant])

  return (
    <Badge
      variant={getBadgeVariant(variant)}
      size="md"
      className="text-[11px]"
      srLabel={state.label}
    >
      {children}
    </Badge>
  )
}

export default StatusBadge
