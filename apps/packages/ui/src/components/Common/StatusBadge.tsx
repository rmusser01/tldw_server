import React from "react"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import {
  Badge,
  getBadgeVariantForDesignSystemSeverity,
  type BadgeVariant
} from "@/components/ui/primitives"

export interface StatusBadgeProps {
  variant: "demo" | "warning" | "error"
  children: React.ReactNode
}

const VARIANT_STATES: Record<StatusBadgeProps["variant"], DesignSystemStateKey> = {
  demo: "degraded",
  warning: "degraded",
  error: "error",
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  variant,
  children
}) => {
  const state = getDesignSystemState(VARIANT_STATES[variant])
  const badgeVariant: BadgeVariant =
    variant === "demo"
      ? "demo"
      : getBadgeVariantForDesignSystemSeverity(state.severity)

  return (
    <Badge
      variant={badgeVariant}
      size="md"
      className="text-[11px]"
      srLabel={state.label}
    >
      {children}
    </Badge>
  )
}

export default StatusBadge
